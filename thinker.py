"""The single, shared mind: one Mini-LLM instance thinking about life forever.

The model itself is a 50M-parameter text completer; it cannot reason on its own.
The thinking is a scaffold built around it:

* Inquiries (chains of thought). Each inquiry is a question followed by steps
  -- perhaps / because / but / then / therefore -- and every step is written
  with the whole chain so far in front of the model.
* Deliberation. For every step the model drafts several candidates in silence
  and keeps the one that follows best from the chain (scored by its own
  probabilities); the rest are set aside but remembered.
* Memory. Each conclusion raises the next question, and every few inquiries the
  mind returns to the root question while recalling older conclusions.

Finished thoughts are appended to a JSONL file so the full history survives
restarts; live progress is fanned out to every connected viewer.
"""
import asyncio
import fcntl
import json
import random
import re
import threading
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
HISTORY_FILE = ROOT / "data" / "thoughts.jsonl"

ROOT_QUESTIONS = [
    "What is the reason for life?",
    "What is the meaning of life?",
    "Why are we alive?",
    "What is the purpose of being alive?",
    "Why does life exist at all?",
]
# Used when the model fails to turn a conclusion into a question.
FALLBACK_QUESTIONS = [
    "Why do living things want to go on living?",
    "Does a life need a reason in order to be good?",
    "What would be lost if there were no life at all?",
    "Is the meaning of life found or made?",
    "Why does it matter that life ends?",
    "Can a life have meaning if nobody remembers it?",
    "What do we owe to those who come after us?",
    "Is happiness the reason for life, or only a part of it?",
    "Why do we ask for a reason at all?",
    "What makes one life different from another?",
]
# Openers keep every question, first answer and conclusion tied to life itself,
# otherwise the small model wanders off to whatever it last mentioned.
QUESTION_OPENERS = ["But why does life", "Why do we live", "What does it mean to live", "If so, why are we alive",
                    "How can a life", "And what is life if", "But why do living things", "Then what in life"]

# One inquiry = question, then these steps in order ("then" is sometimes skipped).
STEPS = [
    ("perhaps", ["Perhaps life", "I think we live", "One answer is that life", "It may be that we are alive",
                 "Maybe to be alive is", "Perhaps we exist"]),
    ("because", ["This is because", "The reason is that", "I believe this because"]),
    ("but", ["But", "However,", "And yet", "On the other hand,"]),
    ("then", ["If that is true, then", "This means that", "It follows that"]),
    ("therefore", ["Therefore, life", "So the reason for life is", "In the end, to live is",
                   "And so we are alive because", "Therefore, the meaning of life is"]),
]

TEMPERATURE = 0.8
TOP_P = 0.95
REPETITION_PENALTY = 1.2
CANDIDATES = 3           # drafts weighed for every step
STEP_TOKENS = (12, 55)   # (stop at a sentence end after this many, hard cap)
QUESTION_TOKENS = 30
REFLECT_EVERY = 5        # every n-th inquiry returns to the root question
RECALL = 3               # ...recalling this many earlier conclusions
TOKEN_DELAY = 0.14       # seconds between tokens: a readable, contemplative pace
STEP_PAUSE = 2.0         # silence between steps of one inquiry
INQUIRY_PAUSE = 6.0      # silence between inquiries

_SENTENCE_END = re.compile(r"[.!?][\"”’')\]]*$")


def tidy(text: str) -> str:
    """Undo the WikiText-style spacing the model learned (" , " / " @-@ ")."""
    text = re.sub(r" @([-.,])@ ", r"\1", text)
    text = re.sub(r" +([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[]) +", r"\1", text)
    text = re.sub(r" +('(?:s|t|re|ve|ll|d|m)\b)", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.replace("�", "")


class Thinker:
    def __init__(self, history_file: Path = HISTORY_FILE, model_dir: Path = MODEL_DIR):
        torch.set_num_threads(2)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).eval()
        self.history_file = history_file
        # There must only ever be one mind writing this history.
        history_file.parent.mkdir(parents=True, exist_ok=True)
        self._lockfile = history_file.with_suffix(".lock").open("w")
        try:
            fcntl.flock(self._lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f"Another instance is already thinking into {history_file}") from None
        self.history: list[dict] = []
        if history_file.exists():
            with history_file.open() as f:
                self.history = [json.loads(line) for line in f if line.strip()]
        self.total_tokens = sum(t.get("tokens", 0) for t in self.history)
        self.chains = max((t.get("chain", 0) for t in self.history), default=0)
        self.current: dict | None = None
        self.phase = ""
        self.seq = 0
        self._lock = threading.Lock()
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ---- drafting and weighing ----------------------------------------

    @torch.no_grad()
    def _draft(self, context: str, seed: str, min_tokens: int, max_tokens: int, question: bool = False):
        """Sample one continuation of `seed` given `context`.

        Returns None if unusable, else a dict with the text, the frame-by-frame
        text for replaying it live, and a score for how well it follows.
        """
        tok = self.tokenizer
        ids = tok(context + seed, return_tensors="pt").input_ids
        past, new_ids, frames, logp, text = None, [], [], 0.0, ""
        inputs, complete = ids, False
        for _ in range(max_tokens):
            out = self.model(input_ids=inputs, past_key_values=past, use_cache=True)
            past = out.past_key_values
            raw = out.logits[0, -1]
            logits = raw.clone()
            for t in set(new_ids[-48:]):  # discourage loops
                logits[t] = logits[t] / REPETITION_PENALTY if logits[t] > 0 else logits[t] * REPETITION_PENALTY
            probs = torch.softmax(logits / TEMPERATURE, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < TOP_P
            sorted_probs = sorted_probs * keep
            choice = sorted_idx[torch.multinomial(sorted_probs / sorted_probs.sum(), 1)].item()
            if choice == tok.eos_token_id:
                complete = not question and bool(new_ids) and bool(_SENTENCE_END.search(text))
                break
            continuation = tok.decode(new_ids + [choice], skip_special_tokens=True)
            if "\n" in continuation.strip():  # the model started a new paragraph
                break
            new_ids.append(choice)
            logp += torch.log_softmax(raw, dim=-1)[choice].item()
            inputs = torch.tensor([[choice]])
            sep = "" if continuation[:1] in (" ", ",", ".", "'", "’", "?") or not continuation else " "
            text = tidy(seed + sep + continuation).strip()
            frames.append(text)
            if question:
                if "?" in text:
                    text = text[: text.index("?") + 1]
                    frames[-1] = text
                    complete = True
                    break
            elif len(new_ids) >= min_tokens and _SENTENCE_END.search(text):
                complete = True
                break
        if not new_ids or (question and not complete):
            return None
        if not complete:
            # Ran out of budget mid-sentence: fall back to the last full sentence.
            ends = [m.end() for m in re.finditer(r"[.!?][\"”’')\]]*(?= |$)", text)]
            if ends and ends[-1] > len(seed) + 20:
                text = text[: ends[-1]]
                complete = True
            else:
                text = text.rstrip(" ,;:") + "…"
            frames.append(text)

        # How much does this draft depend on the chain before it?  Compare its
        # probability given the chain with its probability given nothing.
        n = len(new_ids)
        alone = tok(seed, return_tensors="pt").input_ids
        full = torch.cat([alone, torch.tensor([new_ids])], dim=1)
        lp = torch.log_softmax(self.model(input_ids=full).logits[0], dim=-1)
        start = alone.shape[1]
        logp_alone = sum(lp[start + i - 1, t].item() for i, t in enumerate(new_ids))
        fluency = logp / n
        relevance = (logp - logp_alone) / n
        score = fluency + relevance + 0.02 * min(n, 30) - (0.0 if complete else 1.5)
        # Guard against the model's favourite failures: going in circles,
        # parroting the chain, and "concluding" with a question.
        words = re.findall(r"[a-z]{4,}", text[len(seed):].lower())
        if words:
            score -= 4.0 * (1 - len(set(words)) / len(words))
            seen = set(re.findall(r"[a-z]{4,}", context.lower()))
            score -= 5.0 * max(0.0, sum(w in seen for w in words) / len(words) - 0.6)
        else:
            score -= 2.0
        if not question and text.endswith("?"):
            score -= 2.0
        if text.count("“") != text.count("”") or text.count('"') % 2 or "(" in text or ")" in text:
            score -= 1.5
        return {"text": text, "frames": frames, "tokens": n, "score": score,
                "fluency": fluency, "relevance": relevance}

    def _deliberate(self, context: str, seeds: list[str], min_tokens: int, max_tokens: int, question=False):
        """Draft CANDIDATES continuations and return (best, the others)."""
        self._set_phase(f"weighing {CANDIDATES} possibilities")
        drafts = []
        for _ in range(CANDIDATES):
            seed = random.choice(seeds)
            draft = self._draft(context, seed, min_tokens, max_tokens, question)
            if draft:
                draft["seed"] = seed
                drafts.append(draft)
        drafts.sort(key=lambda d: d["score"], reverse=True)
        return (drafts[0], drafts[1:]) if drafts else (None, [])

    # ---- the endless loop ---------------------------------------------

    def start(self, loop: asyncio.AbstractEventLoop):
        """Begin thinking (idempotent). Called when the first viewer arrives."""
        with self._lock:
            if self._thread is not None:
                return
            self._loop = loop
            self._thread = threading.Thread(target=self._run, name="thinker", daemon=True)
            self._thread.start()

    def _run(self):
        while True:
            try:
                self.inquire(TOKEN_DELAY, STEP_PAUSE)
            except Exception as exc:  # never let the mind die
                print(f"[thinker] error: {exc!r}")
                time.sleep(5)
            self._set_phase("resting before the next question")
            time.sleep(INQUIRY_PAUSE)

    def _conclusions(self) -> list[dict]:
        return [t for t in self.history if t.get("step") == "therefore"]

    def inquire(self, delay: float = 0.0, pause: float = 0.0) -> list[dict]:
        """Think one full chain: question -> steps -> conclusion."""
        chain = self.chains + 1
        conclusions = self._conclusions()
        last = conclusions[-1] if conclusions else None
        recalls: list[dict] = []
        memory = ""

        # 1. The question.
        question, rejected = None, []
        if last and chain % REFLECT_EVERY == 0 and len(conclusions) >= RECALL:
            # Reflection: back to the root question, with older conclusions in mind.
            recalls = sorted(random.sample(conclusions, RECALL), key=lambda t: t["id"])
            memory = " ".join(t["text"] for t in recalls) + "\n\n"
            text = random.choice(ROOT_QUESTIONS)
        elif last:
            # The last conclusion raises the next question.
            memory = last["text"] + "\n\n"
            recalls = [last]
            question, rejected = self._deliberate(last["text"] + " ", QUESTION_OPENERS, 0, QUESTION_TOKENS, True)
            text = question["text"] if question else random.choice(FALLBACK_QUESTIONS)
        else:
            text = ROOT_QUESTIONS[0]
        if question is None:
            words = text.split(" ")
            question = {"text": text, "seed": "", "tokens": 0,
                        "frames": [" ".join(words[: i + 1]) for i in range(len(words))]}
        steps = [self._speak(chain, "question", question, rejected, delay, [t["id"] for t in recalls])]

        # 2. The reasoning, each step seeing the whole chain so far.
        for role, seeds in STEPS:
            if role == "then" and random.random() < 0.5:
                continue
            time.sleep(pause)
            context = memory + " ".join(s["text"] for s in steps) + " "
            best, rejected = self._deliberate(context, seeds, *STEP_TOKENS)
            if best is None:
                continue
            steps.append(self._speak(chain, role, best, rejected, delay))
        self.chains = chain
        return steps

    def _speak(self, chain, role, draft, rejected, delay, recalls=None) -> dict:
        """Say the chosen draft aloud, token by token, then commit it to history."""
        thought = {
            "id": (self.history[-1]["id"] + 1) if self.history else 1,
            "ts": time.time(),
            "chain": chain,
            "step": role,
            "seed": draft["seed"],
            "text": draft["seed"],
        }
        if recalls:
            thought["recalls"] = recalls
        with self._lock:
            self.current = thought
            self.phase = ""
            self._publish({"type": "start", "thought": dict(thought)})
        for frame in draft["frames"]:
            with self._lock:
                thought["text"] = frame
                self._publish({"type": "text", "id": thought["id"], "text": frame})
            time.sleep(delay)
        thought.update(text=draft["text"], tokens=draft["tokens"], done=time.time())
        if rejected:
            thought["rejected"] = [d["text"] for d in rejected]
        with self.history_file.open("a") as f:
            f.write(json.dumps(thought, ensure_ascii=False) + "\n")
        with self._lock:
            self.history.append(thought)
            self.total_tokens += draft["tokens"]
            self.current = None
            if role == "question":
                self.chains = max(self.chains, chain)
            self._publish({"type": "end", "thought": thought, "stats": self._stats()})
        return thought

    def _set_phase(self, label: str):
        with self._lock:
            self.phase = label
            self._publish({"type": "phase", "label": label})

    # ---- fan-out to viewers -------------------------------------------

    def _publish(self, event: dict):
        """Call with the lock held: stamps the event and hands it to the loop."""
        self.seq += 1
        event["seq"] = self.seq
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._fanout, event)

    def _fanout(self, event: dict):
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:  # hopelessly slow viewer: make it reconnect
                self._subscribers.discard(q)

    def _stats(self) -> dict:
        return {
            "thoughts": len(self.history),
            "inquiries": self.chains,
            "tokens": self.total_tokens,
            "since": self.history[0]["ts"] if self.history else None,
        }

    def subscribe(self) -> tuple[asyncio.Queue, dict]:
        """Register a viewer (event-loop thread). Returns its queue and a snapshot."""
        q: asyncio.Queue = asyncio.Queue(maxsize=2000)
        with self._lock:
            snapshot = {
                "type": "snapshot",
                "seq": self.seq,
                "current": dict(self.current) if self.current else None,
                "phase": self.phase,
                "next_id": (self.history[-1]["id"] + 1) if self.history else 1,
                "stats": self._stats(),
            }
            self._subscribers.add(q)
        self.broadcast_viewers()
        return q, snapshot

    def unsubscribe(self, q: asyncio.Queue):
        self._subscribers.discard(q)
        self.broadcast_viewers()

    def is_subscribed(self, q: asyncio.Queue) -> bool:
        return q in self._subscribers

    def broadcast_viewers(self):
        event = {"type": "viewers", "count": len(self._subscribers), "seq": 0}
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def page(self, before: int | None, limit: int) -> list[dict]:
        """Up to `limit` finished thoughts with id < before, oldest first."""
        with self._lock:
            items = self.history if before is None else [t for t in self.history if t["id"] < before]
            return items[-limit:]
