"""The single, shared mind: one small language model thinking about life forever.

The model (SmolLM2-135M-Instruct) writes the sentences; the thinking is a
scaffold built around it:

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
# Stamped on every thought. Thoughts without it came from the first mind
# (Mini-LLM, 50M parameters) and are kept as history but not used as memory.
MODEL_TAG = "smollm2-135m"
MODEL_LABEL = "SmolLM2 · 135M parameters"

SYSTEM = (
    "You are a small, sincere mind thinking carefully about why life exists and what makes it meaningful. "
    "You reason step by step in plain, clear sentences. Each step is one sentence that builds on the step before it. "
    "You speak only for yourself, in the first person, and never address anyone."
)

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
QUESTION_OPENERS = ["Why does", "Why do we", "What does it mean", "How can", "But why", "If so, what", "What would", "Is"]

# One inquiry = question, then these steps in order ("then" is sometimes skipped).
STEPS = [
    ("perhaps", ["Perhaps", "I think", "One answer is that", "It may be that", "Maybe"]),
    ("because", ["This is because", "The reason is that", "I believe this because"]),
    ("but", ["But", "However,", "And yet", "On the other hand,"]),
    ("then", ["If that is true, then", "This means that", "It follows that"]),
    ("therefore", ["Therefore,", "So", "In the end,", "So the reason for life is"]),
]

TEMPERATURE = 0.7
TOP_P = 0.9
REPETITION_PENALTY = 1.15
CANDIDATES = 3           # drafts weighed for every step
STEP_TOKENS = (10, 60)   # (stop at a sentence end after this many, hard cap)
QUESTION_TOKENS = 32
REFLECT_EVERY = 5        # every n-th inquiry returns to the root question
RECALL = 3               # ...recalling this many earlier conclusions
TOKEN_DELAY = 0.14       # seconds between tokens: a readable, contemplative pace
STEP_PAUSE = 2.0         # silence between steps of one inquiry
INQUIRY_PAUSE = 6.0      # silence between inquiries

_SENTENCE_END = re.compile(r"[.!?][\"”’')\]]*$")


def tidy(text: str) -> str:
    """Normalise stray spacing around punctuation."""
    text = re.sub(r" +([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[]) +", r"\1", text)
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

    def _chat(self, user: str) -> str:
        """The prompt up to the point where the mind starts to speak."""
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )

    @torch.no_grad()
    def _deliberate(self, prompt: str, said: str, seeds: list[str], min_tokens: int, max_tokens: int,
                    question: bool = False):
        """Draft CANDIDATES continuations side by side; return (best, the others).

        `prompt` is the chat prompt, `said` what the mind has already said in
        this chain, and the drafts all continue one opener picked from `seeds`.
        """
        self._set_phase(f"weighing {CANDIDATES} possibilities")
        tok = self.tokenizer
        seed = random.choice(seeds)
        lead = (said + " " if said else "") + seed
        ids = tok(prompt + lead, return_tensors="pt", add_special_tokens=False).input_ids
        rows = [{"ids": [], "frames": [], "logp": 0.0, "text": "", "done": False, "complete": False}
                for _ in range(CANDIDATES)]
        inputs, past = ids.repeat(CANDIDATES, 1), None
        for _ in range(max_tokens):
            out = self.model(input_ids=inputs, past_key_values=past, use_cache=True)
            past = out.past_key_values
            step = []
            for b, row in enumerate(rows):
                if row["done"]:
                    step.append(tok.eos_token_id)
                    continue
                raw = out.logits[b, -1]
                logits = raw.clone()
                for t in set(row["ids"][-48:]):  # discourage loops
                    logits[t] = logits[t] / REPETITION_PENALTY if logits[t] > 0 else logits[t] * REPETITION_PENALTY
                probs = torch.softmax(logits / TEMPERATURE, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                keep = torch.cumsum(sorted_probs, dim=-1) - sorted_probs < TOP_P
                sorted_probs = sorted_probs * keep
                choice = sorted_idx[torch.multinomial(sorted_probs / sorted_probs.sum(), 1)].item()
                step.append(choice)
                continuation = tok.decode(row["ids"] + [choice], skip_special_tokens=True)
                if choice == tok.eos_token_id or "\n" in continuation.strip():
                    row["done"] = True
                    row["complete"] = not question and bool(_SENTENCE_END.search(row["text"]))
                    continue
                row["ids"].append(choice)
                row["logp"] += torch.log_softmax(raw, dim=-1)[choice].item()
                sep = "" if continuation[:1] in (" ", ",", ".", "'", "’", "?") or not continuation else " "
                row["text"] = tidy(seed + sep + continuation).strip()
                row["frames"].append(row["text"])
                if question:
                    if "?" in row["text"]:
                        row["text"] = row["text"][: row["text"].index("?") + 1]
                        row["frames"][-1] = row["text"]
                        row["done"] = row["complete"] = True
                elif len(row["ids"]) >= min_tokens and _SENTENCE_END.search(row["text"]):
                    row["done"] = row["complete"] = True
            if all(row["done"] for row in rows):
                break
            inputs = torch.tensor(step).unsqueeze(1)

        seen = set(re.findall(r"[a-z]{4,}", said.lower()))
        drafts = [d for d in (self._judge(row, seed, seen, question) for row in rows) if d]
        # Identical drafts are one draft.
        unique = list({d["text"]: d for d in drafts}.values())
        unique.sort(key=lambda d: d["score"], reverse=True)
        return (unique[0], unique[1:]) if unique else (None, [])

    def _judge(self, row: dict, seed: str, seen: set, question: bool):
        """Turn a raw draft into a scored candidate, or None if it is unusable."""
        text, complete, new_ids = row["text"], row["complete"], row["ids"]
        if not new_ids or (question and not complete):
            return None
        frames = row["frames"]
        if not complete:
            # Ran out of budget mid-sentence: fall back to the last full sentence.
            ends = [m.end() for m in re.finditer(r"[.!?][\"”’')\]]*(?= |$)", text)]
            if ends and ends[-1] > len(seed) + 20:
                text, complete = text[: ends[-1]], True
            else:
                text = text.rstrip(" ,;:-–—") + "…"
            frames = frames + [text]

        # How much does this draft depend on the chain before it?  Compare its
        # probability given the whole prompt with its probability given nothing.
        tok, n = self.tokenizer, len(new_ids)
        alone = tok(seed, return_tensors="pt", add_special_tokens=False).input_ids
        full = torch.cat([alone, torch.tensor([new_ids])], dim=1)
        lp = torch.log_softmax(self.model(input_ids=full).logits[0], dim=-1)
        start = alone.shape[1]
        logp_alone = sum(lp[start + i - 1, t].item() for i, t in enumerate(new_ids))
        fluency = row["logp"] / n
        relevance = (row["logp"] - logp_alone) / n
        score = fluency + relevance + 0.02 * min(n, 30) - (0.0 if complete else 2.0)
        # Guard against the usual failures: going in circles, parroting the
        # chain, "concluding" with a question, and slipping into chat manners.
        words = re.findall(r"[a-z]{4,}", text[len(seed):].lower())
        if words:
            score -= 4.0 * (1 - len(set(words)) / len(words))
            score -= 5.0 * max(0.0, sum(w in seen for w in words) / len(words) - 0.6)
        else:
            score -= 2.0
        if not question and "?" in text:
            score -= 2.0
        if question:  # a real question about life, not "What is that?"
            if len(text.split()) < 6:
                score -= 2.0
            if not re.search(r"\b(li[fv]e|living|alive|exist|mean|purpose|death|die|dying)", text.lower()):
                score -= 1.5
        if text.count("“") != text.count("”") or text.count('"') % 2 or ":" in text[-2:]:
            score -= 1.5
        if re.search(r"\b(you|your|let me|here's|here is|step \d|(great|good|interesting) question|as (we|i)('ve| have)? (already )?discussed)\b", text.lower()):
            score -= 1.5
        return {"text": text, "frames": frames, "tokens": n, "score": score, "seed": seed}

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
        return [t for t in self.history if t.get("step") == "therefore" and t.get("model") == MODEL_TAG]

    def inquire(self, delay: float = 0.0, pause: float = 0.0) -> list[dict]:
        """Think one full chain: question -> steps -> conclusion."""
        chain = self.chains + 1
        conclusions = self._conclusions()
        last = conclusions[-1] if conclusions else None
        recalls: list[dict] = []
        note = None
        if self.history and self.history[-1].get("model") != MODEL_TAG:
            note = f"a new mind continues from here · {MODEL_LABEL}"

        # 1. The question.
        question, rejected = None, []
        if last and chain % REFLECT_EVERY == 0 and len(conclusions) >= RECALL:
            # Reflection: back to the root question, with older conclusions in mind.
            recalls = sorted(random.sample(conclusions, RECALL), key=lambda t: t["id"])
            text = random.choice(ROOT_QUESTIONS)
            task = ("Earlier I reached these conclusions:\n" + "\n".join(f"- {t['text']}" for t in recalls)
                    + f"\n\nWith these in mind, think again, step by step, about this question: {text}")
        elif last:
            # The last conclusion raises the next question.
            recalls = [last]
            ask = self._chat(f"I concluded: \"{last['text']}\"\n"
                             "Ask one short, deeper question about life that this conclusion raises.")
            question, rejected = self._deliberate(ask, "", QUESTION_OPENERS, 0, QUESTION_TOKENS, True)
            text = question["text"] if question else random.choice(FALLBACK_QUESTIONS)
            task = f"Earlier I concluded: \"{last['text']}\"\n\nNow think through this question step by step: {text}"
        else:
            text = ROOT_QUESTIONS[0]
            task = f"Think through this question step by step: {text}"
        if question is None:
            words = text.split(" ")
            question = {"text": text, "seed": "", "tokens": 0,
                        "frames": [" ".join(words[: i + 1]) for i in range(len(words))]}
        steps = [self._speak(chain, "question", question, rejected, delay, [t["id"] for t in recalls], note)]

        # 2. The reasoning, each step seeing the whole chain so far.
        prompt = self._chat(task)
        for role, seeds in STEPS:
            if role == "then" and random.random() < 0.5:
                continue
            time.sleep(pause)
            said = " ".join(s["text"] for s in steps[1:])
            best, rejected = self._deliberate(prompt, said, seeds, *STEP_TOKENS)
            if best is None:
                continue
            steps.append(self._speak(chain, role, best, rejected, delay))
        self.chains = chain
        return steps

    def _speak(self, chain, role, draft, rejected, delay, recalls=None, note=None) -> dict:
        """Say the chosen draft aloud, token by token, then commit it to history."""
        thought = {
            "id": (self.history[-1]["id"] + 1) if self.history else 1,
            "ts": time.time(),
            "model": MODEL_TAG,
            "chain": chain,
            "step": role,
            "seed": draft["seed"],
            "text": draft["seed"],
        }
        if recalls:
            thought["recalls"] = recalls
        if note:
            thought["note"] = note
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
