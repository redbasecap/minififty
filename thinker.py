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
import copy
import fcntl
import json
import os
import random
import re
import threading
import time
import urllib.parse
import urllib.request
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
QUESTION_OPENERS = ["Why does", "Why do we", "What does it mean", "How can", "But why", "What would", "Is", "Can a life"]

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
# Pacing when live. The drafts are shown to viewers while they are written.
WORD_DELAY = 0.14        # per word when a fixed question is spoken
DRAFT_STEP_MIN = 0.09    # minimum seconds per token while drafting, to stay readable
CHOICE_PAUSE = 1.8       # how long the chosen draft is shown next to the rejected ones
STEP_PAUSE = 1.2         # silence between steps of one inquiry
INQUIRY_PAUSE = 6.0      # silence between inquiries

BELIEFS = 5              # conclusions the mind currently holds true
EXPLORE = 0.2            # how often it tries an opener regardless of what it has learned
# Its one tool: looking a topic up. Free, keyless, read-only.
WIKI_API = "https://en.wikipedia.org/w/api.php"
SEARCH_TIMEOUT = 6
SEARCH_EVERY = 2         # look something up in every n-th inquiry

# Words that carry no idea of their own; everything else is a concept in the mind map.
STOPWORDS = set("""
about above after again against allow almost along already also although always among another answer anyone anything
around aspect because been before being believe below between beyond both came cannot come could does doing done down
during each either else even ever every everything find first from further gave give given goes going gone good great
have having hence here however indeed into itself just keep kind know known lead least less like little made make many
maybe mean might more most much must need never next nothing often once only onto other others otherwise ours ourselves
over part perhaps question rather really reason same seem seems shall should simply since some someone something still
such take than that their them themselves then there therefore these they thing this those though through thus together
true truly under unless until upon very want well were what whatever when where whether which while whom whose will
with within without would your yourself hand follow mean means think thought itself aren didn doesn hasn haven wasn
weren couldn shouldn wouldn able also back away sure every bring brings ways lives living live exist exists
finding seen said says seeing looking look looks called
""".split())


def concept(word: str) -> str:
    """Fold simple plurals so 'experiences' and 'experience' are one concept."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    return word[:-1] if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is")) else word


def concepts(text: str) -> set[str]:
    words = set(re.findall(r"[a-z]{4,}", text.lower())) - STOPWORDS
    return {concept(w) for w in words} - STOPWORDS

_SENTENCE_END = re.compile(r"[.!?][\"”’')\]]*$")


def tidy(text: str) -> str:
    """Normalise stray spacing around punctuation."""
    text = re.sub(r" +([,.;:!?%)\]])", r"\1", text)
    text = re.sub(r"([(\[]) +", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.replace("�", "")


class Thinker:
    def __init__(self, history_file: Path = HISTORY_FILE, model_dir: Path = MODEL_DIR):
        torch.set_num_threads(min(2, os.cpu_count() or 1))
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
        self.drafting: dict | None = None   # the drafts being weighed right now
        self._live = False
        self._map_cache: tuple | None = None
        self.seq = 0
        self._lock = threading.Lock()
        self._subscribers: set[asyncio.Queue] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        # What it has learned about its own thinking: opener -> [times used, summed score].
        self._arms: dict[str, list[float]] = {}
        seeds_by_chain: dict[int, list[str]] = {}
        for t in self.history:
            if t.get("model") == MODEL_TAG and t.get("seed") and t.get("step") not in ("search",):
                seeds_by_chain.setdefault(t["chain"], []).append(t["seed"])
        for t in self._conclusions():
            if "score" in t:
                self._credit(seeds_by_chain.get(t["chain"], []), t["score"])
        self._echo = ""            # conclusions the current inquiry started from
        self._kv = None            # attention cache of the prompt read so far...
        self._kv_ids: list[int] = []  # ...and the tokens it covers

    # ---- drafting and weighing ----------------------------------------

    def _chat(self, user: str) -> str:
        """The prompt up to the point where the mind starts to speak."""
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True,
        )

    @torch.no_grad()
    def _deliberate(self, role: str, prompt: str, said: str, seeds: list[str], min_tokens: int, max_tokens: int,
                    question: bool = False):
        """Draft CANDIDATES continuations side by side; return (best, the others).

        `prompt` is the chat prompt, `said` what the mind has already said in
        this chain, and the drafts all continue one opener picked from `seeds`.
        """
        tok = self.tokenizer
        seed = self._pick(seeds)
        with self._lock:
            self.phase = f"weighing {CANDIDATES} possibilities"
            self.drafting = {"step": role, "seed": seed, "texts": [seed] * CANDIDATES, "chosen": None}
            self._publish({"type": "drafts", **self.drafting})
        lead = (said + " " if said else "") + seed
        ids = tok(prompt + lead, add_special_tokens=False).input_ids
        rows = [{"ids": [], "frames": [], "logp": 0.0, "text": "", "done": False, "complete": False}
                for _ in range(CANDIDATES)]
        # Read the prompt once, and only the part not already read for the
        # previous step: within an inquiry the chain just grows at the end.
        common = 0
        while common < min(len(ids), len(self._kv_ids)) - 1 and ids[common] == self._kv_ids[common]:
            common += 1
        if self._kv is not None and common > 0:
            if common < len(self._kv_ids):
                self._kv.crop(common - len(self._kv_ids))  # negative: drop that many from the end
        else:
            self._kv, common = None, 0
        out = self.model(input_ids=torch.tensor([ids[common:]]), past_key_values=self._kv, use_cache=True)
        self._kv, self._kv_ids = out.past_key_values, ids
        # The drafts branch off from that shared reading.
        past = copy.deepcopy(self._kv)
        past.batch_repeat_interleave(CANDIDATES)
        logits_now = out.logits[:, -1].expand(CANDIDATES, -1)
        for _ in range(max_tokens):
            tick = time.time()
            step = []
            for b, row in enumerate(rows):
                if row["done"]:
                    step.append(tok.eos_token_id)
                    continue
                raw = logits_now[b]
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
            with self._lock:  # let everyone watch the drafts grow
                self.drafting["texts"] = [row["text"] or seed for row in rows]
                self._publish({"type": "drafts", **self.drafting})
            if all(row["done"] for row in rows):
                break
            out = self.model(input_ids=torch.tensor(step).unsqueeze(1), past_key_values=past, use_cache=True)
            past, logits_now = out.past_key_values, out.logits[:, -1]
            if self._live:
                time.sleep(max(0.0, DRAFT_STEP_MIN - (time.time() - tick)))

        # Everything already said in this inquiry, plus the conclusions it started from.
        before = re.findall(r"[a-z']+", (self._echo + " " + said).lower())
        seen = ({w for w in before if len(w) >= 4}, {tuple(before[i:i + 4]) for i in range(len(before) - 3)})
        judged = [self._judge(row, seed, seen, question) for row in rows]
        # Identical drafts are one draft.
        unique = list({d["text"]: d for d in reversed(judged) if d}.values())
        unique.sort(key=lambda d: d["score"], reverse=True)
        if not unique:
            return None, []
        with self._lock:  # show which draft won before it becomes the thought
            self.drafting["texts"] = [d["text"] if d else (rows[i]["text"] or seed) for i, d in enumerate(judged)]
            self.drafting["chosen"] = judged.index(unique[0])
            self._publish({"type": "drafts", **self.drafting})
        if self._live:
            time.sleep(CHOICE_PAUSE)
        return unique[0], unique[1:]

    def _judge(self, row: dict, seed: str, seen: tuple[set, set], question: bool):
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
            score -= 6.0 * max(0.0, sum(w in seen[0] for w in words) / len(words) - 0.5)
            said_words = re.findall(r"[a-z']+", text.lower())
            echoes = sum(tuple(said_words[i:i + 4]) in seen[1] for i in range(len(said_words) - 3))
            score -= min(30.0, 3.0 * echoes)  # repeating whole phrases is not a new thought
        else:
            score -= 2.0
        if not question and "?" in text:
            score -= 2.0
        if question and re.search(r"\b(you|your|me|my answer|conclusions?|statement|this question)\b", text.lower()):
            return None  # it is asking the room, or about the task, not about life
        if question:  # a real question about life, not "What is that?"
            if len(text.split()) < 7:
                score -= 2.0
            if re.search(r"\b(conclusion|question|statement)", text.lower()):  # talking about the task
                score -= 2.0
            if not re.search(r"\b(li[fv]e|living|alive|exist|mean|purpose|death|die|dying)", text.lower()):
                score -= 1.5
        if text.count("“") != text.count("”") or text.count('"') % 2 or ":" in text[-2:]:
            score -= 1.5
        if not question and ('"' in text or "“" in text):  # quoting itself instead of thinking
            score -= 1.5
        if re.search(r"\b(you|your|let me|here's|here is|step \d|(great|good|interesting) question|as (we|i)('ve| have)? (already )?discussed)\b", text.lower()):
            score -= 1.5
        return {"text": text, "frames": frames, "tokens": n, "score": score, "seed": seed}

    # ---- learning from itself -----------------------------------------

    def _pick(self, seeds: list[str]) -> str:
        """Choose an opener, preferring the ones whose inquiries scored well."""
        if random.random() < EXPLORE or not self._arms:
            return random.choice(seeds)
        uses = sum(n for n, _ in self._arms.values())
        prior = sum(total for _, total in self._arms.values()) / uses if uses else 0.5

        def value(seed):
            n, total = self._arms.get(seed, (0, 0.0))
            return (total + 2 * prior) / (n + 2) + random.gauss(0, 0.12 / (n + 1) ** 0.5)
        return max(seeds, key=value)

    def _credit(self, seeds: list[str], score: float):
        for seed in seeds:
            arm = self._arms.setdefault(seed, [0, 0.0])
            arm[0] += 1
            arm[1] += score

    @torch.no_grad()
    def _clarity(self, question: str, conclusion: str) -> float:
        """Ask the model itself whether the conclusion answers the question (0..1)."""
        tok = self.tokenizer
        prompt = self._chat(f"Question: {question}\nAnswer: {conclusion}\n\n"
                            "Is this a clear, specific and direct answer to the question? Reply with Yes or No.")
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        logits = self.model(input_ids=ids).logits[0, -1]
        yes = tok("Yes", add_special_tokens=False).input_ids[0]
        no = tok("No", add_special_tokens=False).input_ids[0]
        return torch.softmax(logits[[yes, no]], dim=0)[0].item()

    @staticmethod
    def _overlap(a: str, b: str) -> float:
        ca, cb = concepts(a), concepts(b)
        return len(ca & cb) / len(ca | cb) if ca and cb else 0.0

    def _grade(self, question: str, conclusion: str) -> dict:
        """Score a conclusion: is it a clear answer, and is it a new idea?"""
        earlier = [t["text"] for t in self._conclusions()]
        clarity = self._clarity(question, conclusion)
        novelty = 1 - max((self._overlap(conclusion, e) for e in earlier), default=0.0)
        novelty *= min(1.0, len(concepts(conclusion)) / 5)  # saying almost nothing is not a new idea
        score = round(0.5 * clarity + 0.5 * novelty, 3)
        grade = {"score": score, "clarity": round(clarity, 3), "novelty": round(novelty, 3)}
        # Does it earn a place among the beliefs?
        if any(t["text"] == conclusion for t in self._select(self._conclusions() + [{"text": conclusion, **grade}])):
            grade["learned"] = True
        return grade

    def _select(self, conclusions: list[dict]) -> list[dict]:
        """The best-scored conclusions that are not restatements of each other."""
        chosen: list[dict] = []
        for t in sorted((t for t in conclusions if "score" in t), key=lambda t: t["score"], reverse=True):
            if all(self._overlap(t["text"], c["text"]) < 0.5 for c in chosen):
                chosen.append(t)
            if len(chosen) == BELIEFS:
                break
        return chosen

    def beliefs(self) -> list[dict]:
        return self._select(self._conclusions())

    def learned(self) -> dict:
        """Everything the mind has learned about life and about its own thinking."""
        with self._lock:
            mine = [t for t in self.history if t.get("model") == MODEL_TAG]
        scores = [t["score"] for t in mine if t.get("step") == "therefore" and "score" in t]
        roles = {seed: role for role, seeds in STEPS for seed in seeds}
        roles.update({seed: "question" for seed in QUESTION_OPENERS})
        openers = [{"role": roles[seed], "seed": seed, "uses": n, "mean": round(total / n, 3)}
                   for seed, (n, total) in self._arms.items() if seed in roles and n]
        openers.sort(key=lambda o: o["mean"], reverse=True)
        return {
            "beliefs": [{"id": t["id"], "text": t["text"], "score": t["score"]} for t in self.beliefs()],
            "scored": len(scores),
            "first": round(sum(scores[:10]) / 10, 3) if len(scores) >= 20 else None,
            "recent": round(sum(scores[-10:]) / 10, 3) if len(scores) >= 20 else None,
            "openers": openers,
            "lookups": sum(1 for t in mine if t.get("step") == "search"),
        }

    # ---- its one tool: looking something up ---------------------------

    def _look_up(self, question: str) -> dict | None:
        """Search for the question's key words and fetch a two-sentence summary.

        This is the only thing it can do in the outside world: one read-only
        request to Wikipedia's public search API. It cannot run commands.
        """
        # The query is the question's own content words, in order.
        words = [w for w in re.findall(r"[a-z]{4,}", question.lower()) if w not in STOPWORDS]
        query = " ".join(list(dict.fromkeys(words))[:5]) or "meaning of life"
        self._set_phase(f'search "{query}"')
        if self._live:
            time.sleep(1.5)
        params = urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "search", "gsrsearch": query, "gsrlimit": 6,
            "prop": "extracts", "exintro": 1, "explaintext": 1, "exsentences": 2, "exlimit": 6, "redirects": 1,
        })
        try:
            request = urllib.request.Request(f"{WIKI_API}?{params}", headers={"User-Agent": "small-mind-reader/1.0"})
            with urllib.request.urlopen(request, timeout=SEARCH_TIMEOUT) as response:
                pages = json.load(response).get("query", {}).get("pages", {}).values()
        except Exception as exc:
            print(f"[thinker] lookup failed: {exc!r}")
            return None
        read = {t.get("source") for t in self.history if t.get("step") == "search"}
        for page in sorted(pages, key=lambda p: p.get("index", 99)):
            title, extract = page.get("title", ""), " ".join(page.get("extract", "").split())
            if not title or title in read or len(extract) < 60 or "may refer to" in extract:
                continue  # prefer something it has not read before
            if len(extract) > 360:
                extract = extract[:360].rsplit(" ", 1)[0] + "…"
            return {"query": query, "title": title, "text": extract}
        return None

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
                self.inquire(live=True)
            except Exception as exc:  # never let the mind die
                print(f"[thinker] error: {exc!r}")
                time.sleep(5)
            self._set_phase("resting before the next question")
            time.sleep(INQUIRY_PAUSE)

    def _conclusions(self) -> list[dict]:
        return [t for t in self.history if t.get("step") == "therefore" and t.get("model") == MODEL_TAG]

    def inquire(self, live: bool = False) -> list[dict]:
        """Think one full chain: question -> steps -> conclusion.

        With `live` the pacing pauses apply; without, it runs as fast as it can.
        """
        self._live = live
        delay = WORD_DELAY if live else 0.0
        pause = STEP_PAUSE if live else 0.0
        chain = self.chains + 1
        conclusions = self._conclusions()
        last = conclusions[-1] if conclusions else None
        recalls: list[dict] = []
        note = None
        if self.history and self.history[-1].get("model") != MODEL_TAG:
            note = f"a new mind continues from here · {MODEL_LABEL}"

        # 1. The question.
        self._echo = last["text"] if last else ""
        question, rejected = None, []
        if last and chain % REFLECT_EVERY == 0 and len(conclusions) >= RECALL:
            # Reflection: back to the root question, with older conclusions in mind.
            held = self.beliefs()  # what it holds true, once it has graded enough of its conclusions
            recalls = sorted(held[:RECALL] if len(held) >= RECALL else random.sample(conclusions, RECALL),
                             key=lambda t: t["id"])
            self._echo = " ".join(t["text"] for t in recalls)
            text = random.choice(ROOT_QUESTIONS)
            task = ("Earlier I reached these conclusions:\n" + "\n".join(f"- {t['text']}" for t in recalls)
                    + f"\n\nWith these in mind, think again, step by step, about this question: {text}")
        elif last:
            # The last conclusion raises the next question.
            recalls = [last]
            ask = self._chat(f"I concluded: \"{last['text']}\"\n"
                             "Ask one short, deeper question about life that this conclusion raises.")
            question, rejected = self._deliberate("question", ask, "", QUESTION_OPENERS, 0, QUESTION_TOKENS, True)
            text = question["text"] if question else random.choice(FALLBACK_QUESTIONS)
            task = f"Earlier I concluded: \"{last['text']}\"\n\nNow think through this question step by step: {text}"
            self._echo += " " + " ".join(t["text"] for t in self.beliefs())
        else:
            text = ROOT_QUESTIONS[0]
            task = f"Think through this question step by step: {text}"
        if question is None:
            words = text.split(" ")
            question = {"text": text, "seed": "", "tokens": 0, "spoken": True,
                        "frames": [" ".join(words[: i + 1]) for i in range(len(words))]}
        steps = [self._speak(chain, "question", question, rejected, delay, [t["id"] for t in recalls], note)]

        # 2. Every other inquiry, look something up before thinking.
        time.sleep(pause)
        found = self._look_up(text) if chain % SEARCH_EVERY == 0 and chain % REFLECT_EVERY else None
        if found:
            self._speak(chain, "search", {"text": found["text"], "seed": found["query"], "tokens": 0}, [], delay,
                        extra={"source": found["title"]})
            task += f"\n\nI looked this up first. {found['title']}: {found['text']}"

        # 3. The reasoning, each step seeing the whole chain so far.
        prompt = self._chat(task)
        for role, seeds in STEPS:
            if role == "then" and random.random() < 0.5:
                continue
            time.sleep(pause)
            said = " ".join(s["text"] for s in steps[1:])
            best, rejected = self._deliberate(role, prompt, said, seeds, *STEP_TOKENS)
            if best is None:
                continue
            # 4. Grade the conclusion, and learn from the grade.
            grade = self._grade(text, best["text"]) if role == "therefore" else None
            steps.append(self._speak(chain, role, best, rejected, delay, extra=grade))
            if grade:
                self._credit([s["seed"] for s in steps if s["seed"]], grade["score"])
        self.chains = chain
        return steps

    def _speak(self, chain, role, draft, rejected, delay, recalls=None, note=None, extra=None) -> dict:
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
        if extra:
            thought.update(extra)
        if draft.get("spoken"):
            # A fixed question nobody has seen yet: say it word by word.
            with self._lock:
                self.current = thought
                self.phase = ""
                self.drafting = None
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
            self.drafting = None
            self.phase = ""
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
                "drafting": dict(self.drafting) if self.drafting else None,
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

    def concept_detail(self, word: str) -> dict:
        """Where a concept appears and what it is linked to."""
        with self._lock:
            mine = [t for t in self.history if t.get("model") == MODEL_TAG]
        links: dict[str, int] = {}
        hits = []
        for t in mine:
            found = concepts(t["text"])
            if word in found:
                hits.append({"id": t["id"], "step": t["step"], "text": t["text"]})
                for other in found - {word}:
                    links[other] = links.get(other, 0) + 1
        top = sorted(links, key=lambda w: (-links[w], w))[:10]
        return {"word": word, "count": len(hits), "links": [{"word": w, "weight": links[w]} for w in top],
                "thoughts": hits[-6:]}

    def mind_map(self, max_nodes: int = 34, max_edges: int = 60) -> dict:
        """The concepts this mind keeps returning to and which of them it links.

        A concept is a content word; two concepts are linked each time they
        appear in the same thought. Only the current mind's thoughts count.
        """
        with self._lock:
            if self._map_cache and self._map_cache[0] == (len(self.history), max_nodes):
                return self._map_cache[1]
            texts = [t["text"] for t in self.history if t.get("model") == MODEL_TAG]
            size = len(self.history)
        counts: dict[str, int] = {}
        pairs: dict[tuple[str, str], int] = {}
        for text in texts:
            found = sorted(concepts(text))
            for w in found:
                counts[w] = counts.get(w, 0) + 1
            for i, a in enumerate(found):
                for b in found[i + 1:]:
                    pairs[(a, b)] = pairs.get((a, b), 0) + 1
        top = sorted(counts, key=lambda w: (-counts[w], w))[:max_nodes]
        keep = set(top)
        links = sorted(((n, a, b) for (a, b), n in pairs.items() if a in keep and b in keep), reverse=True)[:max_edges]
        result = {"nodes": [{"word": w, "count": counts[w]} for w in top],
                  "edges": [{"a": a, "b": b, "weight": n} for n, a, b in links]}
        with self._lock:
            self._map_cache = ((size, max_nodes), result)
        return result

    def page(self, before: int | None, limit: int) -> list[dict]:
        """Up to `limit` finished thoughts with id < before, oldest first."""
        with self._lock:
            items = self.history if before is None else [t for t in self.history if t["id"] < before]
            return items[-limit:]
