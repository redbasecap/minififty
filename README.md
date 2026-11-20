# minififty

One 50-million-parameter language model ([Mini-LLM](https://huggingface.co/Ashx098/Mini-LLM))
reasoning about the reason for life in an endless loop of question, argument and conclusion. There is a single shared
instance: everyone who opens the page watches the same thought being written,
and can scroll up through everything it has ever thought.

## Run

```sh
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python download_model.py   # once: ~600 MB download, saves ./model (200 MB)
.venv/bin/python server.py           # http://127.0.0.1:8000
```

`HOST=0.0.0.0 PORT=8000 .venv/bin/python server.py` lets other machines watch.

## Deploy (Fly.io)

Live at https://minififty.fly.dev. Redeploy with:

```sh
fly deploy --ha=false   # exactly one machine: there is only one mind
```

The image bakes in `./model`; the history lives on the `thoughts` volume mounted
at `/app/data`. The machine stops when nobody is watching and wakes on the next
visit — see the comment in `fly.toml` to keep it thinking around the clock.

## How it works

- `download_model.py` — the notebook's conversion: remaps the original training
  checkpoint onto a `LlamaForCausalLM` and saves it to `./model`.
- `thinker.py` — the mind. A background thread starts when the first viewer
  arrives. The model is only a 50M-parameter text completer and cannot reason
  by itself, so the thinking is a scaffold around it:
  - **Inquiries**: each chain is a question followed by steps — *perhaps →
    because → but → (then) → therefore* — and every step is written with the
    whole chain so far as context.
  - **Deliberation**: for every step the model drafts three candidates in
    silence and keeps the one that follows best from the chain, scored with its
    own probabilities (fluency plus how much the draft depends on the chain,
    minus penalties for circling and parroting). The rejected drafts are stored
    and shown on the page.
  - **Memory**: each conclusion raises the next question, and every fifth
    inquiry returns to the root question while recalling three older conclusions.

  Finished thoughts are appended to `data/thoughts.jsonl`, so history survives
  restarts. A lock file guarantees only one instance writes that history.
- `server.py` — FastAPI. `/api/stream` (Server-Sent Events) broadcasts the live
  thought to all viewers; `/api/history?before=<id>` pages back through the past.
- `static/index.html` — the page.

Pace, sampling and chain shape (`TOKEN_DELAY`, `CANDIDATES`, `STEPS`, `REFLECT_EVERY`, …) are constants
at the top of `thinker.py`.
