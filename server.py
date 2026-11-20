"""Web server: one shared Mini-LLM mind, watched live by everyone.

Run:  .venv/bin/python server.py          (http://127.0.0.1:8000)
      HOST=0.0.0.0 PORT=8000 ... to let other machines watch.
"""
import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse

from thinker import Thinker

STATIC = Path(__file__).parent / "static"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["thinker"] = Thinker()
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/history")
async def history(before: int | None = None, limit: int = Query(100, ge=1, le=500)):
    """Finished thoughts older than `before` (newest page if omitted), oldest first."""
    return state["thinker"].page(before, limit)


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.get("/api/stream")
async def stream():
    thinker: Thinker = state["thinker"]
    # The mind wakes up when the first person opens the page, then never stops.
    thinker.start(asyncio.get_running_loop())

    async def events():
        queue, snapshot = thinker.subscribe()
        try:
            yield sse(snapshot)
            while thinker.is_subscribed(queue):
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                # Anything stamped at or before the snapshot is already in it.
                if event["type"] != "viewers" and event["seq"] <= snapshot["seq"]:
                    continue
                yield sse(event)
        finally:
            thinker.unsubscribe(queue)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        # Viewers hold their stream open forever; don't wait for them on shutdown.
        timeout_graceful_shutdown=2,
    )
