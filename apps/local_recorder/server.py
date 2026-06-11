"""Local recorder & realtime transcription server.

Run on your laptop (CPU only) and open http://localhost:8765 in a browser:

    python server.py                 # real FunASR models (downloads on first run)
    python server.py --mock          # no ML deps; for trying out the UI
    python server.py --no-partials   # skip the streaming model (less CPU)

WebSocket protocol (/ws/transcribe):
    client -> {"type": "start", "name": "..."}        JSON, once
    client -> <binary PCM16 mono @ 16 kHz>            repeatedly
    client -> {"type": "stop"}                        JSON, once
    server -> {"type": "ready", "session_id": ...}
    server -> {"type": "partial", "text": ...}
    server -> {"type": "segment", "start_ms": ..., "end_ms": ..., "text": ...}
    server -> {"type": "stopped", "session_id": ...}
    server -> {"type": "error", "message": ...}
"""

import argparse
import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from config import Config
from engine import create_engine
from storage import SessionStore, SessionWriter
from summarizer import Summarizer

import os

cfg = Config()
engine = None
store: SessionStore = None
summarizer: Summarizer = None
# One inference call at a time keeps a no-GPU laptop responsive; bump
# max_workers if you have many cores and want concurrent sessions.
EXECUTOR = ThreadPoolExecutor(max_workers=2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, store, summarizer
    engine = create_engine(cfg)
    store = SessionStore(cfg.data_dir)
    summarizer = Summarizer(cfg)
    threading.Thread(target=_load_models, daemon=True).start()
    yield
    EXECUTOR.shutdown(wait=False, cancel_futures=True)


def _load_models():
    print(f"[engine] loading models ({engine.info()['engine']}) ...")
    try:
        engine.load()
        print("[engine] models loaded")
    except Exception as e:
        print(f"[engine] FAILED to load models: {e}")


app = FastAPI(title="Local Recorder", lifespan=lifespan)


# ---------------------------------------------------------------------------
# REST API
# ---------------------------------------------------------------------------


@app.get("/api/status")
async def api_status():
    return {
        "asr": engine.info(),
        "summarizer": await asyncio.get_running_loop().run_in_executor(None, summarizer.status),
        "sample_rate": cfg.sample_rate,
    }


@app.get("/api/sessions")
async def api_sessions():
    return store.list_sessions()


@app.get("/api/sessions/{session_id}")
async def api_session(session_id: str):
    try:
        return store.get(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")


@app.delete("/api/sessions/{session_id}")
async def api_delete(session_id: str):
    try:
        store.delete(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")
    return {"ok": True}


@app.patch("/api/sessions/{session_id}")
async def api_rename(session_id: str, body: dict):
    try:
        return store.rename(session_id, str(body.get("name", "")))
    except KeyError:
        raise HTTPException(404, "session not found")


@app.get("/api/sessions/{session_id}/audio")
async def api_audio(session_id: str):
    try:
        path = store.audio_path(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")
    if not os.path.isfile(path):
        raise HTTPException(404, "audio not found")
    return FileResponse(path, media_type="audio/wav", filename=f"{session_id}.wav")


@app.get("/api/sessions/{session_id}/transcript.md")
async def api_transcript(session_id: str):
    try:
        meta = store.get(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")
    body = f"# {meta.get('name', session_id)}\n\n{SessionStore.transcript_text(meta)}\n"
    for style, s in (meta.get("summaries") or {}).items():
        body += f"\n---\n\n# Summary ({style})\n\n{s.get('text', '')}\n"
    return PlainTextResponse(body, media_type="text/markdown")


@app.post("/api/sessions/{session_id}/summarize")
async def api_summarize(session_id: str, body: dict | None = None):
    style = (body or {}).get("style", "meeting")
    try:
        meta = store.get(session_id)
    except KeyError:
        raise HTTPException(404, "session not found")
    transcript = SessionStore.transcript_text(meta)
    if not transcript.strip():
        raise HTTPException(400, "session has no transcript to summarize")
    loop = asyncio.get_running_loop()
    summary = await loop.run_in_executor(EXECUTOR, summarizer.summarize, transcript, style)
    store.set_summary(session_id, summary["style"], summary)
    return summary


# ---------------------------------------------------------------------------
# WebSocket: live recording + transcription
# ---------------------------------------------------------------------------


@app.websocket("/ws/transcribe")
async def ws_transcribe(ws: WebSocket):
    await ws.accept()

    info = engine.info()
    if not info["loaded"]:
        msg = info.get("load_error") or "models are still loading, try again shortly"
        await ws.send_json({"type": "error", "message": msg})
        await ws.close()
        return

    # First message must be the start config.
    try:
        start = json.loads(await ws.receive_text())
        assert start.get("type") == "start"
    except Exception:
        await ws.send_json({"type": "error", "message": "expected {'type':'start'} first"})
        await ws.close()
        return

    stream = engine.create_stream()
    writer = SessionWriter(cfg.data_dir, cfg.sample_rate, name=start.get("name", ""))
    await ws.send_json({"type": "ready", "session_id": writer.id})

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    send_lock = asyncio.Lock()

    async def send_events(events):
        for ev in events:
            if ev["type"] == "segment":
                writer.add_segment(ev["start_ms"], ev["end_ms"], ev["text"])
            async with send_lock:
                try:
                    await ws.send_json(ev)
                except Exception:
                    return  # client gone; keep transcribing so it gets saved

    async def consumer():
        while True:
            data = await queue.get()
            if data is None:
                events = await loop.run_in_executor(EXECUTOR, stream.finish)
                await send_events(events)
                return
            # Coalesce whatever has queued up while inference was busy, so a
            # slow CPU degrades to bigger batches instead of growing lag.
            parts = [data]
            while not queue.empty():
                nxt = queue.get_nowait()
                if nxt is None:
                    queue.put_nowait(None)
                    break
                parts.append(nxt)
            pcm = b"".join(parts)
            writer.write_audio(pcm)
            events = await loop.run_in_executor(EXECUTOR, stream.process, pcm)
            await send_events(events)

    consumer_task = asyncio.create_task(consumer())
    stopped_cleanly = False
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await queue.put(msg["bytes"])
            elif msg.get("text"):
                try:
                    cmd = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                if cmd.get("type") == "stop":
                    stopped_cleanly = True
                    break
    except WebSocketDisconnect:
        pass
    finally:
        # Always flush + persist, even if the tab crashed mid-recording.
        await queue.put(None)
        try:
            await consumer_task
        except Exception as e:
            print(f"[ws] consumer error: {e}")
        writer.finish()
        if stopped_cleanly:
            try:
                async with send_lock:
                    await ws.send_json({"type": "stopped", "session_id": writer.id})
                await ws.close()
            except Exception:
                pass
        print(f"[ws] session {writer.id} saved ({writer.meta['duration_ms']} ms)")


# Static frontend (registered last so /api and /ws win).
app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True))


def main():
    parser = argparse.ArgumentParser(description="Local recording & transcription app")
    parser.add_argument("--host", default=cfg.host)
    parser.add_argument("--port", type=int, default=cfg.port)
    parser.add_argument("--data-dir", default=cfg.data_dir)
    parser.add_argument("--mock", action="store_true", default=cfg.mock,
                        help="run without ML models (UI/dev mode)")
    parser.add_argument("--no-partials", action="store_true",
                        help="disable the streaming model; finals only (less CPU)")
    parser.add_argument("--offline-model", default=cfg.offline_model)
    parser.add_argument("--ollama-model", default=cfg.ollama_model)
    args = parser.parse_args()

    cfg.host, cfg.port, cfg.data_dir = args.host, args.port, args.data_dir
    cfg.mock = args.mock
    cfg.offline_model = args.offline_model
    cfg.ollama_model = args.ollama_model
    if args.no_partials:
        cfg.enable_partials = False

    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
