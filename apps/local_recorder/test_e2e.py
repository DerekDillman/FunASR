"""End-to-end smoke test against a running server (mock or real engine).

    python server.py --mock --port 8765 &
    python test_e2e.py
"""

import asyncio
import json
import sys

import numpy as np
import urllib.request
import websockets

BASE = "http://127.0.0.1:8765"
WS = "ws://127.0.0.1:8765/ws/transcribe"
SR = 16000


def http(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{BASE}{path}", data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        ct = resp.headers.get("Content-Type", "")
        raw = resp.read()
        return json.loads(raw) if "json" in ct else raw


def make_audio():
    """2s silence, 3s 'speech' (loud noise), 1.5s silence, 2s speech, 1s silence."""
    rng = np.random.default_rng(0)

    def speech(sec):
        return (rng.normal(0, 4000, int(SR * sec))).clip(-32767, 32767).astype(np.int16)

    def silence(sec):
        return (rng.normal(0, 30, int(SR * sec))).astype(np.int16)

    return np.concatenate([silence(2), speech(3), silence(1.5), speech(2), silence(1)])


async def main():
    # Wait for models.
    for _ in range(60):
        st = http("GET", "/api/status")
        if st["asr"]["loaded"]:
            break
        await asyncio.sleep(1)
    else:
        sys.exit("engine never loaded")
    print("status:", json.dumps(st["asr"]))

    audio = make_audio()
    events = []
    async with websockets.connect(WS) as ws:
        await ws.send(json.dumps({"type": "start", "name": "e2e test"}))
        ready = json.loads(await ws.recv())
        assert ready["type"] == "ready", ready
        session_id = ready["session_id"]
        print("session:", session_id)

        chunk = int(SR * 0.12)  # 120 ms, like the browser worklet

        async def reader():
            try:
                async for raw in ws:
                    ev = json.loads(raw)
                    events.append(ev)
                    if ev["type"] == "stopped":
                        return
            except websockets.ConnectionClosed:
                pass

        reader_task = asyncio.create_task(reader())
        for i in range(0, len(audio), chunk):
            await ws.send(audio[i : i + chunk].tobytes())
            await asyncio.sleep(0.01)  # faster than realtime
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.wait_for(reader_task, timeout=60)

    partials = [e for e in events if e["type"] == "partial" and e["text"]]
    segments = [e for e in events if e["type"] == "segment"]
    print(f"events: {len(partials)} partials, {len(segments)} segments")
    for s in segments:
        print(f"  [{s['start_ms']}-{s['end_ms']}ms] {s['text']}")
    assert partials, "no partial events received"
    assert len(segments) >= 1, "no segments produced"

    # REST checks
    meta = http("GET", f"/api/sessions/{session_id}")
    assert meta["finished"] and len(meta["segments"]) == len(segments)
    assert abs(meta["duration_ms"] - len(audio) * 1000 // SR) < 200, meta["duration_ms"]
    sessions = http("GET", "/api/sessions")
    assert any(s["id"] == session_id for s in sessions)

    wav = http("GET", f"/api/sessions/{session_id}/audio")
    assert wav[:4] == b"RIFF" and len(wav) > len(audio), "bad wav"

    md = http("GET", f"/api/sessions/{session_id}/transcript.md").decode()
    assert "e2e test" in md

    http("PATCH", f"/api/sessions/{session_id}", {"name": "renamed e2e"})
    assert http("GET", f"/api/sessions/{session_id}")["name"] == "renamed e2e"

    summary = http("POST", f"/api/sessions/{session_id}/summarize", {"style": "meeting"})
    print("summary method:", summary["method"])
    assert summary["text"]
    meta = http("GET", f"/api/sessions/{session_id}")
    assert "meeting" in meta["summaries"]

    http("DELETE", f"/api/sessions/{session_id}")
    try:
        http("GET", f"/api/sessions/{session_id}")
        sys.exit("delete failed")
    except urllib.error.HTTPError as e:
        assert e.code == 404

    print("ALL E2E CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
