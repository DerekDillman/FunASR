"""Persistence for recording sessions.

Layout on disk (one folder per session under ``data_dir``):

    data/
      20260611-153012-a1b2c3/
        audio.wav        # 16 kHz mono PCM16, written incrementally
        session.json     # metadata + transcript segments + summaries

``session.json`` is rewritten atomically (tmp file + rename) after every
finalized segment, so a crash mid-recording loses at most the segment in
flight.
"""

import json
import os
import threading
import time
import uuid
import wave


class SessionWriter:
    """Incrementally writes one recording session (audio + transcript)."""

    def __init__(self, data_dir: str, sample_rate: int, name: str = ""):
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.id = f"{ts}-{uuid.uuid4().hex[:6]}"
        self.dir = os.path.join(data_dir, self.id)
        os.makedirs(self.dir, exist_ok=True)
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._closed = False

        self._wav = wave.open(os.path.join(self.dir, "audio.wav"), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(sample_rate)
        self._samples_written = 0

        self.meta = {
            "id": self.id,
            "name": name or f"Recording {time.strftime('%Y-%m-%d %H:%M')}",
            "created_at": time.time(),
            "sample_rate": sample_rate,
            "duration_ms": 0,
            "segments": [],
            "summaries": {},
            "finished": False,
        }
        self._flush_meta()

    def write_audio(self, pcm16: bytes):
        with self._lock:
            if self._closed:
                return
            self._wav.writeframes(pcm16)
            self._samples_written += len(pcm16) // 2
            self.meta["duration_ms"] = int(self._samples_written * 1000 / self.sample_rate)

    def add_segment(self, start_ms: int, end_ms: int, text: str):
        with self._lock:
            self.meta["segments"].append(
                {"start_ms": int(start_ms), "end_ms": int(end_ms), "text": text}
            )
            self._flush_meta()

    def finish(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._wav.close()
            self.meta["finished"] = True
            self._flush_meta()

    def _flush_meta(self):
        _atomic_write_json(os.path.join(self.dir, "session.json"), self.meta)


def _atomic_write_json(path: str, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class SessionStore:
    """Read access + mutations for previously recorded sessions."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _meta_path(self, session_id: str) -> str:
        return os.path.join(self.data_dir, session_id, "session.json")

    def list_sessions(self):
        sessions = []
        for entry in os.listdir(self.data_dir):
            meta = self._load(entry)
            if meta is not None:
                sessions.append(
                    {
                        "id": meta["id"],
                        "name": meta.get("name", meta["id"]),
                        "created_at": meta.get("created_at", 0),
                        "duration_ms": meta.get("duration_ms", 0),
                        "num_segments": len(meta.get("segments", [])),
                        "has_summary": bool(meta.get("summaries")),
                        "finished": meta.get("finished", False),
                    }
                )
        sessions.sort(key=lambda s: s["created_at"], reverse=True)
        return sessions

    def get(self, session_id: str):
        meta = self._load(session_id)
        if meta is None:
            raise KeyError(session_id)
        return meta

    def _load(self, session_id: str):
        # Guard against path traversal in ids coming from the URL.
        if not session_id or "/" in session_id or ".." in session_id:
            return None
        path = self._meta_path(session_id)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def audio_path(self, session_id: str) -> str:
        self.get(session_id)  # validates id
        return os.path.join(self.data_dir, session_id, "audio.wav")

    def rename(self, session_id: str, name: str):
        meta = self.get(session_id)
        meta["name"] = name.strip() or meta["name"]
        _atomic_write_json(self._meta_path(session_id), meta)
        return meta

    def set_summary(self, session_id: str, style: str, summary: dict):
        meta = self.get(session_id)
        meta.setdefault("summaries", {})[style] = summary
        _atomic_write_json(self._meta_path(session_id), meta)
        return meta

    def delete(self, session_id: str):
        meta = self.get(session_id)  # validates id
        folder = os.path.join(self.data_dir, session_id)
        for fname in os.listdir(folder):
            os.remove(os.path.join(folder, fname))
        os.rmdir(folder)
        return meta

    @staticmethod
    def transcript_text(meta: dict, with_timestamps: bool = True) -> str:
        lines = []
        for seg in meta.get("segments", []):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            if with_timestamps:
                lines.append(f"[{_fmt_ms(seg['start_ms'])} - {_fmt_ms(seg['end_ms'])}] {text}")
            else:
                lines.append(text)
        return "\n".join(lines)


def _fmt_ms(ms: int) -> str:
    s = int(ms) // 1000
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
