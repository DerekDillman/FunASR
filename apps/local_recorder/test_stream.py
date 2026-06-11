"""Unit tests for the _FunASRStream segmentation logic, using a fake model
backend (no funasr/torch needed):

    python test_stream.py
"""

import numpy as np

from config import Config
from engine import _FunASRStream


class FakeEngine:
    """Stands in for FunASREngine: scripted VAD, recording offline calls."""

    def __init__(self, cfg, vad_script, with_streaming=True):
        self.cfg = cfg
        self.vad_script = list(vad_script)  # one entry (list of [beg,end]) per VAD call
        self._streaming = object() if with_streaming else None
        self.offline_calls = []  # (num_samples,)
        self.vad_calls = 0
        self.streaming_calls = 0

    def vad_generate(self, chunk, cache, is_final, chunk_ms):
        self.vad_calls += 1
        value = self.vad_script.pop(0) if self.vad_script else []
        return [{"value": value}]

    def streaming_generate(self, chunk, cache, is_final):
        self.streaming_calls += 1
        assert len(chunk) > 0
        return [{"text": f"p{self.streaming_calls} "}]

    def offline_transcribe(self, audio_f32):
        self.offline_calls.append(len(audio_f32))
        return f"final-{len(self.offline_calls)}"


def feed(stream, seconds, chunk_ms=120):
    sr = stream.sr
    events = []
    total = int(seconds * sr)
    step = int(chunk_ms * sr / 1000)
    audio = np.zeros(total, dtype=np.int16)
    for i in range(0, total, step):
        events += stream.process(audio[i : i + step].tobytes())
    return events


def test_basic_segmentation():
    cfg = Config()
    # 10 s of audio = 50 VAD calls (200 ms each). Speech at 1000-3000 ms and
    # 5000-6500 ms, reported the way fsmn-vad streams it: [beg,-1] ... [-1,end].
    script = [[] for _ in range(50)]
    script[5] = [[1000, -1]]
    script[15] = [[-1, 3000]]
    script[25] = [[5000, -1]]
    script[33] = [[-1, 6500]]
    eng = FakeEngine(cfg, script)
    stream = _FunASRStream(eng)

    events = feed(stream, 10)
    events += stream.finish()

    segs = [e for e in events if e["type"] == "segment"]
    assert [(s["start_ms"], s["end_ms"]) for s in segs] == [(1000, 3000), (5000, 6500)], segs
    assert [s["text"] for s in segs] == ["final-1", "final-2"]

    # Offline got each segment + padding, sample-accurate.
    pad = int(cfg.segment_pad_ms * 16)
    assert eng.offline_calls[0] == 2000 * 16 + 2 * pad, eng.offline_calls
    assert eng.offline_calls[1] == 1500 * 16 + 2 * pad, eng.offline_calls

    partials = [e for e in events if e["type"] == "partial" and e["text"]]
    assert partials and eng.streaming_calls >= 16  # 10 s / 600 ms chunks
    print("test_basic_segmentation OK")


def test_finish_closes_open_segment():
    cfg = Config()
    script = [[] for _ in range(20)]
    script[5] = [[1000, -1]]  # VAD never reports the end
    eng = FakeEngine(cfg, script)
    stream = _FunASRStream(eng)
    events = feed(stream, 4)
    events += stream.finish()
    segs = [e for e in events if e["type"] == "segment"]
    assert len(segs) == 1 and segs[0]["start_ms"] == 1000 and segs[0]["end_ms"] == 4000, segs
    print("test_finish_closes_open_segment OK")


def test_max_segment_cap():
    cfg = Config()
    cfg.max_segment_ms = 2000
    script = [[] for _ in range(40)]
    script[0] = [[0, -1]]  # continuous speech, VAD never closes
    eng = FakeEngine(cfg, script)
    stream = _FunASRStream(eng)
    events = feed(stream, 8)
    events += stream.finish()
    segs = [e for e in events if e["type"] == "segment"]
    assert len(segs) == 4, segs  # forced cut every 2 s
    assert all(s["end_ms"] - s["start_ms"] == 2000 for s in segs), segs
    print("test_max_segment_cap OK")


def test_no_partials_mode_trims_buffer():
    cfg = Config()
    script = [[] for _ in range(300)]
    eng = FakeEngine(cfg, script, with_streaming=False)
    stream = _FunASRStream(eng)
    feed(stream, 60)  # one minute of silence
    # Rolling buffer must stay bounded (vs 960k samples unbounded).
    assert len(stream.audio) < 5 * stream.sr, len(stream.audio)
    assert eng.streaming_calls == 0
    print("test_no_partials_mode_trims_buffer OK")


def test_partials_mode_buffer_bounded_and_aligned():
    cfg = Config()
    script = [[] for _ in range(300)]
    eng = FakeEngine(cfg, script)
    stream = _FunASRStream(eng)
    feed(stream, 60)
    assert len(stream.audio) < 5 * stream.sr, len(stream.audio)
    # The streaming read head must never be behind the trimmed buffer start.
    assert stream.fed_asr >= stream.trim
    print("test_partials_mode_buffer_bounded_and_aligned OK")


if __name__ == "__main__":
    test_basic_segmentation()
    test_finish_closes_open_segment()
    test_max_segment_cap()
    test_no_partials_mode_trims_buffer()
    test_partials_mode_buffer_bounded_and_aligned()
    print("ALL STREAM TESTS PASSED")
