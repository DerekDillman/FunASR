"""Speech-to-text engines.

``FunASREngine`` runs three CPU-friendly FunASR models:

  * ``fsmn-vad``                — streaming voice-activity detection, used to
                                  cut the audio into utterance segments.
  * ``paraformer-zh-streaming`` — low-latency partial results while you speak.
  * ``SenseVoiceSmall``         — accurate multilingual (zh/en/yue/ja/ko)
                                  decoding of each finished segment, with
                                  punctuation and inverse text normalization.

``MockEngine`` emits fake events from audio energy alone, so the whole app
(mic capture, websocket plumbing, storage, UI) can be exercised without
installing torch/funasr or downloading models.

All ``process``/``finish`` calls are blocking and meant to be run in a worker
thread; per-model locks make sharing one engine across sessions safe.
"""

import threading
import time

import numpy as np

from config import Config


class BaseStream:
    """One live transcription stream. Returns lists of event dicts:

    {"type": "partial", "text": str}
    {"type": "segment", "start_ms": int, "end_ms": int, "text": str}
    """

    def process(self, pcm16: bytes) -> list:
        raise NotImplementedError

    def finish(self) -> list:
        raise NotImplementedError


# --------------------------------------------------------------------------
# Real engine
# --------------------------------------------------------------------------


class FunASREngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.loaded = False
        self.load_error = None
        self._vad = None
        self._streaming = None
        self._offline = None
        self._punc = None
        self._vad_lock = threading.Lock()
        self._streaming_lock = threading.Lock()
        self._offline_lock = threading.Lock()
        self._is_sensevoice = "sensevoice" in cfg.offline_model.lower()

    def load(self):
        """Blocking; call once from a worker thread at startup."""
        try:
            from funasr import AutoModel

            common = dict(device="cpu", ngpu=0, ncpu=self.cfg.ncpu, disable_pbar=True, disable_log=True)

            self._vad = AutoModel(model=self.cfg.vad_model, **common)
            self._offline = AutoModel(model=self.cfg.offline_model, **common)
            if self.cfg.enable_partials:
                self._streaming = AutoModel(model=self.cfg.streaming_model, **common)
            if not self._is_sensevoice:
                # SenseVoice already punctuates; other offline models need help.
                self._punc = AutoModel(model="ct-punc", **common)
            self.loaded = True
        except Exception as e:  # surfaced via /api/status
            self.load_error = f"{type(e).__name__}: {e}"
            raise

    def info(self):
        return {
            "engine": "funasr",
            "loaded": self.loaded,
            "load_error": self.load_error,
            "offline_model": self.cfg.offline_model,
            "streaming_model": self.cfg.streaming_model if self.cfg.enable_partials else None,
            "vad_model": self.cfg.vad_model,
        }

    def create_stream(self) -> BaseStream:
        if not self.loaded:
            raise RuntimeError("models are still loading")
        return _FunASRStream(self)

    # -- model calls (locked: AutoModel.generate is not re-entrant) --

    def vad_generate(self, chunk, cache, is_final, chunk_ms):
        with self._vad_lock:
            return self._vad.generate(
                input=chunk, cache=cache, is_final=is_final, chunk_size=chunk_ms
            )

    def streaming_generate(self, chunk, cache, is_final):
        with self._streaming_lock:
            return self._streaming.generate(
                input=chunk,
                cache=cache,
                is_final=is_final,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1,
            )

    def offline_transcribe(self, audio_f32) -> str:
        with self._offline_lock:
            if self._is_sensevoice:
                res = self._offline.generate(
                    input=audio_f32, cache={}, language=self.cfg.language, use_itn=True
                )
                from funasr.utils.postprocess_utils import rich_transcription_postprocess

                return rich_transcription_postprocess(res[0]["text"]).strip()
            res = self._offline.generate(input=audio_f32)
            text = (res[0].get("text") or "").strip()
            if text and self._punc is not None:
                try:
                    text = self._punc.generate(input=text)[0]["text"]
                except Exception:
                    pass  # punctuation is best-effort
            return text


class _FunASRStream(BaseStream):
    def __init__(self, engine: FunASREngine):
        self.engine = engine
        cfg = engine.cfg
        self.sr = cfg.sample_rate
        self.vad_chunk = int(cfg.vad_chunk_ms * self.sr / 1000)
        self.asr_chunk = int(cfg.chunk_size_ms * self.sr / 1000)
        self.pad = int(cfg.segment_pad_ms * self.sr / 1000)
        self.max_seg = cfg.max_segment_ms

        # Rolling audio buffer; ``trim`` = absolute sample index of buffer[0].
        self.audio = np.empty(0, dtype=np.int16)
        self.trim = 0
        self.fed_vad = 0  # absolute samples already sent to VAD
        self.fed_asr = 0  # absolute samples already sent to streaming ASR

        self.vad_cache = {}
        self.asr_cache = {}
        self.partial = ""
        self.seg_start_ms = -1  # -1 = not inside speech
        self.last_seg_end_ms = 0

    # -- helpers --

    def _abs_len(self):
        return self.trim + len(self.audio)

    def _slice_ms(self, start_ms, end_ms):
        a = max(self.trim, int(start_ms * self.sr / 1000) - self.pad)
        b = min(self._abs_len(), int(end_ms * self.sr / 1000) + self.pad)
        seg = self.audio[a - self.trim : b - self.trim]
        return seg.astype(np.float32) / 32768.0

    def _trim_buffer(self):
        # Keep audio back to the start of the open segment (or the VAD read
        # head when idle) plus padding, so hours-long recordings stay cheap.
        if self.seg_start_ms >= 0:
            keep_from = int(self.seg_start_ms * self.sr / 1000) - self.pad
        else:
            keep_from = self.fed_vad - 2 * self.pad
        if self.engine._streaming is not None:
            # Never trim audio the streaming model hasn't consumed yet.
            keep_from = min(keep_from, self.fed_asr)
        keep_from = max(self.trim, keep_from)
        if keep_from > self.trim:
            self.audio = self.audio[keep_from - self.trim :]
            self.trim = keep_from

    def _emit_segment(self, start_ms, end_ms, events):
        text = self.engine.offline_transcribe(self._slice_ms(start_ms, end_ms))
        self.last_seg_end_ms = end_ms
        if text:
            events.append(
                {"type": "segment", "start_ms": int(start_ms), "end_ms": int(end_ms), "text": text}
            )
        # Partials covered this stretch of audio; start fresh.
        self.partial = ""
        self.asr_cache = {}
        events.append({"type": "partial", "text": ""})

    # -- main entry points --

    def process(self, pcm16: bytes) -> list:
        events = []
        chunk = np.frombuffer(pcm16, dtype=np.int16)
        if len(chunk) == 0:
            return events
        self.audio = np.concatenate([self.audio, chunk])

        self._run_vad(events, final=False)
        self._run_streaming(events, final=False)
        self._trim_buffer()
        return events

    def finish(self) -> list:
        events = []
        self._run_vad(events, final=True)
        self._run_streaming(events, final=True)
        # Close any segment VAD left open.
        if self.seg_start_ms >= 0:
            end_ms = int(self._abs_len() * 1000 / self.sr)
            if end_ms - self.seg_start_ms > 100:
                self._emit_segment(self.seg_start_ms, end_ms, events)
            self.seg_start_ms = -1
        return events

    # -- model feeding --

    def _run_vad(self, events, final: bool):
        while True:
            avail = self._abs_len() - self.fed_vad
            if avail >= self.vad_chunk:
                n = self.vad_chunk
            elif final and avail > 0:
                n = avail
            else:
                break
            a = self.fed_vad - self.trim
            chunk = self.audio[a : a + n].astype(np.float32) / 32768.0
            self.fed_vad += n
            is_final = final and (self._abs_len() - self.fed_vad) == 0
            try:
                res = self.engine.vad_generate(chunk, self.vad_cache, is_final, self.engine.cfg.vad_chunk_ms)
                segments = res[0].get("value", [])
            except Exception:
                segments = []
            for beg, end in segments:
                if beg != -1:
                    self.seg_start_ms = max(beg, self.last_seg_end_ms)
                if end != -1 and self.seg_start_ms >= 0:
                    self._emit_segment(self.seg_start_ms, end, events)
                    self.seg_start_ms = -1
            # Safety net: VAD never closed the segment (continuous speech).
            now_ms = int(self.fed_vad * 1000 / self.sr)
            if self.seg_start_ms >= 0 and now_ms - self.seg_start_ms >= self.max_seg:
                self._emit_segment(self.seg_start_ms, now_ms, events)
                self.seg_start_ms = now_ms

    def _run_streaming(self, events, final: bool):
        if self.engine._streaming is None:
            return
        while True:
            avail = self._abs_len() - self.fed_asr
            if avail >= self.asr_chunk:
                n = self.asr_chunk
            elif final and avail > 0:
                n = avail
            else:
                break
            a = self.fed_asr - self.trim
            chunk = self.audio[a : a + n].astype(np.float32) / 32768.0
            self.fed_asr += n
            is_final = final and (self._abs_len() - self.fed_asr) == 0
            try:
                res = self.engine.streaming_generate(chunk, self.asr_cache, is_final)
                piece = res[0].get("text", "")
            except Exception:
                piece = ""
            if piece:
                self.partial += piece
                events.append({"type": "partial", "text": self.partial})


# --------------------------------------------------------------------------
# Mock engine (no ML dependencies)
# --------------------------------------------------------------------------


class MockEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.loaded = False
        self.load_error = None

    def load(self):
        time.sleep(0.2)  # pretend to load
        self.loaded = True

    def info(self):
        return {"engine": "mock", "loaded": self.loaded, "load_error": None}

    def create_stream(self) -> BaseStream:
        return _MockStream(self.cfg.sample_rate)


class _MockStream(BaseStream):
    """Energy-gated fake transcription: useful for testing the pipeline."""

    THRESHOLD = 300  # int16 RMS
    HANG_MS = 500

    def __init__(self, sr):
        self.sr = sr
        self.pos_ms = 0.0
        self.seg_start_ms = -1
        self.silence_ms = 0.0
        self.seg_count = 0

    def process(self, pcm16: bytes) -> list:
        events = []
        chunk = np.frombuffer(pcm16, dtype=np.int16)
        if len(chunk) == 0:
            return events
        dur_ms = len(chunk) * 1000.0 / self.sr
        rms = float(np.sqrt(np.mean(chunk.astype(np.float64) ** 2)))
        self.pos_ms += dur_ms

        if rms >= self.THRESHOLD:
            self.silence_ms = 0.0
            if self.seg_start_ms < 0:
                self.seg_start_ms = self.pos_ms - dur_ms
            secs = (self.pos_ms - self.seg_start_ms) / 1000.0
            events.append({"type": "partial", "text": f"(mock) speaking… {secs:.1f}s"})
        elif self.seg_start_ms >= 0:
            self.silence_ms += dur_ms
            if self.silence_ms >= self.HANG_MS:
                events.extend(self._close_segment(self.pos_ms - self.silence_ms))
        return events

    def finish(self) -> list:
        if self.seg_start_ms >= 0:
            return self._close_segment(self.pos_ms)
        return []

    def _close_segment(self, end_ms):
        self.seg_count += 1
        secs = (end_ms - self.seg_start_ms) / 1000.0
        ev = [
            {
                "type": "segment",
                "start_ms": int(self.seg_start_ms),
                "end_ms": int(end_ms),
                "text": f"Mock segment {self.seg_count} ({secs:.1f}s of speech detected).",
            },
            {"type": "partial", "text": ""},
        ]
        self.seg_start_ms = -1
        self.silence_ms = 0.0
        return ev


def create_engine(cfg: Config):
    return MockEngine(cfg) if cfg.mock else FunASREngine(cfg)
