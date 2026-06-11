"""Configuration for the local recorder app.

Everything is overridable from the command line (see server.py) or via
environment variables prefixed with ``RECORDER_``.
"""

import os
from dataclasses import dataclass, field


def _env(name: str, default):
    value = os.environ.get(f"RECORDER_{name}")
    if value is None:
        return default
    if isinstance(default, bool):
        return value.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(value)
    return value


@dataclass
class Config:
    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env("PORT", 8765))

    # Where recordings + transcripts are stored.
    data_dir: str = field(
        default_factory=lambda: _env("DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
    )

    sample_rate: int = 16000

    # --- ASR models (all run on CPU) ---
    # Offline model used for high-quality final segments. SenseVoiceSmall is
    # multilingual (zh/en/yue/ja/ko), includes punctuation, and is fast on CPU.
    offline_model: str = field(default_factory=lambda: _env("OFFLINE_MODEL", "iic/SenseVoiceSmall"))
    # Streaming model used for low-latency live partials.
    streaming_model: str = field(
        default_factory=lambda: _env("STREAMING_MODEL", "paraformer-zh-streaming")
    )
    vad_model: str = field(default_factory=lambda: _env("VAD_MODEL", "fsmn-vad"))
    # Set to False to skip the streaming model entirely (lower CPU usage;
    # transcript then appears segment-by-segment as you pause speaking).
    enable_partials: bool = field(default_factory=lambda: _env("ENABLE_PARTIALS", True))
    language: str = field(default_factory=lambda: _env("LANGUAGE", "auto"))
    ncpu: int = field(default_factory=lambda: _env("NCPU", max(1, (os.cpu_count() or 4) - 1)))

    # Streaming chunk geometry: [0, 10, 5] -> 600 ms latency chunks.
    chunk_size_ms: int = 600
    vad_chunk_ms: int = 200
    # Audio padding added around VAD segments before offline decoding (ms).
    segment_pad_ms: int = 240
    # Hard cap for a single segment if VAD never fires (ms).
    max_segment_ms: int = 30000

    # --- Summarization (local LLM via Ollama) ---
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.2:3b"))
    # Max characters of transcript per LLM call before map-reduce chunking.
    summarize_chunk_chars: int = 16000

    # Use the mock engine (no ML deps; for UI testing / development).
    mock: bool = field(default_factory=lambda: _env("MOCK", False))
