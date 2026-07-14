"""Processing pipeline: video -> audio -> diarized transcript -> Ollama analysis.

Stages:
  1. ffmpeg strips the audio track to 16 kHz mono WAV (what FunASR expects).
  2. FunASR (paraformer + fsmn-vad + ct-punc + cam++) produces sentence-level
     text with millisecond timestamps and a speaker id per sentence.
  3. The diarized transcript is sent to an Ollama server, which analyzes the
     speakers (how many, who is likely who, what each one talks about).
"""

import json
import os
import subprocess
import threading
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
# qwen3:30b-a3b is a MoE model (~19 GB at Q4) that Ollama splits across
# 2x RTX 3060 12GB while only activating 3B params per token, so it is both
# high quality and fast on that hardware. See README for alternatives.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:30b-a3b")

ASR_MODEL = os.environ.get("ASR_MODEL", "paraformer-zh")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "")  # "" = auto-detect

_model = None
_vad_model = None
_live_model = None
_model_lock = threading.Lock()


def extract_audio(video_path: str, wav_path: str) -> None:
    """Strip the audio track from the video into a 16 kHz mono WAV file."""
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                # drop the video stream
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        wav_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed or not on PATH. "
            "Install it with e.g. 'sudo apt install ffmpeg'."
        ) from None
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "unknown error"
        raise RuntimeError(f"ffmpeg failed to extract audio: {tail}")


def _detect_device() -> str:
    if ASR_DEVICE:
        return ASR_DEVICE
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def get_model():
    """Lazily load the FunASR pipeline once and reuse it across requests."""
    global _model
    with _model_lock:
        if _model is None:
            from funasr import AutoModel
            _model = AutoModel(
                model=ASR_MODEL,
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 60000},
                punc_model="ct-punc",
                spk_model="cam++",
                device=_detect_device(),
                disable_update=True,
            )
        return _model


def get_vad_model():
    """Small voice-activity-detection model, used for the fast pre-scan."""
    global _vad_model
    with _model_lock:
        if _vad_model is None:
            from funasr import AutoModel
            _vad_model = AutoModel(model="fsmn-vad", device=_detect_device(),
                                   disable_update=True)
        return _vad_model


def get_live_model():
    """ASR-only model used to stream text out region-by-region while the
    full diarization pass runs at the end. Loaded separately so a failure
    in speaker clustering can't break the live pass."""
    global _live_model
    with _model_lock:
        if _live_model is None:
            from funasr import AutoModel
            _live_model = AutoModel(model=ASR_MODEL, device=_detect_device(),
                                    disable_update=True)
        return _live_model


def detect_speech(wav_path: str) -> list:
    """Fast VAD scan of the whole file. Returns [(start_ms, end_ms), ...]."""
    res = get_vad_model().generate(input=wav_path)
    if not res:
        return []
    return [(int(s), int(e)) for s, e in (res[0].get("value") or [])]


def transcribe_regions(wav_path: str, regions: list, on_update) -> list:
    """Transcribe each detected speech region individually, calling
    on_update(live_segments, regions_done, regions_total, done_ms, total_ms)
    after every region so the UI can show voices as they are heard."""
    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    model = get_live_model()
    total = len(regions)
    total_ms = sum(e - s for s, e in regions)
    done_ms = 0
    live = []
    pad_ms = 150

    for i, (start, end) in enumerate(regions):
        a = max(0, int((start - pad_ms) * sr / 1000))
        b = min(len(audio), int((end + pad_ms) * sr / 1000))
        text = ""
        try:
            res = model.generate(input=audio[a:b], fs=sr, disable_pbar=True)
            if res:
                text = (res[0].get("text") or "").strip()
        except Exception as exc:
            print(f"[live-asr] region {i} ({start}-{end}ms) failed: {exc}",
                  flush=True)
        done_ms += end - start
        if text:
            live.append({
                "speaker_id": -1,
                "speaker": "Voice",
                "start_ms": start,
                "end_ms": end,
                "start": _ms_to_clock(start),
                "end": _ms_to_clock(end),
                "text": text,
            })
        on_update(live, i + 1, total, done_ms, total_ms)
    return live


def _ms_to_clock(ms: int) -> str:
    seconds = int(ms) // 1000
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def transcribe(wav_path: str) -> dict:
    """Run ASR + speaker diarization. Returns segments merged into speaker turns."""
    model = get_model()
    res = model.generate(
        input=wav_path,
        # Tunable via env for long recordings: lower these if you hit
        # CUDA out-of-memory on multi-hour audio.
        batch_size_s=int(os.environ.get("ASR_BATCH_S", "300")),
        batch_size_threshold_s=int(os.environ.get("ASR_BATCH_THRESHOLD_S", "60")),
    )
    if not res:
        return {"segments": [], "speakers": [], "full_text": ""}

    result = res[0]
    sentences = result.get("sentence_info") or []

    segments = []
    for sent in sentences:
        text = (sent.get("text") or "").strip()
        if not text:
            continue
        spk = sent.get("spk", 0)
        start = int(sent.get("start", 0))
        end = int(sent.get("end", start))
        # Merge consecutive sentences from the same speaker into one turn.
        if segments and segments[-1]["speaker_id"] == spk and start - segments[-1]["end_ms"] < 1500:
            segments[-1]["text"] += " " + text
            segments[-1]["end_ms"] = end
            segments[-1]["end"] = _ms_to_clock(end)
        else:
            segments.append({
                "speaker_id": spk,
                "speaker": f"Speaker {spk + 1}",
                "start_ms": start,
                "end_ms": end,
                "start": _ms_to_clock(start),
                "end": _ms_to_clock(end),
                "text": text,
            })

    if not segments and result.get("text"):
        # Diarization can come back empty (e.g. no speech detected by cam++);
        # fall back to a single undifferentiated segment so the user still
        # gets the transcript.
        segments.append({
            "speaker_id": 0,
            "speaker": "Speaker 1",
            "start_ms": 0,
            "end_ms": 0,
            "start": "0:00",
            "end": "0:00",
            "text": result["text"].strip(),
        })

    speakers = sorted({seg["speaker"] for seg in segments})
    full_text = " ".join(seg["text"] for seg in segments)
    return {"segments": segments, "speakers": speakers, "full_text": full_text}


def format_transcript(segments: list) -> str:
    lines = []
    for seg in segments:
        lines.append(f"[{seg['start']} - {seg['end']}] {seg['speaker']}: {seg['text']}")
    return "\n".join(lines)


def analyze_with_ollama(segments: list, speakers: list) -> str:
    """Ask the Ollama server to analyze who is speaking and what they say."""
    transcript = format_transcript(segments)
    # Keep the prompt within a sane context window for a local model.
    max_chars = int(os.environ.get("OLLAMA_MAX_TRANSCRIPT_CHARS", "24000"))
    truncated = ""
    if len(transcript) > max_chars:
        transcript = transcript[:max_chars]
        truncated = "\n(Note: the transcript was truncated to fit the model's context window.)"

    prompt = (
        "Below is a transcript of a video, produced by automatic speech "
        "recognition with speaker diarization. Each line has a timestamp "
        "range and an anonymous speaker label.\n\n"
        f"Detected speakers: {', '.join(speakers) if speakers else 'unknown'}\n\n"
        "Transcript:\n"
        f"{transcript}{truncated}\n\n"
        "Please provide:\n"
        "1. **Speaker profiles** — for each speaker: their likely role/identity "
        "based on what they say (e.g. interviewer, host, expert), their tone, "
        "and the main points they make. Cite timestamps as evidence.\n"
        "2. **Conversation summary** — what the video is about overall.\n"
        "3. **Key moments** — a short list of notable timestamps worth "
        "cross-referencing with the video, and why.\n"
        "4. **Diarization sanity check** — if any lines look misattributed "
        "(e.g. a reply that clearly belongs to the other speaker), point them out."
    )

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert at analyzing conversation transcripts "
                           "and speaker diarization output. Be concrete and always "
                           "cite timestamps.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=int(os.environ.get("OLLAMA_TIMEOUT", "600"))) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL} ({exc}). "
            f"Is the server running and is model '{OLLAMA_MODEL}' pulled?"
        ) from exc

    content = (body.get("message") or {}).get("content", "")
    # Strip <think>...</think> blocks that reasoning models such as qwen3 emit.
    if "<think>" in content and "</think>" in content:
        content = content.split("</think>", 1)[1]
    return content.strip()
