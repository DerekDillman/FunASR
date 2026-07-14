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

# Fun-ASR-Nano is multilingual (31 languages incl. English) with automatic
# language detection, and supports the cam++ speaker-diarization hookup
# (see tests_models/test_fun_asr_nano_spk.py). Set ASR_MODEL=paraformer-zh
# for the Mandarin-focused pipeline instead.
ASR_MODEL = os.environ.get("ASR_MODEL", "FunAudioLLM/Fun-ASR-Nano-2512")
ASR_DEVICE = os.environ.get("ASR_DEVICE", "")  # "" = auto-detect
# Optional language hint, e.g. "English". Empty = auto-detect per utterance.
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "")
_IS_LLM_ASR = "fun-asr" in ASR_MODEL.lower()

_model = None
_vad_model = None
_live_model = None
_model_lock = threading.Lock()


# Far-field camera audio is faint: cut low-frequency rumble and strongly
# boost quiet passages so both the VAD and the ASR get a usable signal.
# Set AUDIO_FILTER="" to disable.
AUDIO_FILTER = os.environ.get("AUDIO_FILTER", "highpass=f=100,dynaudnorm=m=30")

# fsmn-vad speech/noise threshold: default 0.6 is tuned for close-mic
# speech; lower values detect fainter speech (range roughly -1..1).
VAD_THRES = float(os.environ.get("VAD_SPEECH_NOISE_THRES", "0.2"))


def probe_audio_stream(video_path: str):
    """Codec and duration of the source file's first audio stream, so a
    short *source* stream can be told apart from a short *extraction*."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=codec_name,duration",
             "-of", "json", video_path],
            capture_output=True, text=True)
        streams = json.loads(proc.stdout).get("streams") or []
        if not streams:
            return None
        info = streams[0]
        dur = info.get("duration")
        return {"codec": info.get("codec_name"),
                "duration_s": round(float(dur), 1) if dur else None}
    except Exception:
        return None


def extract_audio(video_path: str, wav_path: str, expected_s=None) -> str:
    """Strip the audio track into a 16 kHz mono WAV file.

    Segmented exports (e.g. Frigate) often carry timestamp gaps that make a
    plain decode stop early, so several strategies are tried and the one
    yielding the longest audio wins. `aresample=async=1` pads the gaps with
    silence, which also keeps transcript timestamps aligned with the video.
    Returns a description of the strategy used.
    """
    out_args = ["-vn", "-sn", "-dn", "-acodec", "pcm_s16le",
                "-ar", "16000", "-ac", "1"]
    gap_fill = "aresample=async=1:first_pts=0"
    filt = gap_fill + ("," + AUDIO_FILTER if AUDIO_FILTER else "")
    attempts = [
        ("gap-filling",
         ["-fflags", "+genpts", "-i", video_path, "-map", "0:a:0"]
         + out_args + ["-af", filt]),
        ("error-tolerant",
         ["-ignore_editlist", "1", "-fflags", "+genpts+discardcorrupt",
          "-err_detect", "ignore_err", "-i", video_path, "-map", "0:a:0"]
         + out_args + ["-af", filt]),
        ("plain",
         ["-i", video_path] + out_args
         + (["-af", AUDIO_FILTER] if AUDIO_FILTER else [])),
    ]

    tmp_path = wav_path + ".try.wav"
    best_dur, best_name, last_err = 0.0, None, "unknown error"
    for name, args in attempts:
        try:
            proc = subprocess.run(["ffmpeg", "-y"] + args + [tmp_path],
                                  capture_output=True, text=True)
        except FileNotFoundError:
            raise RuntimeError(
                "ffmpeg is not installed or not on PATH. "
                "Install it with e.g. 'winget install ffmpeg'."
            ) from None
        if proc.returncode != 0:
            if proc.stderr.strip():
                last_err = proc.stderr.strip().splitlines()[-1]
            print(f"[extract:{name}] failed: {last_err}", flush=True)
            continue
        dur = probe_duration(tmp_path) or 0.0
        print(f"[extract:{name}] got {dur / 60:.1f} min of audio", flush=True)
        if dur > best_dur:
            best_dur, best_name = dur, name
            os.replace(tmp_path, wav_path)
        # Good enough — no need to try the remaining strategies.
        if expected_s and dur >= 0.95 * expected_s:
            break
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    if best_name is None:
        raise RuntimeError(f"ffmpeg could not extract any audio: {last_err}")
    return best_name


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


def _model_kwargs(with_diarization: bool) -> dict:
    kwargs = dict(model=ASR_MODEL, device=_detect_device(), disable_update=True)
    if _IS_LLM_ASR:
        # Mirrors tests_models/test_fun_asr_nano_spk.py
        kwargs.update(trust_remote_code=True, remote_code="./model.py", hub="hf")
        if with_diarization:
            kwargs.update(
                vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                vad_kwargs={"max_single_segment_time": 30000,
                            "speech_noise_thres": VAD_THRES},
                spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
            )
    elif with_diarization:
        kwargs.update(
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 60000,
                        "speech_noise_thres": VAD_THRES},
            punc_model="ct-punc",
            spk_model="cam++",
        )
    return kwargs


def _generate_kwargs() -> dict:
    if _IS_LLM_ASR:
        kwargs = {"cache": {}, "batch_size": 1}
        if ASR_LANGUAGE:
            kwargs["language"] = ASR_LANGUAGE
        return kwargs
    return {
        # Tunable via env for long recordings: lower these if you hit
        # CUDA out-of-memory on multi-hour audio.
        "batch_size_s": int(os.environ.get("ASR_BATCH_S", "300")),
        "batch_size_threshold_s": int(os.environ.get("ASR_BATCH_THRESHOLD_S", "60")),
    }


def get_model():
    """Lazily load the FunASR pipeline once and reuse it across requests."""
    global _model
    with _model_lock:
        if _model is None:
            from funasr import AutoModel
            _model = AutoModel(**_model_kwargs(with_diarization=True))
        return _model


def probe_duration(media_path: str):
    """Duration in seconds via ffprobe, or None if it can't be determined."""
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", media_path],
            capture_output=True, text=True)
        return float(proc.stdout.strip())
    except Exception:
        return None


def audio_stats(wav_path: str) -> dict:
    """Duration plus peak/RMS levels so silent or truncated extractions
    are visible instead of a mystery."""
    import math

    import soundfile as sf

    peak = 0.0
    sumsq = 0.0
    n = 0
    with sf.SoundFile(wav_path) as f:
        samplerate = f.samplerate
        frames = f.frames
        while True:
            block = f.read(samplerate * 60, dtype="float32")
            if len(block) == 0:
                break
            peak = max(peak, float(abs(block).max()))
            sumsq += float((block.astype("float64") ** 2).sum())
            n += len(block)
    rms = math.sqrt(sumsq / n) if n else 0.0
    to_db = lambda x: round(20 * math.log10(x), 1) if x > 0 else -120.0
    return {
        "duration_s": round(frames / samplerate, 1),
        "peak_db": to_db(peak),
        "rms_db": to_db(rms),
    }


def fallback_windows(duration_s: float, window_ms: int = 30000) -> list:
    """Fixed windows covering the whole file, used when VAD hears nothing —
    lets the ASR model make its own judgement about every second of audio."""
    total_ms = int(duration_s * 1000)
    return [(start, min(start + window_ms, total_ms))
            for start in range(0, total_ms, window_ms)
            if min(start + window_ms, total_ms) - start >= 1000]


def get_vad_model():
    """Small voice-activity-detection model, used for the fast pre-scan."""
    global _vad_model
    with _model_lock:
        if _vad_model is None:
            from funasr import AutoModel
            _vad_model = AutoModel(model="fsmn-vad", device=_detect_device(),
                                   speech_noise_thres=VAD_THRES,
                                   max_single_segment_time=30000,
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
            _live_model = AutoModel(**_model_kwargs(with_diarization=False))
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
    import tempfile

    import soundfile as sf

    audio, sr = sf.read(wav_path, dtype="float32")
    model = get_live_model()
    total = len(regions)
    total_ms = sum(e - s for s, e in regions)
    done_ms = 0
    live = []
    pad_ms = 150
    clip_path = os.path.join(tempfile.gettempdir(), "vva_live_clip.wav")

    for i, (start, end) in enumerate(regions):
        a = max(0, int((start - pad_ms) * sr / 1000))
        b = min(len(audio), int((end + pad_ms) * sr / 1000))
        text = ""
        try:
            # Written to a file because that is the most universally
            # supported input path across FunASR model types.
            sf.write(clip_path, audio[a:b], sr)
            res = model.generate(input=clip_path, disable_pbar=True,
                                 **_generate_kwargs())
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
    res = model.generate(input=wav_path, **_generate_kwargs())
    if not res:
        return {"segments": [], "speakers": [], "full_text": ""}

    result = res[0]
    sentences = result.get("sentence_info") or []

    segments = []
    for sent in sentences:
        text = (sent.get("text") or sent.get("sentence") or "").strip()
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
