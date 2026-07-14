# Video Voice Analyzer

A drag-and-drop web app that takes an MP4 video and:

1. **Strips the audio** into its own file (`ffmpeg` → 16 kHz mono WAV, downloadable from the UI).
2. **Transcribes the speech** with FunASR, including **sentence-level timestamps**.
3. **Separates the voices** (speaker diarization with the CAM++ speaker model) so each line is attributed to `Speaker 1`, `Speaker 2`, ...
4. **Analyzes the conversation with your Ollama server** — who each speaker likely is, what they talk about, key moments, and a diarization sanity check.
5. Shows everything next to a **video player**: click any transcript line (or any timestamp in the analysis) and the video jumps to that moment, so voices can be cross-referenced with the picture.

```
browser ──drop mp4──▶ FastAPI ──ffmpeg──▶ audio.wav
                                   │
                                   ▼
                     FunASR (paraformer + fsmn-vad + ct-punc + cam++)
                                   │  text + timestamps + speaker ids
                                   ▼
                     Ollama /api/chat  ──▶  speaker analysis
```

## Requirements

- Python 3.9+
- `ffmpeg` on PATH (`sudo apt install ffmpeg`)
- An [Ollama](https://ollama.com) server (local or remote)
- A GPU is strongly recommended for FunASR (it falls back to CPU automatically)

```bash
cd video_voice_analyzer
pip install -r requirements.txt
python app.py
# open http://localhost:8000
```

The FunASR models (~2 GB total) download automatically from ModelScope on the first run.

## Ollama model for 2x RTX 3060 12 GB (24 GB total)

Set the model with `OLLAMA_MODEL`. Recommendations for this hardware:

| Model | VRAM (Q4) | Notes |
|---|---|---|
| **`qwen3:30b-a3b`** ⭐ recommended | ~19 GB | MoE — 30B quality but only 3B active params per token, so it's *fast*. Ollama splits it across both 3060s automatically. Best quality/speed on this rig. |
| `qwen3:14b` | ~9.3 GB | Fits entirely on **one** GPU, leaving the other free for FunASR. Pick this if you want to run ASR and the LLM concurrently with zero contention. |
| `llama3.1:8b` | ~4.9 GB | Lightest option, still fine for transcript summarization. |

```bash
ollama pull qwen3:30b-a3b
```

FunASR itself (paraformer + cam++) only needs ~2–3 GB of VRAM, so a comfortable
split is FunASR on GPU 0 and the LLM across both / on GPU 1:

```bash
# optional: pin FunASR to the first GPU
ASR_DEVICE=cuda:0 python app.py
```

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Your Ollama server ("ollama server to be entered") |
| `OLLAMA_MODEL` | `qwen3:30b-a3b` | Model used for the speaker analysis |
| `OLLAMA_TIMEOUT` | `600` | Seconds to wait for the analysis |
| `OLLAMA_MAX_TRANSCRIPT_CHARS` | `24000` | Transcript truncation limit for the LLM prompt |
| `ASR_MODEL` | `FunAudioLLM/Fun-ASR-Nano-2512` | FunASR ASR model (see note below) |
| `ASR_LANGUAGE` | *(auto-detect)* | Optional language hint for Fun-ASR-Nano, e.g. `English` |
| `ASR_DEVICE` | auto | `cuda:0`, `cuda:1`, or `cpu` |
| `PORT` / `HOST` | `8000` / `0.0.0.0` | Web server bind |
| `DATA_DIR` | `./data` | Where uploads, extracted audio, and results live |
| `MAX_UPLOAD_MB` | `20480` | Upload size limit (20 GB default — hour-long camera exports are fine) |
| `ASR_BATCH_S` | `300` | FunASR batch size in seconds of audio; lower it if you hit CUDA OOM on very long recordings |
| `ASR_BATCH_THRESHOLD_S` | `60` | See FunASR OOM guidance in `docs/tutorial` |

Example pointing at a remote Ollama box:

```bash
OLLAMA_URL=http://192.168.1.50:11434 OLLAMA_MODEL=qwen3:14b python app.py
```

### Language note

The default model is `Fun-ASR-Nano` — multilingual (31 languages including
English) with per-utterance automatic language detection, and it supports the
cam++ speaker-diarization hookup (`sentence_info` with speaker labels). Its
weights download from Hugging Face on first run. You can pin the language with
`ASR_LANGUAGE=English` if auto-detection ever misfires.

Alternative: `ASR_MODEL=paraformer-zh` switches to the classic
Mandarin-focused paraformer pipeline (downloads from ModelScope).

## API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/upload` | POST (multipart `file`) | Start a job, returns `{job_id}` |
| `/api/jobs/{id}` | GET | Status + transcript + analysis when done |
| `/media/{id}/video` | GET | The uploaded video (for the player) |
| `/media/{id}/audio` | GET | The extracted WAV file |
| `/api/config` | GET | Current Ollama/ASR configuration |

The transcript in the job result is a list of speaker turns:

```json
{
  "speaker": "Speaker 2",
  "start_ms": 12340, "end_ms": 15200,
  "start": "0:12", "end": "0:15",
  "text": "..."
}
```
