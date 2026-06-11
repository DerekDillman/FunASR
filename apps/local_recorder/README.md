# Local Recorder — record, transcribe & summarize, 100% on your laptop

A small web app on top of [FunASR](https://github.com/modelscope/FunASR) that:

- 🎙️ records your microphone in the browser,
- 📝 transcribes **in real time** (live partials while you speak, accurate
  punctuated finals after each pause),
- ✨ summarizes recordings (meetings, trainings, instructions) with a fast
  **local** LLM via [Ollama](https://ollama.com),
- 💾 stores everything (WAV audio + JSON transcripts + summaries) in a local
  folder — nothing ever leaves your machine, no GPU required.

```
Browser (mic → AudioWorklet → 16 kHz PCM over WebSocket)
   └── FastAPI server (this folder)
         ├── fsmn-vad                  voice activity detection → utterance segments
         ├── paraformer-zh-streaming   live partial text (~600 ms latency)
         ├── SenseVoiceSmall           final text per segment (zh/en/yue/ja/ko, punctuation, ITN)
         ├── data/<session>/           audio.wav + session.json (atomic, crash-safe)
         └── Ollama (optional)         summaries; extractive fallback if absent
```

## Quick start

```bash
cd apps/local_recorder
python -m venv .venv && source .venv/bin/activate   # optional but recommended

# CPU-only torch keeps the install small:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python server.py
# open http://localhost:8765
```

The first run downloads ~1.5 GB of models from ModelScope (SenseVoiceSmall,
paraformer streaming, fsmn-vad); after that it is fully offline. Model loading
takes a little while — the **ASR** status chip in the header turns green when
ready.

> **Mic permission:** browsers only expose the microphone on secure origins.
> `http://localhost` counts as secure, so no SSL setup is needed.

### Try the UI without installing models

```bash
pip install fastapi 'uvicorn[standard]' numpy
python server.py --mock
```

Mock mode fakes transcription from audio energy — useful for checking your
microphone, the live view, and storage before downloading models.

## Summaries with a local LLM

Install [Ollama](https://ollama.com) and pull a small model — 3B-class models
summarize comfortably on a CPU laptop:

```bash
ollama pull llama3.2:3b          # default
# or: ollama pull qwen2.5:3b-instruct  (then: python server.py --ollama-model qwen2.5:3b-instruct)
```

Open a recording and click **✨ Summarize**, choosing *Meeting*,
*Training / lecture*, *Instructions*, or *General*. Long transcripts are
summarized map-reduce style so they fit small context windows. If Ollama
isn't running, a built-in extractive summarizer is used as a fallback so the
button always works.

## Options

```
python server.py --help

--host / --port        bind address (default 127.0.0.1:8765 — local only)
--data-dir             where recordings are stored (default ./data)
--no-partials          skip the streaming model: lower CPU use, transcript
                       appears segment-by-segment after each pause
--offline-model        e.g. iic/SenseVoiceSmall (default) or paraformer-zh
--ollama-model         local LLM for summaries (default llama3.2:3b)
--mock                 run without ML models
```

Environment variables (`RECORDER_HOST`, `RECORDER_PORT`, `RECORDER_DATA_DIR`,
`RECORDER_OFFLINE_MODEL`, `RECORDER_OLLAMA_MODEL`, `RECORDER_OLLAMA_URL`,
`RECORDER_ENABLE_PARTIALS`, `RECORDER_LANGUAGE`, `RECORDER_NCPU`,
`RECORDER_MOCK`) override the same settings.

## Notes on robustness

- Audio is written to `data/<session>/audio.wav` **as it arrives**, and the
  transcript JSON is rewritten atomically after every finalized segment — a
  browser or server crash loses at most the segment in flight.
- If the browser tab disconnects mid-recording, the server still flushes,
  finalizes and saves the session.
- Inference runs in a worker thread pool with per-model locks; when the CPU
  falls behind, incoming audio is coalesced into bigger batches instead of
  building up unbounded lag.
- The server binds to `127.0.0.1` by default, so nothing is exposed to your
  network.

## Language notes

`SenseVoiceSmall` (the default final-pass model) auto-detects Chinese,
English, Cantonese, Japanese and Korean and adds punctuation. The live
*partials* model (`paraformer-zh-streaming`) is strongest on Mandarin; for
mostly-English recordings the live preview may be rough, but the **final
transcript of every segment comes from SenseVoice** and stays accurate. If
the partials aren't useful to you, run with `--no-partials` to save CPU.
