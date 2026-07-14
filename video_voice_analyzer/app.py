"""Web app: drop an MP4, get a diarized transcript with timestamps plus an
Ollama-powered analysis of who is speaking and what they say.

Run with:  python app.py   (or: uvicorn app:app --host 0.0.0.0 --port 8000)
"""

import os
import shutil
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", BASE_DIR / "data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "20480")) * 1024 * 1024

app = FastAPI(title="Video Voice Analyzer")

# In-memory job registry. Jobs are also cheap to re-run, so no persistence.
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def _job_dir(job_id: str) -> Path:
    return DATA_DIR / job_id


def _set(job_id: str, **fields):
    with jobs_lock:
        jobs[job_id].update(fields)


def process_job(job_id: str, video_path: Path, original_name: str):
    wav_path = _job_dir(job_id) / "audio.wav"
    try:
        _set(job_id, status="extracting_audio",
             detail="Stripping audio track with ffmpeg (gap-tolerant decode "
                    "+ far-field cleanup: rumble filter, loudness boost)...")
        video_dur = pipeline.probe_duration(str(video_path))
        src_audio = pipeline.probe_audio_stream(str(video_path))
        print(f"[{job_id}] source: video {video_dur and round(video_dur/60,1)} "
              f"min, audio stream: {src_audio}", flush=True)
        method = pipeline.extract_audio(str(video_path), str(wav_path),
                                        expected_s=video_dur)

        stats = pipeline.audio_stats(str(wav_path))
        stats["video_duration_s"] = video_dur
        stats["source_audio"] = src_audio
        stats["extract_method"] = method
        _set(job_id, audio_stats=stats)
        dur_note = (f"{stats['duration_s'] / 60:.1f} min of audio, "
                    f"peak {stats['peak_db']} dB, average {stats['rms_db']} dB")
        mismatch = ""
        if video_dur and video_dur - stats["duration_s"] > 10:
            src_dur = (src_audio or {}).get("duration_s")
            if src_dur and video_dur - src_dur > 10:
                mismatch = (f" WARNING: the export itself only contains "
                            f"{src_dur / 60:.1f} min of audio for a "
                            f"{video_dur / 60:.1f} min video — the recording "
                            f"source (e.g. Frigate) did not record audio for "
                            f"most of it. Check the camera/NVR audio "
                            f"recording settings.")
            else:
                mismatch = (f" WARNING: the video is {video_dur / 60:.1f} min "
                            f"long but only {stats['duration_s'] / 60:.1f} min "
                            f"of audio could be decoded — the export's audio "
                            f"may be damaged.")
        print(f"[{job_id}] audio extracted ({method}): {dur_note}.{mismatch}",
              flush=True)

        _set(job_id, status="detecting_speech",
             detail=f"Audio extracted ({dur_note}).{mismatch} Scanning for "
                    f"voices (VAD sensitivity {pipeline.VAD_THRES}). First "
                    f"run downloads the models, which can take a while...")
        regions = pipeline.detect_speech(str(wav_path))
        speech_ms = sum(e - s for s, e in regions)
        print(f"[{job_id}] VAD found {len(regions)} speech regions, "
              f"{speech_ms // 1000}s of speech total", flush=True)

        brute_force = False
        if not regions:
            # Don't trust silence: let the ASR model judge every second
            # of audio itself in fixed 30 s windows.
            brute_force = True
            regions = pipeline.fallback_windows(stats["duration_s"])
            speech_ms = sum(e - s for s, e in regions)
            print(f"[{job_id}] VAD heard nothing — brute-force scanning "
                  f"{len(regions)} windows of 30s", flush=True)

        if not regions:
            _set(job_id, status="done",
                 transcript={"segments": [], "speakers": [], "full_text": ""},
                 analysis=None,
                 analysis_error=f"The extracted audio track is empty or "
                                f"near-zero length ({dur_note}).{mismatch}",
                 detail="Complete — no usable audio",
                 finished_at=time.time())
            return

        def on_update(live, done, total, done_ms, total_ms):
            pct = int(done_ms * 100 / total_ms) if total_ms else 100
            _set(job_id,
                 live_segments=list(live),
                 progress={"regions_done": done, "regions_total": total,
                           "speech_done_ms": done_ms,
                           "speech_total_ms": total_ms, "percent": pct},
                 detail=f"Transcribing speech region {done} of {total} "
                        f"— {len(live)} voice segments heard so far")

        if brute_force:
            start_detail = (f"VAD heard nothing above the noise floor, so "
                            f"every second of audio is being run through the "
                            f"speech recognizer directly — "
                            f"{len(regions)} windows of 30s...")
        else:
            start_detail = (f"Found {len(regions)} speech regions "
                            f"({speech_ms // 60000}m "
                            f"{speech_ms % 60000 // 1000}s of actual speech). "
                            f"Transcribing them one by one...")
        _set(job_id, status="transcribing",
             progress={"regions_done": 0, "regions_total": len(regions),
                       "speech_done_ms": 0, "speech_total_ms": speech_ms,
                       "percent": 0},
             detail=start_detail)
        live = pipeline.transcribe_regions(str(wav_path), regions, on_update)
        print(f"[{job_id}] live pass done: {len(live)} segments with text",
              flush=True)

        _set(job_id, status="diarizing",
             detail="All speech transcribed. Now separating the voices to "
                    "figure out who said what (second pass with the speaker "
                    "model)...")
        transcript = pipeline.transcribe(str(wav_path))
        if not transcript["segments"] and live:
            # Diarization pass came back empty but the live pass heard text;
            # keep the live result rather than discarding it.
            transcript = {"segments": live, "speakers": ["Voice"],
                          "full_text": " ".join(s["text"] for s in live)}
        _set(job_id, transcript=transcript)
        print(f"[{job_id}] diarization done: {len(transcript['segments'])} "
              f"segments, speakers: {transcript['speakers']}", flush=True)

        if not transcript["segments"]:
            _set(job_id, status="done", analysis=None,
                 analysis_error=f"Nothing could be transcribed into words "
                                f"even after scanning all of the audio "
                                f"({dur_note}).{mismatch} Download the "
                                f"extracted audio above and listen to it — "
                                f"if you can clearly hear the voices in that "
                                f"file, report back; if the voices are barely "
                                f"audible even to you, the recording is below "
                                f"what speech recognition can recover.",
                 detail="Complete — no transcribable speech",
                 finished_at=time.time())
            return

        _set(job_id, status="analyzing",
             detail=f"Asking Ollama ({pipeline.OLLAMA_MODEL}) to analyze the "
                    f"speakers...")
        try:
            analysis = pipeline.analyze_with_ollama(
                transcript["segments"], transcript["speakers"])
            analysis_error = None
        except Exception as exc:  # Ollama being down shouldn't lose the transcript
            analysis = None
            analysis_error = str(exc)

        _set(job_id, status="done", detail="Complete",
             analysis=analysis, analysis_error=analysis_error,
             finished_at=time.time())
    except Exception as exc:
        print(f"[{job_id}] FAILED: {exc}", flush=True)
        _set(job_id, status="error", detail=str(exc), finished_at=time.time())


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. "
                                 f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")

    job_id = uuid.uuid4().hex[:12]
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True)
    video_path = job_dir / f"video{ext}"

    size = 0
    with open(video_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                out.close()
                shutil.rmtree(job_dir, ignore_errors=True)
                raise HTTPException(413, "File too large")
            out.write(chunk)

    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "filename": file.filename,
            "video_ext": ext,
            "status": "queued",
            "detail": "Queued",
            "created_at": time.time(),
        }

    threading.Thread(target=process_job,
                     args=(job_id, video_path, file.filename),
                     daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Job not found")
        return JSONResponse(dict(job))


@app.get("/media/{job_id}/video")
def get_video(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found")
    path = _job_dir(job_id) / f"video{job['video_ext']}"
    if not path.exists():
        raise HTTPException(404, "Video not found")
    return FileResponse(path)


@app.get("/media/{job_id}/audio")
def get_audio(job_id: str):
    path = _job_dir(job_id) / "audio.wav"
    if not path.exists():
        raise HTTPException(404, "Audio not extracted yet")
    return FileResponse(path, filename="extracted_audio.wav")


@app.get("/api/config")
def config():
    return {
        "ollama_url": pipeline.OLLAMA_URL,
        "ollama_model": pipeline.OLLAMA_MODEL,
        "asr_model": pipeline.ASR_MODEL,
    }


app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"),
                port=int(os.environ.get("PORT", "8000")))
