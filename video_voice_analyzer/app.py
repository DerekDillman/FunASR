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
             detail="Stripping audio track with ffmpeg...")
        pipeline.extract_audio(str(video_path), str(wav_path))

        _set(job_id, status="transcribing",
             detail="Running FunASR speech recognition + speaker diarization "
                    "(first run downloads the models)...")
        transcript = pipeline.transcribe(str(wav_path))

        _set(job_id, status="analyzing", transcript=transcript,
             detail=f"Asking Ollama ({pipeline.OLLAMA_MODEL}) to analyze the speakers...")
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
