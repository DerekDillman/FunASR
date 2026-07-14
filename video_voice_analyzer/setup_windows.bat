@echo off
setlocal
REM ============================================================
REM  Video Voice Analyzer - one-shot Windows setup
REM  Creates C:\ai, clones the repo, builds a venv, installs
REM  dependencies, verifies GPU support, and starts the app.
REM  Safe to re-run: it updates instead of re-cloning.
REM ============================================================

set "AI_DIR=C:\ai"
set "BRANCH=claude/video-audio-voice-analysis-j6k739"
set "REPO_URL=https://github.com/DerekDillman/FunASR.git"
set "APP_DIR=%AI_DIR%\FunASR\video_voice_analyzer"

where git >nul 2>nul || (
  echo [ERROR] git is not installed. Install it from https://git-scm.com and re-run.
  pause & exit /b 1
)
where python >nul 2>nul || (
  echo [ERROR] python is not installed. Install Python 3.10+ from https://python.org
  echo         and be sure to check "Add python.exe to PATH" in the installer.
  pause & exit /b 1
)

echo === [1/6] Creating %AI_DIR% and fetching the code...
if not exist "%AI_DIR%" mkdir "%AI_DIR%"
if exist "%AI_DIR%\FunASR\.git" (
  echo Repo already exists, updating it instead...
  git -C "%AI_DIR%\FunASR" fetch origin %BRANCH% || (echo [ERROR] git fetch failed & pause & exit /b 1)
  git -C "%AI_DIR%\FunASR" checkout %BRANCH%
  git -C "%AI_DIR%\FunASR" pull origin %BRANCH%
) else (
  git clone -b %BRANCH% %REPO_URL% "%AI_DIR%\FunASR" || (echo [ERROR] git clone failed & pause & exit /b 1)
)

cd /d "%APP_DIR%"

echo === [2/6] Creating the virtual environment...
if not exist venv python -m venv venv
call venv\Scripts\activate.bat

echo === [3/6] Installing dependencies (several GB, be patient)...
python -m pip install --upgrade pip
pip install -r requirements.txt || (echo [ERROR] pip install failed & pause & exit /b 1)

echo === [4/6] Checking that PyTorch can see the GPUs...
python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
  echo GPU not visible - installing the CUDA build of PyTorch...
  pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu121
  python -c "import torch; print('CUDA available:', torch.cuda.is_available(), '- GPUs:', torch.cuda.device_count())"
) else (
  python -c "import torch; print('CUDA available: True - GPUs:', torch.cuda.device_count())"
)

echo === [5/6] Checking ffmpeg...
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg not found - installing via winget...
  winget install --accept-source-agreements --accept-package-agreements Gyan.FFmpeg
  echo.
  echo [NOTE] ffmpeg was just installed. If video processing fails with an
  echo        ffmpeg error, close this window and run run.bat to restart
  echo        with a fresh PATH.
)

echo === [6/6] Checking the Ollama model...
where ollama >nul 2>nul
if errorlevel 1 (
  echo [WARN] ollama command not found on PATH. Make sure Ollama is running
  echo        and pull the model yourself:  ollama pull qwen3:30b-a3b
) else (
  ollama list | findstr /c:"qwen3:30b-a3b" >nul || (
    echo Pulling qwen3:30b-a3b - about 19 GB, one-time download...
    ollama pull qwen3:30b-a3b
  )
)

echo.
echo ============================================================
echo  Setup complete. Starting the app now.
echo  From any device on your network, open:
echo      http://192.168.40.137:8000
echo  To start it again later, run:  %APP_DIR%\run.bat
echo ============================================================
echo.
python app.py
pause
