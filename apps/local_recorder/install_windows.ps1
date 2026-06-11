<#
.SYNOPSIS
    One-shot installer for the Local Recorder app on Windows.

    Installs everything under C:\ai\HarvardRecorder:
      - checks/installs Git and Python 3.12 (via winget) if missing
      - clones the FunASR repo (app branch)
      - creates a virtual environment
      - installs CPU-only PyTorch + all app dependencies
      - optionally installs Ollama and pulls llama3.2:3b for summaries
      - creates Start-HarvardRecorder.bat and a desktop shortcut

.USAGE
    Run from any PowerShell window (no admin needed unless winget installs require it):

        irm https://raw.githubusercontent.com/DerekDillman/FunASR/refs/heads/claude/friendly-einstein-rcmnq0/apps/local_recorder/install_windows.ps1 | iex

    Or download this file and run:  powershell -ExecutionPolicy Bypass -File install_windows.ps1

    Safe to re-run: it updates the repo and dependencies in place.
#>

$ErrorActionPreference = "Stop"

$InstallDir = "C:\ai\HarvardRecorder"
$RepoUrl    = "https://github.com/DerekDillman/FunASR.git"
$Branch     = "claude/friendly-einstein-rcmnq0"
$RepoDir    = Join-Path $InstallDir "FunASR"
$VenvDir    = Join-Path $InstallDir ".venv"
$AppDir     = Join-Path $RepoDir "apps\local_recorder"

function Step($msg)  { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Ok($msg)    { Write-Host "    $msg" -ForegroundColor Green }
function Warn($msg)  { Write-Host "    $msg" -ForegroundColor Yellow }

function Refresh-Path {
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
}

function Have($cmd) { return [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function Winget-Install($id, $name) {
    if (-not (Have "winget")) {
        throw "$name is not installed and winget is unavailable. Install $name manually, then re-run this script."
    }
    Warn "$name not found - installing via winget (a UAC prompt may appear)..."
    winget install --id $id -e --accept-source-agreements --accept-package-agreements
    Refresh-Path
}

# Returns a command-line array (e.g. @("py","-3.12") or @("python")) for a
# supported interpreter, or $null. funasr/torch are happiest on 3.10-3.12.
function Find-Python {
    if (Have "py") {
        foreach ($v in "3.12", "3.11", "3.10") {
            & py "-$v" -c "pass" 2>$null
            if ($LASTEXITCODE -eq 0) { return ,@("py", "-$v") }
        }
    }
    if (Have "python") {
        try {
            $ver = & python -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and [version]$ver -ge [version]"3.10" -and [version]$ver -lt [version]"3.13") {
                return ,@("python")
            }
        } catch {}
    }
    return $null
}

Write-Host ""
Write-Host "  Harvard Recorder - local recording, realtime transcription & summaries" -ForegroundColor Magenta
Write-Host "  Installing into $InstallDir (CPU-only, fully local)" -ForegroundColor Magenta

# --- 1. Prerequisites -------------------------------------------------------

Step "Checking Git"
if (-not (Have "git")) { Winget-Install "Git.Git" "Git" }
if (-not (Have "git")) { throw "Git still not on PATH. Open a NEW PowerShell window and re-run this script." }
Ok (git --version)

Step "Checking Python (3.10 - 3.12)"
$py = Find-Python
if (-not $py) {
    Winget-Install "Python.Python.3.12" "Python 3.12"
    $py = Find-Python
}
if (-not $py) { throw "No suitable Python found. Open a NEW PowerShell window and re-run this script." }
$pyExe  = $py[0]
$pyArgs = @(); if ($py.Count -gt 1) { $pyArgs = $py[1..($py.Count - 1)] }
Ok ((& $pyExe @pyArgs --version) | Out-String).Trim()

# --- 2. Clone / update the repo --------------------------------------------

Step "Fetching the app ($RepoUrl, branch $Branch)"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
if (Test-Path (Join-Path $RepoDir ".git")) {
    git -C $RepoDir fetch origin $Branch
    git -C $RepoDir checkout $Branch
    git -C $RepoDir pull origin $Branch
    Ok "Repo updated"
} else {
    git clone --depth 1 --branch $Branch $RepoUrl $RepoDir
    Ok "Repo cloned"
}

# --- 3. Virtual environment + dependencies ----------------------------------

Step "Creating virtual environment"
$venvPy = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    & $pyExe @pyArgs -m venv $VenvDir
}
Ok "venv at $VenvDir"

Step "Installing dependencies (CPU-only PyTorch; this can take several minutes)"
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
& $venvPy -m pip install -r (Join-Path $AppDir "requirements.txt")

Step "Verifying the installation"
& $venvPy -c "import torch, funasr, fastapi, uvicorn; print('torch', torch.__version__, '| funasr', funasr.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Dependency verification failed - see errors above." }
Ok "All Python packages import cleanly"

# --- 4. Launcher + shortcut --------------------------------------------------

Step "Creating launcher"
$bat = @"
@echo off
title Harvard Recorder
set MODELSCOPE_CACHE=%~dp0models
cd /d "%~dp0FunASR\apps\local_recorder"
echo.
echo  Harvard Recorder - http://localhost:8765
echo  First launch downloads ~1.5 GB of speech models; watch this window.
echo  Close this window (or press Ctrl+C) to stop the recorder.
echo.
start "" /min cmd /c "timeout /t 6 >nul && start http://localhost:8765"
"%~dp0.venv\Scripts\python.exe" server.py --data-dir "%~dp0recordings"
"@
$batPath = Join-Path $InstallDir "Start-HarvardRecorder.bat"
Set-Content -Path $batPath -Value $bat -Encoding ASCII
New-Item -ItemType Directory -Force -Path (Join-Path $InstallDir "recordings") | Out-Null
Ok $batPath

try {
    $desktop = [Environment]::GetFolderPath("Desktop")
    $shell = New-Object -ComObject WScript.Shell
    $sc = $shell.CreateShortcut((Join-Path $desktop "Harvard Recorder.lnk"))
    $sc.TargetPath = $batPath
    $sc.WorkingDirectory = $InstallDir
    $sc.Description = "Local recording, realtime transcription and summaries"
    $sc.Save()
    Ok "Desktop shortcut created"
} catch {
    Warn "Could not create desktop shortcut ($_). Use $batPath instead."
}

# --- 5. Optional: Ollama for LLM summaries -----------------------------------

Step "Summarization LLM (optional)"
if (Have "ollama") {
    Ok "Ollama already installed"
    $pull = Read-Host "    Pull the summary model llama3.2:3b (~2 GB)? [Y/n]"
    if ($pull -notmatch '^[nN]') { ollama pull llama3.2:3b }
} else {
    $ans = Read-Host "    Install Ollama + llama3.2:3b (~2 GB) for AI summaries? Without it a basic fallback is used. [Y/n]"
    if ($ans -notmatch '^[nN]') {
        Winget-Install "Ollama.Ollama" "Ollama"
        if (Have "ollama") {
            ollama pull llama3.2:3b
            Ok "Ollama ready with llama3.2:3b"
        } else {
            Warn "Ollama installed but not on PATH yet. Open a new terminal and run: ollama pull llama3.2:3b"
        }
    } else {
        Warn "Skipped. You can add it later: winget install Ollama.Ollama && ollama pull llama3.2:3b"
    }
}

# --- Done ---------------------------------------------------------------------

Write-Host ""
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Start it:   double-click 'Harvard Recorder' on your desktop"
Write-Host "              (or run $batPath)"
Write-Host "  Then open:  http://localhost:8765  (opens automatically)"
Write-Host ""
Write-Host "  Notes:"
Write-Host "   - First launch downloads ~1.5 GB of speech models, then it's fully offline."
Write-Host "   - Recordings + transcripts are saved in $InstallDir\recordings"
Write-Host "   - Everything runs on CPU; nothing leaves your machine."
Write-Host ""
