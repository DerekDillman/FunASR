@echo off
REM Start the Video Voice Analyzer (after setup_windows.bat has been run once).
cd /d %~dp0
call venv\Scripts\activate.bat
python app.py
pause
