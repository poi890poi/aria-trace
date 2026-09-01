@echo off
setlocal
cd /d "%~dp0"
python -B -m aria_tools game-calibration %*
exit /b %errorlevel%
