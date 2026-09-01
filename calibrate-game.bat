@echo off
setlocal
cd /d "%~dp0"
python -B -m iris_tools game-calibration %*
exit /b %errorlevel%
