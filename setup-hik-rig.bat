@echo off
setlocal
set "ARIA_RIG_ROOT=%~dp0"
set "ARIA_RIG_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_RIG_PYTHON%" set "ARIA_RIG_PYTHON=python"

"%ARIA_RIG_PYTHON%" -B -m pip install --target "%ARIA_RIG_ROOT%.tools\rig-calibration-opencv-gui" --no-deps -r "%ARIA_RIG_ROOT%requirements-hik-rig-calibration.txt"
exit /b %ERRORLEVEL%
