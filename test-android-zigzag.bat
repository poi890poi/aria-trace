@echo off
setlocal
cd /d "%~dp0"

set "ARIA_ZIGZAG_ROOT=%~dp0"
set "ARIA_ZIGZAG_OPENCV=%ARIA_ZIGZAG_ROOT%.tools\rig-calibration-opencv-gui"
set "PYTHONPATH=%ARIA_ZIGZAG_OPENCV%;%ARIA_ZIGZAG_ROOT%.tools;%ARIA_ZIGZAG_ROOT%"
set "ARIA_ZIGZAG_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_ZIGZAG_PYTHON%" set "ARIA_ZIGZAG_PYTHON=python"

"%ARIA_ZIGZAG_PYTHON%" -B -m acquisition.capture_game_minimap_zigzag --control-only %*
exit /b %errorlevel%
