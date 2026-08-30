@echo off
setlocal
cd /d "%~dp0"

set "ARIA_WAKE_ROOT=%~dp0"
set "ARIA_WAKE_OPENCV=%ARIA_WAKE_ROOT%.tools\rig-calibration-opencv-gui"
if not exist "%ARIA_WAKE_OPENCV%\cv2\cv2.pyd" (
  echo HIK calibration environment is not installed.
  echo Run setup-hik-rig.bat once, then retry.
  exit /b 2
)

set "PYTHONPATH=%ARIA_WAKE_OPENCV%;%ARIA_WAKE_ROOT%.tools;%ARIA_WAKE_ROOT%"
set "ARIA_WAKE_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_WAKE_PYTHON%" set "ARIA_WAKE_PYTHON=python"

"%ARIA_WAKE_PYTHON%" -B -m acquisition.detect_game_display_dim %*
exit /b %errorlevel%
