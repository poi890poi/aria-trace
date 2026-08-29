@echo off
setlocal
cd /d "%~dp0"

set "ARIA_CAPTURE_ROOT=%~dp0"
set "ARIA_CAPTURE_OPENCV=%ARIA_CAPTURE_ROOT%.tools\rig-calibration-opencv-gui"
if not exist "%ARIA_CAPTURE_OPENCV%\cv2\cv2.pyd" (
  echo HIK calibration environment is not installed.
  echo Run setup-hik-rig.bat once, then retry.
  exit /b 2
)
set "PYTHONPATH=%ARIA_CAPTURE_OPENCV%;%ARIA_CAPTURE_ROOT%.tools;%ARIA_CAPTURE_ROOT%"
set "ARIA_CAPTURE_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_CAPTURE_PYTHON%" set "ARIA_CAPTURE_PYTHON=python"

if "%~1"=="" (
  echo Usage: capture-game-minimap-zigzag.bat ^<HIK calibration folder or JSON^>
  echo This records synchronized source data only. It does not run calibration.
  exit /b 2
)

"%ARIA_CAPTURE_PYTHON%" -B -m acquisition.capture_game_minimap_zigzag %*
exit /b %errorlevel%
