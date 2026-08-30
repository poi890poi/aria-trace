@echo off
setlocal
cd /d "%~dp0"

set "ARIA_MINIMAP_ROOT=%~dp0"
set "ARIA_MINIMAP_OPENCV=%ARIA_MINIMAP_ROOT%.tools\rig-calibration-opencv-gui"
if not exist "%ARIA_MINIMAP_OPENCV%\cv2\cv2.pyd" (
  echo HIK calibration environment is not installed.
  echo Run setup-hik-rig.bat once, then retry.
  exit /b 2
)
set "PYTHONPATH=%ARIA_MINIMAP_OPENCV%;%ARIA_MINIMAP_ROOT%.tools;%ARIA_MINIMAP_ROOT%"
set "ARIA_MINIMAP_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_MINIMAP_PYTHON%" set "ARIA_MINIMAP_PYTHON=python"

if "%~1"=="" (
  echo Usage: calibrate-game-minimap.bat SESSION [options]
  echo This analyzes one completed fresh Android or Android-plus-HIK zigzag session.
  exit /b 2
)

"%ARIA_MINIMAP_PYTHON%" -B -m acquisition.session_minimap_localization %*
exit /b %errorlevel%
