@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "ARIA_ANDROIDCAM_ROOT=%~dp0"
set "ARIA_ANDROIDCAM_GUI_OPENCV=%ARIA_ANDROIDCAM_ROOT%.tools\rig-calibration-opencv-gui"
set "ARIA_ANDROIDCAM_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_ANDROIDCAM_PYTHON%" set "ARIA_ANDROIDCAM_PYTHON=python"
set "PYTHONPATH=%ARIA_ANDROIDCAM_GUI_OPENCV%;%ARIA_ANDROIDCAM_ROOT%.tools;%ARIA_ANDROIDCAM_ROOT%"

"%ARIA_ANDROIDCAM_PYTHON%" -c "import cv2,numpy; x=[v for v in cv2.getBuildInformation().splitlines() if v.strip().startswith('GUI:')]; raise SystemExit(0 if x and x[0].split(':',1)[1].strip().upper() not in ('NONE','') else 1)" >nul 2>&1
if errorlevel 1 (
  echo GUI-capable OpenCV is unavailable.
  echo Run setup-hik-rig.bat once or install opencv-python, then retry.
  exit /b 2
)

pushd "%ARIA_ANDROIDCAM_ROOT%"
"%ARIA_ANDROIDCAM_PYTHON%" -B -m androidcam.stream %*
set "ARIA_ANDROIDCAM_EXIT=%ERRORLEVEL%"
popd
exit /b %ARIA_ANDROIDCAM_EXIT%
