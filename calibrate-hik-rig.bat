@echo off
setlocal
set "ARIA_RIG_ROOT=%~dp0"
set "ARIA_RIG_OPENCV=%ARIA_RIG_ROOT%.tools\rig-calibration-opencv-gui"
if not exist "%ARIA_RIG_OPENCV%\cv2\cv2.pyd" (
  echo GUI-capable OpenCV is not installed for HIK rig calibration.
  echo Run setup-hik-rig.bat once, then retry.
  exit /b 2
)
set "PYTHONPATH=%ARIA_RIG_OPENCV%;%ARIA_RIG_ROOT%.tools;%ARIA_RIG_ROOT%"

set "ARIA_RIG_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_RIG_PYTHON%" set "ARIA_RIG_PYTHON=python"

pushd "%ARIA_RIG_ROOT%"
"%ARIA_RIG_PYTHON%" -B -m acquisition.rig_calibration.hik.calibrate %*
set "ARIA_RIG_EXIT=%ERRORLEVEL%"
popd
exit /b %ARIA_RIG_EXIT%
