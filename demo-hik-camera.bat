@echo off
setlocal
set "ARIA_HIK_ROOT=%~dp0"
set "ARIA_HIK_OPENCV=%ARIA_HIK_ROOT%.tools\rig-calibration-opencv-gui"
if not exist "%ARIA_HIK_OPENCV%\cv2\cv2.pyd" (
  echo GUI-capable OpenCV is not installed.
  echo Run setup-hik-rig.bat once, then retry.
  exit /b 2
)

set "ARIA_HIK_SELECTION=%~1"
set "ARIA_HIK_STREAM_ARGS="
if defined ARIA_HIK_SELECTION if exist "%ARIA_HIK_SELECTION%\hik_camera_calibration.json" set "ARIA_HIK_SELECTION=%ARIA_HIK_SELECTION%\hik_camera_calibration.json"
if defined ARIA_HIK_SELECTION (
  if exist "%ARIA_HIK_SELECTION%" (
    set "ARIA_HIK_STREAM_ARGS="%ARIA_HIK_SELECTION%""
  ) else (
    set "ARIA_HIK_STREAM_ARGS=--game-id "%ARIA_HIK_SELECTION%""
  )
)

set "PYTHONPATH=%ARIA_HIK_OPENCV%;%ARIA_HIK_ROOT%.tools;%ARIA_HIK_ROOT%"
set "ARIA_HIK_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_HIK_PYTHON%" set "ARIA_HIK_PYTHON=python"

if defined ARIA_HIK_SELECTION (echo Selection: %ARIA_HIK_SELECTION%) else (echo Selection: automatic unique active rig profile)
echo The demo only controls Android display power. It does not launch apps or send touch input.
pushd "%ARIA_HIK_ROOT%"
"%ARIA_HIK_PYTHON%" -B -m acquisition.rig_calibration.hik.stream %ARIA_HIK_STREAM_ARGS% --gui --manage-phone-display %2 %3 %4 %5 %6 %7 %8 %9
set "ARIA_HIK_EXIT=%ERRORLEVEL%"
popd
exit /b %ARIA_HIK_EXIT%
