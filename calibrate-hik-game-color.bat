@echo off
setlocal
set "ARIA_COLOR_ROOT=%~dp0"
set "ARIA_COLOR_OPENCV=%ARIA_COLOR_ROOT%.tools\rig-calibration-opencv-gui"
if not exist "%ARIA_COLOR_OPENCV%\cv2\cv2.pyd" (
  echo OpenCV is not installed. Run setup-hik-rig.bat once, then retry.
  exit /b 2
)
set "PYTHONPATH=%ARIA_COLOR_OPENCV%;%ARIA_COLOR_ROOT%.tools;%ARIA_COLOR_ROOT%"
set "ARIA_COLOR_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_COLOR_PYTHON%" set "ARIA_COLOR_PYTHON=python"
pushd "%ARIA_COLOR_ROOT%"
"%ARIA_COLOR_PYTHON%" -B -m rig_runtime.workflows.hik_game_color_calibration %*
set "ARIA_COLOR_EXIT=%ERRORLEVEL%"
popd
exit /b %ARIA_COLOR_EXIT%
