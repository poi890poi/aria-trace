@echo off
setlocal
set "ARIA_GAME_CHECK_ROOT=%~dp0"
set "ARIA_GAME_CHECK_OPENCV=%ARIA_GAME_CHECK_ROOT%.tools\rig-calibration-opencv-gui"
if not exist "%ARIA_GAME_CHECK_OPENCV%\cv2\cv2.pyd" (
  echo OpenCV is not installed. Run setup-hik-rig.bat once, then retry.
  exit /b 2
)
set "PYTHONPATH=%ARIA_GAME_CHECK_OPENCV%;%ARIA_GAME_CHECK_ROOT%.tools;%ARIA_GAME_CHECK_ROOT%"
set "ARIA_GAME_CHECK_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_GAME_CHECK_PYTHON%" set "ARIA_GAME_CHECK_PYTHON=python"
pushd "%ARIA_GAME_CHECK_ROOT%"
"%ARIA_GAME_CHECK_PYTHON%" -B -m aria_trace.workflows.game_repeatability %*
set "ARIA_GAME_CHECK_EXIT=%ERRORLEVEL%"
popd
exit /b %ARIA_GAME_CHECK_EXIT%
