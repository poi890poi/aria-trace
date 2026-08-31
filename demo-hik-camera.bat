@echo off
setlocal EnableExtensions EnableDelayedExpansion
set "ARIA_HIK_ROOT=%~dp0"
set "ARIA_HIK_OPENCV=%ARIA_HIK_ROOT%.tools\rig-calibration-opencv-gui"
if not exist "%ARIA_HIK_OPENCV%\cv2\cv2.pyd" (
  echo GUI-capable OpenCV is not installed.
  echo Run setup-hik-rig.bat once, then retry.
  exit /b 2
)

set "ARIA_HIK_STREAM_ARGS=%*"
set "ARIA_HIK_MANAGE_PHONE=--manage-phone-display"
set "ARIA_HIK_FIRST=%~1"
if not "%ARIA_HIK_FIRST%"=="" if not "%ARIA_HIK_FIRST:~0,2%"=="--" set "ARIA_HIK_STREAM_ARGS=--game-id "%~1" %2 %3 %4 %5 %6 %7 %8 %9"

set "ARIA_HIK_EXPECT_LIBRARY="
for %%A in (%*) do (
  if defined ARIA_HIK_EXPECT_LIBRARY (
    if /i "%%~A"=="native" set "ARIA_HIK_MANAGE_PHONE="
    set "ARIA_HIK_EXPECT_LIBRARY="
  ) else if /i "%%~A"=="--camera-library" (
    set "ARIA_HIK_EXPECT_LIBRARY=1"
  )
)

set "PYTHONPATH=%ARIA_HIK_OPENCV%;%ARIA_HIK_ROOT%.tools;%ARIA_HIK_ROOT%"
set "ARIA_HIK_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_HIK_PYTHON%" set "ARIA_HIK_PYTHON=python"

if defined ARIA_HIK_MANAGE_PHONE (echo The calibrated demo only controls Android display power. It does not launch apps or send touch input.) else (echo Native Hikrobot MVS full-sensor mode; no profile or phone operation.)
pushd "%ARIA_HIK_ROOT%"
"%ARIA_HIK_PYTHON%" -B -m acquisition.rig_calibration.hik.stream %ARIA_HIK_STREAM_ARGS% --gui %ARIA_HIK_MANAGE_PHONE%
set "ARIA_HIK_EXIT=%ERRORLEVEL%"
popd
exit /b %ARIA_HIK_EXIT%
