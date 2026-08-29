@echo off
setlocal
set "ARIA_HIK_ROOT=%~dp0"
set "ARIA_HIK_OPENCV=%ARIA_HIK_ROOT%.tools\rig-calibration-opencv-gui"
if not exist "%ARIA_HIK_OPENCV%\cv2\cv2.pyd" (
  echo GUI-capable OpenCV is not installed.
  echo Run setup-hik-rig.bat once, then retry.
  exit /b 2
)

set "ARIA_HIK_CALIBRATION=%~1"
set "ARIA_HIK_STREAM_OPTION=%~2"
if defined ARIA_HIK_CALIBRATION if exist "%ARIA_HIK_CALIBRATION%\hik_camera_calibration.json" set "ARIA_HIK_CALIBRATION=%ARIA_HIK_CALIBRATION%\hik_camera_calibration.json"
if not defined ARIA_HIK_CALIBRATION (
  for /f "usebackq delims=" %%F in (`powershell.exe -NoProfile -Command "$f=Get-ChildItem -LiteralPath '%ARIA_HIK_ROOT%artifacts' -Filter 'hik_camera_calibration.json' -Recurse -File -ErrorAction SilentlyContinue ^| Sort-Object LastWriteTime -Descending ^| Select-Object -First 1; if($f){$f.FullName}"`) do set "ARIA_HIK_CALIBRATION=%%F"
)
if not defined ARIA_HIK_CALIBRATION (
  echo No saved HIK calibration was found under artifacts.
  echo Run calibrate-hik-rig.bat and save a calibration first.
  exit /b 3
)
if not exist "%ARIA_HIK_CALIBRATION%" (
  echo Calibration does not exist: %ARIA_HIK_CALIBRATION%
  exit /b 3
)

set "PYTHONPATH=%ARIA_HIK_OPENCV%;%ARIA_HIK_ROOT%.tools;%ARIA_HIK_ROOT%"
set "ARIA_HIK_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_HIK_PYTHON%" set "ARIA_HIK_PYTHON=python"

echo Calibration: %ARIA_HIK_CALIBRATION%
echo The demo only controls Android display power. It does not launch apps or send touch input.
pushd "%ARIA_HIK_ROOT%"
"%ARIA_HIK_PYTHON%" -B -m acquisition.rig_calibration.hik.stream "%ARIA_HIK_CALIBRATION%" --gui --manage-phone-display %ARIA_HIK_STREAM_OPTION%
set "ARIA_HIK_EXIT=%ERRORLEVEL%"
popd
exit /b %ARIA_HIK_EXIT%
