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

set "ARIA_WAKE_CALIBRATION=%~1"
if defined ARIA_WAKE_CALIBRATION if exist "%ARIA_WAKE_CALIBRATION%\hik_camera_calibration.json" set "ARIA_WAKE_CALIBRATION=%ARIA_WAKE_CALIBRATION%\hik_camera_calibration.json"
if not defined ARIA_WAKE_CALIBRATION (
  for /f "usebackq delims=" %%F in (`powershell.exe -NoProfile -Command "$f=Get-ChildItem -LiteralPath '%ARIA_WAKE_ROOT%artifacts' -Filter 'hik_camera_calibration.json' -Recurse -File -ErrorAction SilentlyContinue ^| Sort-Object LastWriteTime -Descending ^| Select-Object -First 1; if($f){$f.FullName}"`) do set "ARIA_WAKE_CALIBRATION=%%F"
)
if not defined ARIA_WAKE_CALIBRATION (
  echo No saved HIK calibration was found under artifacts.
  exit /b 3
)

set "PYTHONPATH=%ARIA_WAKE_OPENCV%;%ARIA_WAKE_ROOT%.tools;%ARIA_WAKE_ROOT%"
set "ARIA_WAKE_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_WAKE_PYTHON%" set "ARIA_WAKE_PYTHON=python"

"%ARIA_WAKE_PYTHON%" -B -m acquisition.detect_game_display_dim "%ARIA_WAKE_CALIBRATION%"
exit /b %errorlevel%
