@echo off
setlocal
set "ARIA_RIG_REVIEW_ROOT=%~dp0"
set "ARIA_RIG_REVIEW_OPENCV=%ARIA_RIG_REVIEW_ROOT%.tools\rig-calibration-opencv-gui"
if not exist "%ARIA_RIG_REVIEW_OPENCV%\cv2\cv2.pyd" (
  echo OpenCV is not installed. Run setup-hik-rig.bat once, then retry.
  exit /b 2
)
set "PYTHONPATH=%ARIA_RIG_REVIEW_OPENCV%;%ARIA_RIG_REVIEW_ROOT%.tools;%ARIA_RIG_REVIEW_ROOT%"
set "ARIA_RIG_REVIEW_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_RIG_REVIEW_PYTHON%" set "ARIA_RIG_REVIEW_PYTHON=python"
pushd "%ARIA_RIG_REVIEW_ROOT%"
"%ARIA_RIG_REVIEW_PYTHON%" -B -m rig_runtime.workflows.rig_evidence_review %*
set "ARIA_RIG_REVIEW_EXIT=%ERRORLEVEL%"
popd
exit /b %ARIA_RIG_REVIEW_EXIT%
