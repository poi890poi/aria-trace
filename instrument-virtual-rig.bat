@echo off
setlocal
set "IRIS_INSTRUMENT_ROOT=%~dp0"
set "IRIS_INSTRUMENT_OPENCV=%IRIS_INSTRUMENT_ROOT%.tools\rig-calibration-opencv-gui"
if not exist "%IRIS_INSTRUMENT_OPENCV%\cv2\cv2.pyd" (
  echo GUI-capable OpenCV is not installed for virtual rig instrumentation.
  echo Run setup-hik-rig.bat once, then retry.
  exit /b 2
)
set "PYTHONPATH=%IRIS_INSTRUMENT_OPENCV%;%IRIS_INSTRUMENT_ROOT%.tools;%IRIS_INSTRUMENT_ROOT%"
set "IRIS_INSTRUMENT_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%IRIS_INSTRUMENT_PYTHON%" set "IRIS_INSTRUMENT_PYTHON=python"
pushd "%IRIS_INSTRUMENT_ROOT%"
"%IRIS_INSTRUMENT_PYTHON%" -B -m virtualhikcam.instrument %*
set "IRIS_INSTRUMENT_EXIT=%ERRORLEVEL%"
popd
exit /b %IRIS_INSTRUMENT_EXIT%
