@echo off
setlocal
set "IRIS_VIRTUAL_ROOT=%~dp0"
set "IRIS_VIRTUAL_OPENCV=%IRIS_VIRTUAL_ROOT%.tools\rig-calibration-opencv-gui"
set "PYTHONPATH=%IRIS_VIRTUAL_OPENCV%;%IRIS_VIRTUAL_ROOT%.tools;%IRIS_VIRTUAL_ROOT%"
set "IRIS_VIRTUAL_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%IRIS_VIRTUAL_PYTHON%" set "IRIS_VIRTUAL_PYTHON=python"
pushd "%IRIS_VIRTUAL_ROOT%"
"%IRIS_VIRTUAL_PYTHON%" -B -m virtualhikcam.control %*
set "IRIS_VIRTUAL_EXIT=%ERRORLEVEL%"
popd
exit /b %IRIS_VIRTUAL_EXIT%
