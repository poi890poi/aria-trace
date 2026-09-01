@echo off
setlocal
cd /d "%~dp0"
if defined IRIS_PYTHON (
  set "IRIS_ADAPTER_PYTHON=%IRIS_PYTHON%"
) else (
  set "IRIS_ADAPTER_PYTHON=python"
)
"%IRIS_ADAPTER_PYTHON%" -m pip install -r "%~dp0python\requirements-hik-camera-adapter.txt"
exit /b %errorlevel%
