@echo off
setlocal
cd /d "%~dp0"
call "%~dp0iris-runtime-env.bat"
if defined IRIS_PYTHON (
  set "IRIS_ADAPTER_PYTHON=%IRIS_PYTHON%"
) else (
  set "IRIS_ADAPTER_PYTHON=python"
)
set "PYTHONPATH=%~dp0python;%PYTHONPATH%"
"%IRIS_ADAPTER_PYTHON%" -B "%~dp0python\iris_tools.py" camera-adapter-demo --gui %*
exit /b %errorlevel%
