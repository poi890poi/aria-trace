@echo off
setlocal
cd /d "%~dp0"
if defined ARIA_PYTHON (
  set "ARIA_ADAPTER_PYTHON=%ARIA_PYTHON%"
) else (
  set "ARIA_ADAPTER_PYTHON=python"
)
set "PYTHONPATH=%~dp0python;%PYTHONPATH%"
"%ARIA_ADAPTER_PYTHON%" -B -m acquisition.rig_calibration.hik.stream --gui %*
exit /b %errorlevel%
