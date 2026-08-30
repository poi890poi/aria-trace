@echo off
setlocal
cd /d "%~dp0"
if defined ARIA_PYTHON (
  set "ARIA_ADAPTER_PYTHON=%ARIA_PYTHON%"
) else (
  set "ARIA_ADAPTER_PYTHON=python"
)
"%ARIA_ADAPTER_PYTHON%" -m pip install -r "%~dp0python\requirements-hik-camera-adapter.txt"
exit /b %errorlevel%
