@echo off
setlocal
cd /d "%~dp0"
if defined ARIA_PYTHON (
  set "ARIA_USER_PYTHON=%ARIA_PYTHON%"
) else (
  set "ARIA_USER_PYTHON=python"
)
"%ARIA_USER_PYTHON%" -m pip install -r "%~dp0python\requirements-python-tools.txt"
exit /b %errorlevel%
