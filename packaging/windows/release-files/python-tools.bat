@echo off
setlocal
cd /d "%~dp0"
if defined ARIA_PYTHON (
  set "ARIA_USER_PYTHON=%ARIA_PYTHON%"
) else (
  set "ARIA_USER_PYTHON=python"
)
"%ARIA_USER_PYTHON%" -B "%~dp0python\aria_tools.py" %*
exit /b %errorlevel%
