@echo off
setlocal
cd /d "%~dp0"
if defined IRIS_PYTHON (
  set "IRIS_USER_PYTHON=%IRIS_PYTHON%"
) else (
  set "IRIS_USER_PYTHON=python"
)
"%IRIS_USER_PYTHON%" -m pip install -r "%~dp0python\requirements-python-tools.txt"
exit /b %errorlevel%
