@echo off
setlocal
cd /d "%~dp0"
call "%~dp0iris-runtime-env.bat"
if defined IRIS_PYTHON (
  set "IRIS_USER_PYTHON=%IRIS_PYTHON%"
) else (
  set "IRIS_USER_PYTHON=python"
)
"%IRIS_USER_PYTHON%" -B "%~dp0python\iris_tools.py" %*
exit /b %errorlevel%
