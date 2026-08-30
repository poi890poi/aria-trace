@echo off
setlocal
cd /d "%~dp0"
set "ARIA_PROFILE_PYTHON=C:\Program Files\Python37\python.exe"
if not exist "%ARIA_PROFILE_PYTHON%" set "ARIA_PROFILE_PYTHON=python"
set "PYTHONPATH=%~dp0.tools\rig-calibration-opencv-gui;%~dp0.tools;%~dp0"
"%ARIA_PROFILE_PYTHON%" -B -m acquisition.profile_manager %*
exit /b %ERRORLEVEL%
