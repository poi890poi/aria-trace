@echo off
setlocal
cd /d "%~dp0"
"%~dp0apps\aria-rig-calibration\aria-rig-calibration.exe" %*
exit /b %errorlevel%
