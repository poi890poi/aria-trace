@echo off
setlocal
cd /d "%~dp0"
"%~dp0apps\iris-rig-calibration\iris-rig-calibration.exe" %*
exit /b %errorlevel%
