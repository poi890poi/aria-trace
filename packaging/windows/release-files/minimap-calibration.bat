@echo off
setlocal
cd /d "%~dp0"
"%~dp0apps\iris-minimap-calibration\iris-minimap-calibration.exe" %*
exit /b %errorlevel%
