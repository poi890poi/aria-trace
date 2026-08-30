@echo off
setlocal
cd /d "%~dp0"
"%~dp0apps\aria-minimap-calibration\aria-minimap-calibration.exe" %*
exit /b %errorlevel%
