@echo off
setlocal
cd /d "%~dp0"
"%~dp0apps\iris-game-color-calibration\iris-game-color-calibration.exe" %*
exit /b %errorlevel%
