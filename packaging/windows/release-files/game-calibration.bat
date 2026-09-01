@echo off
setlocal
cd /d "%~dp0"
"%~dp0apps\iris-game-calibration\iris-game-calibration.exe" %*
exit /b %errorlevel%
