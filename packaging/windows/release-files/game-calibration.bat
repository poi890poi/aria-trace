@echo off
setlocal
cd /d "%~dp0"
"%~dp0apps\aria-game-calibration\aria-game-calibration.exe" %*
exit /b %errorlevel%
