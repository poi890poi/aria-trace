@echo off
setlocal
cd /d "%~dp0"
"%~dp0apps\aria-game-color-calibration\aria-game-color-calibration.exe" %*
exit /b %errorlevel%
