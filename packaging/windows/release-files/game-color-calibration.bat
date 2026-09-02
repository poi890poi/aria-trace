@echo off
setlocal
cd /d "%~dp0"
call "%~dp0iris-runtime-env.bat"
"%~dp0apps\iris-game-color-calibration\iris-game-color-calibration.exe" %*
exit /b %errorlevel%
