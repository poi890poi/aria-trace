@echo off
setlocal
cd /d "%~dp0"
call "%~dp0iris-runtime-env.bat"
"%~dp0apps\iris-game-calibration\iris-game-calibration.exe" %*
exit /b %errorlevel%
