@echo off
setlocal
cd /d "%~dp0"
call "%~dp0iris-runtime-env.bat"
"%~dp0apps\iris-minimap-calibration\iris-minimap-calibration.exe" %*
exit /b %errorlevel%
