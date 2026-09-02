@echo off
setlocal
cd /d "%~dp0"
call "%~dp0iris-runtime-env.bat"
"%~dp0apps\iris-rig-calibration\iris-rig-calibration.exe" %*
exit /b %errorlevel%
