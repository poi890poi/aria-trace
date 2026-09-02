@echo off
setlocal
cd /d "%~dp0"
call "%~dp0iris-runtime-env.bat"
"%~dp0apps\iris-camera-adapter-demo\iris-camera-adapter-demo.exe" --gui %*
exit /b %errorlevel%
