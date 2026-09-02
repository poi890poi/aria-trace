@echo off
setlocal
cd /d "%~dp0"
call "%~dp0iris-runtime-env.bat"
"%~dp0apps\iris-zigzag-acquisition\iris-zigzag-acquisition.exe" %*
exit /b %errorlevel%
