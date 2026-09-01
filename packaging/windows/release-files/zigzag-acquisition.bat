@echo off
setlocal
cd /d "%~dp0"
"%~dp0apps\iris-zigzag-acquisition\iris-zigzag-acquisition.exe" %*
exit /b %errorlevel%
