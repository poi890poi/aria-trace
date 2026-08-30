@echo off
setlocal
cd /d "%~dp0"
"%~dp0apps\aria-zigzag-acquisition\aria-zigzag-acquisition.exe" %*
exit /b %errorlevel%
