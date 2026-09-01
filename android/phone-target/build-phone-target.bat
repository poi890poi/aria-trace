@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-phone-target.ps1" %*
exit /b %ERRORLEVEL%
