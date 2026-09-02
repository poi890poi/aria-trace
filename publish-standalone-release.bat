@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0publish-standalone-release.ps1" %*
exit /b %errorlevel%
