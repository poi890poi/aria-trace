@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-python-3.12.10.ps1" %*
exit /b %errorlevel%
