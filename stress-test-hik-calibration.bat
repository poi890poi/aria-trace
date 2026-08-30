@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stress-test-hik-calibration.ps1" %*
exit /b %errorlevel%
