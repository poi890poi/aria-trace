@echo off
setlocal
cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0calibrate-hik-game-camera.ps1" %*
exit /b %errorlevel%
