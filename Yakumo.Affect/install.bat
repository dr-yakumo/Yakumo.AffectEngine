@echo off
chcp 65001 > nul
echo Yakumo Affect Engine -- Installer
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause
