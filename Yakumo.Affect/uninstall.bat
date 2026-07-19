@echo off
chcp 65001 > nul
echo Yakumo Affect Engine -- Uninstaller
echo.
powershell -ExecutionPolicy Bypass -File "%~dp0uninstall.ps1" %*
pause
