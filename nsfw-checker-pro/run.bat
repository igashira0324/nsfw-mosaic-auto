@echo off
setlocal
cd /d "%~dp0"

echo Loading nsfw-checker-pro...
echo This may take a few seconds during engine initialization.

python main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application crashed.
    pause
)
