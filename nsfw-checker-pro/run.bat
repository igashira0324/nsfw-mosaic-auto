@echo off
setlocal
cd /d "%~dp0"

REM Check for venv in parent directory
set VENV_PATH=..\venv\Scripts\python.exe
if exist "%VENV_PATH%" (
    echo Using virtual environment: %VENV_PATH%
    set PYTHON_CMD="%VENV_PATH%"
) else (
    echo Virtual environment not found. Using system python.
    set PYTHON_CMD=python
)

echo Loading nsfw-checker-pro...
echo This may take a few seconds during engine initialization.

%PYTHON_CMD% main.py

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application crashed.
    pause
)
