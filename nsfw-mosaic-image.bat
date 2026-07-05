@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"

REM Python仮想環境があれば有効化
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

python mosaic-image.py %*
