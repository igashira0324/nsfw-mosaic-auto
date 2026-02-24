@echo off
chcp 65001 > nul
setlocal

REM Python仮想環境があれば有効化
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

python mosaic-video-speek.py

if exist mosaic-video-speek.bat del mosaic-video-speek.bat