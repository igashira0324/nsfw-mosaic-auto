@echo off
setlocal
title NSFW Auto Mosaic & Checker - Unified Starter
color 0b

:MENU
cls
echo ============================================================
echo   NSFW Auto Mosaic & Checker Unified Starter
echo ============================================================
echo.
echo  [1] 画像自動モザイク (nsfw-mosaic-image.bat)
echo  [2] 動画自動モザイク - 標準 (nsfw-mosaic-video.bat)
echo  [3] 動画自動モザイク - 音声調整付き (nsfw-mosaic-video-speek.bat)
echo  [4] 統合型NSFWチェッカー (nsfw-checker-pro/run.bat)
echo.
echo  [Q] 終了
echo.
echo ============================================================
set /p choice="選択肢を入力してください (1-4, Q): "

if "%choice%"=="1" (
    start "" "nsfw-mosaic-image.bat"
    goto MENU
)
if "%choice%"=="2" (
    start "" "nsfw-mosaic-video.bat"
    goto MENU
)
if "%choice%"=="3" (
    start "" "nsfw-mosaic-video-speek.bat"
    goto MENU
)
if "%choice%"=="4" (
    cd nsfw-checker-pro
    start "" "run.bat"
    cd ..
    goto MENU
)
if /i "%choice%"=="Q" exit

echo 無効な選択です。
pause
goto MENU
