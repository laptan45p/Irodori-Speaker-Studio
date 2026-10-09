@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set GRADIO_ANALYTICS_ENABLED=False
if not exist "irodori\.venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
"irodori\.venv\Scripts\python.exe" -u app.py --open-browser %*
set "app_status=%errorlevel%"
if not "%app_status%"=="0" (
    echo.
    echo Application stopped with an error. Copy the error above.
    pause
)
exit /b %app_status%
