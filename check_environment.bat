@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if not exist "irodori\.venv\Scripts\python.exe" (
    echo Run setup.bat first.
    pause
    exit /b 1
)
"irodori\.venv\Scripts\python.exe" -u doctor.py --setup-check
set "check_status=%errorlevel%"
pause
exit /b %check_status%
