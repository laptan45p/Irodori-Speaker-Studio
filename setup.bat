@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
where python >nul 2>nul
if not errorlevel 1 (
    python -u setup_client.py
    goto completed
)
where py >nul 2>nul
if not errorlevel 1 (
    py -3 -u setup_client.py
    goto completed
)
where python3 >nul 2>nul
if not errorlevel 1 (
    python3 -u setup_client.py
    goto completed
)
echo Python was not found. Install Python for Windows first.
pause
exit /b 1
:completed
set "setup_status=%errorlevel%"
if not "%setup_status%"=="0" (
    echo.
    echo Setup failed. See setup.log above.
    pause
    exit /b %setup_status%
)
echo.
echo Setup complete. Double-click start.bat to launch.
pause
