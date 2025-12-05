@echo off
REM Run web application script for Windows

REM Run init.bat to set up the environment
call init.bat
if %errorlevel% neq 0 (
    echo [ERROR] Failed to initialize environment
    exit /b 1
)

REM Check if Flask is installed
python -c "import flask" >nul 2>&1
if %errorlevel% neq 0 (
    echo Installing Flask...
    pip install flask
)

REM Run the web app
echo.
echo Starting web application...
echo If port 5000 is in use, the app will automatically use the next available port.
echo.
python webapp.py

