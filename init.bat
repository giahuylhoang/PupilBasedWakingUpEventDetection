@echo off
REM Initialize virtual environment and set up Python path for Windows
REM This script creates/activates venv and installs dependencies

setlocal enabledelayedexpansion

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Find Python command
set PYTHON_CMD=
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
    goto :python_found
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
    for /f "tokens=1 delims=." %%m in ("!PYTHON_VERSION!") do set PYTHON_MAJOR=%%m
    if "!PYTHON_MAJOR!"=="3" (
        set PYTHON_CMD=python
        goto :python_found
    )
)

py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3
    goto :python_found
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%v in ('py --version 2^>^&1') do set PYTHON_VERSION=%%v
    for /f "tokens=1 delims=." %%m in ("!PYTHON_VERSION!") do set PYTHON_MAJOR=%%m
    if "!PYTHON_MAJOR!"=="3" (
        set PYTHON_CMD=py
        goto :python_found
    )
)

echo [ERROR] Python 3 is not installed or not in PATH.
echo Please install Python 3 to continue.
exit /b 1

:python_found
REM Set PYTHONPATH to current directory
set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo Virtual environment found. Activating...
    call venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo Virtual environment not found. Creating a new one...
    !PYTHON_CMD! -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
    call venv\Scripts\activate.bat
    echo Virtual environment created and activated.
    
    REM Upgrade pip
    echo Upgrading pip...
    python -m pip install --upgrade pip --quiet
)

REM Install or upgrade dependencies from requirements.txt
if exist "requirements.txt" (
    echo Installing/updating dependencies from requirements.txt...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [WARNING] Some packages from requirements.txt may have failed to install.
        echo You may need to install them manually.
    ) else (
        echo Dependencies installed successfully.
    )
) else (
    echo [WARNING] requirements.txt not found. Skipping dependency installation.
)

endlocal

