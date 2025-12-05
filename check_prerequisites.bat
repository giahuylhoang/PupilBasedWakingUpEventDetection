@echo off
REM Quick prerequisite check script for Windows
REM Run this before setup.bat to check if you have everything needed

setlocal enabledelayedexpansion

echo ==========================================
echo Prerequisites Check (Windows)
echo ==========================================
echo.

set ALL_OK=true

REM Check Python 3
echo Checking: Python 3
set PYTHON_CMD=
set PYTHON_VERSION=

python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
    for /f "tokens=2" %%v in ('python3 --version 2^>^&1') do set PYTHON_VERSION=%%v
    goto :python_check_version
)

python --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
    for /f "tokens=1 delims=." %%m in ("!PYTHON_VERSION!") do set PYTHON_MAJOR=%%m
    if "!PYTHON_MAJOR!"=="3" (
        set PYTHON_CMD=python
        goto :python_check_version
    )
)

py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3
    for /f "tokens=2" %%v in ('py -3 --version 2^>^&1') do set PYTHON_VERSION=%%v
    goto :python_check_version
)

py --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%v in ('py --version 2^>^&1') do set PYTHON_VERSION=%%v
    for /f "tokens=1 delims=." %%m in ("!PYTHON_VERSION!") do set PYTHON_MAJOR=%%m
    if "!PYTHON_MAJOR!"=="3" (
        set PYTHON_CMD=py
        goto :python_check_version
    )
)

echo [ERROR] Python 3 is not installed
set ALL_OK=false
echo.
echo Installation instructions:
echo   Option 1 - Microsoft Store:
echo     Open Microsoft Store and search for "Python 3.11" or "Python 3.12"
echo.
echo   Option 2 - Official Installer:
echo     Visit: https://www.python.org/downloads/
echo     Download and run the Windows installer
echo     IMPORTANT: Check "Add Python to PATH" during installation
echo.
goto :check_pip

:python_check_version
echo [OK] !PYTHON_VERSION! is installed (using: !PYTHON_CMD!)

REM Check version (must be 3.7+)
for /f "tokens=1,2 delims=." %%a in ("!PYTHON_VERSION!") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

set PYTHON_MAJOR=!PYTHON_MAJOR: =!
set PYTHON_MINOR=!PYTHON_MINOR: =!

if "!PYTHON_MAJOR!" lss "3" (
    echo [ERROR] Python 3.7+ required, found !PYTHON_VERSION!
    set ALL_OK=false
) else if "!PYTHON_MAJOR!" equ "3" (
    if "!PYTHON_MINOR!" lss "7" (
        echo [ERROR] Python 3.7+ required, found !PYTHON_VERSION!
        set ALL_OK=false
    )
)

:check_pip
echo.

REM Check pip
echo Checking: pip
!PYTHON_CMD! -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip is not available
    set ALL_OK=false
    echo Try: !PYTHON_CMD! -m ensurepip --upgrade
) else (
    for /f "tokens=2" %%v in ('!PYTHON_CMD! -m pip --version 2^>^&1') do set PIP_VERSION=%%v
    echo [OK] pip is available (version !PIP_VERSION!)
)
echo.

REM Check venv
echo Checking: venv module
!PYTHON_CMD! -m venv --help >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] venv module is not available
    set ALL_OK=false
    echo venv should come with Python 3. Reinstall Python if missing.
) else (
    echo [OK] venv module is available
)
echo.

REM Summary
echo ==========================================
if "!ALL_OK!"=="true" (
    echo [OK] All prerequisites are met!
    echo.
    echo You can now run:
    echo   setup.bat
    exit /b 0
) else (
    echo [ERROR] Some prerequisites are missing
    echo.
    echo Please install the missing components and run this check again:
    echo   check_prerequisites.bat
    echo.
    echo For detailed installation instructions, see:
    echo   INSTALL_PYTHON.md
    exit /b 1
)

endlocal

