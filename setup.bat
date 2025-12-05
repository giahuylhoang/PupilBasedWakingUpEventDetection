@echo off
REM Setup script for Windows
REM This script ensures everything is set up correctly for running the analysis
REM It checks all prerequisites and sets up the complete environment

setlocal enabledelayedexpansion

echo ==========================================
echo Pupil-Based Waking Up Event Detection
echo First-Time Setup (Windows)
echo ==========================================
echo.

REM Get the directory where this script is located
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

set SETUP_CAN_CONTINUE=true

echo Checking prerequisites...
echo.

REM Check for Python - try multiple common names
echo Checking Python 3 installation...
set PYTHON_CMD=
set PYTHON_VERSION=

REM Try python3 first
python3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=python3
    for /f "tokens=2" %%v in ('python3 --version 2^>^&1') do set PYTHON_VERSION=%%v
    goto :python_found
)

REM Try python
python --version >nul 2>&1
if %errorlevel% equ 0 (
    REM Check if it's Python 3
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
    for /f "tokens=1 delims=." %%m in ("!PYTHON_VERSION!") do set PYTHON_MAJOR=%%m
    if "!PYTHON_MAJOR!"=="3" (
        set PYTHON_CMD=python
        goto :python_found
    )
)

REM Try py launcher with -3 flag
py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set PYTHON_CMD=py -3
    for /f "tokens=2" %%v in ('py -3 --version 2^>^&1') do set PYTHON_VERSION=%%v
    goto :python_found
)

REM Try py launcher without flag
py --version >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=2" %%v in ('py --version 2^>^&1') do set PYTHON_VERSION=%%v
    for /f "tokens=1 delims=." %%m in ("!PYTHON_VERSION!") do set PYTHON_MAJOR=%%m
    if "!PYTHON_MAJOR!"=="3" (
        set PYTHON_CMD=py
        goto :python_found
    )
)

REM Python not found
echo [ERROR] Python 3 is not installed!
echo.
echo ================================================
echo Python 3 Installation Required
echo ================================================
echo.
echo To install Python 3 on Windows:
echo.
echo Option 1 - Using Microsoft Store (Recommended):
echo   1. Open Microsoft Store
echo   2. Search for "Python 3.11" or "Python 3.12"
echo   3. Click Install
echo.
echo Option 2 - Using Official Installer:
echo   1. Visit: https://www.python.org/downloads/
echo   2. Download the latest Python 3.x Windows installer
echo   3. Run the installer
echo   4. IMPORTANT: Check "Add Python to PATH" during installation
echo   5. Click "Install Now"
echo.
echo Option 3 - Using py launcher (if you have multiple Python versions):
echo   The py launcher should work automatically if Python is installed
echo.
echo ================================================
echo.
echo NEXT STEP: Install Python 3 using the instructions above,
echo then run this script again:
echo   setup.bat
echo.
echo For detailed installation instructions, see:
echo   INSTALL_PYTHON.md
echo.
exit /b 1

:python_found
echo [OK] Python found: !PYTHON_VERSION!
echo      Using command: !PYTHON_CMD!
echo.

REM Check Python version (must be 3.7+)
echo Checking Python version...
for /f "tokens=1,2 delims=." %%a in ("!PYTHON_VERSION!") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

REM Remove any non-numeric characters from version numbers
set PYTHON_MAJOR=!PYTHON_MAJOR: =!
set PYTHON_MINOR=!PYTHON_MINOR: =!

if "!PYTHON_MAJOR!" lss "3" (
    echo [ERROR] Python 3.7 or higher is required. Found: Python !PYTHON_VERSION!
    set SETUP_CAN_CONTINUE=false
) else if "!PYTHON_MAJOR!" equ "3" (
    if "!PYTHON_MINOR!" lss "7" (
        echo [ERROR] Python 3.7 or higher is required. Found: Python !PYTHON_VERSION!
        set SETUP_CAN_CONTINUE=false
    )
)

if "!SETUP_CAN_CONTINUE!"=="false" (
    echo.
    echo Please upgrade Python 3 to version 3.7 or higher.
    echo Download from: https://www.python.org/downloads/
    echo.
    exit /b 1
)

REM Check for venv module
echo.
echo Checking for venv module...
!PYTHON_CMD! -m venv --help >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] venv module is not available
    set SETUP_CAN_CONTINUE=false
    echo.
    echo venv should be included with Python 3.
    echo Try reinstalling Python 3 from python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
) else (
    echo [OK] venv module is available
)

REM Check for pip
echo.
echo Checking for pip...
!PYTHON_CMD! -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] pip is not available
    echo Attempting to install pip...
    !PYTHON_CMD! -m ensurepip --upgrade >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Could not install pip
        set SETUP_CAN_CONTINUE=false
        echo.
        echo Please install pip manually or reinstall Python 3
        echo.
    ) else (
        echo [OK] pip installed successfully
    )
) else (
    echo [OK] pip is available
)

REM If prerequisites are missing, exit
if "!SETUP_CAN_CONTINUE!"=="false" (
    echo.
    echo ================================================
    echo [ERROR] Setup cannot continue - missing prerequisites
    echo ================================================
    echo.
    echo Please fix the issues above and run setup.bat again
    echo.
    exit /b 1
)

REM Run init.bat to set up the environment
echo.
echo ================================================
echo Setting up virtual environment and installing dependencies...
echo ================================================
echo.

call init.bat
if %errorlevel% neq 0 (
    echo [ERROR] Environment setup failed!
    echo.
    echo Try running manually:
    echo   init.bat
    echo   pip install -r requirements.txt
    exit /b 1
)

echo [OK] Environment setup completed!
echo.

REM Verify installation
echo Verifying installation...
call venv\Scripts\activate.bat

REM Check if key packages are installed
set ALL_INSTALLED=true

echo.
echo Checking installed packages...
python -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] numpy is NOT installed
    set ALL_INSTALLED=false
) else (
    echo [OK] numpy is installed
)

python -c "import pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pandas is NOT installed
    set ALL_INSTALLED=false
) else (
    echo [OK] pandas is installed
)

python -c "import scipy" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] scipy is NOT installed
    set ALL_INSTALLED=false
) else (
    echo [OK] scipy is installed
)

python -c "import matplotlib" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] matplotlib is NOT installed
    set ALL_INSTALLED=false
) else (
    echo [OK] matplotlib is installed
)

if "!ALL_INSTALLED!"=="false" (
    echo.
    echo [WARNING] Some required packages are missing!
    echo.
    echo Attempting to install missing packages...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install packages automatically
        echo.
        echo Try running manually:
        echo   init.bat
        echo   pip install -r requirements.txt
        exit /b 1
    ) else (
        echo [OK] Packages installed successfully
    )
)

REM Final verification
echo.
echo ================================================
echo Final Verification
echo ================================================

set FINAL_CHECK_PASSED=true
for %%p in (numpy pandas scipy matplotlib) do (
    python -c "import %%p" >nul 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] %%p import failed
        set FINAL_CHECK_PASSED=false
    ) else (
        echo [OK] %%p is working
    )
)

if "!FINAL_CHECK_PASSED!"=="false" (
    echo.
    echo [ERROR] Some packages are not working correctly
    echo Try: init.bat ^&^& pip install -r requirements.txt --force-reinstall
    exit /b 1
)

echo.
echo ================================================
echo [OK] Setup completed successfully!
echo ================================================
echo.
echo Summary:
echo   [OK] Python 3 installed and working
echo   [OK] Virtual environment created
echo   [OK] All dependencies installed
echo.
echo Next steps:
echo   1. Activate the environment:
echo      init.bat
echo.
echo   2. Run an analysis:
echo      run_individual.bat -d "data\raw\test\cycle_9"
echo.
echo   3. Or run the web app:
echo      run_webapp.bat
echo.

endlocal

