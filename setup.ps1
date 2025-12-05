# PowerShell setup script for Windows
# This script provides better error handling and user experience than the batch file
# Run with: powershell -ExecutionPolicy Bypass -File setup.ps1

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Pupil-Based Waking Up Event Detection" -ForegroundColor Cyan
Write-Host "First-Time Setup (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Get script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$SetupCanContinue = $true

Write-Host "Checking prerequisites..." -ForegroundColor Yellow
Write-Host ""

# Function to find Python
function Find-Python {
    $pythonCommands = @("python3", "python", "py -3", "py")
    
    foreach ($cmd in $pythonCommands) {
        try {
            $versionOutput = & cmd /c "$cmd --version 2>&1"
            if ($LASTEXITCODE -eq 0) {
                # Check if it's Python 3
                if ($versionOutput -match "Python 3\.(\d+)") {
                    $minorVersion = [int]$matches[1]
                    if ($minorVersion -ge 7) {
                        return @{
                            Command = $cmd
                            Version = $versionOutput.Trim()
                        }
                    }
                }
            }
        } catch {
            continue
        }
    }
    return $null
}

# Check Python
Write-Host "Checking Python 3 installation..." -ForegroundColor Yellow
$python = Find-Python

if ($null -eq $python) {
    Write-Host "[ERROR] Python 3 is not installed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "===============================================" -ForegroundColor Yellow
    Write-Host "Python 3 Installation Required" -ForegroundColor Yellow
    Write-Host "===============================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To install Python 3 on Windows:" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Option 1 - Using Microsoft Store (Recommended):" -ForegroundColor Green
    Write-Host "  1. Open Microsoft Store"
    Write-Host "  2. Search for 'Python 3.11' or 'Python 3.12'"
    Write-Host "  3. Click Install"
    Write-Host ""
    Write-Host "Option 2 - Using Official Installer:" -ForegroundColor Green
    Write-Host "  1. Visit: https://www.python.org/downloads/"
    Write-Host "  2. Download the latest Python 3.x Windows installer"
    Write-Host "  3. Run the installer"
    Write-Host "  4. IMPORTANT: Check 'Add Python to PATH' during installation"
    Write-Host "  5. Click 'Install Now'"
    Write-Host ""
    Write-Host "===============================================" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "NEXT STEP: Install Python 3, then run this script again:" -ForegroundColor Yellow
    Write-Host "  powershell -ExecutionPolicy Bypass -File setup.ps1" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

Write-Host "[OK] Python found: $($python.Version)" -ForegroundColor Green
Write-Host "     Using command: $($python.Command)" -ForegroundColor Gray
Write-Host ""

# Check venv module
Write-Host "Checking for venv module..." -ForegroundColor Yellow
try {
    $null = & cmd /c "$($python.Command) -m venv --help 2>&1"
    Write-Host "[OK] venv module is available" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] venv module is not available" -ForegroundColor Red
    $SetupCanContinue = $false
    Write-Host "venv should be included with Python 3." -ForegroundColor Yellow
    Write-Host "Try reinstalling Python 3 from python.org" -ForegroundColor Yellow
}

# Check pip
Write-Host ""
Write-Host "Checking for pip..." -ForegroundColor Yellow
try {
    $pipVersion = & cmd /c "$($python.Command) -m pip --version 2>&1"
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] pip is available" -ForegroundColor Green
    } else {
        throw "pip not found"
    }
} catch {
    Write-Host "[WARNING] pip is not available" -ForegroundColor Yellow
    Write-Host "Attempting to install pip..." -ForegroundColor Yellow
    try {
        $null = & cmd /c "$($python.Command) -m ensurepip --upgrade 2>&1"
        Write-Host "[OK] pip installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Could not install pip" -ForegroundColor Red
        $SetupCanContinue = $false
    }
}

if (-not $SetupCanContinue) {
    Write-Host ""
    Write-Host "===============================================" -ForegroundColor Red
    Write-Host "[ERROR] Setup cannot continue - missing prerequisites" -ForegroundColor Red
    Write-Host "===============================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please fix the issues above and run setup.ps1 again" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

# Run init.bat to set up environment
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Setting up virtual environment and installing dependencies..." -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

try {
    & cmd /c "init.bat"
    if ($LASTEXITCODE -ne 0) {
        throw "init.bat failed"
    }
    Write-Host "[OK] Environment setup completed!" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Environment setup failed!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Try running manually:" -ForegroundColor Yellow
    Write-Host "  init.bat" -ForegroundColor Cyan
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Cyan
    exit 1
}

# Verify installation
Write-Host ""
Write-Host "Verifying installation..." -ForegroundColor Yellow

$requiredPackages = @("numpy", "pandas", "scipy", "matplotlib")
$allInstalled = $true

foreach ($package in $requiredPackages) {
    try {
        $null = & cmd /c "python -c `"import $package`" 2>&1"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] $package is installed" -ForegroundColor Green
        } else {
            Write-Host "[ERROR] $package is NOT installed" -ForegroundColor Red
            $allInstalled = $false
        }
    } catch {
        Write-Host "[ERROR] $package is NOT installed" -ForegroundColor Red
        $allInstalled = $false
    }
}

if (-not $allInstalled) {
    Write-Host ""
    Write-Host "[WARNING] Some required packages are missing!" -ForegroundColor Yellow
    Write-Host "Attempting to install missing packages..." -ForegroundColor Yellow
    try {
        & cmd /c "pip install -r requirements.txt"
        if ($LASTEXITCODE -ne 0) {
            throw "pip install failed"
        }
        Write-Host "[OK] Packages installed successfully" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Failed to install packages automatically" -ForegroundColor Red
        Write-Host ""
        Write-Host "Try running manually:" -ForegroundColor Yellow
        Write-Host "  init.bat" -ForegroundColor Cyan
        Write-Host "  pip install -r requirements.txt" -ForegroundColor Cyan
        exit 1
    }
}

# Final verification
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Final Verification" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

$finalCheckPassed = $true
foreach ($package in $requiredPackages) {
    try {
        $null = & cmd /c "python -c `"import $package`" 2>&1"
        if ($LASTEXITCODE -eq 0) {
            Write-Host "[OK] $package is working" -ForegroundColor Green
        } else {
            Write-Host "[ERROR] $package import failed" -ForegroundColor Red
            $finalCheckPassed = $false
        }
    } catch {
        Write-Host "[ERROR] $package import failed" -ForegroundColor Red
        $finalCheckPassed = $false
    }
}

if (-not $finalCheckPassed) {
    Write-Host ""
    Write-Host "[ERROR] Some packages are not working correctly" -ForegroundColor Red
    Write-Host "Try: init.bat && pip install -r requirements.txt --force-reinstall" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "===============================================" -ForegroundColor Green
Write-Host "[OK] Setup completed successfully!" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Summary:" -ForegroundColor Cyan
Write-Host "  [OK] Python 3 installed and working" -ForegroundColor Green
Write-Host "  [OK] Virtual environment created" -ForegroundColor Green
Write-Host "  [OK] All dependencies installed" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Activate the environment:" -ForegroundColor Yellow
Write-Host "     init.bat" -ForegroundColor Cyan
Write-Host ""
Write-Host "  2. Run an analysis:" -ForegroundColor Yellow
Write-Host "     run_individual.bat -d `"data\raw\test\cycle_9`"" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Or run the web app:" -ForegroundColor Yellow
Write-Host "     run_webapp.bat" -ForegroundColor Cyan
Write-Host ""

