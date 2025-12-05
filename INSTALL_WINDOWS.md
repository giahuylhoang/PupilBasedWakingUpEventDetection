# Installing and Setting Up on Windows

This guide will help you set up the Pupil-Based Waking Up Event Detection tool on Windows.

## Prerequisites

- **Windows 7 or later** (Windows 10/11 recommended)
- **Python 3.7 or higher** (Python 3.8+ recommended)
- **Internet connection** (for downloading dependencies)

## Step 1: Install Python 3

### Check if Python is Already Installed

Open Command Prompt or PowerShell and run:

```cmd
python --version
```

or

```cmd
python3 --version
```

If you see a version number (e.g., `Python 3.11.5`), you're good to go! Skip to [Step 2](#step-2-setup-the-project).

### Install Python 3

If Python is not installed, choose one of these methods:

#### Option 1: Microsoft Store (Recommended - Easiest)

1. Open **Microsoft Store**
2. Search for **"Python 3.11"** or **"Python 3.12"**
3. Click **Install**
4. The Python launcher will be automatically added to your PATH

#### Option 2: Official Python Installer

1. **Download Python:**
   - Visit: https://www.python.org/downloads/
   - Click "Download Python 3.x.x" (latest version)
   - Download the Windows installer (.exe file)

2. **Install Python:**
   - Run the downloaded installer
   - **IMPORTANT:** Check the box **"Add Python to PATH"** at the bottom of the installer
   - Click **"Install Now"**
   - Wait for installation to complete

3. **Verify Installation:**
   - Open a **new** Command Prompt or PowerShell window
   - Run: `python --version`
   - You should see the Python version number

### Troubleshooting Python Installation

**Problem: `python` command not found**

- Make sure you checked "Add Python to PATH" during installation
- Restart your computer after installation
- Try using `py` launcher instead: `py --version`
- Manually add Python to PATH:
  1. Find Python installation (usually `C:\Users\YourName\AppData\Local\Programs\Python\Python3xx`)
  2. Add it to System Environment Variables PATH

**Problem: Multiple Python versions**

- Use the `py` launcher: `py -3` for Python 3
- Or specify version: `py -3.11` for Python 3.11

## Step 2: Setup the Project

### Quick Setup (Recommended)

1. **Open Command Prompt or PowerShell** in the project directory

2. **Run the setup script:**

   **Using Batch file (Command Prompt):**
   ```cmd
   setup.bat
   ```

   **Using PowerShell (Better error handling):**
   ```powershell
   powershell -ExecutionPolicy Bypass -File setup.ps1
   ```

   The setup script will:
   - ✓ Check if Python 3 is installed
   - ✓ Check for pip and venv
   - ✓ Create a virtual environment
   - ✓ Install all required dependencies
   - ✓ Verify all packages are working

3. **If Python is not found**, the script will show you installation instructions. Install Python and run the setup script again.

### Manual Setup (Alternative)

If you prefer to set up manually:

1. **Initialize the environment:**
   ```cmd
   init.bat
   ```

   This will:
   - Create a virtual environment (`venv` folder)
   - Install dependencies from `requirements.txt`
   - Set up PYTHONPATH

2. **Verify installation:**
   ```cmd
   python -c "import numpy, pandas, scipy, matplotlib; print('All packages installed!')"
   ```

## Step 3: Verify Installation

Run the prerequisite checker:

```cmd
check_prerequisites.bat
```

This will verify that:
- Python 3.7+ is installed
- pip is available
- venv module is available

## Step 4: Run the Application

### Run Individual Analysis

```cmd
run_individual.bat -d "data\raw\test\cycle_10"
```

For more options:
```cmd
run_individual.bat --help
```

### Run Batch Analysis

Process multiple folders at once:

```cmd
REM Process all folders with CSV files in a directory
run_batch.bat -r "data\raw\test" -o "data\results\batch_run"

REM Process specific folders
run_batch.bat --folders "data\raw\test\cycle_9" "data\raw\test\cycle_10" -o "data\results\batch_run"
```

For more options:
```cmd
run_batch.bat --help
```

### Run Web Application

```cmd
run_webapp.bat
```

Then open your browser and go to: `http://localhost:5000`

## Common Issues and Solutions

### Issue: "python is not recognized"

**Solution:**
- Make sure Python is added to PATH
- Restart Command Prompt/PowerShell after installing Python
- Try using `py` launcher: `py -3` instead of `python`
- Use full path: `C:\Python3xx\python.exe`

### Issue: "pip is not recognized"

**Solution:**
```cmd
python -m ensurepip --upgrade
```

Or reinstall Python with "Add Python to PATH" checked.

### Issue: "venv module not found"

**Solution:**
- Reinstall Python 3 from python.org
- Make sure you're using Python 3.7 or higher
- Try: `python -m venv venv` manually

### Issue: Package installation fails

**Solution:**
- Make sure you're connected to the internet
- Try upgrading pip: `python -m pip install --upgrade pip`
- Install packages one by one to identify the problematic package
- Check if you have enough disk space

### Issue: Permission errors

**Solution:**
- Don't run as Administrator (not recommended)
- Make sure you have write permissions in the project directory
- Use virtual environment (which the setup script does automatically)

### Issue: Different Python versions

If you have multiple Python versions installed:

- Use `py -3` to use Python 3
- Use `py -3.11` to use Python 3.11 specifically
- The setup scripts will try to find the correct Python automatically

## Activating the Virtual Environment

After setup, whenever you want to use the project:

```cmd
init.bat
```

This activates the virtual environment and sets up the environment variables.

## Uninstalling

To remove the project:

1. Delete the project folder
2. The virtual environment (`venv` folder) will be deleted with it
3. No system-wide changes are made (everything is contained in the project folder)

## Getting Help

If you encounter issues:

1. Check this guide first
2. Run `check_prerequisites.bat` to verify your setup
3. Check the error messages - they usually indicate what's missing
4. Make sure you're using Python 3.7 or higher
5. Try running the setup script again

## System Requirements

- **Minimum**: Python 3.7
- **Recommended**: Python 3.8 or higher
- **Tested with**: Python 3.9, 3.10, 3.11, 3.12
- **Windows**: Windows 7 or later (Windows 10/11 recommended)

## Next Steps

After successful setup:

1. Read the [README.md](README.md) for usage instructions
2. Check [QUICK_START.md](QUICK_START.md) for a quick tutorial
3. Run your first analysis!

