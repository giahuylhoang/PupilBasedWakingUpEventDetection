# Installing Python 3

This guide will help you install Python 3 on your system if you don't have it already.

## Check if Python 3 is Already Installed

First, check if Python 3 is already installed:

```bash
python3 --version
```

If this shows a version number (e.g., `Python 3.9.7`), you're good to go! Skip to the [Setup](#setup) section.

If you get a "command not found" error, follow the instructions below for your operating system.

## Installation by Operating System

### macOS

#### Option 1: Using Homebrew (Recommended)

1. **Install Homebrew** (if you don't have it):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3**:
   ```bash
   brew install python3
   ```

3. **Verify installation**:
   ```bash
   python3 --version
   pip3 --version
   ```

#### Option 2: Using Official Installer

1. **Download Python**:
   - Visit: https://www.python.org/downloads/
   - Click "Download Python 3.x.x" (latest version)
   - Download the macOS installer (.pkg file)

2. **Install Python**:
   - Open the downloaded .pkg file
   - Follow the installation wizard
   - **Important**: Check "Add Python to PATH" during installation

3. **Verify installation**:
   ```bash
   python3 --version
   pip3 --version
   ```

### Linux (Ubuntu/Debian)

```bash
# Update package list
sudo apt-get update

# Install Python 3 and pip
sudo apt-get install python3 python3-pip python3-venv

# Verify installation
python3 --version
pip3 --version
```

### Linux (Fedora/CentOS/RHEL)

```bash
# Install Python 3 and pip
sudo dnf install python3 python3-pip

# Or for older systems:
sudo yum install python3 python3-pip

# Verify installation
python3 --version
pip3 --version
```

### Linux (Arch Linux)

```bash
# Install Python 3 and pip
sudo pacman -S python python-pip

# Verify installation
python3 --version
pip3 --version
```

### Other Linux Distributions

Use your distribution's package manager:
- **OpenSUSE**: `sudo zypper install python3 python3-pip`
- **Alpine**: `sudo apk add python3 py3-pip`
- **Gentoo**: `sudo emerge dev-lang/python:3.11`

## Troubleshooting

### Python is installed but `python3` command not found

**On macOS:**
```bash
# Check if Python is in /usr/local/bin
ls -la /usr/local/bin/python*

# If found, add to PATH (add to ~/.zshrc or ~/.bash_profile):
export PATH="/usr/local/bin:$PATH"
```

**On Linux:**
```bash
# Check Python installation location
which python
whereis python3

# You might need to create a symlink:
sudo ln -s /usr/bin/python /usr/bin/python3
```

### pip3 not found

**Install pip separately:**
```bash
# Download get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

# Install pip
python3 get-pip.py

# Verify
pip3 --version
```

### Permission Errors

If you get permission errors, you might need to use `sudo` (Linux) or check your PATH settings.

**On macOS**, avoid using `sudo` with pip. Instead, use:
```bash
python3 -m pip install --user <package>
```

## After Installing Python

Once Python 3 is installed:

1. **Verify installation**:
   ```bash
   python3 --version
   pip3 --version
   ```

2. **Run the setup script**:
   ```bash
   cd PupilBasedWakingUpEventDetection
   ./setup.sh
   ```

## Still Having Issues?

If you're still having trouble:

1. Check the [Python.org documentation](https://www.python.org/about/gettingstarted/)
2. Make sure you're using a terminal/command prompt (not Python shell)
3. Restart your terminal after installation
4. Check that Python was added to your PATH environment variable

## System Requirements

- **Minimum**: Python 3.7
- **Recommended**: Python 3.8 or higher
- **Tested with**: Python 3.9, 3.10, 3.11, 3.12

