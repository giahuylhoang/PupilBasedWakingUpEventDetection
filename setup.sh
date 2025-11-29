#!/bin/bash

# Setup script for first-time users
# This script ensures everything is set up correctly for running the analysis

set -e  # Exit on error

echo "=========================================="
echo "Pupil-Based Waking Up Event Detection"
echo "First-Time Setup"
echo "=========================================="
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Check if Python 3 is installed
echo "Checking Python 3 installation..."
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python 3 found: $PYTHON_VERSION"
else
    print_error "Python 3 is not installed!"
    echo ""
    echo "Please install Python 3.7 or higher:"
    echo "  macOS: brew install python3"
    echo "  Ubuntu/Debian: sudo apt-get install python3"
    echo "  Or download from: https://www.python.org/downloads/"
    exit 1
fi

# Check Python version
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 7 ]); then
    print_error "Python 3.7 or higher is required. Found: Python $PYTHON_MAJOR.$PYTHON_MINOR"
    exit 1
fi

# Check if venv module is available
echo ""
echo "Checking for venv module..."
if python3 -m venv --help >/dev/null 2>&1; then
    print_success "venv module is available"
else
    print_error "venv module is not available"
    echo ""
    echo "Please install python3-venv:"
    echo "  macOS: Usually included with Python 3"
    echo "  Ubuntu/Debian: sudo apt-get install python3-venv"
    exit 1
fi

# Check if pip is available
echo ""
echo "Checking for pip..."
if python3 -m pip --version >/dev/null 2>&1; then
    print_success "pip is available"
else
    print_error "pip is not available"
    echo ""
    echo "Please install pip:"
    echo "  python3 -m ensurepip --upgrade"
    exit 1
fi

# Run init.sh to set up the environment
echo ""
echo "Setting up virtual environment and installing dependencies..."
echo ""

# Source init.sh (this will create venv and install dependencies)
if source ./init.sh; then
    print_success "Environment setup completed successfully!"
else
    print_error "Environment setup failed!"
    exit 1
fi

# Verify installation
echo ""
echo "Verifying installation..."
source venv/bin/activate

# Check if key packages are installed
REQUIRED_PACKAGES=("numpy" "pandas" "scipy" "matplotlib")
ALL_INSTALLED=true

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        print_success "$package is installed"
    else
        print_error "$package is NOT installed"
        ALL_INSTALLED=false
    fi
done

if [ "$ALL_INSTALLED" = false ]; then
    echo ""
    print_error "Some required packages are missing. Try running:"
    echo "  source init.sh"
    echo "  pip install -r requirements.txt"
    exit 1
fi

echo ""
echo "=========================================="
print_success "Setup completed successfully!"
echo "=========================================="
echo ""
echo "You can now run the analysis using:"
echo "  ./run_individual.sh -d \"data/raw/test/cycle_9\""
echo ""
echo "Or activate the environment manually with:"
echo "  source init.sh"
echo ""

