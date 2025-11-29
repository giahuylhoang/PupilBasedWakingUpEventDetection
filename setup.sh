#!/bin/bash

# Setup script for first-time users
# This script ensures everything is set up correctly for running the analysis
# It checks all prerequisites and sets up the complete environment

# We'll handle errors manually to provide better user guidance

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

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macOS"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if [ -f /etc/debian_version ]; then
            echo "Debian/Ubuntu"
        elif [ -f /etc/redhat-release ]; then
            echo "RedHat/Fedora"
        else
            echo "Linux"
        fi
    else
        echo "Unknown"
    fi
}

OS_TYPE=$(detect_os)

# Track if setup can continue
SETUP_CAN_CONTINUE=true

echo "Checking prerequisites..."
echo ""

# Check if Python 3 is installed
echo "Checking Python 3 installation..."
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python 3 found: $PYTHON_VERSION"
else
    SETUP_CAN_CONTINUE=false
    print_error "Python 3 is not installed!"
    echo ""
    echo "════════════════════════════════════════"
    echo "Python 3 Installation Required"
    echo "════════════════════════════════════════"
    echo ""
    echo "Detected OS: $OS_TYPE"
    echo ""
    
    case "$OS_TYPE" in
        "macOS")
            echo "To install Python 3 on macOS:"
            echo ""
            echo "Option 1 - Using Homebrew (Recommended):"
            echo "  1. Install Homebrew if needed:"
            echo "     /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            echo "  2. Install Python 3:"
            echo "     brew install python3"
            echo ""
            echo "Option 2 - Using Official Installer:"
            echo "  1. Visit: https://www.python.org/downloads/"
            echo "  2. Download and run the macOS installer"
            echo "  3. Make sure to check 'Add Python to PATH' during installation"
            ;;
        "Debian/Ubuntu")
            echo "To install Python 3 on Ubuntu/Debian:"
            echo "  sudo apt-get update"
            echo "  sudo apt-get install python3 python3-pip python3-venv"
            ;;
        "RedHat/Fedora")
            echo "To install Python 3 on Fedora/CentOS:"
            echo "  sudo dnf install python3 python3-pip"
            echo ""
            echo "Or on older systems:"
            echo "  sudo yum install python3 python3-pip"
            ;;
        *)
            echo "To install Python 3:"
            echo "  Visit: https://www.python.org/downloads/"
            echo "  Download and install Python 3.7 or higher"
            echo "  Make sure to add Python to your PATH"
            ;;
    esac
    
    echo ""
    echo "════════════════════════════════════════"
    echo ""
    echo "For detailed installation instructions, see:"
    echo "  INSTALL_PYTHON.md"
    echo ""
    echo "════════════════════════════════════════"
    echo ""
    echo "NEXT STEP: Install Python 3 using the instructions above,"
    echo "then run this script again:"
    echo "  ./setup.sh"
    echo ""
    echo "Or see INSTALL_PYTHON.md for detailed instructions."
    echo ""
    exit 1
fi

# Check Python version
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)" 2>/dev/null)
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null)

if [ -z "$PYTHON_MAJOR" ] || [ -z "$PYTHON_MINOR" ]; then
    print_error "Could not determine Python version"
    SETUP_CAN_CONTINUE=false
elif [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 7 ]); then
    print_error "Python 3.7 or higher is required. Found: Python $PYTHON_MAJOR.$PYTHON_MINOR"
    echo ""
    echo "Please upgrade Python 3 to version 3.7 or higher."
    SETUP_CAN_CONTINUE=false
fi

# Check if venv module is available
echo ""
echo "Checking for venv module..."
if python3 -m venv --help >/dev/null 2>&1; then
    print_success "venv module is available"
else
    print_error "venv module is not available"
    SETUP_CAN_CONTINUE=false
    echo ""
    case "$OS_TYPE" in
        "Debian/Ubuntu")
            echo "Install with: sudo apt-get install python3-venv"
            ;;
        "macOS")
            echo "venv should be included with Python 3."
            echo "Try reinstalling Python 3 from python.org"
            ;;
        *)
            echo "Please install python3-venv for your system"
            ;;
    esac
fi

# Check if pip is available
echo ""
echo "Checking for pip..."
if python3 -m pip --version >/dev/null 2>&1; then
    PIP_VERSION=$(python3 -m pip --version 2>/dev/null | head -n1)
    print_success "pip is available"
else
    print_error "pip is not available"
    SETUP_CAN_CONTINUE=false
    echo ""
    echo "Attempting to install pip..."
    if python3 -m ensurepip --upgrade 2>/dev/null; then
        print_success "pip installed successfully"
        SETUP_CAN_CONTINUE=true
    else
        print_error "Could not auto-install pip"
        case "$OS_TYPE" in
            "Debian/Ubuntu")
                echo "Install with: sudo apt-get install python3-pip"
                ;;
            "RedHat/Fedora")
                echo "Install with: sudo dnf install python3-pip"
                ;;
            *)
                echo "See INSTALL_PYTHON.md for installation instructions"
                ;;
        esac
    fi
fi

# If prerequisites are missing, exit with helpful message
if [ "$SETUP_CAN_CONTINUE" = false ]; then
    echo ""
    echo "════════════════════════════════════════"
    print_error "Setup cannot continue - missing prerequisites"
    echo "════════════════════════════════════════"
    echo ""
    echo "Please fix the issues above and run ./setup.sh again"
    echo ""
    exit 1
fi

# Run init.sh to set up the environment
echo ""
echo "════════════════════════════════════════"
echo "Setting up virtual environment and installing dependencies..."
echo "════════════════════════════════════════"
echo ""

# Source init.sh (this will create venv and install dependencies)
if source ./init.sh; then
    print_success "Environment setup completed!"
else
    print_error "Environment setup failed!"
    echo ""
    echo "Try running manually:"
    echo "  source init.sh"
    echo "  pip install -r requirements.txt"
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
    print_error "Some required packages are missing!"
    echo ""
    echo "Attempting to install missing packages..."
    if pip install -r requirements.txt; then
        print_success "Packages installed successfully"
    else
        print_error "Failed to install packages automatically"
        echo ""
        echo "Try running manually:"
        echo "  source init.sh"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
fi

# Final verification
echo ""
echo "════════════════════════════════════════"
echo "Final Verification"
echo "════════════════════════════════════════"

FINAL_CHECK_PASSED=true
for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        print_success "$package is working"
    else
        print_error "$package import failed"
        FINAL_CHECK_PASSED=false
    fi
done

if [ "$FINAL_CHECK_PASSED" = false ]; then
    echo ""
    print_error "Some packages are not working correctly"
    echo "Try: source init.sh && pip install -r requirements.txt --force-reinstall"
    exit 1
fi

echo ""
echo "════════════════════════════════════════"
print_success "✓ Setup completed successfully!"
echo "════════════════════════════════════════"
echo ""
echo "Summary:"
echo "  ✓ Python 3 installed and working"
echo "  ✓ Virtual environment created"
echo "  ✓ All dependencies installed"
echo ""
echo "Next steps:"
echo "  1. Run an analysis:"
echo "     ./run_individual.sh -d \"data/raw/test/cycle_9\""
echo ""
echo "  2. Or activate the environment manually:"
echo "     source init.sh"
echo ""
echo "  3. Run tests to verify everything:"
echo "     ./test_setup.sh"
echo ""

