#!/bin/bash

# Quick prerequisite check script
# Run this before setup.sh to check if you have everything needed

echo "=========================================="
echo "Prerequisites Check"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_check() {
    echo -e "${BLUE}Checking:${NC} $1"
}

print_ok() {
    echo -e "${GREEN}✓${NC} $1"
}

print_missing() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Detect OS
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
echo "Operating System: $OS_TYPE"
echo ""

ALL_OK=true

# Check Python 3
print_check "Python 3"
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version)
    print_ok "$PYTHON_VERSION is installed"
    
    # Check version
    PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
    PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 7 ]); then
        print_missing "Python 3.7+ required, found $PYTHON_MAJOR.$PYTHON_MINOR"
        ALL_OK=false
    fi
else
    print_missing "Python 3 is not installed"
    ALL_OK=false
    echo ""
    print_info "Installation instructions for $OS_TYPE:"
    case "$OS_TYPE" in
        "macOS")
            echo "  brew install python3"
            echo "  Or download from: https://www.python.org/downloads/"
            ;;
        "Debian/Ubuntu")
            echo "  sudo apt-get update"
            echo "  sudo apt-get install python3 python3-pip python3-venv"
            ;;
        "RedHat/Fedora")
            echo "  sudo dnf install python3 python3-pip"
            ;;
        *)
            echo "  Visit: https://www.python.org/downloads/"
            ;;
    esac
    echo ""
    print_info "See INSTALL_PYTHON.md for detailed instructions"
fi
echo ""

# Check pip
print_check "pip"
if python3 -m pip --version >/dev/null 2>&1; then
    PIP_VERSION=$(python3 -m pip --version | cut -d' ' -f2)
    print_ok "pip is available (version $PIP_VERSION)"
else
    print_missing "pip is not available"
    ALL_OK=false
    print_info "Try: python3 -m ensurepip --upgrade"
fi
echo ""

# Check venv
print_check "venv module"
if python3 -m venv --help >/dev/null 2>&1; then
    print_ok "venv module is available"
else
    print_missing "venv module is not available"
    ALL_OK=false
    case "$OS_TYPE" in
        "Debian/Ubuntu")
            print_info "Install with: sudo apt-get install python3-venv"
            ;;
        *)
            print_info "Usually comes with Python 3. Reinstall Python if missing."
            ;;
    esac
fi
echo ""

# Summary
echo "=========================================="
if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}✓ All prerequisites are met!${NC}"
    echo ""
    echo "You can now run:"
    echo "  ./setup.sh"
    exit 0
else
    echo -e "${RED}✗ Some prerequisites are missing${NC}"
    echo ""
    echo "Please install the missing components and run this check again:"
    echo "  ./check_prerequisites.sh"
    echo ""
    echo "For detailed installation instructions, see:"
    echo "  INSTALL_PYTHON.md"
    exit 1
fi

