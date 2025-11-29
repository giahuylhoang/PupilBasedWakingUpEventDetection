#!/bin/bash

# Quick test script to verify the setup works on a new computer
# This script checks all components and runs a basic functionality test

set -e  # Exit on error

echo "=========================================="
echo "Quick Setup Test"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_test() {
    echo -e "${BLUE}Testing:${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓ PASS${NC}: $1"
}

print_error() {
    echo -e "${RED}✗ FAIL${NC}: $1"
}

print_warning() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
}

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

TESTS_PASSED=0
TESTS_FAILED=0

# Test 1: Check Python 3
print_test "Python 3 installation"
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version)
    print_success "Python 3 found: $PYTHON_VERSION"
    ((TESTS_PASSED++))
else
    print_error "Python 3 not found"
    ((TESTS_FAILED++))
    echo "  → Install Python 3 to continue"
fi
echo ""

# Test 2: Check if virtual environment exists
print_test "Virtual environment"
if [ -d "venv" ]; then
    print_success "Virtual environment found"
    ((TESTS_PASSED++))
else
    print_warning "Virtual environment not found"
    echo "  → Run './setup.sh' or 'source init.sh' to create it"
fi
echo ""

# Test 3: Check if venv is activated or can be activated
print_test "Virtual environment activation"
if [ -d "venv" ]; then
    if source venv/bin/activate 2>/dev/null; then
        print_success "Virtual environment can be activated"
        ((TESTS_PASSED++))
        
        # Check pip in venv
        if command -v pip >/dev/null 2>&1 || command -v pip3 >/dev/null 2>&1; then
            print_success "pip is available in venv"
            ((TESTS_PASSED++))
        else
            print_error "pip not found in virtual environment"
            ((TESTS_FAILED++))
        fi
    else
        print_error "Cannot activate virtual environment"
        ((TESTS_FAILED++))
    fi
else
    print_warning "Skipping (venv not found)"
fi
echo ""

# Test 4: Check required Python packages
print_test "Required Python packages"
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null
    
    REQUIRED_PACKAGES=("numpy" "pandas" "scipy" "matplotlib" "seaborn" "sklearn")
    PACKAGES_INSTALLED=0
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        # Handle sklearn which is imported as scikit-learn
        if [ "$package" = "sklearn" ]; then
            if python3 -c "import sklearn" 2>/dev/null; then
                ((PACKAGES_INSTALLED++))
            fi
        else
            if python3 -c "import $package" 2>/dev/null; then
                ((PACKAGES_INSTALLED++))
            fi
        fi
    done
    
    if [ $PACKAGES_INSTALLED -eq ${#REQUIRED_PACKAGES[@]} ]; then
        print_success "All required packages installed ($PACKAGES_INSTALLED/${#REQUIRED_PACKAGES[@]})"
        ((TESTS_PASSED++))
    else
        print_error "Some packages missing ($PACKAGES_INSTALLED/${#REQUIRED_PACKAGES[@]})"
        echo "  → Run 'source init.sh' to install dependencies"
        ((TESTS_FAILED++))
    fi
else
    print_warning "Skipping (venv not found)"
fi
echo ""

# Test 5: Check if project modules can be imported
print_test "Project module imports"
if [ -d "venv" ]; then
    source venv/bin/activate 2>/dev/null
    
    # Set PYTHONPATH
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    
    MODULES=("src.data.data_loader" "src.utils.utilities")
    MODULES_IMPORTED=0
    
    for module in "${MODULES[@]}"; do
        if python3 -c "import $module" 2>/dev/null; then
            ((MODULES_IMPORTED++))
        fi
    done
    
    if [ $MODULES_IMPORTED -eq ${#MODULES[@]} ]; then
        print_success "All project modules can be imported ($MODULES_IMPORTED/${#MODULES[@]})"
        ((TESTS_PASSED++))
    else
        print_error "Cannot import some modules ($MODULES_IMPORTED/${#MODULES[@]})"
        ((TESTS_FAILED++))
    fi
else
    print_warning "Skipping (venv not found)"
fi
echo ""

# Test 6: Check if scripts exist and are executable
print_test "Script files"
SCRIPTS=("init.sh" "setup.sh" "run_individual.sh")
SCRIPTS_OK=0

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        if [ -x "$script" ]; then
            ((SCRIPTS_OK++))
        else
            print_warning "$script exists but is not executable"
            echo "  → Run 'chmod +x $script'"
        fi
    else
        print_error "$script not found"
        ((TESTS_FAILED++))
    fi
done

if [ $SCRIPTS_OK -eq ${#SCRIPTS[@]} ]; then
    print_success "All scripts exist and are executable"
    ((TESTS_PASSED++))
else
    print_warning "Some scripts may need fixing"
fi
echo ""

# Test 7: Check if test data exists (optional)
print_test "Test data availability"
if [ -d "data/raw/test/cycle_9" ]; then
    TEST_FILES=("calcium.csv" "pupil_size.csv" "arteriole_diameter.csv")
    TEST_DATA_FOUND=0
    
    for file in "${TEST_FILES[@]}"; do
        if find "data/raw/test/cycle_9" -name "*${file%%.csv}*" -name "*.csv" | grep -q .; then
            ((TEST_DATA_FOUND++))
        fi
    done
    
    if [ $TEST_DATA_FOUND -gt 0 ]; then
        print_success "Test data found ($TEST_DATA_FOUND/${#TEST_FILES[@]} file types)"
        ((TESTS_PASSED++))
    else
        print_warning "Test data directory exists but no matching files found"
    fi
else
    print_warning "Test data directory not found (optional for testing)"
fi
echo ""

# Test 8: Quick functionality test (if everything else passes)
if [ $TESTS_FAILED -eq 0 ] && [ -d "venv" ]; then
    print_test "Quick functionality test"
    source venv/bin/activate 2>/dev/null
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
    
    # Try to import and use a simple function
    if python3 -c "
from src.data.data_loader import _find_file_by_pattern
import os
import tempfile

# Create a temporary directory with a test file
with tempfile.TemporaryDirectory() as tmpdir:
    test_file = os.path.join(tmpdir, 'test_calcium.csv')
    with open(test_file, 'w') as f:
        f.write('time,value\n0,1\n1,2\n')
    
    # Test the function
    result = _find_file_by_pattern(tmpdir, 'calcium')
    if result and 'calcium' in result.lower():
        print('SUCCESS')
    else:
        print('FAILED')
        exit(1)
" 2>/dev/null; then
        print_success "Basic functionality works"
        ((TESTS_PASSED++))
    else
        print_error "Basic functionality test failed"
        ((TESTS_FAILED++))
    fi
else
    print_warning "Skipping functionality test (setup incomplete)"
fi
echo ""

# Summary
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Tests Passed: ${GREEN}$TESTS_PASSED${NC}"
echo -e "Tests Failed: ${RED}$TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed! Your setup is working correctly.${NC}"
    echo ""
    echo "You can now run the analysis with:"
    echo "  ./run_individual.sh -d \"data/raw/test/cycle_9\""
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Please fix the issues above.${NC}"
    echo ""
    echo "Quick fix:"
    echo "  1. Run: ./setup.sh"
    echo "  2. Or run: source init.sh"
    echo "  3. Then run this test again: ./test_setup.sh"
    exit 1
fi

