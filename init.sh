#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 [-p <path>]"
    echo "  -p <path>  Specify a custom path to add to PYTHONPATH (optional)"
    echo "  If no path is provided, the current directory will be used."
    exit 1
}

# Parse command-line options
while getopts ":p:" opt; do
    case ${opt} in
        p )
            custom_path=$OPTARG
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            usage
            ;;
        : )
            echo "Invalid option: $OPTARG requires an argument" 1>&2
            usage
            ;;
    esac
done

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Function to get absolute path (works on both Mac and Linux)
get_abs_path() {
    local path="$1"
    # Try realpath first (Linux)
    if command -v realpath >/dev/null 2>&1; then
        realpath "$path"
    # Try grealpath (Mac with coreutils)
    elif command -v grealpath >/dev/null 2>&1; then
        grealpath "$path"
    # Fallback to Python (works on both Mac and Linux)
    else
        python3 -c "import os; print(os.path.abspath('$path'))"
    fi
}

# If no custom path is provided, use the directory where the script is located
if [ -z "$custom_path" ]; then
    export_path="$SCRIPT_DIR"
else
    export_path=$(get_abs_path "$custom_path")
fi

# Check if the path exists
if [ ! -d "$export_path" ]; then
    echo "Error: The specified path does not exist or is not a directory." 1>&2
    exit 1
fi

# Export the path to PYTHONPATH
export PYTHONPATH="$export_path:$PYTHONPATH"

echo "PYTHONPATH has been updated. The following path has been added:"
echo "$export_path"

# Check if python3 is available
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is not installed or not in PATH." 1>&2
    echo "Please install Python 3 to continue." 1>&2
    exit 1
fi

# Change to script directory to ensure relative paths work
cd "$SCRIPT_DIR" || {
    echo "Error: Cannot change to script directory: $SCRIPT_DIR" 1>&2
    exit 1
}

# Initialize the virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    # Check if pip is available
    if ! command -v pip >/dev/null 2>&1 && ! command -v pip3 >/dev/null 2>&1; then
        echo "Error: pip is not available in the virtual environment." 1>&2
        exit 1
    fi
    # Use pip3 if pip is not available, otherwise use pip
    PIP_CMD=$(command -v pip3 || command -v pip)
else
    echo "Virtual environment not found. Creating a new one..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment." 1>&2
        exit 1
    fi
    source venv/bin/activate
    echo "Virtual environment created and activated."
    
    # Use pip3 if pip is not available, otherwise use pip
    PIP_CMD=$(command -v pip3 || command -v pip)
    
    # Upgrade pip to latest version
    echo "Upgrading pip..."
    $PIP_CMD install --upgrade pip --quiet
fi

# Install or upgrade dependencies from requirements.txt if it exists
if [ -f "requirements.txt" ]; then
    echo "Installing/updating dependencies from requirements.txt..."
    $PIP_CMD install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Warning: Some packages from requirements.txt may have failed to install." 1>&2
        echo "You may need to install them manually." 1>&2
    else
        echo "Dependencies installed successfully."
    fi
else
    echo "Warning: requirements.txt not found. Skipping dependency installation."
fi

# Optionally, you can source this script in your shell to apply the changes
# to the current session. Add this line to your .bashrc or .bash_profile:
# source /path/to/this/init.sh
