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

# If no custom path is provided, use the current directory
if [ -z "$custom_path" ]; then
    export_path=$(pwd)
else
    export_path=$(realpath "$custom_path")
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

# Activate conda base environment
echo "Activating conda base environment..."
conda init
conda activate base

# Initialize the virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found. Creating a new one..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment created and activated."
fi

# Optionally, you can source this script in your shell to apply the changes
# to the current session. Add this line to your .bashrc or .bash_profile:
# source /path/to/this/init.sh
