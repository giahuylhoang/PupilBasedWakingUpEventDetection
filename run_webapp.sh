#!/bin/bash

# Run init.sh to set up the environment
source ./init.sh

# Check if Flask is installed
python3 -c "import flask" 2>/dev/null || {
    echo "Installing Flask..."
    pip install flask
}

# Run the web app
echo ""
echo "Starting web application..."
echo "If port 5000 is in use, the app will automatically use the next available port."
echo ""
python3 webapp.py