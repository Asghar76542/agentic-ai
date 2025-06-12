#!/bin/bash

# ChromeDriver Fix Script for AgenticSeek
# This script helps automatically resolve ChromeDriver compatibility issues

echo "🔧 AgenticSeek ChromeDriver Auto-Fix"
echo "===================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Check if pip is available and install required packages
echo "📦 Checking required packages..."

# Install required packages if not present
python3 -c "import requests" 2>/dev/null || {
    echo "📥 Installing requests package..."
    pip3 install requests --user
}

# Run the Python script
python3 "$(dirname "$0")/fix_chromedriver.py" "$@"
