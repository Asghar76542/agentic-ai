#!/bin/bash

# SearxNG Auto-Fix Shell Wrapper Script
# This script provides a convenient shell interface for the SearxNG Python auto-fix tool

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$SCRIPT_DIR/fix_searxng.py"

echo "🔍 SearxNG Configuration Auto-Fix Tool"
echo "======================================="

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: fix_searxng.py not found at $PYTHON_SCRIPT"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT" || {
    echo "❌ Error: Cannot change to project root directory: $PROJECT_ROOT"
    exit 1
}

echo "📁 Working directory: $PROJECT_ROOT"

# Run the Python fix script with any arguments passed to this script
echo "🚀 Running SearxNG auto-fix..."
echo ""

python3 "$PYTHON_SCRIPT" "$@"
exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ SearxNG auto-fix completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. You can test SearxNG integration with: ./scripts/fix_searxng.sh --test"
    echo "2. Start AgenticSeek services with: ./start_services.sh"
    echo "3. Run AgenticSeek with: python3 cli.py"
else
    echo "❌ SearxNG auto-fix encountered issues (exit code: $exit_code)"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Make sure Docker is running: docker info"
    echo "2. Check if port 8080 is available: netstat -ln | grep :8080"
    echo "3. Try restarting Docker services: ./start_services.sh"
    echo "4. Check the SearxNG logs: docker logs agenticseek-searxng-1"
fi

exit $exit_code
