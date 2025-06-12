#!/bin/bash

# AgenticSeek Master Auto-Fix Shell Wrapper
# This script provides a convenient way to run all auto-fix tools

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$SCRIPT_DIR/master_fix.py"

echo "🔧 AgenticSeek Master Auto-Fix Tool"
echo "===================================="
echo ""

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: master_fix.py not found at $PYTHON_SCRIPT"
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
echo ""

# Show what will be fixed
if [ $# -eq 0 ]; then
    echo "🎯 This will run ALL auto-fixes:"
    echo "   • Configuration validation and auto-fix"
    echo "   • ChromeDriver compatibility fixes"  
    echo "   • SearxNG configuration fixes"
    echo "   • MCP server setup fixes"
    echo ""
    echo "⏱️  This may take several minutes..."
    echo ""
fi

# Run the master fix script
python3 "$PYTHON_SCRIPT" "$@"
exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "🎉 Auto-fix process completed successfully!"
    echo ""
    echo "🚀 Ready to start AgenticSeek:"
    echo "   1. Start services: ./start_services.sh"
    echo "   2. Run AgenticSeek: python3 cli.py"
    echo "   3. Or access web UI: http://localhost:3000"
else
    echo "⚠️  Auto-fix process encountered some issues"
    echo ""
    echo "💡 You can run individual fixes:"
    echo "   ./scripts/master_fix.sh config    # Configuration only"
    echo "   ./scripts/master_fix.sh chrome    # ChromeDriver only"
    echo "   ./scripts/master_fix.sh searxng   # SearxNG only"
    echo "   ./scripts/master_fix.sh mcp       # MCP servers only"
fi

exit $exit_code
