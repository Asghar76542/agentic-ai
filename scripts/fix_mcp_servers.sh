#!/bin/bash

# MCP Server Auto-Fix Shell Wrapper Script
# This script provides a convenient shell interface for the MCP server Python auto-fix tool

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_SCRIPT="$SCRIPT_DIR/fix_mcp_servers.py"

echo "🔧 MCP Server Configuration Auto-Fix Tool"
echo "=========================================="

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: fix_mcp_servers.py not found at $PYTHON_SCRIPT"
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
echo "🚀 Running MCP server auto-fix..."
echo ""

python3 "$PYTHON_SCRIPT" "$@"
exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "✅ MCP server auto-fix completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Test MCP functionality: ./scripts/fix_mcp_servers.sh --test"
    echo "2. Try MCP test script: python3 test_mcp_simple.py"
    echo "3. Run AgenticSeek with MCP support: python3 cli.py"
    echo "4. Set MCP_FINDER_API_KEY in .env if you have an API key"
else
    echo "❌ MCP server auto-fix encountered issues (exit code: $exit_code)"
    echo ""
    echo "Troubleshooting tips:"
    echo "1. Install Node.js: https://nodejs.org/"
    echo "2. Check Node.js installation: node --version && npm --version && npx --version"
    echo "3. Restart your shell to pick up PATH changes"
    echo "4. Check if npm global packages are in PATH"
    echo "5. Try installing MCP servers manually: npx @modelcontextprotocol/create-server"
fi

exit $exit_code
