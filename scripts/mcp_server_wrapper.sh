#!/bin/bash

# MCP Server Wrapper Script
# This script ensures that Node.js tools are available in PATH when running MCP servers

# Find Node.js installation
find_nodejs() {
    # Check common installation locations
    local node_paths=(
        "/usr/local/bin"
        "/usr/bin"
        "$HOME/.nvm/versions/node/*/bin"
        "$HOME/.npm-global/bin"
        "/opt/nodejs/bin"
    )
    
    for path_pattern in "${node_paths[@]}"; do
        if [[ "$path_pattern" == *"*"* ]]; then
            # Handle glob patterns
            for expanded_path in $path_pattern; do
                if [[ -x "$expanded_path/node" ]]; then
                    echo "$expanded_path"
                    return 0
                fi
            done
        else
            if [[ -x "$path_pattern/node" ]]; then
                echo "$path_pattern"
                return 0
            fi
        fi
    done
    
    return 1
}

# Add Node.js to PATH if found
NODE_PATH=$(find_nodejs)
if [[ -n "$NODE_PATH" ]]; then
    export PATH="$NODE_PATH:$PATH"
fi

# Verify npx is available
if ! command -v npx &> /dev/null; then
    echo "Error: npx not found even after PATH setup" >&2
    echo "Available PATH: $PATH" >&2
    exit 1
fi

# Execute the original command
exec "$@"
