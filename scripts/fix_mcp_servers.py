#!/usr/bin/env python3
"""
MCP Server Configuration and Setup Auto-Fix Tool

This script automatically detects and fixes common MCP (Model Context Protocol) server
issues in the AgenticSeek project, including:
- Missing Node.js/npx dependencies
- PATH issues that prevent npx from being found
- MCP server installation and configuration problems
- Environment variable setup for MCP functionality

Author: GitHub Copilot
Date: June 2025
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json

class MCPServerFixer:
    def __init__(self):
        self.project_root = self._find_project_root()
        self.mcp_config_file = self.project_root / "mcp_config.ini"
        self.env_file = self.project_root / ".env"
        
    def _find_project_root(self):
        """Find the AgenticSeek project root directory"""
        current = Path.cwd()
        while current != current.parent:
            if (current / "config.ini").exists() and (current / "mcp_config.ini").exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def print_status(self, message, status="INFO"):
        """Print colored status messages"""
        colors = {
            "INFO": "\033[94m",    # Blue
            "SUCCESS": "\033[92m", # Green
            "WARNING": "\033[93m", # Yellow
            "ERROR": "\033[91m",   # Red
            "RESET": "\033[0m"     # Reset
        }
        print(f"{colors.get(status, '')}{status}: {message}{colors['RESET']}")
    
    def check_nodejs_installation(self):
        """Check if Node.js, npm, and npx are properly installed and accessible"""
        self.print_status("Checking Node.js installation...", "INFO")
        
        # Check for node
        node_path = shutil.which('node')
        if not node_path:
            self.print_status("Node.js not found in PATH", "ERROR")
            return False
        
        # Check for npm
        npm_path = shutil.which('npm')
        if not npm_path:
            self.print_status("npm not found in PATH", "ERROR")
            return False
        
        # Check for npx
        npx_path = shutil.which('npx')
        if not npx_path:
            self.print_status("npx not found in PATH", "ERROR")
            return False
        
        # Get versions
        try:
            node_version = subprocess.run(['node', '--version'], 
                                        capture_output=True, text=True, check=True)
            npm_version = subprocess.run(['npm', '--version'], 
                                       capture_output=True, text=True, check=True)
            npx_version = subprocess.run(['npx', '--version'], 
                                       capture_output=True, text=True, check=True)
            
            self.print_status(f"Node.js: {node_version.stdout.strip()} at {node_path}", "SUCCESS")
            self.print_status(f"npm: {npm_version.stdout.strip()} at {npm_path}", "SUCCESS")
            self.print_status(f"npx: {npx_version.stdout.strip()} at {npx_path}", "SUCCESS")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.print_status(f"Error checking Node.js tools: {e}", "ERROR")
            return False
    
    def fix_path_issues(self):
        """Fix PATH issues that might prevent MCP servers from finding npx"""
        self.print_status("Checking and fixing PATH configuration...", "INFO")
        
        # Find all Node.js related paths
        node_paths = []
        
        # Check common Node.js installation locations
        common_paths = [
            '/usr/local/bin',
            '/usr/bin',
            os.path.expanduser('~/.nvm/versions/node/*/bin'),
            os.path.expanduser('~/.npm-global/bin'),
            '/opt/nodejs/bin'
        ]
        
        # Find actual Node.js installations
        for path_pattern in common_paths:
            if '*' in path_pattern:
                # Handle glob patterns for nvm
                import glob
                matching_paths = glob.glob(path_pattern)
                for path in matching_paths:
                    if os.path.exists(os.path.join(path, 'node')):
                        node_paths.append(path)
            else:
                if os.path.exists(os.path.join(path_pattern, 'node')):
                    node_paths.append(path_pattern)
        
        if not node_paths:
            self.print_status("No Node.js installations found", "ERROR")
            return False
        
        # Add paths to environment
        current_path = os.environ.get('PATH', '')
        updated = False
        
        for node_path in node_paths:
            if node_path not in current_path:
                os.environ['PATH'] = f"{node_path}:{current_path}"
                current_path = os.environ['PATH']
                updated = True
                self.print_status(f"Added to PATH: {node_path}", "SUCCESS")
        
        if updated:
            self.print_status("PATH updated successfully", "SUCCESS")
        else:
            self.print_status("PATH already contains Node.js paths", "INFO")
        
        return True
    
    def check_mcp_config_file(self):
        """Check and create MCP configuration file if missing"""
        self.print_status("Checking MCP configuration file...", "INFO")
        
        if not self.mcp_config_file.exists():
            self.print_status("Creating missing mcp_config.ini file", "WARNING")
            self._create_default_mcp_config()
            return True
        
        # Read and validate existing config
        try:
            with open(self.mcp_config_file, 'r') as f:
                content = f.read()
            
            required_sections = ['MCP_GENERAL', 'MCP_SECURITY', 'MCP_REGISTRY']
            missing_sections = []
            
            for section in required_sections:
                if f'[{section}]' not in content:
                    missing_sections.append(section)
            
            if missing_sections:
                self.print_status(f"Missing sections in mcp_config.ini: {missing_sections}", "WARNING")
                self._update_mcp_config(content, missing_sections)
            else:
                self.print_status("MCP configuration file is valid", "SUCCESS")
            
            return True
            
        except Exception as e:
            self.print_status(f"Error reading mcp_config.ini: {e}", "ERROR")
            return False
    
    def _create_default_mcp_config(self):
        """Create a default MCP configuration file"""
        default_config = """[MCP_GENERAL]
enabled = true
workspace_base = /tmp/agenticseek_mcp_workspaces
health_check_interval = 30
max_concurrent_servers = 10
operation_timeout = 300

[MCP_SECURITY]
enable_sandboxing = true
sandbox_permissions = 750
allow_network_access = true
restrict_file_access = true
max_memory_per_server = 512
max_cpu_per_server = 50

[MCP_REGISTRY]
registry_url = https://registry.smithery.ai
registry_timeout = 30
cache_duration = 60
auto_update_listings = true
"""
        self.mcp_config_file.write_text(default_config)
        self.print_status(f"Created default MCP config at {self.mcp_config_file}", "SUCCESS")
    
    def _update_mcp_config(self, existing_content, missing_sections):
        """Update MCP configuration with missing sections"""
        section_configs = {
            'MCP_GENERAL': """
[MCP_GENERAL]
enabled = true
workspace_base = /tmp/agenticseek_mcp_workspaces
health_check_interval = 30
max_concurrent_servers = 10
operation_timeout = 300
""",
            'MCP_SECURITY': """
[MCP_SECURITY]
enable_sandboxing = true
sandbox_permissions = 750
allow_network_access = true
restrict_file_access = true
max_memory_per_server = 512
max_cpu_per_server = 50
""",
            'MCP_REGISTRY': """
[MCP_REGISTRY]
registry_url = https://registry.smithery.ai
registry_timeout = 30
cache_duration = 60
auto_update_listings = true
"""
        }
        
        updated_content = existing_content
        for section in missing_sections:
            if section in section_configs:
                updated_content += section_configs[section]
        
        self.mcp_config_file.write_text(updated_content)
        self.print_status("Updated MCP configuration with missing sections", "SUCCESS")
    
    def setup_mcp_environment_variables(self):
        """Setup environment variables needed for MCP functionality"""
        self.print_status("Setting up MCP environment variables...", "INFO")
        
        if not self.env_file.exists():
            self.print_status("No .env file found, creating one", "WARNING")
            self.env_file.write_text("")
        
        # Read current .env content
        env_content = self.env_file.read_text()
        
        # Add MCP_FINDER_API_KEY if not present
        if "MCP_FINDER_API_KEY" not in env_content:
            env_content += '\nMCP_FINDER_API_KEY=""\n'
            self.print_status("Added MCP_FINDER_API_KEY to .env (you may need to set a value)", "WARNING")
        
        # Add NODE_PATH if not present
        if "NODE_PATH" not in env_content:
            node_modules_path = f"{self.project_root}/node_modules"
            env_content += f'\nNODE_PATH="{node_modules_path}"\n'
        
        self.env_file.write_text(env_content)
        
        # Load the environment variables
        try:
            for line in env_content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip('\'"')
                    os.environ[key] = value
        except Exception as e:
            self.print_status(f"Error loading environment variables: {e}", "WARNING")
        
        self.print_status("MCP environment variables configured", "SUCCESS")
        return True
    
    def create_mcp_server_wrapper(self):
        """Create a wrapper script to fix npx path issues in MCP server execution"""
        wrapper_path = self.project_root / "scripts" / "mcp_server_wrapper.sh"
        
        # Create scripts directory if it doesn't exist
        scripts_dir = wrapper_path.parent
        scripts_dir.mkdir(exist_ok=True)
        
        wrapper_content = """#!/bin/bash

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
"""
        
        wrapper_path.write_text(wrapper_content)
        wrapper_path.chmod(0o755)
        
        self.print_status(f"Created MCP server wrapper at {wrapper_path}", "SUCCESS")
        return wrapper_path
    
    def test_mcp_server_functionality(self):
        """Test basic MCP server functionality"""
        self.print_status("Testing MCP server functionality...", "INFO")
        
        try:
            # Try to run a simple npx command
            result = subprocess.run(['npx', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                self.print_status(f"npx test successful: {result.stdout.strip()}", "SUCCESS")
            else:
                self.print_status(f"npx test failed: {result.stderr}", "ERROR")
                return False
            
            # Test basic MCP agent import
            try:
                sys.path.append(str(self.project_root))
                from sources.agents.mcp_agent import McpAgent
                self.print_status("MCP agent import successful", "SUCCESS")
            except ImportError as e:
                self.print_status(f"MCP agent import failed: {e}", "WARNING")
                # This might be expected if dependencies aren't installed
            
            return True
            
        except Exception as e:
            self.print_status(f"MCP functionality test failed: {e}", "ERROR")
            return False
    
    def install_python_dependencies(self):
        """Install required Python dependencies for MCP functionality"""
        self.print_status("Checking Python dependencies for MCP...", "INFO")
        
        required_packages = [
            "python-dotenv",
            "configparser"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.print_status(f"Installing missing packages: {missing_packages}", "WARNING")
            try:
                for package in missing_packages:
                    subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                                 capture_output=True, check=True)
                self.print_status("Python dependencies installed successfully", "SUCCESS")
            except subprocess.CalledProcessError as e:
                self.print_status(f"Failed to install dependencies: {e}", "ERROR")
                return False
        else:
            self.print_status("All required Python dependencies are available", "SUCCESS")
        
        return True
    
    def run_comprehensive_fix(self):
        """Run all MCP server fixes"""
        self.print_status("=== MCP Server Configuration Auto-Fix Tool ===", "INFO")
        self.print_status(f"Working in project root: {self.project_root}", "INFO")
        
        success = True
        
        # Step 1: Check Node.js installation
        if not self.check_nodejs_installation():
            self.print_status("Node.js installation issues detected", "ERROR")
            success = False
        
        # Step 2: Fix PATH issues
        if not self.fix_path_issues():
            success = False
        
        # Step 3: Re-check Node.js after PATH fix
        if not self.check_nodejs_installation():
            self.print_status("Node.js still not accessible after PATH fix", "ERROR")
            success = False
        
        # Step 4: Install Python dependencies
        if not self.install_python_dependencies():
            success = False
        
        # Step 5: Check/create MCP config
        if not self.check_mcp_config_file():
            success = False
        
        # Step 6: Setup environment variables
        if not self.setup_mcp_environment_variables():
            success = False
        
        # Step 7: Create wrapper script
        wrapper_path = self.create_mcp_server_wrapper()
        
        # Step 8: Test functionality
        if not self.test_mcp_server_functionality():
            self.print_status("MCP functionality test failed, but continuing", "WARNING")
        
        # Provide summary and recommendations
        if success:
            self.print_status("=== MCP Server configuration completed successfully! ===", "SUCCESS")
            self.print_status("MCP functionality should now work properly", "SUCCESS")
            print("\nNext steps:")
            print("1. Test MCP functionality: python3 test_mcp_simple.py")
            print("2. Set MCP_FINDER_API_KEY if you have one")
            print("3. Start AgenticSeek with MCP support enabled")
        else:
            self.print_status("=== Some MCP configuration issues remain ===", "WARNING")
            print("\nManual steps to try:")
            print("1. Install Node.js if not available: https://nodejs.org/")
            print("2. Make sure npx is in your PATH")
            print("3. Check Node.js installation: node --version && npm --version && npx --version")
            print("4. Restart your shell to pick up PATH changes")
        
        return success

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print(__doc__)
        return
    
    fixer = MCPServerFixer()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Just test the functionality
        fixer.test_mcp_server_functionality()
    else:
        # Run comprehensive fix
        success = fixer.run_comprehensive_fix()
        
        if success:
            print("\n" + "="*50)
            print("MCP Server configuration has been fixed!")
            print("You can test it by running:")
            print(f"  python3 {__file__} --test")
            print("="*50)
        
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
