#!/usr/bin/env python3
"""
AgenticSeek Master Auto-Fix Tool

This script runs all available auto-fix tools to resolve common issues in AgenticSeek:
- ChromeDriver compatibility problems
- SearxNG configuration issues  
- MCP server setup problems
- LLM provider connection issues

Author: GitHub Copilot
Date: June 2025
"""

import os
import sys
import subprocess
from pathlib import Path

class MasterFixer:
    def __init__(self):
        self.project_root = self._find_project_root()
        self.scripts_dir = self.project_root / "scripts"
        
    def _find_project_root(self):
        """Find the AgenticSeek project root directory"""
        current = Path.cwd()
        while current != current.parent:
            if (current / "config.ini").exists() and (current / "docker-compose.yml").exists():
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
    
    def run_fix_script(self, script_name, description):
        """Run a specific fix script and return success status"""
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            self.print_status(f"Script not found: {script_name}", "ERROR")
            return False
        
        self.print_status(f"Running {description}...", "INFO")
        
        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                self.print_status(f"{description} completed successfully", "SUCCESS")
                return True
            else:
                self.print_status(f"{description} failed (exit code {result.returncode})", "ERROR")
                if result.stderr:
                    print(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.print_status(f"{description} timed out", "ERROR")
            return False
        except Exception as e:
            self.print_status(f"{description} error: {e}", "ERROR")
            return False
    
    def run_shell_script(self, script_name, description):
        """Run a shell script and return success status"""
        script_path = self.scripts_dir / script_name
        
        if not script_path.exists():
            self.print_status(f"Script not found: {script_name}", "ERROR")
            return False
        
        self.print_status(f"Running {description}...", "INFO")
        
        try:
            result = subprocess.run(
                [str(script_path)],
                cwd=self.project_root,
                shell=True,
                timeout=300
            )
            
            if result.returncode == 0:
                self.print_status(f"{description} completed successfully", "SUCCESS")
                return True
            else:
                self.print_status(f"{description} failed (exit code {result.returncode})", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.print_status(f"{description} timed out", "ERROR")
            return False
        except Exception as e:
            self.print_status(f"{description} error: {e}", "ERROR")
            return False
    
    def run_all_fixes(self):
        """Run all available auto-fix scripts"""
        self.print_status("=== AgenticSeek Master Auto-Fix Tool ===", "INFO")
        self.print_status(f"Working in project root: {self.project_root}", "INFO")
        print()
        
        fix_scripts = [
            ("validate_config.py", "Configuration Validation and Auto-Fix"),
            ("fix_chromedriver.py", "ChromeDriver Auto-Fix"),
            ("fix_searxng.py", "SearxNG Configuration Auto-Fix"),
            ("fix_mcp_servers.py", "MCP Server Setup Auto-Fix")
        ]
        
        results = {}
        
        for script, description in fix_scripts:
            print("=" * 60)
            success = self.run_fix_script(script, description)
            results[description] = success
            print()
        
        # Summary
        print("=" * 60)
        self.print_status("=== SUMMARY ===", "INFO")
        
        all_success = True
        for description, success in results.items():
            status_icon = "✅" if success else "❌"
            print(f"{status_icon} {description}")
            if not success:
                all_success = False
        
        print()
        if all_success:
            self.print_status("🎉 All auto-fixes completed successfully!", "SUCCESS")
            print("\nNext steps:")
            print("1. Start AgenticSeek services: ./start_services.sh")
            print("2. Run AgenticSeek: python3 cli.py")
            print("3. Access web interface: http://localhost:3000")
        else:
            self.print_status("⚠️  Some auto-fixes encountered issues", "WARNING")
            print("\nTroubleshooting:")
            print("1. Check error messages above")
            print("2. Run individual fix scripts manually")
            print("3. Check the documentation for manual setup steps")
            print("4. Ensure Docker is running: docker info")
            print("5. Ensure Node.js is installed: node --version")
        
        return all_success
    
    def run_individual_fix(self, fix_type):
        """Run a specific type of fix"""
        fix_map = {
            "config": ("validate_config.py", "Configuration Validation"),
            "chrome": ("fix_chromedriver.py", "ChromeDriver Auto-Fix"),
            "searxng": ("fix_searxng.py", "SearxNG Configuration Auto-Fix"),
            "mcp": ("fix_mcp_servers.py", "MCP Server Setup Auto-Fix")
        }
        
        if fix_type not in fix_map:
            self.print_status(f"Unknown fix type: {fix_type}", "ERROR")
            self.print_status(f"Available types: {', '.join(fix_map.keys())}", "INFO")
            return False
        
        script, description = fix_map[fix_type]
        return self.run_fix_script(script, description)

def main():
    """Main function"""
    fixer = MasterFixer()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command in ['-h', '--help']:
            print(__doc__)
            print("\nUsage:")
            print(f"  {sys.argv[0]}              # Run all auto-fixes")
            print(f"  {sys.argv[0]} config       # Fix configuration issues only")
            print(f"  {sys.argv[0]} chrome       # Fix ChromeDriver issues only")
            print(f"  {sys.argv[0]} searxng      # Fix SearxNG issues only")
            print(f"  {sys.argv[0]} mcp          # Fix MCP server issues only")
            return
        
        elif command in ['config', 'chrome', 'searxng', 'mcp']:
            success = fixer.run_individual_fix(command)
            sys.exit(0 if success else 1)
        
        else:
            print(f"Unknown command: {command}")
            print("Use --help for usage information")
            sys.exit(1)
    
    else:
        # Run all fixes
        success = fixer.run_all_fixes()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
