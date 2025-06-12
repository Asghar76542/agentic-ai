#!/usr/bin/env python3
"""
SearxNG Configuration and Setup Auto-Fix Tool

This script automatically detects and fixes common SearxNG configuration issues
in the AgenticSeek project, including:
- Missing or incorrect SEARXNG_BASE_URL environment variable
- SearxNG service startup problems
- Docker configuration issues
- Environment variable loading issues

Author: GitHub Copilot
Date: June 2025
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

class SearxNGFixer:
    def __init__(self):
        self.project_root = self._find_project_root()
        self.env_file = self.project_root / ".env"
        self.default_searxng_url = "http://127.0.0.1:8080"
        self.default_redis_url = "redis://redis:6379/0"
        
    def _find_project_root(self):
        """Find the AgenticSeek project root directory"""
        current = Path.cwd()
        while current != current.parent:
            if (current / "config.ini").exists() and (current / "docker-compose.yml").exists():
                return current
            current = current.parent
        return Path.cwd()  # Fallback to current directory
    
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
    
    def check_docker_running(self):
        """Check if Docker is running"""
        try:
            result = subprocess.run(['docker', 'info'], 
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def check_env_file(self):
        """Check and fix .env file configuration"""
        self.print_status("Checking .env file configuration...")
        
        if not self.env_file.exists():
            self.print_status("Creating missing .env file", "WARNING")
            self._create_default_env_file()
            return True
        
        # Read current .env file
        env_content = self.env_file.read_text()
        needs_update = False
        
        # Check for SEARXNG_BASE_URL
        if "SEARXNG_BASE_URL" not in env_content:
            self.print_status("Adding missing SEARXNG_BASE_URL to .env", "WARNING")
            env_content += f'\nSEARXNG_BASE_URL="{self.default_searxng_url}"\n'
            needs_update = True
        
        # Check for REDIS_BASE_URL
        if "REDIS_BASE_URL" not in env_content:
            self.print_status("Adding missing REDIS_BASE_URL to .env", "WARNING")
            env_content += f'\nREDIS_BASE_URL="{self.default_redis_url}"\n'
            needs_update = True
        
        if needs_update:
            self.env_file.write_text(env_content)
            self.print_status("Updated .env file with missing variables", "SUCCESS")
        
        return True
    
    def _create_default_env_file(self):
        """Create a default .env file with necessary SearxNG configuration"""
        default_content = f'''SEARXNG_BASE_URL="{self.default_searxng_url}"
REDIS_BASE_URL="{self.default_redis_url}"
WORK_DIR="{self.project_root}/workspace"
OLLAMA_PORT="11434"
LM_STUDIO_PORT="1234"
CUSTOM_ADDITIONAL_LLM_PORT="11435"
OPENAI_API_KEY=""
DEEPSEEK_API_KEY=""
OPENROUTER_API_KEY=""
TOGETHER_API_KEY=""
GOOGLE_API_KEY=""
ANTHROPIC_API_KEY=""
MCP_FINDER_API_KEY=""
'''
        self.env_file.write_text(default_content)
        self.print_status(f"Created default .env file at {self.env_file}", "SUCCESS")
    
    def load_env_variables(self):
        """Load environment variables from .env file"""
        if not self.env_file.exists():
            return False
        
        self.print_status("Loading environment variables from .env file...")
        
        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip('\'"')
                        os.environ[key] = value
            
            self.print_status("Environment variables loaded successfully", "SUCCESS")
            return True
        except Exception as e:
            self.print_status(f"Error loading .env file: {e}", "ERROR")
            return False
    
    def check_searxng_service(self):
        """Check if SearxNG service is running"""
        searxng_url = os.getenv("SEARXNG_BASE_URL", self.default_searxng_url)
        
        self.print_status(f"Checking SearxNG service at {searxng_url}...")
        
        try:
            response = requests.get(f"{searxng_url}/", timeout=10)
            if response.status_code == 200:
                self.print_status("SearxNG service is running correctly", "SUCCESS")
                return True
            else:
                self.print_status(f"SearxNG returned status code {response.status_code}", "WARNING")
                return False
        except requests.exceptions.RequestException as e:
            self.print_status(f"SearxNG service is not accessible: {e}", "ERROR")
            return False
    
    def start_searxng_service(self):
        """Start SearxNG service using docker-compose"""
        self.print_status("Attempting to start SearxNG service...")
        
        try:
            os.chdir(self.project_root)
            
            # Check if we should use docker-compose or docker compose
            compose_cmd = "docker-compose"
            try:
                subprocess.run([compose_cmd, '--version'], 
                             capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                compose_cmd = "docker compose"
            
            # Start SearxNG and Redis services
            result = subprocess.run(
                [compose_cmd, 'up', '-d', 'searxng', 'redis'],
                capture_output=True, text=True, timeout=120
            )
            
            if result.returncode == 0:
                self.print_status("SearxNG service started successfully", "SUCCESS")
                # Wait a moment for the service to fully start
                time.sleep(10)
                return True
            else:
                self.print_status(f"Failed to start SearxNG: {result.stderr}", "ERROR")
                return False
                
        except subprocess.TimeoutExpired:
            self.print_status("Timeout while starting SearxNG service", "ERROR")
            return False
        except Exception as e:
            self.print_status(f"Error starting SearxNG service: {e}", "ERROR")
            return False
    
    def fix_searxng_python_import(self):
        """Fix Python import issues with SearxNG tool"""
        searxng_tool_path = self.project_root / "sources" / "tools" / "searxSearch.py"
        
        if not searxng_tool_path.exists():
            self.print_status("searxSearch.py not found", "ERROR")
            return False
        
        self.print_status("Checking searxSearch.py for environment loading...")
        
        # Read the current file
        content = searxng_tool_path.read_text()
        
        # Check if it already has dotenv loading
        if "python-dotenv" in content or "load_dotenv" in content:
            self.print_status("Environment loading already present in searxSearch.py", "INFO")
            return True
        
        # Add environment loading at the top
        lines = content.split('\n')
        import_section_end = 0
        
        # Find where imports end
        for i, line in enumerate(lines):
            if line.strip().startswith('from sources.tools.tools import Tools'):
                import_section_end = i + 1
                break
        
        # Insert dotenv loading
        new_lines = lines[:import_section_end]
        new_lines.extend([
            "",
            "# Load environment variables",
            "try:",
            "    from dotenv import load_dotenv",
            "    load_dotenv()",
            "except ImportError:",
            "    pass  # python-dotenv not available",
            ""
        ])
        new_lines.extend(lines[import_section_end:])
        
        # Write back the modified content
        searxng_tool_path.write_text('\n'.join(new_lines))
        self.print_status("Enhanced searxSearch.py with environment loading", "SUCCESS")
        return True
    
    def install_python_dotenv(self):
        """Install python-dotenv if not present"""
        try:
            import dotenv
            self.print_status("python-dotenv is already installed", "INFO")
            return True
        except ImportError:
            self.print_status("Installing python-dotenv...", "INFO")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', 'python-dotenv'], 
                             capture_output=True, check=True)
                self.print_status("python-dotenv installed successfully", "SUCCESS")
                return True
            except subprocess.CalledProcessError as e:
                self.print_status(f"Failed to install python-dotenv: {e}", "ERROR")
                return False
    
    def run_comprehensive_fix(self):
        """Run all SearxNG fixes"""
        self.print_status("=== SearxNG Configuration Auto-Fix Tool ===", "INFO")
        self.print_status(f"Working in project root: {self.project_root}", "INFO")
        
        success = True
        
        # Step 1: Check Docker
        if not self.check_docker_running():
            self.print_status("Docker is not running. Please start Docker first.", "ERROR")
            success = False
        
        # Step 2: Fix .env file
        if not self.check_env_file():
            success = False
        
        # Step 3: Load environment variables
        self.load_env_variables()
        
        # Step 4: Install python-dotenv
        if not self.install_python_dotenv():
            success = False
        
        # Step 5: Fix Python import
        if not self.fix_searxng_python_import():
            success = False
        
        # Step 6: Check/start SearxNG service
        if not self.check_searxng_service():
            if self.check_docker_running():
                if not self.start_searxng_service():
                    success = False
                else:
                    # Recheck after starting
                    time.sleep(5)
                    if not self.check_searxng_service():
                        success = False
        
        # Step 7: Provide summary
        if success:
            self.print_status("=== All SearxNG issues have been resolved! ===", "SUCCESS")
            self.print_status("You can now use the SearxNG search functionality", "SUCCESS")
            self.print_status(f"SearxNG URL: {os.getenv('SEARXNG_BASE_URL', self.default_searxng_url)}", "INFO")
        else:
            self.print_status("=== Some issues could not be automatically resolved ===", "WARNING")
            self.print_status("Please check the errors above and resolve them manually", "WARNING")
        
        return success
    
    def test_searxng_integration(self):
        """Test SearxNG integration with AgenticSeek"""
        self.print_status("Testing SearxNG integration...", "INFO")
        
        try:
            # Import and test the searxSearch tool
            sys.path.append(str(self.project_root))
            from sources.tools.searxSearch import searxSearch
            
            # Create searxSearch instance
            search_tool = searxSearch()
            
            # Test a simple search
            result = search_tool.execute(["test query"])
            
            if "web search failed" in result.lower() or "error" in result.lower():
                self.print_status("SearxNG integration test failed", "ERROR")
                self.print_status(f"Result: {result}", "ERROR")
                return False
            else:
                self.print_status("SearxNG integration test successful", "SUCCESS")
                return True
                
        except Exception as e:
            self.print_status(f"SearxNG integration test failed: {e}", "ERROR")
            return False

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print(__doc__)
        return
    
    fixer = SearxNGFixer()
    
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        # Just test the integration
        fixer.load_env_variables()
        fixer.test_searxng_integration()
    else:
        # Run comprehensive fix
        success = fixer.run_comprehensive_fix()
        
        if success:
            print("\n" + "="*50)
            print("SearxNG configuration has been fixed!")
            print("You can test it by running:")
            print(f"  python3 {__file__} --test")
            print("="*50)
        
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
