#!/usr/bin/env python3
"""
ChromeDriver Auto-Fix Utility for AgenticSeek

This utility helps automatically resolve ChromeDriver compatibility issues
by detecting the Chrome version and installing the matching ChromeDriver.
"""

import os
import sys
import subprocess
import re
import requests
import zipfile
import tempfile
import shutil
import platform
from pathlib import Path

def get_chrome_version():
    """Detect the installed Chrome browser version."""
    chrome_paths = {
        'Windows': [
            r'C:\Program Files\Google\Chrome\Application\chrome.exe',
            r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe',
        ],
        'Darwin': [  # macOS
            '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
        ],
        'Linux': [
            '/usr/bin/google-chrome',
            '/usr/bin/google-chrome-stable',
            '/usr/bin/chromium-browser',
            '/snap/bin/chromium',
        ]
    }
    
    system = platform.system()
    paths = chrome_paths.get(system, [])
    
    for chrome_path in paths:
        if os.path.exists(chrome_path):
            try:
                result = subprocess.run(
                    [chrome_path, '--version'], 
                    capture_output=True, 
                    text=True,
                    timeout=10
                )
                version_line = result.stdout.strip()
                version_match = re.search(r'(\d+\.[\d.]+)', version_line)
                if version_match:
                    return version_match.group(1), chrome_path
            except Exception:
                continue
    
    print("❌ Chrome browser not found. Please install Google Chrome first.")
    return None, None

def get_chromedriver_download_url(chrome_version):
    """Get the download URL for ChromeDriver matching the Chrome version."""
    major_version = chrome_version.split('.')[0]
    
    # For Chrome 115+, use the new Chrome for Testing API
    if int(major_version) >= 115:
        try:
            api_url = f"https://googlechromelabs.github.io/chrome-for-testing/known-good-versions-with-downloads.json"
            response = requests.get(api_url, timeout=30)
            data = response.json()
            
            # Find the closest matching version
            for version_info in reversed(data['versions']):
                if version_info['version'].startswith(f"{major_version}."):
                    downloads = version_info.get('downloads', {})
                    chromedriver_downloads = downloads.get('chromedriver', [])
                    
                    system = platform.system().lower()
                    if system == 'darwin':
                        platform_name = 'mac-x64'
                    elif system == 'windows':
                        platform_name = 'win64' if platform.machine().endswith('64') else 'win32'
                    else:  # Linux
                        platform_name = 'linux64'
                    
                    for download in chromedriver_downloads:
                        if download['platform'] == platform_name:
                            return download['url'], version_info['version']
            
        except Exception as e:
            print(f"⚠️  Failed to get ChromeDriver URL from Chrome for Testing: {e}")
    
    # Fallback for older versions or if API fails
    system = platform.system()
    if system == 'Windows':
        platform_name = 'win32'
    elif system == 'Darwin':
        platform_name = 'mac64'
    else:
        platform_name = 'linux64'
    
    # Try the old ChromeDriver API for older versions
    url = f"https://chromedriver.storage.googleapis.com/LATEST_RELEASE_{major_version}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            driver_version = response.text.strip()
            download_url = f"https://chromedriver.storage.googleapis.com/{driver_version}/chromedriver_{platform_name}.zip"
            return download_url, driver_version
    except Exception:
        pass
    
    return None, None

def download_and_install_chromedriver(download_url, version):
    """Download and install ChromeDriver."""
    print(f"📥 Downloading ChromeDriver version {version}...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, 'chromedriver.zip')
        
        # Download the zip file
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Find the chromedriver executable
        chromedriver_name = 'chromedriver.exe' if platform.system() == 'Windows' else 'chromedriver'
        chromedriver_path = None
        
        for root, dirs, files in os.walk(temp_dir):
            if chromedriver_name in files:
                chromedriver_path = os.path.join(root, chromedriver_name)
                break
        
        if not chromedriver_path:
            raise FileNotFoundError(f"ChromeDriver executable not found in downloaded archive")
        
        # Make it executable on Unix systems
        if platform.system() != 'Windows':
            os.chmod(chromedriver_path, 0o755)
        
        # Determine installation directory
        install_dirs = []
        
        # Try to install in the same directory as this script
        script_dir = Path(__file__).parent.parent
        install_dirs.append(script_dir)
        
        # Try common PATH directories
        if platform.system() != 'Windows':
            install_dirs.extend([
                Path.home() / '.local' / 'bin',
                Path('/usr/local/bin'),
                Path('/usr/bin'),
            ])
        else:
            # On Windows, try to install in the current directory
            install_dirs.append(Path.cwd())
        
        # Install ChromeDriver
        for install_dir in install_dirs:
            try:
                install_dir.mkdir(parents=True, exist_ok=True)
                destination = install_dir / chromedriver_name
                shutil.copy2(chromedriver_path, destination)
                
                print(f"✅ ChromeDriver installed successfully at: {destination}")
                
                # Test the installation
                result = subprocess.run(
                    [str(destination), '--version'], 
                    capture_output=True, 
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"✅ ChromeDriver test successful: {result.stdout.strip()}")
                    return str(destination)
                
            except PermissionError:
                print(f"⚠️  Permission denied writing to {install_dir}")
                continue
            except Exception as e:
                print(f"⚠️  Failed to install in {install_dir}: {e}")
                continue
        
        raise Exception("Failed to install ChromeDriver in any accessible directory")

def main():
    """Main function to fix ChromeDriver issues."""
    print("🔧 AgenticSeek ChromeDriver Auto-Fix Utility")
    print("=" * 50)
    
    # Step 1: Detect Chrome version
    print("📍 Step 1: Detecting Chrome browser version...")
    chrome_version, chrome_path = get_chrome_version()
    
    if not chrome_version:
        print("\n❌ Cannot proceed without Chrome browser installed.")
        print("Please install Google Chrome from: https://www.google.com/chrome/")
        return False
    
    print(f"✅ Found Chrome version: {chrome_version}")
    print(f"   Chrome path: {chrome_path}")
    
    # Step 2: Get ChromeDriver download URL
    print(f"\n📍 Step 2: Finding matching ChromeDriver...")
    download_url, driver_version = get_chromedriver_download_url(chrome_version)
    
    if not download_url:
        print(f"❌ Could not find ChromeDriver for Chrome version {chrome_version}")
        print("\nManual installation required:")
        print("1. Visit: https://googlechromelabs.github.io/chrome-for-testing/")
        print("2. Download ChromeDriver matching your Chrome version")
        print("3. Place it in your PATH or project directory")
        return False
    
    print(f"✅ Found ChromeDriver version: {driver_version}")
    
    # Step 3: Download and install
    print(f"\n📍 Step 3: Installing ChromeDriver...")
    try:
        chromedriver_path = download_and_install_chromedriver(download_url, driver_version)
        print(f"\n🎉 ChromeDriver successfully installed!")
        print(f"   Installation path: {chromedriver_path}")
        print(f"   Chrome version: {chrome_version}")
        print(f"   ChromeDriver version: {driver_version}")
        
        print(f"\n✅ AgenticSeek should now work properly with web browsing!")
        return True
        
    except Exception as e:
        print(f"\n❌ Installation failed: {e}")
        print("\nTry manual installation:")
        print("1. Visit: https://googlechromelabs.github.io/chrome-for-testing/")
        print("2. Download ChromeDriver matching your Chrome version")
        print("3. Place it in your PATH or project directory")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
