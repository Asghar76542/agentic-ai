#!/usr/bin/env python3
"""
AgenticSeek Configuration Validator

This utility helps validate and fix common configuration issues in AgenticSeek,
particularly around LLM provider settings and connection adapters.
"""

import os
import sys
import configparser
from urllib.parse import urlparse
import requests
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sources.utility import pretty_print
except ImportError:
    def pretty_print(msg, color="info"):
        colors = {
            "success": "\033[92m",
            "warning": "\033[93m", 
            "failure": "\033[91m",
            "info": "\033[94m",
            "status": "\033[96m"
        }
        reset = "\033[0m"
        print(f"{colors.get(color, '')}{msg}{reset}")

def read_config(config_path="config.ini"):
    """Read and parse the configuration file."""
    if not os.path.exists(config_path):
        pretty_print(f"❌ Configuration file not found: {config_path}", color="failure")
        return None
    
    config = configparser.ConfigParser()
    try:
        config.read(config_path)
        return config
    except Exception as e:
        pretty_print(f"❌ Error reading config file: {e}", color="failure")
        return None

def validate_provider_config(config):
    """Validate the provider configuration."""
    issues = []
    fixes = []
    
    if 'MAIN' not in config:
        issues.append("❌ [MAIN] section missing from config.ini")
        return issues, fixes
    
    main_config = config['MAIN']
    
    # Check required fields
    required_fields = ['provider_name', 'provider_model', 'provider_server_address']
    for field in required_fields:
        if field not in main_config:
            issues.append(f"❌ Missing required field: {field}")
    
    if issues:
        return issues, fixes
    
    provider_name = main_config.get('provider_name', '').lower()
    server_address = main_config.get('provider_server_address', '')
    
    # Check provider-specific requirements
    http_required_providers = ['lm-studio', 'server', 'openai']
    
    if provider_name in http_required_providers:
        if not server_address.startswith(('http://', 'https://')):
            issues.append(f"❌ Provider '{provider_name}' requires HTTP prefix in server address")
            fixes.append(f"🔧 Fix: Change 'provider_server_address = {server_address}' to 'provider_server_address = http://{server_address}'")
    
    # Validate URL format for HTTP providers
    if provider_name in http_required_providers and server_address.startswith(('http://', 'https://')):
        try:
            parsed = urlparse(server_address)
            if not parsed.netloc:
                issues.append(f"❌ Invalid server address format: {server_address}")
        except Exception as e:
            issues.append(f"❌ Invalid URL: {e}")
    
    # Provider-specific validations
    if provider_name == 'lm-studio':
        if not server_address.endswith(('/v1', '/v1/')):
            expected_url = server_address.rstrip('/') + '/v1/chat/completions'
            pretty_print(f"ℹ️  LM Studio will use endpoint: {expected_url}", color="info")
    
    return issues, fixes

def test_connection(config):
    """Test connection to the configured provider."""
    if 'MAIN' not in config:
        return False
    
    main_config = config['MAIN']
    provider_name = main_config.get('provider_name', '').lower()
    server_address = main_config.get('provider_server_address', '')
    
    if provider_name == 'lm-studio':
        test_url = f"{server_address}/v1/models"
        pretty_print(f"🔗 Testing LM Studio connection: {test_url}", color="status")
        
        try:
            response = requests.get(test_url, timeout=10)
            if response.status_code == 200:
                pretty_print("✅ LM Studio connection successful!", color="success")
                return True
            else:
                pretty_print(f"❌ LM Studio responded with status {response.status_code}", color="failure")
                return False
        except requests.exceptions.ConnectionError:
            pretty_print("❌ Could not connect to LM Studio. Is it running?", color="failure")
            return False
        except Exception as e:
            pretty_print(f"❌ Connection test failed: {e}", color="failure")
            return False
    
    elif provider_name == 'ollama':
        # Test Ollama connection
        if not server_address.startswith('http'):
            test_url = f"http://{server_address}"
        else:
            test_url = server_address
        
        pretty_print(f"🔗 Testing Ollama connection: {test_url}", color="status")
        
        try:
            response = requests.get(f"{test_url}/api/version", timeout=10)
            if response.status_code == 200:
                pretty_print("✅ Ollama connection successful!", color="success")
                return True
            else:
                pretty_print(f"❌ Ollama responded with status {response.status_code}", color="failure")
                return False
        except Exception as e:
            pretty_print(f"❌ Ollama connection test failed: {e}", color="failure")
            return False
    
    pretty_print(f"⚠️  Connection test not implemented for provider: {provider_name}", color="warning")
    return None

def check_env_file():
    """Check if .env file exists and has required variables."""
    env_file = ".env"
    env_example = ".env.example"
    
    if not os.path.exists(env_file):
        if os.path.exists(env_example):
            pretty_print(f"❌ .env file missing. Found {env_example}", color="failure")
            pretty_print(f"🔧 Fix: Copy {env_example} to {env_file}", color="info")
            return False
        else:
            pretty_print("❌ Both .env and .env.example files are missing", color="failure")
            return False
    
    pretty_print("✅ .env file found", color="success")
    return True

def auto_fix_config(config_path="config.ini"):
    """Attempt to automatically fix common configuration issues."""
    config = read_config(config_path)
    if not config:
        return False
    
    modified = False
    
    if 'MAIN' in config:
        main_config = config['MAIN']
        provider_name = main_config.get('provider_name', '').lower()
        server_address = main_config.get('provider_server_address', '')
        
        # Auto-fix missing HTTP prefix
        http_required_providers = ['lm-studio', 'server', 'openai']
        if (provider_name in http_required_providers and 
            server_address and 
            not server_address.startswith(('http://', 'https://'))):
            
            new_address = f"http://{server_address}"
            config['MAIN']['provider_server_address'] = new_address
            pretty_print(f"🔧 Auto-fixed server address: {server_address} → {new_address}", color="success")
            modified = True
    
    if modified:
        try:
            with open(config_path, 'w') as f:
                config.write(f)
            pretty_print(f"✅ Configuration file updated: {config_path}", color="success")
            return True
        except Exception as e:
            pretty_print(f"❌ Failed to save config file: {e}", color="failure")
            return False
    
    return False

def main():
    """Main validation function."""
    print("🔧 AgenticSeek Configuration Validator")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("config.ini"):
        pretty_print("❌ config.ini not found. Are you in the AgenticSeek directory?", color="failure")
        return False
    
    # Check .env file
    print("\n📍 Step 1: Checking .env file...")
    env_ok = check_env_file()
    
    # Read and validate config
    print("\n📍 Step 2: Validating configuration...")
    config = read_config()
    if not config:
        return False
    
    issues, fixes = validate_provider_config(config)
    
    if not issues:
        pretty_print("✅ Configuration looks good!", color="success")
    else:
        pretty_print("❌ Configuration issues found:", color="failure")
        for issue in issues:
            print(f"  {issue}")
        
        if fixes:
            print("\n💡 Suggested fixes:")
            for fix in fixes:
                print(f"  {fix}")
        
        # Offer to auto-fix
        print("\n🔧 Attempt automatic fixes? (y/n): ", end="")
        if input().lower().startswith('y'):
            if auto_fix_config():
                print("\n📍 Re-validating after fixes...")
                config = read_config()
                issues, _ = validate_provider_config(config)
                if not issues:
                    pretty_print("✅ All issues fixed!", color="success")
                else:
                    pretty_print("⚠️  Some issues remain:", color="warning")
                    for issue in issues:
                        print(f"  {issue}")
    
    # Test connection
    print("\n📍 Step 3: Testing provider connection...")
    connection_ok = test_connection(config)
    
    # Summary
    print("\n📋 Summary:")
    print(f"  .env file: {'✅' if env_ok else '❌'}")
    print(f"  Configuration: {'✅' if not issues else '❌'}")
    print(f"  Connection: {'✅' if connection_ok else '❌' if connection_ok is False else '⚠️ '}")
    
    if env_ok and not issues and connection_ok:
        pretty_print("\n🎉 AgenticSeek configuration is ready!", color="success")
        return True
    else:
        pretty_print("\n⚠️  Please fix the issues above before running AgenticSeek.", color="warning")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⚠️  Validation cancelled by user")
        sys.exit(1)
    except Exception as e:
        pretty_print(f"\n❌ Unexpected error: {e}", color="failure")
        sys.exit(1)
