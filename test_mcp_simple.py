#!/usr/bin/env python3
"""
Simplified MCP Test Script
Tests core MCP components without full AgenticSeek dependencies
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_mcp_config():
    """Test MCP configuration management"""
    print("Testing MCP Configuration...")
    
    try:
        # Create a minimal config file for testing
        config_content = """[MCP_GENERAL]
enabled = true
workspace_base = /tmp/test_mcp
health_check_interval = 10
max_concurrent_servers = 3
operation_timeout = 120

[MCP_SECURITY]
enable_sandboxing = true
sandbox_permissions = 750
allow_network_access = false
restrict_file_access = true

[MCP_REGISTRY]
registry_url = https://registry.smithery.ai
registry_timeout = 15
"""
        
        # Import and test config
        from sources.agents.mcp_agent import MCPConfig
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ini', delete=False) as f:
            f.write(config_content)
            config_file = f.name
        
        try:
            config = MCPConfig(config_file)
            
            # Test configuration values
            assert config.getboolean('MCP_GENERAL', 'enabled') == True
            assert config.getint('MCP_GENERAL', 'health_check_interval') == 10
            assert config.get('MCP_SECURITY', 'sandbox_permissions') == '750'
            assert config.get('MCP_REGISTRY', 'registry_url') == 'https://registry.smithery.ai'
            
            print("✅ MCP Configuration test passed!")
            return True
            
        finally:
            os.unlink(config_file)
            
    except Exception as e:
        print(f"❌ MCP Configuration test failed: {e}")
        return False

def test_mcp_enums():
    """Test MCP enumerations"""
    print("Testing MCP Enumerations...")
    
    try:
        from sources.agents.mcp_agent import MCPServerStatus
        
        # Test enum values
        assert MCPServerStatus.STOPPED.value == "stopped"
        assert MCPServerStatus.RUNNING.value == "running"
        assert MCPServerStatus.ERROR.value == "error"
        
        print("✅ MCP Enumerations test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MCP Enumerations test failed: {e}")
        return False

def test_mcp_dataclasses():
    """Test MCP data structures"""
    print("Testing MCP Data Structures...")
    
    try:
        from sources.agents.mcp_agent import MCPServerInfo, MCPServerStatus
        
        # Create test server info
        server_info = MCPServerInfo(
            qualified_name="test/server",
            display_name="Test Server",
            description="A test server",
            install_command="npm install test-server",
            run_command="node test-server",
            tools=[{"name": "test_tool", "description": "A test tool"}],
            resources=[{"name": "test_resource", "type": "database"}]
        )
        
        # Test default values
        assert server_info.status == MCPServerStatus.STOPPED
        assert server_info.process is None
        assert server_info.session_id is None
        
        print("✅ MCP Data Structures test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MCP Data Structures test failed: {e}")
        return False

def test_mcp_sandbox():
    """Test MCP sandbox functionality"""
    print("Testing MCP Security Sandbox...")
    
    try:
        from sources.agents.mcp_agent import MCPConfig, MCPSecuritySandbox
        
        # Create test config
        config = MCPConfig()
        config._create_default_config()
        
        # Create sandbox
        sandbox = MCPSecuritySandbox(config)
        
        # Test sandbox creation
        sandbox_path = sandbox.create_sandbox("test_server")
        assert os.path.exists(sandbox_path)
        assert os.path.isdir(sandbox_path)
        
        # Test cleanup
        sandbox.cleanup_sandbox(sandbox_path)
        assert not os.path.exists(sandbox_path)
        
        print("✅ MCP Security Sandbox test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MCP Security Sandbox test failed: {e}")
        return False

def test_import_completeness():
    """Test that all MCP classes can be imported"""
    print("Testing MCP Import Completeness...")
    
    try:
        from sources.agents.mcp_agent import (
            MCPConfig, MCPServerStatus, MCPServerInfo, 
            MCPSecuritySandbox, MCPRegistry, MCPLifecycleManager
        )
        
        # Test that classes exist and are callable
        assert callable(MCPConfig)
        assert callable(MCPSecuritySandbox)
        assert callable(MCPRegistry)
        assert callable(MCPLifecycleManager)
        
        print("✅ MCP Import Completeness test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MCP Import Completeness test failed: {e}")
        return False

def main():
    """Run simplified MCP tests"""
    print("=" * 50)
    print("🧪 AgenticSeek MCP Protocol Test Suite")
    print("=" * 50)
    
    tests = [
        test_mcp_config,
        test_mcp_enums,
        test_mcp_dataclasses,
        test_mcp_sandbox,
        test_import_completeness
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All MCP core tests passed!")
        print("✅ MCP Protocol Implementation is functional!")
    else:
        print("⚠️  Some tests failed. Implementation needs review.")
    
    print("=" * 50)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
