#!/usr/bin/env python3
"""
Test Script for Enhanced MCP Protocol Implementation
Tests all major components of the MCP system
"""

import os
import sys
import asyncio
import tempfile
import shutil
from pathlib import Path

# Add the project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sources.agents.mcp_agent import (
    McpAgent, MCPConfig, MCPRegistry, MCPSecuritySandbox, 
    MCPLifecycleManager, MCPServerStatus
)
from sources.llm_provider import Provider
from sources.utility import pretty_print

class MCPTestSuite:
    """Comprehensive test suite for MCP implementation"""
    
    def __init__(self):
        self.test_results = {}
        self.temp_dir = None
        
    def setup_test_environment(self):
        """Set up test environment"""
        pretty_print("Setting up test environment...", color="info")
        
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp(prefix="mcp_test_")
        
        # Set test environment variables
        os.environ['MCP_FINDER_API_KEY'] = 'test_key_123'
        
        # Create test configuration
        test_config_path = os.path.join(self.temp_dir, "test_mcp_config.ini")
        with open(test_config_path, 'w') as f:
            f.write("""[MCP_GENERAL]
enabled = true
workspace_base = {}/mcp_workspaces
health_check_interval = 5
max_concurrent_servers = 5
operation_timeout = 60

[MCP_SECURITY]
enable_sandboxing = true
sandbox_permissions = 750
allow_network_access = false
restrict_file_access = true
max_memory_per_server = 256
max_cpu_per_server = 25

[MCP_REGISTRY]
registry_url = https://registry.smithery.ai
registry_timeout = 15
cache_duration = 30
auto_update_listings = false
""".format(self.temp_dir))
        
        return test_config_path
    
    def test_config_loading(self, config_path):
        """Test MCP configuration loading"""
        pretty_print("Testing configuration loading...", color="status")
        
        try:
            config = MCPConfig(config_path)
            
            # Test basic configuration values
            assert config.getboolean('MCP_GENERAL', 'enabled') == True
            assert config.getint('MCP_GENERAL', 'health_check_interval') == 5
            assert config.get('MCP_SECURITY', 'sandbox_permissions') == '750'
            
            self.test_results['config_loading'] = True
            pretty_print("✓ Configuration loading test passed", color="success")
            
        except Exception as e:
            self.test_results['config_loading'] = False
            pretty_print(f"✗ Configuration loading test failed: {e}", color="error")
    
    def test_sandbox_creation(self, config_path):
        """Test security sandbox functionality"""
        pretty_print("Testing security sandbox...", color="status")
        
        try:
            config = MCPConfig(config_path)
            sandbox = MCPSecuritySandbox(config)
            
            # Create test sandbox
            sandbox_path = sandbox.create_sandbox("test_server")
            
            # Verify sandbox was created
            assert os.path.exists(sandbox_path)
            assert os.path.isdir(sandbox_path)
            
            # Check permissions
            stat_info = os.stat(sandbox_path)
            permissions = oct(stat_info.st_mode)[-3:]
            assert permissions == '750'
            
            # Cleanup
            sandbox.cleanup_sandbox(sandbox_path)
            assert not os.path.exists(sandbox_path)
            
            self.test_results['sandbox_creation'] = True
            pretty_print("✓ Security sandbox test passed", color="success")
            
        except Exception as e:
            self.test_results['sandbox_creation'] = False
            pretty_print(f"✗ Security sandbox test failed: {e}", color="error")
    
    def test_registry_initialization(self, config_path):
        """Test MCP registry initialization"""
        pretty_print("Testing registry initialization...", color="status")
        
        try:
            config = MCPConfig(config_path)
            registry = MCPRegistry("test_api_key", config)
            
            # Verify registry components
            assert registry.config is not None
            assert registry.finder is not None
            assert registry.sandbox is not None
            assert isinstance(registry.installed_servers, dict)
            assert isinstance(registry.running_servers, dict)
            assert registry.max_concurrent == 5
            assert registry.operation_timeout == 60
            
            self.test_results['registry_initialization'] = True
            pretty_print("✓ Registry initialization test passed", color="success")
            
        except Exception as e:
            self.test_results['registry_initialization'] = False
            pretty_print(f"✗ Registry initialization test failed: {e}", color="error")
    
    def test_lifecycle_manager(self, config_path):
        """Test MCP lifecycle manager"""
        pretty_print("Testing lifecycle manager...", color="status")
        
        try:
            config = MCPConfig(config_path)
            registry = MCPRegistry("test_api_key", config)
            lifecycle_manager = MCPLifecycleManager(registry, config)
            
            # Verify lifecycle manager
            assert lifecycle_manager.registry is registry
            assert lifecycle_manager.config is config
            assert lifecycle_manager.health_check_interval == 5
            assert lifecycle_manager.monitoring_thread is None
            assert lifecycle_manager.stop_monitoring == False
            
            # Test monitoring start/stop
            lifecycle_manager.start_monitoring()
            assert lifecycle_manager.monitoring_thread is not None
            assert lifecycle_manager.monitoring_thread.is_alive()
            
            lifecycle_manager.stop_monitoring = True
            lifecycle_manager.monitoring_thread.join(timeout=2)
            
            self.test_results['lifecycle_manager'] = True
            pretty_print("✓ Lifecycle manager test passed", color="success")
            
        except Exception as e:
            self.test_results['lifecycle_manager'] = False
            pretty_print(f"✗ Lifecycle manager test failed: {e}", color="error")
    
    def test_agent_initialization(self, config_path):
        """Test MCP agent initialization"""
        pretty_print("Testing agent initialization...", color="status")
        
        try:
            # Create mock provider
            class MockProvider:
                def get_model_name(self):
                    return "test_model"
            
            provider = MockProvider()
            
            # Create agent (will use actual MCP_FINDER_API_KEY from environment)
            agent = McpAgent(
                name="Test MCP Agent",
                prompt_path="prompts/base/mcp_agent.txt",
                provider=provider,
                verbose=True
            )
            
            # Verify agent initialization
            assert agent.agent_name == "Test MCP Agent"
            assert agent.role == "mcp"
            assert agent.type == "mcp_agent"
            assert agent.mcp_config is not None
            
            # Verify tools are available
            expected_tools = [
                "mcp_finder", "search_mcp_servers", "install_mcp_server",
                "start_mcp_server", "stop_mcp_server", "list_mcp_servers",
                "get_mcp_server_status", "list_available_tools", "execute_mcp_tool"
            ]
            
            for tool in expected_tools:
                assert tool in agent.tools
            
            self.test_results['agent_initialization'] = True
            pretty_print("✓ Agent initialization test passed", color="success")
            
        except Exception as e:
            self.test_results['agent_initialization'] = False
            pretty_print(f"✗ Agent initialization test failed: {e}", color="error")
    
    def test_tool_execution_simulation(self, config_path):
        """Test MCP tool execution simulation"""
        pretty_print("Testing tool execution simulation...", color="status")
        
        try:
            config = MCPConfig(config_path)
            registry = MCPRegistry("test_api_key", config)
            
            # Test search functionality
            search_result = registry.search_servers("weather")
            assert isinstance(search_result, list)  # Should return empty list for test API key
            
            # Test server status
            status = registry.get_server_status("nonexistent_server")
            assert status == MCPServerStatus.STOPPED
            
            # Test tools listing
            tools = registry.list_available_tools()
            assert isinstance(tools, dict)
            assert len(tools) == 0  # No running servers
            
            self.test_results['tool_execution_simulation'] = True
            pretty_print("✓ Tool execution simulation test passed", color="success")
            
        except Exception as e:
            self.test_results['tool_execution_simulation'] = False
            pretty_print(f"✗ Tool execution simulation test failed: {e}", color="error")
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        pretty_print("Cleaning up test environment...", color="info")
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Remove test environment variables
        if 'MCP_FINDER_API_KEY' in os.environ:
            del os.environ['MCP_FINDER_API_KEY']
    
    def run_all_tests(self):
        """Run all tests in the suite"""
        pretty_print("Starting MCP Test Suite...", color="info")
        pretty_print("=" * 50, color="info")
        
        try:
            # Setup
            config_path = self.setup_test_environment()
            
            # Run individual tests
            self.test_config_loading(config_path)
            self.test_sandbox_creation(config_path)
            self.test_registry_initialization(config_path)
            self.test_lifecycle_manager(config_path)
            self.test_agent_initialization(config_path)
            self.test_tool_execution_simulation(config_path)
            
            # Report results
            self.report_results()
            
        except Exception as e:
            pretty_print(f"Test suite failed with error: {e}", color="error")
            
        finally:
            self.cleanup_test_environment()
    
    def report_results(self):
        """Report test results"""
        pretty_print("=" * 50, color="info")
        pretty_print("Test Results:", color="info")
        
        passed = 0
        total = len(self.test_results)
        
        for test_name, result in self.test_results.items():
            status = "PASS" if result else "FAIL"
            color = "success" if result else "error"
            pretty_print(f"{test_name}: {status}", color=color)
            if result:
                passed += 1
        
        pretty_print("=" * 50, color="info")
        pretty_print(f"Summary: {passed}/{total} tests passed", color="info")
        
        if passed == total:
            pretty_print("🎉 All tests passed! MCP implementation is ready.", color="success")
        else:
            pretty_print("⚠️  Some tests failed. Please review the implementation.", color="warning")

def main():
    """Main test function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("MCP Test Suite")
        print("Usage: python test_mcp_implementation.py")
        print("\nThis script tests the enhanced MCP protocol implementation.")
        print("It validates configuration, sandboxing, registry, lifecycle management,")
        print("and agent initialization components.")
        return
    
    test_suite = MCPTestSuite()
    test_suite.run_all_tests()

if __name__ == "__main__":
    main()
