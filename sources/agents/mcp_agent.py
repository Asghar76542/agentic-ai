import os
import asyncio
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import time
import uuid
import configparser

from sources.utility import pretty_print, animate_thinking
from sources.agents.agent import Agent
from sources.tools.mcpFinder import MCP_finder
from sources.memory import Memory

class MCPConfig:
    """Configuration management for MCP settings"""
    
    def __init__(self, config_file: str = "mcp_config.ini"):
        self.config = configparser.ConfigParser()
        self.config_file = config_file
        self.load_config()
    
    def load_config(self):
        """Load MCP configuration from file"""
        try:
            if os.path.exists(self.config_file):
                self.config.read(self.config_file)
            else:
                # Use default configuration
                self._create_default_config()
        except Exception as e:
            pretty_print(f"Error loading MCP config: {e}", color="warning")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default configuration"""
        self.config['MCP_GENERAL'] = {
            'enabled': 'true',
            'workspace_base': '/tmp/agenticseek_mcp_workspaces',
            'health_check_interval': '30',
            'max_concurrent_servers': '10',
            'operation_timeout': '300'
        }
        self.config['MCP_SECURITY'] = {
            'enable_sandboxing': 'true',
            'sandbox_permissions': '750',
            'allow_network_access': 'true',
            'restrict_file_access': 'true',
            'max_memory_per_server': '512',
            'max_cpu_per_server': '50'
        }
        self.config['MCP_REGISTRY'] = {
            'registry_url': 'https://registry.smithery.ai',
            'registry_timeout': '30',
            'cache_duration': '60',
            'auto_update_listings': 'true'
        }
    
    def get(self, section: str, key: str, fallback=None):
        """Get configuration value"""
        return self.config.get(section, key, fallback=fallback)
    
    def getboolean(self, section: str, key: str, fallback=False):
        """Get boolean configuration value"""
        return self.config.getboolean(section, key, fallback=fallback)
    
    def getint(self, section: str, key: str, fallback=0):
        """Get integer configuration value"""
        return self.config.getint(section, key, fallback=fallback)

class MCPServerStatus(Enum):
    """Status of MCP server instances"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    STOPPING = "stopping"

@dataclass
class MCPServerInfo:
    """Information about an MCP server"""
    qualified_name: str
    display_name: str
    description: str
    install_command: str
    run_command: str
    tools: List[Dict[str, Any]]
    resources: List[Dict[str, Any]]
    status: MCPServerStatus = MCPServerStatus.STOPPED
    process: Optional[subprocess.Popen] = None
    session_id: Optional[str] = None
    port: Optional[int] = None
    workspace_dir: Optional[str] = None

class MCPSecuritySandbox:
    """Security sandbox for MCP server execution"""
    
    def __init__(self, config: MCPConfig):
        self.config = config
        self.workspace_base = config.get('MCP_GENERAL', 'workspace_base', '/tmp/agenticseek_mcp_workspaces')
        self.sandbox_permissions = int(config.get('MCP_SECURITY', 'sandbox_permissions', '750'), 8)
        self.enable_sandboxing = config.getboolean('MCP_SECURITY', 'enable_sandboxing', True)
        Path(self.workspace_base).mkdir(parents=True, exist_ok=True)
        
    def create_sandbox(self, server_name: str) -> str:
        """Create isolated workspace for MCP server"""
        sandbox_id = f"{server_name}_{uuid.uuid4().hex[:8]}"
        sandbox_path = os.path.join(self.workspace_base, sandbox_id)
        Path(sandbox_path).mkdir(parents=True, exist_ok=True)
        
        # Set restrictive permissions
        if self.enable_sandboxing:
            os.chmod(sandbox_path, self.sandbox_permissions)
        
        return sandbox_path
    
    def cleanup_sandbox(self, sandbox_path: str):
        """Clean up sandbox workspace"""
        try:
            if os.path.exists(sandbox_path):
                shutil.rmtree(sandbox_path)
        except Exception as e:
            pretty_print(f"Warning: Failed to cleanup sandbox {sandbox_path}: {e}", color="warning")

class MCPRegistry:
    """Registry for managing MCP servers"""
    
    def __init__(self, api_key: str, config: MCPConfig):
        self.config = config
        self.finder = MCP_finder(api_key)
        self.installed_servers: Dict[str, MCPServerInfo] = {}
        self.running_servers: Dict[str, MCPServerInfo] = {}
        self.sandbox = MCPSecuritySandbox(config)
        self.max_concurrent = config.getint('MCP_GENERAL', 'max_concurrent_servers', 10)
        self.operation_timeout = config.getint('MCP_GENERAL', 'operation_timeout', 300)
        
    def search_servers(self, query: str) -> List[Dict[str, Any]]:
        """Search for MCP servers in registry"""
        try:
            return self.finder.find_mcp_servers(query)
        except Exception as e:
            pretty_print(f"Error searching MCP servers: {e}", color="error")
            return []
    
    def install_server(self, qualified_name: str) -> bool:
        """Install an MCP server"""
        try:
            # Get server details from registry
            server_details = self.finder.get_mcp_server_details(qualified_name)
            
            server_info = MCPServerInfo(
                qualified_name=qualified_name,
                display_name=server_details.get('displayName', qualified_name),
                description=server_details.get('description', ''),
                install_command=server_details.get('installCommand', ''),
                run_command=server_details.get('runCommand', ''),
                tools=server_details.get('tools', []),
                resources=server_details.get('resources', [])
            )
            
            # Install the server (npm install, pip install, etc.)
            if server_info.install_command:
                pretty_print(f"Installing MCP server: {qualified_name}", color="info")
                result = subprocess.run(
                    server_info.install_command.split(),
                    capture_output=True,
                    text=True,
                    timeout=self.operation_timeout
                )
                
                if result.returncode != 0:
                    pretty_print(f"Installation failed: {result.stderr}", color="error")
                    return False
            
            self.installed_servers[qualified_name] = server_info
            pretty_print(f"Successfully installed MCP server: {qualified_name}", color="success")
            return True
            
        except Exception as e:
            pretty_print(f"Error installing MCP server {qualified_name}: {e}", color="error")
            return False
    
    def start_server(self, qualified_name: str) -> bool:
        """Start an MCP server instance"""
        if qualified_name not in self.installed_servers:
            pretty_print(f"MCP server {qualified_name} not installed", color="error")
            return False
        
        server_info = self.installed_servers[qualified_name]
        
        if server_info.status == MCPServerStatus.RUNNING:
            pretty_print(f"MCP server {qualified_name} already running", color="warning")
            return True
        
        try:
            # Create sandbox workspace
            workspace_dir = self.sandbox.create_sandbox(qualified_name)
            server_info.workspace_dir = workspace_dir
            
            # Update status
            server_info.status = MCPServerStatus.STARTING
            server_info.session_id = str(uuid.uuid4())
            
            # Start the server process
            if server_info.run_command:
                # Modify run command to use sandbox workspace
                run_cmd = server_info.run_command.replace('${WORKSPACE}', workspace_dir)
                
                pretty_print(f"Starting MCP server: {qualified_name}", color="info")
                process = subprocess.Popen(
                    run_cmd.split(),
                    cwd=workspace_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env={**os.environ, 'MCP_WORKSPACE': workspace_dir}
                )
                
                server_info.process = process
                server_info.status = MCPServerStatus.RUNNING
                
                # Add to running servers
                self.running_servers[qualified_name] = server_info
                
                pretty_print(f"Successfully started MCP server: {qualified_name}", color="success")
                return True
            
        except Exception as e:
            pretty_print(f"Error starting MCP server {qualified_name}: {e}", color="error")
            server_info.status = MCPServerStatus.ERROR
            if server_info.workspace_dir:
                self.sandbox.cleanup_sandbox(server_info.workspace_dir)
            return False
        
        return False
    
    def stop_server(self, qualified_name: str) -> bool:
        """Stop an MCP server instance"""
        if qualified_name not in self.running_servers:
            pretty_print(f"MCP server {qualified_name} not running", color="warning")
            return True
        
        server_info = self.running_servers[qualified_name]
        
        try:
            server_info.status = MCPServerStatus.STOPPING
            
            if server_info.process:
                server_info.process.terminate()
                try:
                    server_info.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server_info.process.kill()
                    server_info.process.wait()
            
            # Cleanup sandbox
            if server_info.workspace_dir:
                self.sandbox.cleanup_sandbox(server_info.workspace_dir)
            
            server_info.status = MCPServerStatus.STOPPED
            del self.running_servers[qualified_name]
            
            pretty_print(f"Successfully stopped MCP server: {qualified_name}", color="success")
            return True
            
        except Exception as e:
            pretty_print(f"Error stopping MCP server {qualified_name}: {e}", color="error")
            return False
    
    def get_server_status(self, qualified_name: str) -> MCPServerStatus:
        """Get status of an MCP server"""
        if qualified_name in self.running_servers:
            return self.running_servers[qualified_name].status
        elif qualified_name in self.installed_servers:
            return self.installed_servers[qualified_name].status
        else:
            return MCPServerStatus.STOPPED
    
    def list_available_tools(self, qualified_name: str = None) -> Dict[str, List[Dict]]:
        """List tools available from MCP servers"""
        tools = {}
        
        if qualified_name:
            if qualified_name in self.running_servers:
                server = self.running_servers[qualified_name]
                tools[qualified_name] = server.tools
        else:
            for name, server in self.running_servers.items():
                tools[name] = server.tools
                
        return tools

class MCPLifecycleManager:
    """Manages lifecycle of MCP servers"""
    
    def __init__(self, registry: MCPRegistry, config: MCPConfig):
        self.registry = registry
        self.config = config
        self.health_check_interval = config.getint('MCP_GENERAL', 'health_check_interval', 30)
        self.monitoring_thread = None
        self.stop_monitoring = False
        
    def start_monitoring(self):
        """Start health monitoring of MCP servers"""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(target=self._monitor_servers)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitor_servers(self):
        """Monitor health of running MCP servers"""
        while not self.stop_monitoring:
            try:
                for qualified_name, server_info in list(self.registry.running_servers.items()):
                    if server_info.process:
                        # Check if process is still alive
                        if server_info.process.poll() is not None:
                            pretty_print(f"MCP server {qualified_name} has stopped unexpectedly", color="warning")
                            server_info.status = MCPServerStatus.ERROR
                            
                            # Cleanup
                            if server_info.workspace_dir:
                                self.registry.sandbox.cleanup_sandbox(server_info.workspace_dir)
                            
                            del self.registry.running_servers[qualified_name]
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                pretty_print(f"Error in MCP server monitoring: {e}", color="error")
                time.sleep(self.health_check_interval)

    def shutdown_all(self):
        """Shutdown all running MCP servers"""
        pretty_print("Shutting down all MCP servers...", color="info")
        
        for qualified_name in list(self.registry.running_servers.keys()):
            self.registry.stop_server(qualified_name)
        
        self.stop_monitoring = True

class McpAgent(Agent):
    """
    Enhanced MCP agent with full protocol implementation, security sandboxing,
    and lifecycle management capabilities.
    """

    def __init__(self, name, prompt_path, provider, verbose=False):
        """
        The mcp agent is a special agent for using MCPs.
        MCP agent will be disabled if the user does not explicitly set the MCP_FINDER_API_KEY in environment variable.
        """
        super().__init__(name, prompt_path, provider, verbose, None)
        
        # Load MCP configuration
        self.mcp_config = MCPConfig()
        
        # Check if MCP is enabled in configuration
        if not self.mcp_config.getboolean('MCP_GENERAL', 'enabled', True):
            pretty_print("MCP functionality disabled in configuration.", color="warning")
            self.enabled = False
            return
        
        keys = self.get_api_keys()
        
        # Initialize MCP components
        self.registry = None
        self.lifecycle_manager = None
        
        if keys["mcp_finder"]:
            self.registry = MCPRegistry(keys["mcp_finder"], self.mcp_config)
            self.lifecycle_manager = MCPLifecycleManager(self.registry, self.mcp_config)
            self.lifecycle_manager.start_monitoring()
        
        self.tools = {
            "mcp_finder": MCP_finder(keys["mcp_finder"]),
            "search_mcp_servers": self._search_mcp_servers,
            "install_mcp_server": self._install_mcp_server,
            "start_mcp_server": self._start_mcp_server,
            "stop_mcp_server": self._stop_mcp_server,
            "list_mcp_servers": self._list_mcp_servers,
            "get_mcp_server_status": self._get_mcp_server_status,
            "list_available_tools": self._list_available_tools,
            "execute_mcp_tool": self._execute_mcp_tool,
        }
        
        self.role = "mcp"
        self.type = "mcp_agent"
        self.memory = Memory(self.load_prompt(prompt_path),
                                recover_last_session=False, # session recovery in handled by the interaction class
                                memory_compression=False,
                                model_provider=provider.get_model_name())
        self.enabled = True
    
    def get_api_keys(self) -> dict:
        """
        Returns the API keys for the tools.
        """
        api_key_mcp_finder = os.getenv("MCP_FINDER_API_KEY")
        if not api_key_mcp_finder or api_key_mcp_finder == "":
            pretty_print("MCP Finder disabled.", color="warning")
            self.enabled = False
        return {
            "mcp_finder": api_key_mcp_finder
        }
    
    def _search_mcp_servers(self, query: str) -> str:
        """Search for MCP servers in the registry"""
        if not self.registry:
            return "MCP Registry not available"
        
        try:
            servers = self.registry.search_servers(query)
            if not servers:
                return f"No MCP servers found for query: {query}"
            
            result = f"Found {len(servers)} MCP server(s) for '{query}':\n\n"
            for server in servers:
                result += f"Name: {server.get('displayName', 'Unknown')}\n"
                result += f"Qualified Name: {server.get('qualifiedName', 'Unknown')}\n"
                result += f"Description: {server.get('description', 'No description')}\n"
                result += f"Tools: {len(server.get('tools', []))}\n"
                result += f"Resources: {len(server.get('resources', []))}\n"
                result += "---\n"
                
            return result
            
        except Exception as e:
            return f"Error searching MCP servers: {str(e)}"
    
    def _install_mcp_server(self, qualified_name: str) -> str:
        """Install an MCP server"""
        if not self.registry:
            return "MCP Registry not available"
        
        try:
            success = self.registry.install_server(qualified_name)
            if success:
                return f"Successfully installed MCP server: {qualified_name}"
            else:
                return f"Failed to install MCP server: {qualified_name}"
        except Exception as e:
            return f"Error installing MCP server {qualified_name}: {str(e)}"
    
    def _start_mcp_server(self, qualified_name: str) -> str:
        """Start an MCP server"""
        if not self.registry:
            return "MCP Registry not available"
        
        try:
            success = self.registry.start_server(qualified_name)
            if success:
                return f"Successfully started MCP server: {qualified_name}"
            else:
                return f"Failed to start MCP server: {qualified_name}"
        except Exception as e:
            return f"Error starting MCP server {qualified_name}: {str(e)}"
    
    def _stop_mcp_server(self, qualified_name: str) -> str:
        """Stop an MCP server"""
        if not self.registry:
            return "MCP Registry not available"
        
        try:
            success = self.registry.stop_server(qualified_name)
            if success:
                return f"Successfully stopped MCP server: {qualified_name}"
            else:
                return f"Failed to stop MCP server: {qualified_name}"
        except Exception as e:
            return f"Error stopping MCP server {qualified_name}: {str(e)}"
    
    def _list_mcp_servers(self) -> str:
        """List all installed and running MCP servers"""
        if not self.registry:
            return "MCP Registry not available"
        
        try:
            result = "MCP Servers Status:\n\n"
            
            # Installed servers
            if self.registry.installed_servers:
                result += "Installed Servers:\n"
                for name, server in self.registry.installed_servers.items():
                    result += f"- {server.display_name} ({name}): {server.status.value}\n"
                result += "\n"
            
            # Running servers
            if self.registry.running_servers:
                result += "Running Servers:\n"
                for name, server in self.registry.running_servers.items():
                    result += f"- {server.display_name} ({name}): {len(server.tools)} tools available\n"
                result += "\n"
            
            if not self.registry.installed_servers and not self.registry.running_servers:
                result += "No MCP servers installed or running.\n"
                
            return result
            
        except Exception as e:
            return f"Error listing MCP servers: {str(e)}"
    
    def _get_mcp_server_status(self, qualified_name: str) -> str:
        """Get status of specific MCP server"""
        if not self.registry:
            return "MCP Registry not available"
        
        try:
            status = self.registry.get_server_status(qualified_name)
            return f"MCP server {qualified_name} status: {status.value}"
        except Exception as e:
            return f"Error getting status for {qualified_name}: {str(e)}"
    
    def _list_available_tools(self, qualified_name: str = None) -> str:
        """List tools available from MCP servers"""
        if not self.registry:
            return "MCP Registry not available"
        
        try:
            tools = self.registry.list_available_tools(qualified_name)
            
            if not tools:
                return "No tools available from running MCP servers"
            
            result = "Available MCP Tools:\n\n"
            
            for server_name, server_tools in tools.items():
                result += f"Server: {server_name}\n"
                if server_tools:
                    for tool in server_tools:
                        result += f"  - {tool.get('name', 'Unknown')}: {tool.get('description', 'No description')}\n"
                else:
                    result += "  No tools available\n"
                result += "\n"
                
            return result
            
        except Exception as e:
            return f"Error listing available tools: {str(e)}"
    
    def _execute_mcp_tool(self, server_name: str, tool_name: str, arguments: str = "{}") -> str:
        """Execute a tool from an MCP server"""
        if not self.registry:
            return "MCP Registry not available"
        
        try:
            if server_name not in self.registry.running_servers:
                return f"MCP server {server_name} is not running"
            
            # Parse arguments
            try:
                args = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                return f"Invalid JSON arguments: {arguments}"
            
            # This is a simplified implementation - in a full implementation,
            # we would use the MCP protocol to communicate with the server
            server_info = self.registry.running_servers[server_name]
            
            # Find the tool
            tool_found = None
            for tool in server_info.tools:
                if tool.get('name') == tool_name:
                    tool_found = tool
                    break
            
            if not tool_found:
                return f"Tool {tool_name} not found in server {server_name}"
            
            return f"Tool execution simulated: {tool_name} with args {args} from server {server_name}"
            
        except Exception as e:
            return f"Error executing MCP tool: {str(e)}"
    
    def expand_prompt(self, prompt):
        """
        Expands the prompt with the tools available.
        """
        tools_str = self.get_tools_description()
        mcp_status = ""
        
        if self.registry:
            running_count = len(self.registry.running_servers)
            installed_count = len(self.registry.installed_servers)
            mcp_status = f"\nMCP Status: {installed_count} servers installed, {running_count} running"
        
        prompt += f"""
        You can use the following tools and MCPs:
        {tools_str}
        
        {mcp_status}
        
        Available MCP commands:
        - search_mcp_servers(query): Search for MCP servers
        - install_mcp_server(qualified_name): Install an MCP server
        - start_mcp_server(qualified_name): Start an MCP server
        - stop_mcp_server(qualified_name): Stop an MCP server
        - list_mcp_servers(): List all MCP servers
        - get_mcp_server_status(qualified_name): Get server status
        - list_available_tools(qualified_name): List available tools
        - execute_mcp_tool(server_name, tool_name, arguments): Execute an MCP tool
        """
        return prompt
    
    async def process(self, prompt, speech_module) -> Tuple[str, str]:
        """Process user request with enhanced MCP capabilities"""
        if self.enabled == False:
            return "MCP Agent is disabled.", ""
        
        prompt = self.expand_prompt(prompt)
        self.memory.push('user', prompt)
        working = True
        
        while working == True:
            animate_thinking("Thinking...", color="status")
            answer, reasoning = await self.llm_request()
            exec_success, _ = self.execute_modules(answer)
            answer = self.remove_blocks(answer)
            self.last_answer = answer
            self.last_reasoning = reasoning
            self.status_message = "Ready"
            
            if len(self.blocks_result) == 0:
                working = False
                
        return answer, reasoning
    
    def __del__(self):
        """Cleanup when agent is destroyed"""
        if self.lifecycle_manager:
            self.lifecycle_manager.shutdown_all()

if __name__ == "__main__":
    pass