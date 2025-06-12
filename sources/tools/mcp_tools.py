"""
MCP Tools Framework
Provides enhanced MCP tool execution, communication, and management capabilities
"""

import json
import asyncio
import websockets
import subprocess
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import uuid
import logging
from pathlib import Path

from sources.tools.tools import Tools
from sources.utility import pretty_print

@dataclass
class MCPToolCall:
    """Represents an MCP tool call"""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str = None
    
    def __post_init__(self):
        if not self.call_id:
            self.call_id = str(uuid.uuid4())

@dataclass 
class MCPToolResult:
    """Represents the result of an MCP tool call"""
    call_id: str
    success: bool
    result: Any = None
    error: str = None
    execution_time: float = 0.0

class MCPCommunicator:
    """Handles communication with MCP servers using JSON-RPC over stdio/websockets"""
    
    def __init__(self):
        self.active_connections: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
    
    async def connect_to_server(self, server_name: str, connection_info: Dict[str, Any]) -> bool:
        """Connect to an MCP server"""
        try:
            connection_type = connection_info.get('type', 'stdio')
            
            if connection_type == 'stdio':
                return await self._connect_stdio(server_name, connection_info)
            elif connection_type == 'websocket':
                return await self._connect_websocket(server_name, connection_info)
            else:
                pretty_print(f"Unsupported connection type: {connection_type}", color="error")
                return False
                
        except Exception as e:
            pretty_print(f"Error connecting to MCP server {server_name}: {e}", color="error")
            return False
    
    async def _connect_stdio(self, server_name: str, connection_info: Dict[str, Any]) -> bool:
        """Connect to MCP server via stdio"""
        try:
            command = connection_info.get('command', [])
            if not command:
                return False
            
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=connection_info.get('cwd'),
                env=connection_info.get('env')
            )
            
            self.active_connections[server_name] = {
                'type': 'stdio',
                'process': process,
                'request_id': 0
            }
            
            # Initialize connection with MCP server
            init_request = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(server_name),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {}
                    },
                    "clientInfo": {
                        "name": "AgenticSeek",
                        "version": "1.0.0"
                    }
                }
            }
            
            await self._send_request(server_name, init_request)
            response = await self._receive_response(server_name)
            
            if response and response.get('result'):
                pretty_print(f"Successfully connected to MCP server: {server_name}", color="success")
                return True
            
            return False
            
        except Exception as e:
            pretty_print(f"Error in stdio connection to {server_name}: {e}", color="error")
            return False
    
    async def _connect_websocket(self, server_name: str, connection_info: Dict[str, Any]) -> bool:
        """Connect to MCP server via WebSocket"""
        try:
            uri = connection_info.get('uri')
            if not uri:
                return False
            
            websocket = await websockets.connect(uri)
            
            self.active_connections[server_name] = {
                'type': 'websocket',
                'websocket': websocket,
                'request_id': 0
            }
            
            # Initialize connection
            init_request = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(server_name),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {},
                        "resources": {}
                    },
                    "clientInfo": {
                        "name": "AgenticSeek",
                        "version": "1.0.0"
                    }
                }
            }
            
            await websocket.send(json.dumps(init_request))
            response_raw = await websocket.recv()
            response = json.loads(response_raw)
            
            if response and response.get('result'):
                pretty_print(f"Successfully connected to MCP server: {server_name}", color="success")
                return True
                
            return False
            
        except Exception as e:
            pretty_print(f"Error in websocket connection to {server_name}: {e}", color="error")
            return False
    
    def _get_next_id(self, server_name: str) -> int:
        """Get next request ID for server"""
        connection = self.active_connections.get(server_name)
        if connection:
            connection['request_id'] += 1
            return connection['request_id']
        return 1
    
    async def _send_request(self, server_name: str, request: Dict[str, Any]):
        """Send request to MCP server"""
        connection = self.active_connections.get(server_name)
        if not connection:
            raise Exception(f"No connection to server {server_name}")
        
        request_json = json.dumps(request) + '\n'
        
        if connection['type'] == 'stdio':
            connection['process'].stdin.write(request_json.encode())
            await connection['process'].stdin.drain()
        elif connection['type'] == 'websocket':
            await connection['websocket'].send(request_json)
    
    async def _receive_response(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Receive response from MCP server"""
        connection = self.active_connections.get(server_name)
        if not connection:
            return None
        
        try:
            if connection['type'] == 'stdio':
                line = await connection['process'].stdout.readline()
                if line:
                    return json.loads(line.decode().strip())
            elif connection['type'] == 'websocket':
                response_raw = await connection['websocket'].recv()
                return json.loads(response_raw)
        except Exception as e:
            self.logger.error(f"Error receiving response from {server_name}: {e}")
        
        return None
    
    async def call_tool(self, server_name: str, tool_call: MCPToolCall) -> MCPToolResult:
        """Call a tool on an MCP server"""
        start_time = time.time()
        
        try:
            if server_name not in self.active_connections:
                return MCPToolResult(
                    call_id=tool_call.call_id,
                    success=False,
                    error=f"No connection to server {server_name}"
                )
            
            # Prepare tool call request
            request = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(server_name),
                "method": "tools/call",
                "params": {
                    "name": tool_call.tool_name,
                    "arguments": tool_call.arguments
                }
            }
            
            # Send request
            await self._send_request(server_name, request)
            
            # Receive response
            response = await self._receive_response(server_name)
            
            execution_time = time.time() - start_time
            
            if response:
                if 'result' in response:
                    return MCPToolResult(
                        call_id=tool_call.call_id,
                        success=True,
                        result=response['result'],
                        execution_time=execution_time
                    )
                elif 'error' in response:
                    return MCPToolResult(
                        call_id=tool_call.call_id,
                        success=False,
                        error=response['error'].get('message', 'Unknown error'),
                        execution_time=execution_time
                    )
            
            return MCPToolResult(
                call_id=tool_call.call_id,
                success=False,
                error="No response received",
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return MCPToolResult(
                call_id=tool_call.call_id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """List available tools from an MCP server"""
        try:
            if server_name not in self.active_connections:
                return []
            
            request = {
                "jsonrpc": "2.0",
                "id": self._get_next_id(server_name),
                "method": "tools/list"
            }
            
            await self._send_request(server_name, request)
            response = await self._receive_response(server_name)
            
            if response and 'result' in response:
                return response['result'].get('tools', [])
                
        except Exception as e:
            pretty_print(f"Error listing tools from {server_name}: {e}", color="error")
        
        return []
    
    async def disconnect_server(self, server_name: str):
        """Disconnect from an MCP server"""
        try:
            connection = self.active_connections.get(server_name)
            if not connection:
                return
            
            if connection['type'] == 'stdio' and connection.get('process'):
                connection['process'].terminate()
                await connection['process'].wait()
            elif connection['type'] == 'websocket' and connection.get('websocket'):
                await connection['websocket'].close()
            
            del self.active_connections[server_name]
            pretty_print(f"Disconnected from MCP server: {server_name}", color="info")
            
        except Exception as e:
            pretty_print(f"Error disconnecting from {server_name}: {e}", color="error")
    
    async def disconnect_all(self):
        """Disconnect from all MCP servers"""
        for server_name in list(self.active_connections.keys()):
            await self.disconnect_server(server_name)

class MCPToolExecutor(Tools):
    """Enhanced MCP tool executor with communication capabilities"""
    
    def __init__(self):
        super().__init__()
        self.tag = "mcp_tool_executor"
        self.name = "MCP Tool Executor"
        self.description = "Execute tools from MCP servers with full protocol support"
        self.communicator = MCPCommunicator()
        
    async def execute_tool_async(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Execute a tool asynchronously"""
        tool_call = MCPToolCall(tool_name=tool_name, arguments=arguments)
        return await self.communicator.call_tool(server_name, tool_call)
    
    def execute(self, blocks: list, safety: bool = False) -> str:
        """Execute MCP tool calls from code blocks"""
        if not blocks or not isinstance(blocks, list):
            return "Error: No blocks provided\n"
        
        output = ""
        
        for block in blocks:
            try:
                # Parse the block - expect JSON format
                if isinstance(block, str):
                    # Try to parse as JSON
                    try:
                        tool_data = json.loads(block.strip())
                    except json.JSONDecodeError:
                        output += f"Error: Invalid JSON in block: {block}\n"
                        continue
                else:
                    tool_data = block
                
                # Extract required fields
                server_name = tool_data.get('server_name')
                tool_name = tool_data.get('tool_name')
                arguments = tool_data.get('arguments', {})
                
                if not server_name or not tool_name:
                    output += "Error: server_name and tool_name are required\n"
                    continue
                
                # Execute the tool (sync wrapper for async call)
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                result = loop.run_until_complete(
                    self.execute_tool_async(server_name, tool_name, arguments)
                )
                
                if result.success:
                    output += f"Tool executed successfully:\n"
                    output += f"Result: {json.dumps(result.result, indent=2)}\n"
                    output += f"Execution time: {result.execution_time:.2f}s\n"
                else:
                    output += f"Tool execution failed: {result.error}\n"
                
                output += "-------\n"
                
            except Exception as e:
                output += f"Error processing block: {str(e)}\n"
                continue
        
        return output.strip()
    
    def execution_failure_check(self, output: str) -> bool:
        """Check if execution failed"""
        output = output.strip().lower()
        if not output:
            return True
        if "error" in output or "failed" in output:
            return True
        return False

# Example usage and testing
if __name__ == "__main__":
    async def test_mcp_tools():
        """Test MCP tools functionality"""
        executor = MCPToolExecutor()
        
        # Test tool execution
        test_blocks = [
            {
                "server_name": "test_server",
                "tool_name": "test_tool",
                "arguments": {"param1": "value1", "param2": "value2"}
            }
        ]
        
        result = executor.execute(test_blocks)
        print(f"Test result: {result}")
    
    # Run test
    asyncio.run(test_mcp_tools())
