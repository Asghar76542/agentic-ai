"""
Advanced MCP Server Manager
Provides real-time monitoring, auto-scaling, and health management for MCP servers
"""

import asyncio
import json
import psutil
import time
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from pathlib import Path
import signal
import socket
from contextlib import closing

from sources.utility import pretty_print
from sources.tools.mcp_tools import MCPCommunicator, MCPToolCall, MCPToolResult

class ServerHealthStatus(Enum):
    """Health status of MCP servers"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class ServerMetrics:
    """Performance metrics for MCP servers"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_io: Dict[str, int] = None
    response_times: List[float] = None
    uptime: float = 0.0
    error_count: int = 0
    success_rate: float = 100.0
    
    def __post_init__(self):
        if self.network_io is None:
            self.network_io = {"bytes_sent": 0, "bytes_recv": 0}
        if self.response_times is None:
            self.response_times = []

@dataclass
class ServerAlert:
    """Alert information for MCP servers"""
    server_name: str
    alert_type: str
    severity: str
    message: str
    timestamp: float
    resolved: bool = False

class MCPServerMonitor:
    """Real-time monitoring for MCP servers"""
    
    def __init__(self, callback: Optional[Callable] = None):
        self.servers: Dict[str, Any] = {}
        self.metrics: Dict[str, ServerMetrics] = {}
        self.alerts: List[ServerAlert] = []
        self.monitoring_active = False
        self.monitor_thread = None
        self.callback = callback
        self.logger = logging.getLogger(__name__)
        
        # Thresholds
        self.cpu_threshold = 80.0
        self.memory_threshold = 80.0
        self.response_time_threshold = 5.0
        self.error_rate_threshold = 10.0
    
    def add_server(self, server_name: str, process: Any, server_info: Dict[str, Any]):
        """Add server for monitoring"""
        self.servers[server_name] = {
            'process': process,
            'info': server_info,
            'start_time': time.time(),
            'last_check': time.time()
        }
        self.metrics[server_name] = ServerMetrics()
        pretty_print(f"Added server {server_name} to monitoring", color="info")
    
    def remove_server(self, server_name: str):
        """Remove server from monitoring"""
        if server_name in self.servers:
            del self.servers[server_name]
        if server_name in self.metrics:
            del self.metrics[server_name]
        # Mark alerts as resolved
        for alert in self.alerts:
            if alert.server_name == server_name and not alert.resolved:
                alert.resolved = True
        pretty_print(f"Removed server {server_name} from monitoring", color="info")
    
    def start_monitoring(self, interval: float = 30.0):
        """Start real-time monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        pretty_print("Started MCP server monitoring", color="success")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        pretty_print("Stopped MCP server monitoring", color="info")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                for server_name in list(self.servers.keys()):
                    self._check_server_health(server_name)
                
                # Trigger callback if provided
                if self.callback:
                    self.callback(self.get_monitoring_summary())
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def _check_server_health(self, server_name: str):
        """Check health of specific server"""
        try:
            server_data = self.servers.get(server_name)
            if not server_data:
                return
            
            process = server_data['process']
            metrics = self.metrics[server_name]
            
            # Check if process is still running
            if hasattr(process, 'poll') and process.poll() is not None:
                self._create_alert(
                    server_name, "process_died", "critical",
                    f"Server process has died"
                )
                return
            
            # Get process metrics if available
            if hasattr(process, 'pid'):
                try:
                    ps_process = psutil.Process(process.pid)
                    
                    # CPU usage
                    metrics.cpu_usage = ps_process.cpu_percent()
                    if metrics.cpu_usage > self.cpu_threshold:
                        self._create_alert(
                            server_name, "high_cpu", "warning",
                            f"High CPU usage: {metrics.cpu_usage:.1f}%"
                        )
                    
                    # Memory usage
                    memory_info = ps_process.memory_info()
                    metrics.memory_usage = memory_info.rss / 1024 / 1024  # MB
                    memory_percent = ps_process.memory_percent()
                    if memory_percent > self.memory_threshold:
                        self._create_alert(
                            server_name, "high_memory", "warning",
                            f"High memory usage: {memory_percent:.1f}%"
                        )
                    
                    # Network I/O
                    try:
                        net_io = ps_process.io_counters()
                        metrics.network_io = {
                            "bytes_sent": net_io.write_bytes,
                            "bytes_recv": net_io.read_bytes
                        }
                    except (psutil.AccessDenied, AttributeError):
                        pass
                    
                    # Uptime
                    metrics.uptime = time.time() - server_data['start_time']
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    self._create_alert(
                        server_name, "process_access", "warning",
                        f"Cannot access process metrics"
                    )
            
            server_data['last_check'] = time.time()
            
        except Exception as e:
            self.logger.error(f"Error checking health for {server_name}: {e}")
    
    def _create_alert(self, server_name: str, alert_type: str, severity: str, message: str):
        """Create new alert"""
        # Check if similar alert already exists
        for alert in self.alerts:
            if (alert.server_name == server_name and 
                alert.alert_type == alert_type and 
                not alert.resolved):
                return  # Don't create duplicate alerts
        
        alert = ServerAlert(
            server_name=server_name,
            alert_type=alert_type,
            severity=severity,
            message=message,
            timestamp=time.time()
        )
        
        self.alerts.append(alert)
        pretty_print(f"Alert for {server_name}: {message}", color="warning")
    
    def get_server_health(self, server_name: str) -> ServerHealthStatus:
        """Get health status of server"""
        if server_name not in self.servers:
            return ServerHealthStatus.UNKNOWN
        
        # Check for critical alerts
        for alert in self.alerts:
            if (alert.server_name == server_name and 
                alert.severity == "critical" and 
                not alert.resolved):
                return ServerHealthStatus.CRITICAL
        
        # Check for warning alerts
        for alert in self.alerts:
            if (alert.server_name == server_name and 
                alert.severity == "warning" and 
                not alert.resolved):
                return ServerHealthStatus.WARNING
        
        return ServerHealthStatus.HEALTHY
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""
        summary = {
            "total_servers": len(self.servers),
            "healthy_servers": 0,
            "warning_servers": 0,
            "critical_servers": 0,
            "total_alerts": len([a for a in self.alerts if not a.resolved]),
            "server_details": {},
            "recent_alerts": []
        }
        
        # Count server statuses
        for server_name in self.servers:
            health = self.get_server_health(server_name)
            if health == ServerHealthStatus.HEALTHY:
                summary["healthy_servers"] += 1
            elif health == ServerHealthStatus.WARNING:
                summary["warning_servers"] += 1
            elif health == ServerHealthStatus.CRITICAL:
                summary["critical_servers"] += 1
            
            # Add server details
            metrics = self.metrics.get(server_name, ServerMetrics())
            summary["server_details"][server_name] = {
                "health": health.value,
                "metrics": asdict(metrics),
                "uptime": metrics.uptime
            }
        
        # Recent alerts (last 24 hours)
        recent_threshold = time.time() - (24 * 60 * 60)
        summary["recent_alerts"] = [
            asdict(alert) for alert in self.alerts 
            if alert.timestamp > recent_threshold
        ]
        
        return summary

class MCPAutoScaler:
    """Auto-scaling functionality for MCP servers"""
    
    def __init__(self, monitor: MCPServerMonitor, registry: Any):
        self.monitor = monitor
        self.registry = registry
        self.scaling_rules: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
    
    def add_scaling_rule(self, server_name: str, min_instances: int = 1, 
                        max_instances: int = 3, cpu_threshold: float = 70.0,
                        memory_threshold: float = 70.0):
        """Add auto-scaling rule for server"""
        self.scaling_rules[server_name] = {
            "min_instances": min_instances,
            "max_instances": max_instances,
            "cpu_threshold": cpu_threshold,
            "memory_threshold": memory_threshold,
            "current_instances": 1
        }
    
    def check_scaling_needed(self, server_name: str) -> Optional[str]:
        """Check if scaling is needed for server"""
        if server_name not in self.scaling_rules:
            return None
        
        rule = self.scaling_rules[server_name]
        metrics = self.monitor.metrics.get(server_name)
        
        if not metrics:
            return None
        
        # Scale up conditions
        if (metrics.cpu_usage > rule["cpu_threshold"] or 
            metrics.memory_usage > rule["memory_threshold"]):
            if rule["current_instances"] < rule["max_instances"]:
                return "scale_up"
        
        # Scale down conditions
        if (metrics.cpu_usage < rule["cpu_threshold"] * 0.5 and 
            metrics.memory_usage < rule["memory_threshold"] * 0.5):
            if rule["current_instances"] > rule["min_instances"]:
                return "scale_down"
        
        return None
    
    async def scale_server(self, server_name: str, action: str) -> bool:
        """Scale server up or down"""
        try:
            if action == "scale_up":
                # Implementation would depend on how multiple instances are handled
                pretty_print(f"Scaling up {server_name}", color="info")
                return True
            elif action == "scale_down":
                pretty_print(f"Scaling down {server_name}", color="info")
                return True
        except Exception as e:
            self.logger.error(f"Error scaling {server_name}: {e}")
            return False

class MCPLoadBalancer:
    """Load balancer for MCP server instances"""
    
    def __init__(self):
        self.server_pools: Dict[str, List[Any]] = {}
        self.current_index: Dict[str, int] = {}
        self.health_checks: Dict[str, Dict[str, bool]] = {}
    
    def add_server_pool(self, pool_name: str, servers: List[Any]):
        """Add pool of servers for load balancing"""
        self.server_pools[pool_name] = servers
        self.current_index[pool_name] = 0
        self.health_checks[pool_name] = {
            str(i): True for i in range(len(servers))
        }
    
    def get_next_server(self, pool_name: str) -> Optional[Any]:
        """Get next available server using round-robin"""
        if pool_name not in self.server_pools:
            return None
        
        servers = self.server_pools[pool_name]
        if not servers:
            return None
        
        # Find next healthy server
        attempts = 0
        while attempts < len(servers):
            index = self.current_index[pool_name]
            server = servers[index]
            
            # Update index for next call
            self.current_index[pool_name] = (index + 1) % len(servers)
            
            # Check if server is healthy
            if self.health_checks[pool_name].get(str(index), True):
                return server
            
            attempts += 1
        
        return None  # No healthy servers available
    
    def mark_server_unhealthy(self, pool_name: str, server_index: int):
        """Mark server as unhealthy"""
        if pool_name in self.health_checks:
            self.health_checks[pool_name][str(server_index)] = False
    
    def mark_server_healthy(self, pool_name: str, server_index: int):
        """Mark server as healthy"""
        if pool_name in self.health_checks:
            self.health_checks[pool_name][str(server_index)] = True

def find_free_port() -> int:
    """Find a free port for server communication"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

class MCPServerCluster:
    """Manages clusters of MCP servers for high availability"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitor = MCPServerMonitor(callback=self._monitoring_callback)
        self.load_balancer = MCPLoadBalancer()
        self.auto_scaler = None  # Will be set when registry is available
        self.clusters: Dict[str, Dict[str, Any]] = {}
        
    def create_cluster(self, cluster_name: str, server_type: str, 
                      initial_size: int = 2) -> bool:
        """Create a new MCP server cluster"""
        try:
            cluster_info = {
                "server_type": server_type,
                "instances": [],
                "created_at": time.time(),
                "target_size": initial_size
            }
            
            # Create initial instances
            for i in range(initial_size):
                instance_name = f"{cluster_name}-{i}"
                port = find_free_port()
                
                # This would be implemented based on specific server requirements
                instance_info = {
                    "name": instance_name,
                    "port": port,
                    "status": "pending",
                    "created_at": time.time()
                }
                
                cluster_info["instances"].append(instance_info)
            
            self.clusters[cluster_name] = cluster_info
            pretty_print(f"Created cluster {cluster_name} with {initial_size} instances", color="success")
            return True
            
        except Exception as e:
            pretty_print(f"Error creating cluster {cluster_name}: {e}", color="error")
            return False
    
    def _monitoring_callback(self, summary: Dict[str, Any]):
        """Handle monitoring updates"""
        # Implement auto-scaling logic based on monitoring data
        if self.auto_scaler:
            for server_name in summary["server_details"]:
                action = self.auto_scaler.check_scaling_needed(server_name)
                if action:
                    asyncio.create_task(self.auto_scaler.scale_server(server_name, action))
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get status of all clusters"""
        status = {
            "total_clusters": len(self.clusters),
            "clusters": {}
        }
        
        for cluster_name, cluster_info in self.clusters.items():
            healthy_instances = sum(
                1 for instance in cluster_info["instances"] 
                if instance.get("status") == "running"
            )
            
            status["clusters"][cluster_name] = {
                "server_type": cluster_info["server_type"],
                "total_instances": len(cluster_info["instances"]),
                "healthy_instances": healthy_instances,
                "target_size": cluster_info["target_size"],
                "uptime": time.time() - cluster_info["created_at"]
            }
        
        return status

# Example usage
if __name__ == "__main__":
    async def test_cluster_management():
        """Test cluster management functionality"""
        config = {
            "enable_monitoring": True,
            "enable_auto_scaling": True,
            "monitoring_interval": 30
        }
        
        cluster = MCPServerCluster(config)
        
        # Create a test cluster
        success = cluster.create_cluster("weather-cluster", "weather-mcp", 2)
        print(f"Cluster creation: {'Success' if success else 'Failed'}")
        
        # Get cluster status
        status = cluster.get_cluster_status()
        print(f"Cluster status: {json.dumps(status, indent=2)}")
    
    # Run test
    asyncio.run(test_cluster_management())
