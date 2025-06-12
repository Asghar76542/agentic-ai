
"""
Performance Dashboard

Real-time monitoring and analytics dashboard for the multi-layer caching system.
Provides comprehensive metrics, visualizations, and performance insights across
all cache layers and system components.

Key Features:
- Real-time cache performance metrics
- System resource monitoring
- Performance trend analysis
- Cache efficiency recommendations
- Interactive dashboard interface
- Exportable performance reports
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. System monitoring will be limited.")


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    timestamp: datetime
    metric_name: str
    value: float
    category: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class CachePerformanceSnapshot:
    """Snapshot of cache performance at a specific time."""
    timestamp: datetime
    cache_name: str
    hit_rate: float
    miss_rate: float
    size: int
    memory_usage: int
    average_response_time: float
    total_operations: int
    evictions: int
    errors: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class SystemResourceSnapshot:
    """Snapshot of system resource usage."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available: int
    disk_usage_percent: float
    network_io: Dict[str, int]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PerformanceMonitor:
    """
    Real-time performance monitoring system.
    
    Collects and analyzes performance metrics from all cache layers
    and system components to provide comprehensive insights.
    """
    
    def __init__(self, 
                 collection_interval: int = 30,  # seconds
                 max_history_size: int = 1000,
                 enable_system_monitoring: bool = True):
        """
        Initialize performance monitor.
        
        Args:
            collection_interval: How often to collect metrics (seconds)
            max_history_size: Maximum number of historical data points to keep
            enable_system_monitoring: Whether to monitor system resources
        """
        self.collection_interval = collection_interval
        self.max_history_size = max_history_size
        self.enable_system_monitoring = enable_system_monitoring and PSUTIL_AVAILABLE
        
        # Data storage
        self.metrics_history: deque = deque(maxlen=max_history_size)
        self.cache_snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history_size))
        self.system_snapshots: deque = deque(maxlen=max_history_size)
        
        # Registered cache managers
        self.cache_managers: Dict[str, Any] = {}
        self.unified_cache_manager = None
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Performance thresholds
        self.thresholds = {
            "low_hit_rate": 0.7,  # 70%
            "high_memory_usage": 0.8,  # 80%
            "high_cpu_usage": 0.8,  # 80%
            "high_response_time": 0.1  # 100ms
        }
        
        # Alert history
        self.alerts: deque = deque(maxlen=100)
        
        print("Performance Monitor initialized")
    
    def register_cache_manager(self, name: str, cache_manager):
        """
        Register a cache manager for monitoring.
        
        Args:
            name: Name identifier for the cache manager
            cache_manager: Cache manager instance to monitor
        """
        with self.lock:
            self.cache_managers[name] = cache_manager
            print(f"Registered cache manager: {name}")
    
    def register_unified_cache_manager(self, unified_manager):
        """
        Register the unified cache manager.
        
        Args:
            unified_manager: UnifiedCacheManager instance
        """
        with self.lock:
            self.unified_cache_manager = unified_manager
            print("Registered unified cache manager")
    
    def start_monitoring(self):
        """Start real-time performance monitoring."""
        if self.monitoring_active:
            print("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        print("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that runs in background thread."""
        while self.monitoring_active:
            try:
                self._collect_metrics()
                time.sleep(self.collection_interval)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(5)  # Wait before retrying
    
    def _collect_metrics(self):
        """Collect all performance metrics."""
        timestamp = datetime.now()
        
        # Collect cache metrics
        self._collect_cache_metrics(timestamp)
        
        # Collect system metrics
        if self.enable_system_monitoring:
            self._collect_system_metrics(timestamp)
        
        # Check for performance issues
        self._check_alerts(timestamp)
    
    def _collect_cache_metrics(self, timestamp: datetime):
        """Collect metrics from all registered cache managers."""
        with self.lock:
            # Collect from individual cache managers
            for name, manager in self.cache_managers.items():
                try:
                    if hasattr(manager, 'get_cache_stats'):
                        stats = manager.get_cache_stats()
                        snapshot = self._create_cache_snapshot(timestamp, name, stats)
                        self.cache_snapshots[name].append(snapshot)
                except Exception as e:
                    print(f"Error collecting metrics from {name}: {e}")
            
            # Collect from unified cache manager
            if self.unified_cache_manager:
                try:
                    unified_stats = self.unified_cache_manager.get_unified_stats()
                    for cache_name, stats in unified_stats.items():
                        snapshot = self._create_cache_snapshot(timestamp, f"unified_{cache_name}", stats)
                        self.cache_snapshots[f"unified_{cache_name}"].append(snapshot)
                except Exception as e:
                    print(f"Error collecting unified cache metrics: {e}")
    
    def _create_cache_snapshot(self, timestamp: datetime, cache_name: str, stats: Dict) -> CachePerformanceSnapshot:
        """Create cache performance snapshot from stats."""
        return CachePerformanceSnapshot(
            timestamp=timestamp,
            cache_name=cache_name,
            hit_rate=stats.get('hit_rate', 0.0),
            miss_rate=100.0 - stats.get('hit_rate', 0.0),
            size=stats.get('size', 0),
            memory_usage=stats.get('memory_usage', 0),
            average_response_time=stats.get('avg_response_time', 0.0),
            total_operations=stats.get('hits', 0) + stats.get('misses', 0),
            evictions=stats.get('evictions', 0),
            errors=stats.get('errors', 0)
        )
    
    def _collect_system_metrics(self, timestamp: datetime):
        """Collect system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            snapshot = SystemResourceSnapshot(
                timestamp=timestamp,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available=memory_available,
                disk_usage_percent=disk_usage_percent,
                network_io=network_io
            )
            
            self.system_snapshots.append(snapshot)
            
        except Exception as e:
            print(f"Error collecting system metrics: {e}")
    
    def _check_alerts(self, timestamp: datetime):
        """Check for performance issues and generate alerts."""
        alerts = []
        
        # Check cache performance
        for cache_name, snapshots in self.cache_snapshots.items():
            if not snapshots:
                continue
                
            latest = snapshots[-1]
            
            # Low hit rate alert
            if latest.hit_rate < self.thresholds["low_hit_rate"] * 100:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'LOW_HIT_RATE',
                    'cache': cache_name,
                    'value': latest.hit_rate,
                    'threshold': self.thresholds["low_hit_rate"] * 100,
                    'message': f"Low hit rate in {cache_name}: {latest.hit_rate:.1f}%"
                })
            
            # High response time alert
            if latest.average_response_time > self.thresholds["high_response_time"]:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'HIGH_RESPONSE_TIME',
                    'cache': cache_name,
                    'value': latest.average_response_time,
                    'threshold': self.thresholds["high_response_time"],
                    'message': f"High response time in {cache_name}: {latest.average_response_time:.3f}s"
                })
        
        # Check system resources
        if self.system_snapshots:
            latest_system = self.system_snapshots[-1]
            
            # High CPU usage alert
            if latest_system.cpu_percent > self.thresholds["high_cpu_usage"] * 100:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'HIGH_CPU_USAGE',
                    'value': latest_system.cpu_percent,
                    'threshold': self.thresholds["high_cpu_usage"] * 100,
                    'message': f"High CPU usage: {latest_system.cpu_percent:.1f}%"
                })
            
            # High memory usage alert
            if latest_system.memory_percent > self.thresholds["high_memory_usage"] * 100:
                alerts.append({
                    'timestamp': timestamp,
                    'type': 'HIGH_MEMORY_USAGE',
                    'value': latest_system.memory_percent,
                    'threshold': self.thresholds["high_memory_usage"] * 100,
                    'message': f"High memory usage: {latest_system.memory_percent:.1f}%"
                })
        
        # Store alerts
        for alert in alerts:
            self.alerts.append(alert)
            print(f"ALERT: {alert['message']}")
    
    def get_dashboard_data(self, time_range_minutes: int = 60) -> Dict:
        """
        Get dashboard data for the specified time range.
        
        Args:
            time_range_minutes: Time range in minutes for data retrieval
            
        Returns:
            Dictionary containing dashboard data
        """
        cutoff_time = datetime.now() - timedelta(minutes=time_range_minutes)
        
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'time_range_minutes': time_range_minutes,
            'cache_performance': {},
            'system_resources': [],
            'alerts': [],
            'summary': {}
        }
        
        # Collect cache performance data
        with self.lock:
            for cache_name, snapshots in self.cache_snapshots.items():
                recent_snapshots = [
                    s for s in snapshots 
                    if s.timestamp >= cutoff_time
                ]
                
                if recent_snapshots:
                    dashboard_data['cache_performance'][cache_name] = [
                        s.to_dict() for s in recent_snapshots
                    ]
        
        # Collect system resource data
        recent_system = [
            s for s in self.system_snapshots 
            if s.timestamp >= cutoff_time
        ]
        dashboard_data['system_resources'] = [s.to_dict() for s in recent_system]
        
        # Collect recent alerts
        recent_alerts = [
            alert for alert in self.alerts 
            if alert['timestamp'] >= cutoff_time
        ]
        dashboard_data['alerts'] = recent_alerts
        
        # Generate summary statistics
        dashboard_data['summary'] = self._generate_summary_stats(time_range_minutes)
        
        return dashboard_data
    
    def _generate_summary_stats(self, time_range_minutes: int) -> Dict:
        """Generate summary statistics for the dashboard."""
        cutoff_time = datetime.now() - timedelta(minutes=time_range_minutes)
        
        summary = {
            'cache_summary': {},
            'system_summary': {},
            'performance_trends': {},
            'recommendations': []
        }
        
        # Cache summary
        with self.lock:
            total_hits = 0
            total_operations = 0
            cache_count = 0
            
            for cache_name, snapshots in self.cache_snapshots.items():
                recent_snapshots = [
                    s for s in snapshots 
                    if s.timestamp >= cutoff_time
                ]
                
                if recent_snapshots:
                    latest = recent_snapshots[-1]
                    cache_count += 1
                    total_operations += latest.total_operations
                    total_hits += (latest.hit_rate / 100.0) * latest.total_operations
                    
                    # Individual cache stats
                    summary['cache_summary'][cache_name] = {
                        'hit_rate': latest.hit_rate,
                        'size': latest.size,
                        'memory_usage': latest.memory_usage,
                        'total_operations': latest.total_operations
                    }
            
            # Overall cache performance
            overall_hit_rate = (total_hits / total_operations * 100) if total_operations > 0 else 0
            summary['cache_summary']['overall'] = {
                'hit_rate': overall_hit_rate,
                'total_caches': cache_count,
                'total_operations': total_operations
            }
        
        # System summary
        if self.system_snapshots:
            recent_system = [
                s for s in self.system_snapshots 
                if s.timestamp >= cutoff_time
            ]
            
            if recent_system:
                latest_system = recent_system[-1]
                avg_cpu = statistics.mean([s.cpu_percent for s in recent_system])
                avg_memory = statistics.mean([s.memory_percent for s in recent_system])
                
                summary['system_summary'] = {
                    'current_cpu': latest_system.cpu_percent,
                    'current_memory': latest_system.memory_percent,
                    'avg_cpu': avg_cpu,
                    'avg_memory': avg_memory,
                    'memory_available': latest_system.memory_available
                }
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations()
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze cache performance
        with self.lock:
            for cache_name, snapshots in self.cache_snapshots.items():
                if not snapshots:
                    continue
                    
                latest = snapshots[-1]
                
                # Low hit rate recommendation
                if latest.hit_rate < 70:
                    recommendations.append(
                        f"Consider increasing TTL for {cache_name} cache to improve hit rate ({latest.hit_rate:.1f}%)"
                    )
                
                # High memory usage recommendation
                if latest.memory_usage > 100 * 1024 * 1024:  # 100MB
                    recommendations.append(
                        f"Consider enabling compression for {cache_name} cache to reduce memory usage"
                    )
                
                # Small cache size recommendation
                if latest.size < 10 and latest.hit_rate < 80:
                    recommendations.append(
                        f"Consider increasing cache size for {cache_name} to improve performance"
                    )
        
        # System resource recommendations
        if self.system_snapshots:
            latest_system = self.system_snapshots[-1]
            
            if latest_system.memory_percent > 85:
                recommendations.append(
                    "High memory usage detected. Consider reducing cache sizes or enabling more aggressive eviction"
                )
            
            if latest_system.cpu_percent > 80:
                recommendations.append(
                    "High CPU usage detected. Consider optimizing cache operations or reducing monitoring frequency"
                )
        
        return recommendations
    
    def export_performance_report(self, filename: Optional[str] = None, 
                                time_range_hours: int = 24) -> str:
        """
        Export comprehensive performance report.
        
        Args:
            filename: Output filename (auto-generated if None)
            time_range_hours: Time range for report data
            
        Returns:
            Path to generated report file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"
        
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        
        report_data = {
            'report_info': {
                'generated_at': datetime.now().isoformat(),
                'time_range_hours': time_range_hours,
                'monitoring_interval': self.collection_interval
            },
            'cache_performance': {},
            'system_resources': [],
            'alerts': [],
            'summary': self._generate_summary_stats(time_range_hours * 60)
        }
        
        # Export cache data
        with self.lock:
            for cache_name, snapshots in self.cache_snapshots.items():
                relevant_snapshots = [
                    s for s in snapshots 
                    if s.timestamp >= cutoff_time
                ]
                
                if relevant_snapshots:
                    report_data['cache_performance'][cache_name] = [
                        s.to_dict() for s in relevant_snapshots
                    ]
        
        # Export system data
        relevant_system = [
            s for s in self.system_snapshots 
            if s.timestamp >= cutoff_time
        ]
        report_data['system_resources'] = [s.to_dict() for s in relevant_system]
        
        # Export alerts
        relevant_alerts = [
            alert for alert in self.alerts 
            if alert['timestamp'] >= cutoff_time
        ]
        report_data['alerts'] = relevant_alerts
        
        # Write report file
        try:
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            print(f"Performance report exported to: {filename}")
            return filename
        except Exception as e:
            print(f"Error exporting report: {e}")
            raise
    
    def get_real_time_metrics(self) -> Dict:
        """Get current real-time metrics."""
        with self.lock:
            current_time = datetime.now()
            
            metrics = {
                'timestamp': current_time.isoformat(),
                'cache_metrics': {},
                'system_metrics': {},
                'alert_count': len([a for a in self.alerts if current_time - a['timestamp'] < timedelta(minutes=5)])
            }
            
            # Latest cache metrics
            for cache_name, snapshots in self.cache_snapshots.items():
                if snapshots:
                    latest = snapshots[-1]
                    metrics['cache_metrics'][cache_name] = {
                        'hit_rate': latest.hit_rate,
                        'size': latest.size,
                        'memory_usage': latest.memory_usage,
                        'response_time': latest.average_response_time
                    }
            
            # Latest system metrics
            if self.system_snapshots:
                latest_system = self.system_snapshots[-1]
                metrics['system_metrics'] = {
                    'cpu_percent': latest_system.cpu_percent,
                    'memory_percent': latest_system.memory_percent,
                    'memory_available': latest_system.memory_available,
                    'disk_usage': latest_system.disk_usage_percent
                }
            
            return metrics


class PerformanceDashboard:
    """
    Web-based performance dashboard interface.
    
    Provides an interactive dashboard for monitoring cache performance
    and system resources in real-time.
    """
    
    def __init__(self, monitor: PerformanceMonitor, port: int = 8080):
        """
        Initialize performance dashboard.
        
        Args:
            monitor: PerformanceMonitor instance
            port: Port for web dashboard
        """
        self.monitor = monitor
        self.port = port
        self.dashboard_active = False
        
    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard page."""
        dashboard_data = self.monitor.get_dashboard_data(60)  # Last 60 minutes
        
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AgenticSeek Performance Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-title { font-weight: bold; color: #2c3e50; margin-bottom: 10px; }
                .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
                .alert { background: #e74c3c; color: white; padding: 10px; border-radius: 4px; margin: 5px 0; }
                .recommendation { background: #f39c12; color: white; padding: 10px; border-radius: 4px; margin: 5px 0; }
                .status-good { color: #27ae60; }
                .status-warning { color: #f39c12; }
                .status-error { color: #e74c3c; }
                .timestamp { color: #7f8c8d; font-size: 12px; }
            </style>
            <script>
                function refreshDashboard() {
                    location.reload();
                }
                setInterval(refreshDashboard, 30000); // Refresh every 30 seconds
            </script>
        </head>
        <body>
            <div class="header">
                <h1>🚀 AgenticSeek Performance Dashboard</h1>
                <p class="timestamp">Last updated: {timestamp}</p>
            </div>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-title">Overall Cache Performance</div>
                    <div class="metric-value">{overall_hit_rate:.1f}%</div>
                    <p>Hit Rate across all caches</p>
                    <p>Total Operations: {total_operations:,}</p>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">System Resources</div>
                    <p>CPU Usage: <span class="{cpu_status}">{cpu_usage:.1f}%</span></p>
                    <p>Memory Usage: <span class="{memory_status}">{memory_usage:.1f}%</span></p>
                    <p>Available Memory: {memory_available_gb:.1f} GB</p>
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Active Caches</div>
                    <div class="metric-value">{cache_count}</div>
                    <p>Monitoring {cache_count} cache layers</p>
                    {cache_list}
                </div>
                
                <div class="metric-card">
                    <div class="metric-title">Recent Alerts</div>
                    <div class="metric-value">{alert_count}</div>
                    <p>Alerts in last 60 minutes</p>
                    {alerts_html}
                </div>
            </div>
            
            <div class="metric-card" style="margin-top: 20px;">
                <div class="metric-title">Performance Recommendations</div>
                {recommendations_html}
            </div>
            
            <div class="metric-card" style="margin-top: 20px;">
                <div class="metric-title">Cache Details</div>
                {cache_details_html}
            </div>
        </body>
        </html>
        """
        
        # Extract data for template
        summary = dashboard_data.get('summary', {})
        cache_summary = summary.get('cache_summary', {})
        system_summary = summary.get('system_resources', {})
        
        overall_hit_rate = cache_summary.get('overall', {}).get('hit_rate', 0)
        total_operations = cache_summary.get('overall', {}).get('total_operations', 0)
        cache_count = cache_summary.get('overall', {}).get('total_caches', 0)
        
        # System status
        cpu_usage = system_summary.get('current_cpu', 0) if system_summary else 0
        memory_usage = system_summary.get('current_memory', 0) if system_summary else 0
        memory_available_gb = system_summary.get('memory_available', 0) / (1024**3) if system_summary else 0
        
        cpu_status = 'status-good' if cpu_usage < 70 else 'status-warning' if cpu_usage < 90 else 'status-error'
        memory_status = 'status-good' if memory_usage < 70 else 'status-warning' if memory_usage < 90 else 'status-error'
        
        # Cache list
        cache_list = ""
        for cache_name in dashboard_data.get('cache_performance', {}).keys():
            cache_list += f"<p>• {cache_name}</p>"
        
        # Alerts
        alerts_html = ""
        recent_alerts = dashboard_data.get('alerts', [])[-5:]  # Last 5 alerts
        for alert in recent_alerts:
            alerts_html += f'<div class="alert">{alert["message"]}</div>'
        
        # Recommendations
        recommendations_html = ""
        recommendations = summary.get('recommendations', [])
        for rec in recommendations:
            recommendations_html += f'<div class="recommendation">💡 {rec}</div>'
        
        # Cache details
        cache_details_html = ""
        for cache_name, cache_data in cache_summary.items():
            if cache_name != 'overall' and isinstance(cache_data, dict):
                cache_details_html += f"""
                <div style="margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 4px;">
                    <strong>{cache_name}</strong><br>
                    Hit Rate: {cache_data.get('hit_rate', 0):.1f}% | 
                    Size: {cache_data.get('size', 0)} items | 
                    Memory: {cache_data.get('memory_usage', 0) / 1024:.1f} KB
                </div>
                """
        
        return html_template.format(
            timestamp=dashboard_data['timestamp'],
            overall_hit_rate=overall_hit_rate,
            total_operations=total_operations,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_available_gb=memory_available_gb,
            cpu_status=cpu_status,
            memory_status=memory_status,
            cache_count=cache_count,
            cache_list=cache_list,
            alert_count=len(recent_alerts),
            alerts_html=alerts_html,
            recommendations_html=recommendations_html,
            cache_details_html=cache_details_html
        )
    
    def save_dashboard_html(self, filename: str = "performance_dashboard.html"):
        """Save dashboard as HTML file."""
        html_content = self.generate_html_dashboard()
        
        try:
            with open(filename, 'w') as f:
                f.write(html_content)
            print(f"Dashboard saved to: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving dashboard: {e}")
            raise


# Convenience functions for easy setup
def create_performance_monitor(collection_interval: int = 30) -> PerformanceMonitor:
    """Create and configure a performance monitor."""
    return PerformanceMonitor(collection_interval=collection_interval)


def create_dashboard(monitor: PerformanceMonitor) -> PerformanceDashboard:
    """Create a performance dashboard."""
    return PerformanceDashboard(monitor)


if __name__ == "__main__":
    # Example usage
    print("Performance Dashboard - Multi-layer Cache Monitoring")
    print("This module provides comprehensive performance monitoring and analytics")
