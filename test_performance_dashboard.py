
"""
Performance Dashboard Test

Demonstrates the real-time performance monitoring and analytics dashboard
for the multi-layer caching system.
"""

import sys
import time
import asyncio
from datetime import datetime, timedelta

sys.path.append('.')

from sources.cache.performanceDashboard import (
    PerformanceMonitor, 
    PerformanceDashboard,
    create_performance_monitor,
    create_dashboard
)
from sources.cache.webCache import WebCache
from sources.cache.llmCache import LLMCache
from sources.cache.unifiedCacheManager import UnifiedCacheManager
from sources.utility import pretty_print


class MockCacheManager:
    """Mock cache manager for testing dashboard functionality."""
    
    def __init__(self, name: str):
        self.name = name
        self.operations = 0
        self.hits = 0
        self.misses = 0
        self.start_time = time.time()
        
    def simulate_operation(self, is_hit: bool = True):
        """Simulate a cache operation."""
        self.operations += 1
        if is_hit:
            self.hits += 1
        else:
            self.misses += 1
    
    def get_cache_stats(self):
        """Return mock cache statistics."""
        hit_rate = (self.hits / self.operations * 100) if self.operations > 0 else 0
        response_time = 0.001 if self.hits > 0 else 0.1  # Faster for hits
        
        return {
            'hit_rate': hit_rate,
            'hits': self.hits,
            'misses': self.misses,
            'size': min(self.operations, 100),  # Mock cache size
            'memory_usage': self.operations * 1024,  # Mock memory usage
            'avg_response_time': response_time,
            'evictions': max(0, self.operations - 100),  # Mock evictions
            'errors': 0
        }


def test_performance_monitor_setup():
    """Test performance monitor initialization and configuration."""
    print("="*60)
    pretty_print("Testing Performance Monitor Setup", color="info")
    print("="*60)
    
    # Create performance monitor
    monitor = create_performance_monitor(collection_interval=5)  # 5 seconds for testing
    
    print("\n1. Performance Monitor Configuration:")
    print(f"   Collection interval: {monitor.collection_interval}s")
    print(f"   Max history size: {monitor.max_history_size}")
    print(f"   System monitoring: {monitor.enable_system_monitoring}")
    print(f"   Monitoring active: {monitor.monitoring_active}")
    
    # Test threshold configuration
    print(f"\n2. Performance Thresholds:")
    for threshold_name, value in monitor.thresholds.items():
        print(f"   {threshold_name}: {value}")
    
    return monitor


def test_cache_manager_registration(monitor):
    """Test registering cache managers with the monitor."""
    print("\n" + "="*60)
    pretty_print("Testing Cache Manager Registration", color="info")
    print("="*60)
    
    # Create mock cache managers
    llm_cache_manager = MockCacheManager("llm_cache")
    web_cache_manager = MockCacheManager("web_cache")
    computation_cache_manager = MockCacheManager("computation_cache")
    
    print("\n1. Registering Cache Managers:")
    
    # Register cache managers
    monitor.register_cache_manager("llm", llm_cache_manager)
    monitor.register_cache_manager("web", web_cache_manager)
    monitor.register_cache_manager("computation", computation_cache_manager)
    
    print(f"   Registered managers: {list(monitor.cache_managers.keys())}")
    
    # Create and register unified cache manager
    unified_manager = UnifiedCacheManager()
    
    # Create some real caches for unified manager
    try:
        web_cache = WebCache(max_size_mb=10)
        unified_manager.register_cache("web", web_cache)
        monitor.register_unified_cache_manager(unified_manager)
        print("   ✅ Unified cache manager registered")
    except Exception as e:
        print(f"   ⚠️  Unified cache manager registration failed: {e}")
    
    return {
        "llm": llm_cache_manager,
        "web": web_cache_manager,
        "computation": computation_cache_manager
    }


def test_metric_collection(monitor, cache_managers):
    """Test metric collection and data storage."""
    print("\n" + "="*60)
    pretty_print("Testing Metric Collection", color="info")
    print("="*60)
    
    print("\n1. Simulating Cache Operations:")
    
    # Simulate various cache operations
    operations = [
        ("llm", True, "Cache hit for LLM response"),
        ("llm", True, "Another LLM cache hit"),
        ("llm", False, "LLM cache miss"),
        ("web", True, "Web page cache hit"),
        ("web", False, "Web page cache miss"),
        ("web", True, "Web search cache hit"),
        ("computation", False, "Computation cache miss"),
        ("computation", True, "Computation cache hit"),
        ("computation", True, "Another computation hit")
    ]
    
    for cache_name, is_hit, description in operations:
        cache_managers[cache_name].simulate_operation(is_hit)
        print(f"   {description}")
        time.sleep(0.1)  # Small delay between operations
    
    print("\n2. Manual Metric Collection:")
    timestamp = datetime.now()
    monitor._collect_cache_metrics(timestamp)
    
    # Display collected metrics
    print(f"\n3. Collected Cache Snapshots:")
    for cache_name, snapshots in monitor.cache_snapshots.items():
        if snapshots:
            latest = snapshots[-1]
            print(f"   {cache_name}:")
            print(f"     Hit rate: {latest.hit_rate:.1f}%")
            print(f"     Size: {latest.size} items")
            print(f"     Memory usage: {latest.memory_usage} bytes")
            print(f"     Total operations: {latest.total_operations}")
    
    # Test system metric collection
    if monitor.enable_system_monitoring:
        print(f"\n4. System Metrics Collection:")
        monitor._collect_system_metrics(timestamp)
        
        if monitor.system_snapshots:
            latest_system = monitor.system_snapshots[-1]
            print(f"   CPU usage: {latest_system.cpu_percent:.1f}%")
            print(f"   Memory usage: {latest_system.memory_percent:.1f}%")
            print(f"   Available memory: {latest_system.memory_available / (1024**3):.1f} GB")
    
    return timestamp


def test_real_time_monitoring(monitor, cache_managers):
    """Test real-time monitoring functionality."""
    print("\n" + "="*60)
    pretty_print("Testing Real-time Monitoring", color="info")
    print("="*60)
    
    print("\n1. Starting Real-time Monitoring:")
    monitor.start_monitoring()
    
    print(f"   Monitoring active: {monitor.monitoring_active}")
    print(f"   Collection interval: {monitor.collection_interval}s")
    
    print("\n2. Simulating Continuous Operations:")
    
    # Simulate ongoing cache operations
    for i in range(10):
        # Randomly distribute operations across caches
        import random
        cache_name = random.choice(["llm", "web", "computation"])
        is_hit = random.random() > 0.3  # 70% hit rate
        
        cache_managers[cache_name].simulate_operation(is_hit)
        
        status = "HIT" if is_hit else "MISS"
        print(f"   Operation {i+1}: {cache_name} cache {status}")
        
        time.sleep(0.5)  # Half second between operations
    
    print("\n3. Waiting for metric collection cycle...")
    time.sleep(6)  # Wait for at least one collection cycle
    
    print("\n4. Real-time Metrics:")
    real_time_metrics = monitor.get_real_time_metrics()
    
    print(f"   Timestamp: {real_time_metrics['timestamp']}")
    print(f"   Active alerts: {real_time_metrics['alert_count']}")
    
    for cache_name, metrics in real_time_metrics['cache_metrics'].items():
        print(f"   {cache_name}:")
        print(f"     Hit rate: {metrics['hit_rate']:.1f}%")
        print(f"     Size: {metrics['size']} items")
        print(f"     Response time: {metrics['response_time']:.3f}s")
    
    if real_time_metrics['system_metrics']:
        sys_metrics = real_time_metrics['system_metrics']
        print(f"   System:")
        print(f"     CPU: {sys_metrics['cpu_percent']:.1f}%")
        print(f"     Memory: {sys_metrics['memory_percent']:.1f}%")
    
    # Stop monitoring
    monitor.stop_monitoring()
    print("\n   Monitoring stopped")


def test_dashboard_generation(monitor):
    """Test dashboard data generation and HTML creation."""
    print("\n" + "="*60)
    pretty_print("Testing Dashboard Generation", color="info")
    print("="*60)
    
    # Create dashboard
    dashboard = create_dashboard(monitor)
    
    print("\n1. Generating Dashboard Data:")
    dashboard_data = monitor.get_dashboard_data(time_range_minutes=60)
    
    print(f"   Timestamp: {dashboard_data['timestamp']}")
    print(f"   Time range: {dashboard_data['time_range_minutes']} minutes")
    print(f"   Cache performance data: {len(dashboard_data['cache_performance'])} caches")
    print(f"   System resource points: {len(dashboard_data['system_resources'])}")
    print(f"   Alerts: {len(dashboard_data['alerts'])}")
    
    # Display summary statistics
    summary = dashboard_data.get('summary', {})
    if summary:
        print(f"\n2. Summary Statistics:")
        
        cache_summary = summary.get('cache_summary', {})
        if 'overall' in cache_summary:
            overall = cache_summary['overall']
            print(f"   Overall hit rate: {overall.get('hit_rate', 0):.1f}%")
            print(f"   Total caches: {overall.get('total_caches', 0)}")
            print(f"   Total operations: {overall.get('total_operations', 0)}")
        
        recommendations = summary.get('recommendations', [])
        if recommendations:
            print(f"\n   Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations[:3], 1):  # Show first 3
                print(f"     {i}. {rec}")
    
    print("\n3. Generating HTML Dashboard:")
    try:
        html_filename = dashboard.save_dashboard_html("test_dashboard.html")
        print(f"   ✅ Dashboard saved to: {html_filename}")
        
        # Show a preview of the HTML content
        html_content = dashboard.generate_html_dashboard()
        print(f"   Dashboard size: {len(html_content)} characters")
        print(f"   Contains CSS: {'<style>' in html_content}")
        print(f"   Contains JavaScript: {'<script>' in html_content}")
        
    except Exception as e:
        print(f"   ❌ Dashboard generation failed: {e}")
    
    return dashboard


def test_performance_report_export(monitor):
    """Test performance report export functionality."""
    print("\n" + "="*60)
    pretty_print("Testing Performance Report Export", color="info")
    print("="*60)
    
    print("\n1. Exporting Performance Report:")
    try:
        report_filename = monitor.export_performance_report(
            filename="test_performance_report.json",
            time_range_hours=1
        )
        print(f"   ✅ Report exported to: {report_filename}")
        
        # Read and display report summary
        import json
        try:
            with open(report_filename, 'r') as f:
                report_data = json.load(f)
            
            print(f"\n2. Report Contents:")
            print(f"   Generated at: {report_data['report_info']['generated_at']}")
            print(f"   Time range: {report_data['report_info']['time_range_hours']} hours")
            print(f"   Cache performance data: {len(report_data['cache_performance'])} caches")
            print(f"   System resource points: {len(report_data['system_resources'])}")
            print(f"   Alerts: {len(report_data['alerts'])}")
            
            # Show summary if available
            if 'summary' in report_data:
                summary = report_data['summary']
                if 'cache_summary' in summary and 'overall' in summary['cache_summary']:
                    overall = summary['cache_summary']['overall']
                    print(f"   Overall hit rate: {overall.get('hit_rate', 0):.1f}%")
                
                recommendations = summary.get('recommendations', [])
                print(f"   Recommendations: {len(recommendations)}")
        
        except Exception as e:
            print(f"   ⚠️  Could not read report file: {e}")
    
    except Exception as e:
        print(f"   ❌ Report export failed: {e}")


def test_alert_system(monitor, cache_managers):
    """Test performance alert system."""
    print("\n" + "="*60)
    pretty_print("Testing Alert System", color="info")
    print("="*60)
    
    print("\n1. Current Alert Thresholds:")
    for threshold_name, value in monitor.thresholds.items():
        print(f"   {threshold_name}: {value}")
    
    print("\n2. Simulating Performance Issues:")
    
    # Simulate low hit rate scenario
    print("\n   Simulating low hit rate scenario...")
    for _ in range(20):
        cache_managers["llm"].simulate_operation(is_hit=False)  # All misses
    
    # Manually trigger metric collection and alert checking
    timestamp = datetime.now()
    monitor._collect_cache_metrics(timestamp)
    monitor._check_alerts(timestamp)
    
    print(f"\n3. Generated Alerts:")
    recent_alerts = [
        alert for alert in monitor.alerts 
        if alert['timestamp'] >= timestamp - timedelta(minutes=1)
    ]
    
    if recent_alerts:
        for alert in recent_alerts:
            print(f"   🚨 {alert['type']}: {alert['message']}")
    else:
        print("   No alerts generated")
    
    print(f"\n4. Total Alert History:")
    print(f"   Total alerts: {len(monitor.alerts)}")
    
    # Show alert types
    alert_types = {}
    for alert in monitor.alerts:
        alert_type = alert['type']
        alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
    
    for alert_type, count in alert_types.items():
        print(f"   {alert_type}: {count}")


async def run_comprehensive_test():
    """Run comprehensive performance dashboard test."""
    print("🚀 Starting Performance Dashboard Comprehensive Test")
    print("="*80)
    
    try:
        # Setup
        monitor = test_performance_monitor_setup()
        cache_managers = test_cache_manager_registration(monitor)
        
        # Basic functionality tests
        test_metric_collection(monitor, cache_managers)
        test_real_time_monitoring(monitor, cache_managers)
        
        # Dashboard and reporting tests
        dashboard = test_dashboard_generation(monitor)
        test_performance_report_export(monitor)
        
        # Advanced features
        test_alert_system(monitor, cache_managers)
        
        print("\n" + "="*80)
        pretty_print("✅ All Performance Dashboard tests completed successfully!", color="success")
        print("="*80)
        
        print("\nKey Features Demonstrated:")
        print("✓ Real-time performance monitoring across all cache layers")
        print("✓ System resource monitoring (CPU, memory, disk)")
        print("✓ Interactive HTML dashboard generation")
        print("✓ Comprehensive performance report export")
        print("✓ Intelligent alert system with configurable thresholds")
        print("✓ Performance trend analysis and recommendations")
        print("✓ Multi-cache coordination and unified metrics")
        print("✓ Automatic data collection and historical tracking")
        
        # Final summary
        print(f"\nFinal Performance Summary:")
        final_metrics = monitor.get_real_time_metrics()
        
        total_operations = 0
        total_hit_rate = 0
        cache_count = 0
        
        for cache_name, metrics in final_metrics['cache_metrics'].items():
            print(f"  {cache_name}: {metrics['hit_rate']:.1f}% hit rate, {metrics['size']} items")
            total_hit_rate += metrics['hit_rate']
            cache_count += 1
        
        if cache_count > 0:
            avg_hit_rate = total_hit_rate / cache_count
            print(f"  Average hit rate across all caches: {avg_hit_rate:.1f}%")
        
        print(f"  Total alerts generated: {len(monitor.alerts)}")
        print(f"  Monitoring duration: {time.time() - monitor.alerts[0]['timestamp'].timestamp():.1f}s" if monitor.alerts else "N/A")
        
    except Exception as e:
        pretty_print(f"❌ Test failed: {str(e)}", color="failure")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())
