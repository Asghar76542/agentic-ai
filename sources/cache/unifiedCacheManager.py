"""
Unified Cache Manager
Coordinates all cache layers and manages memory pressure across the system
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Mock dependencies for testing environment
try:
    from sources.logger import Logger
except ImportError:
    class Logger:
        def __init__(self, name): pass
        def info(self, msg): pass
        def warning(self, msg): pass
        def error(self, msg): pass

try:
    from sources.cache.llmCache import LLMCache
    from sources.cache.webCache import WebCache
    from sources.cache.computationCache import ComputationCache
    from sources.cache.cacheStats import CacheStats
except ImportError:
    # For testing without full dependencies
    class LLMCache:
        def __init__(self, *args, **kwargs): pass
        def get_stats(self): return {}
        def clear(self): pass
        def get_cache_info(self): return {}
    
    class WebCache:
        def __init__(self, *args, **kwargs): pass
        def get_stats(self): return {}
        def clear(self): pass
        def get_cache_info(self): return {}
    
    class ComputationCache:
        def __init__(self, *args, **kwargs): pass
        def get_stats(self): return {}
        def clear(self): pass
        def get_cache_info(self): return {}
    
    class CacheStats:
        def __init__(self, name): pass
        def get_stats(self): return {}


class MemoryPressureLevel(Enum):
    """Memory pressure levels for cache management."""
    LOW = "low"           # < 60% memory usage
    MODERATE = "moderate" # 60-75% memory usage
    HIGH = "high"         # 75-85% memory usage
    CRITICAL = "critical" # > 85% memory usage


@dataclass
class CacheConfig:
    """Configuration for individual cache instances."""
    max_size: int
    default_ttl: int
    priority: int  # 1 (highest) to 5 (lowest) for eviction order
    memory_limit_mb: int
    auto_cleanup: bool = True


@dataclass
class SystemLimits:
    """System-wide cache limits and thresholds."""
    max_total_memory_mb: int = 512    # Maximum total cache memory
    max_memory_percentage: float = 15.0  # Max % of system memory to use
    warning_threshold: float = 0.8    # Warn when 80% of limit reached
    cleanup_threshold: float = 0.9    # Cleanup when 90% of limit reached
    emergency_threshold: float = 0.95 # Emergency cleanup at 95%


class UnifiedCacheManager:
    """
    Manages all cache layers with intelligent memory pressure handling.
    Coordinates LLM, web, and computation caches for optimal performance.
    """
    
    def __init__(self, system_limits: Optional[SystemLimits] = None):
        """
        Initialize the unified cache manager.
        
        Args:
            system_limits: System-wide limits configuration
        """
        self.system_limits = system_limits or SystemLimits()
        self.logger = Logger("unified_cache_manager.log")
        
        # Cache instances
        self.caches: Dict[str, Any] = {}
        self.cache_configs: Dict[str, CacheConfig] = {}
        
        # Memory monitoring
        self.current_memory_usage = 0
        self.memory_check_interval = 30  # seconds
        self.last_memory_check = 0
        
        # Statistics tracking
        self.stats = CacheStats("UnifiedCacheManager")
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background monitoring
        self.monitoring_enabled = True
        self.monitoring_thread = None
        
        # Initialize default cache configurations
        self._initialize_default_configs()
        
        # Create cache instances
        self._create_cache_instances()
        
        # Start background monitoring
        self._start_monitoring()
    
    def _initialize_default_configs(self):
        """Initialize default configurations for each cache type."""
        self.cache_configs = {
            'llm': CacheConfig(
                max_size=1000,
                default_ttl=3600,
                priority=2,  # High priority
                memory_limit_mb=128,
                auto_cleanup=True
            ),
            'web': CacheConfig(
                max_size=2000,
                default_ttl=1800,
                priority=3,  # Medium priority
                memory_limit_mb=256,
                auto_cleanup=True
            ),
            'computation': CacheConfig(
                max_size=500,
                default_ttl=7200,
                priority=1,  # Highest priority (expensive to recompute)
                memory_limit_mb=128,
                auto_cleanup=True
            )
        }
    
    def _create_cache_instances(self):
        """Create cache instances with configured limits."""
        try:
            with self.lock:
                # LLM Cache
                llm_config = self.cache_configs['llm']
                self.caches['llm'] = LLMCache(
                    max_size=llm_config.max_size,
                    default_ttl=llm_config.default_ttl
                )
                
                # Web Cache
                web_config = self.cache_configs['web']
                self.caches['web'] = WebCache(
                    max_size=web_config.max_size,
                    default_ttl=web_config.default_ttl
                )
                
                # Computation Cache
                comp_config = self.cache_configs['computation']
                self.caches['computation'] = ComputationCache(
                    max_size=comp_config.max_size,
                    default_ttl=comp_config.default_ttl
                )
                
                self.logger.info("All cache instances created successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to create cache instances: {str(e)}")
            # Create mock caches for testing
            for cache_type in ['llm', 'web', 'computation']:
                if cache_type not in self.caches:
                    self.caches[cache_type] = self._create_mock_cache()
    
    def _create_mock_cache(self):
        """Create a mock cache for testing purposes."""
        class MockCache:
            def get_stats(self): return {'cache_size': 0, 'memory_usage_bytes': 0}
            def clear(self): pass
            def get_cache_info(self): return {'total_entries': 0, 'memory_usage_mb': 0}
            def invalidate_expired(self): pass
        return MockCache()
    
    def _start_monitoring(self):
        """Start background memory monitoring thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            return
        
        def monitor_loop():
            while self.monitoring_enabled:
                try:
                    self._check_memory_pressure()
                    time.sleep(self.memory_check_interval)
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {str(e)}")
                    time.sleep(5)  # Short sleep on error
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Memory monitoring started")
    
    def _check_memory_pressure(self):
        """Check system memory pressure and take action if needed."""
        current_time = time.time()
        if current_time - self.last_memory_check < self.memory_check_interval:
            return
        
        self.last_memory_check = current_time
        
        # Get system memory info
        memory_info = psutil.virtual_memory()
        system_memory_percent = memory_info.percent
        
        # Calculate cache memory usage
        cache_memory_mb = self._calculate_total_cache_memory()
        self.current_memory_usage = cache_memory_mb
        
        # Calculate cache memory as percentage of system
        cache_memory_percent = (cache_memory_mb / (memory_info.total / 1024 / 1024)) * 100
        
        # Determine pressure level
        pressure_level = self._get_memory_pressure_level(system_memory_percent, cache_memory_mb)
        
        # Take action based on pressure level
        if pressure_level in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]:
            self._handle_memory_pressure(pressure_level)
        
        # Log memory status periodically
        if current_time % 300 < self.memory_check_interval:  # Every 5 minutes
            self.logger.info(
                f"Memory status - System: {system_memory_percent:.1f}%, "
                f"Cache: {cache_memory_mb:.1f}MB ({cache_memory_percent:.1f}%), "
                f"Pressure: {pressure_level.value}"
            )
    
    def _calculate_total_cache_memory(self) -> float:
        """Calculate total memory usage across all caches in MB."""
        total_bytes = 0
        
        with self.lock:
            for cache_name, cache in self.caches.items():
                try:
                    stats = cache.get_stats()
                    memory_bytes = stats.get('memory_usage_bytes', 0)
                    if isinstance(memory_bytes, (int, float)):
                        total_bytes += memory_bytes
                except Exception as e:
                    self.logger.warning(f"Failed to get memory stats for {cache_name}: {str(e)}")
        
        return total_bytes / 1024 / 1024  # Convert to MB
    
    def _get_memory_pressure_level(self, system_percent: float, cache_memory_mb: float) -> MemoryPressureLevel:
        """Determine memory pressure level based on system and cache usage."""
        # Check system memory pressure
        if system_percent >= 85:
            return MemoryPressureLevel.CRITICAL
        elif system_percent >= 75:
            return MemoryPressureLevel.HIGH
        elif system_percent >= 60:
            return MemoryPressureLevel.MODERATE
        
        # Check cache memory limits
        max_cache_memory = min(
            self.system_limits.max_total_memory_mb,
            psutil.virtual_memory().total / 1024 / 1024 * self.system_limits.max_memory_percentage / 100
        )
        
        cache_usage_ratio = cache_memory_mb / max_cache_memory
        
        if cache_usage_ratio >= self.system_limits.emergency_threshold:
            return MemoryPressureLevel.CRITICAL
        elif cache_usage_ratio >= self.system_limits.cleanup_threshold:
            return MemoryPressureLevel.HIGH
        elif cache_usage_ratio >= self.system_limits.warning_threshold:
            return MemoryPressureLevel.MODERATE
        
        return MemoryPressureLevel.LOW
    
    def _handle_memory_pressure(self, pressure_level: MemoryPressureLevel):
        """Handle memory pressure by cleaning up caches."""
        self.logger.warning(f"Memory pressure detected: {pressure_level.value}")
        
        if pressure_level == MemoryPressureLevel.CRITICAL:
            # Emergency cleanup - clear low priority caches completely
            self._emergency_cleanup()
        elif pressure_level == MemoryPressureLevel.HIGH:
            # Aggressive cleanup - reduce cache sizes
            self._aggressive_cleanup()
        elif pressure_level == MemoryPressureLevel.MODERATE:
            # Gentle cleanup - remove expired and least used entries
            self._gentle_cleanup()
    
    def _emergency_cleanup(self):
        """Emergency cleanup during critical memory pressure."""
        self.logger.warning("Performing emergency cache cleanup")
        
        # Sort caches by priority (higher number = lower priority)
        sorted_caches = sorted(
            self.cache_configs.items(),
            key=lambda x: x[1].priority,
            reverse=True
        )
        
        # Clear lowest priority caches first
        for cache_name, config in sorted_caches[:2]:  # Clear 2 lowest priority
            if cache_name in self.caches:
                try:
                    self.caches[cache_name].clear()
                    self.logger.info(f"Cleared {cache_name} cache (priority {config.priority})")
                except Exception as e:
                    self.logger.error(f"Failed to clear {cache_name} cache: {str(e)}")
        
        # Reduce size of remaining caches
        for cache_name, config in sorted_caches[2:]:
            if cache_name in self.caches:
                self._reduce_cache_size(cache_name, 0.3)  # Keep only 30%
    
    def _aggressive_cleanup(self):
        """Aggressive cleanup during high memory pressure."""
        self.logger.info("Performing aggressive cache cleanup")
        
        # Clean expired entries from all caches
        self._clean_expired_entries()
        
        # Reduce cache sizes based on priority
        for cache_name, config in self.cache_configs.items():
            if cache_name in self.caches:
                reduction_factor = 0.5 + (config.priority * 0.1)  # 50-90% retention
                self._reduce_cache_size(cache_name, reduction_factor)
    
    def _gentle_cleanup(self):
        """Gentle cleanup during moderate memory pressure."""
        self.logger.info("Performing gentle cache cleanup")
        
        # Clean expired entries from all caches
        self._clean_expired_entries()
        
        # Reduce only the lowest priority cache
        lowest_priority_cache = max(
            self.cache_configs.items(),
            key=lambda x: x[1].priority
        )[0]
        
        if lowest_priority_cache in self.caches:
            self._reduce_cache_size(lowest_priority_cache, 0.7)  # Keep 70%
    
    def _clean_expired_entries(self):
        """Clean expired entries from all caches."""
        for cache_name, cache in self.caches.items():
            try:
                if hasattr(cache, 'invalidate_expired'):
                    cache.invalidate_expired()
                elif hasattr(cache, 'cleanup_expired'):
                    cache.cleanup_expired()
            except Exception as e:
                self.logger.warning(f"Failed to clean expired entries from {cache_name}: {str(e)}")
    
    def _reduce_cache_size(self, cache_name: str, retention_factor: float):
        """Reduce cache size by evicting entries."""
        try:
            cache = self.caches[cache_name]
            if hasattr(cache, 'reduce_size'):
                cache.reduce_size(retention_factor)
            elif hasattr(cache, 'cache') and hasattr(cache.cache, 'clear'):
                # For caches that don't have reduce_size, clear partial entries
                current_size = len(cache.cache)
                target_size = int(current_size * retention_factor)
                
                # This is a simplified approach - real implementation would be more sophisticated
                if current_size > target_size:
                    entries_to_remove = current_size - target_size
                    keys_to_remove = list(cache.cache.keys())[:entries_to_remove]
                    for key in keys_to_remove:
                        cache.cache.pop(key, None)
                    
                    self.logger.info(f"Reduced {cache_name} cache from {current_size} to {len(cache.cache)} entries")
        except Exception as e:
            self.logger.error(f"Failed to reduce {cache_name} cache size: {str(e)}")
    
    # Public API methods
    
    def get_cache(self, cache_type: str):
        """Get a specific cache instance."""
        return self.caches.get(cache_type)
    
    def get_all_caches(self) -> Dict[str, Any]:
        """Get all cache instances."""
        return self.caches.copy()
    
    def update_cache_config(self, cache_type: str, config: CacheConfig):
        """Update configuration for a specific cache."""
        with self.lock:
            self.cache_configs[cache_type] = config
            self.logger.info(f"Updated configuration for {cache_type} cache")
    
    def force_cleanup(self, pressure_level: MemoryPressureLevel = MemoryPressureLevel.MODERATE):
        """Force a cleanup operation."""
        self._handle_memory_pressure(pressure_level)
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status and cache statistics."""
        system_memory = psutil.virtual_memory()
        cache_memory_mb = self._calculate_total_cache_memory()
        
        cache_stats = {}
        total_entries = 0
        
        for cache_name, cache in self.caches.items():
            try:
                stats = cache.get_stats()
                cache_stats[cache_name] = stats
                total_entries += stats.get('cache_size', 0)
            except Exception as e:
                cache_stats[cache_name] = {'error': str(e)}
        
        pressure_level = self._get_memory_pressure_level(
            system_memory.percent,
            cache_memory_mb
        )
        
        return {
            'system_memory': {
                'total_gb': system_memory.total / 1024 / 1024 / 1024,
                'used_percent': system_memory.percent,
                'available_mb': system_memory.available / 1024 / 1024
            },
            'cache_memory': {
                'total_mb': cache_memory_mb,
                'limit_mb': self.system_limits.max_total_memory_mb,
                'usage_percent': (cache_memory_mb / self.system_limits.max_total_memory_mb) * 100,
                'total_entries': total_entries
            },
            'pressure_level': pressure_level.value,
            'cache_stats': cache_stats,
            'limits': {
                'max_total_memory_mb': self.system_limits.max_total_memory_mb,
                'max_memory_percentage': self.system_limits.max_memory_percentage,
                'warning_threshold': self.system_limits.warning_threshold,
                'cleanup_threshold': self.system_limits.cleanup_threshold
            }
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics across all caches."""
        stats = {
            'memory_status': self.get_memory_status(),
            'cache_details': {},
            'system_info': {
                'monitoring_enabled': self.monitoring_enabled,
                'memory_check_interval': self.memory_check_interval,
                'last_memory_check': self.last_memory_check,
                'uptime': time.time() - getattr(self, 'start_time', time.time())
            }
        }
        
        # Get detailed cache information
        for cache_name, cache in self.caches.items():
            try:
                if hasattr(cache, 'get_cache_info'):
                    stats['cache_details'][cache_name] = cache.get_cache_info()
                else:
                    stats['cache_details'][cache_name] = cache.get_stats()
            except Exception as e:
                stats['cache_details'][cache_name] = {'error': str(e)}
        
        return stats
    
    def clear_all_caches(self):
        """Clear all caches."""
        with self.lock:
            for cache_name, cache in self.caches.items():
                try:
                    cache.clear()
                    self.logger.info(f"Cleared {cache_name} cache")
                except Exception as e:
                    self.logger.error(f"Failed to clear {cache_name} cache: {str(e)}")
    
    def clear_cache(self, cache_type: str):
        """Clear a specific cache."""
        if cache_type in self.caches:
            try:
                self.caches[cache_type].clear()
                self.logger.info(f"Cleared {cache_type} cache")
            except Exception as e:
                self.logger.error(f"Failed to clear {cache_type} cache: {str(e)}")
    
    def shutdown(self):
        """Shutdown the cache manager and stop monitoring."""
        self.monitoring_enabled = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Unified cache manager shutdown complete")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except:
            pass


# Global instance for easy access
_global_cache_manager: Optional[UnifiedCacheManager] = None

def get_cache_manager() -> UnifiedCacheManager:
    """Get the global cache manager instance."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = UnifiedCacheManager()
    return _global_cache_manager

def initialize_cache_manager(system_limits: Optional[SystemLimits] = None) -> UnifiedCacheManager:
    """Initialize the global cache manager with custom limits."""
    global _global_cache_manager
    _global_cache_manager = UnifiedCacheManager(system_limits)
    return _global_cache_manager

def shutdown_cache_manager():
    """Shutdown the global cache manager."""
    global _global_cache_manager
    if _global_cache_manager:
        _global_cache_manager.shutdown()
        _global_cache_manager = None
