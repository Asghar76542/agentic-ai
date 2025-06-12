"""
Cache Statistics and Performance Monitoring
Comprehensive statistics collection and analysis for cache systems.
"""

import time
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path

from sources.logger import Logger


@dataclass
class PerformanceMetrics:
    """Performance metrics for cache operations."""
    operation_type: str
    duration: float
    timestamp: datetime
    cache_type: str
    result: str  # 'hit', 'miss', 'store', 'eviction'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation_type': self.operation_type,
            'duration': self.duration,
            'timestamp': self.timestamp.isoformat(),
            'cache_type': self.cache_type,
            'result': self.result
        }


class CacheStats:
    """
    Advanced cache statistics collection and analysis.
    
    Tracks performance metrics, hit rates, response times, and trends
    across different cache layers.
    """
    
    def __init__(self, cache_type: str = "generic"):
        """
        Initialize cache statistics.
        
        Args:
            cache_type: Type of cache (llm, web, compute, etc.)
        """
        self.cache_type = cache_type
        self.logger = Logger(f"{cache_type}_cache_stats.log")
        
        # Basic counters
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.evictions = 0
        
        # Detailed metrics
        self.semantic_hits = 0
        self.exact_hits = 0
        self.response_times: List[float] = []
        self.hit_rate_history: List[Tuple[datetime, float]] = []
        self.size_history: List[Tuple[datetime, int]] = []
        
        # Performance tracking
        self.metrics: List[PerformanceMetrics] = []
        self.start_time = datetime.now()
        
        # Configuration
        self.max_metrics_history = 10000
        self.metrics_retention_days = 30
    
    def record_hit(self, is_semantic: bool = False, response_time: float = 0.0):
        """
        Record a cache hit.
        
        Args:
            is_semantic: Whether this was a semantic similarity hit
            response_time: Time taken for the operation
        """
        self.hits += 1
        if is_semantic:
            self.semantic_hits += 1
        else:
            self.exact_hits += 1
        
        if response_time > 0:
            self.response_times.append(response_time)
        
        # Record detailed metrics
        metric = PerformanceMetrics(
            operation_type='get',
            duration=response_time,
            timestamp=datetime.now(),
            cache_type=self.cache_type,
            result='semantic_hit' if is_semantic else 'exact_hit'
        )
        self._add_metric(metric)
        
        self._update_hit_rate_history()
    
    def record_miss(self, response_time: float = 0.0):
        """
        Record a cache miss.
        
        Args:
            response_time: Time taken for the operation
        """
        self.misses += 1
        
        if response_time > 0:
            self.response_times.append(response_time)
        
        # Record detailed metrics
        metric = PerformanceMetrics(
            operation_type='get',
            duration=response_time,
            timestamp=datetime.now(),
            cache_type=self.cache_type,
            result='miss'
        )
        self._add_metric(metric)
        
        self._update_hit_rate_history()
    
    def record_cache_store(self, response_time: float = 0.0):
        """
        Record a cache store operation.
        
        Args:
            response_time: Time taken for the operation
        """
        self.stores += 1
        
        if response_time > 0:
            self.response_times.append(response_time)
        
        # Record detailed metrics
        metric = PerformanceMetrics(
            operation_type='store',
            duration=response_time,
            timestamp=datetime.now(),
            cache_type=self.cache_type,
            result='store'
        )
        self._add_metric(metric)
    
    def record_eviction(self, response_time: float = 0.0):
        """
        Record a cache eviction.
        
        Args:
            response_time: Time taken for the operation
        """
        self.evictions += 1
        
        # Record detailed metrics
        metric = PerformanceMetrics(
            operation_type='eviction',
            duration=response_time,
            timestamp=datetime.now(),
            cache_type=self.cache_type,
            result='eviction'
        )
        self._add_metric(metric)
    
    def record_cache_size(self, size: int):
        """
        Record current cache size.
        
        Args:
            size: Current number of entries in cache
        """
        self.size_history.append((datetime.now(), size))
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.size_history = [
            (timestamp, size) for timestamp, size in self.size_history
            if timestamp > cutoff_time
        ]
    
    def _add_metric(self, metric: PerformanceMetrics):
        """Add a performance metric and manage history size."""
        self.metrics.append(metric)
        
        # Limit metrics history size
        if len(self.metrics) > self.max_metrics_history:
            # Remove old metrics
            cutoff_time = datetime.now() - timedelta(days=self.metrics_retention_days)
            self.metrics = [
                m for m in self.metrics
                if m.timestamp > cutoff_time
            ]
    
    def _update_hit_rate_history(self):
        """Update hit rate history."""
        hit_rate = self.get_hit_rate()
        self.hit_rate_history.append((datetime.now(), hit_rate))
        
        # Keep only recent history (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.hit_rate_history = [
            (timestamp, rate) for timestamp, rate in self.hit_rate_history
            if timestamp > cutoff_time
        ]
    
    def get_hit_rate(self) -> float:
        """Calculate current cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0
    
    def get_semantic_hit_rate(self) -> float:
        """Calculate semantic hit rate among all hits."""
        return (self.semantic_hits / self.hits) if self.hits > 0 else 0.0
    
    def get_average_response_time(self) -> float:
        """Calculate average response time."""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
    
    def get_performance_improvement(self) -> Dict[str, float]:
        """Calculate performance improvements from caching."""
        if not self.metrics:
            return {'cached_avg': 0.0, 'improvement': 0.0}
        
        # Separate hit and miss response times
        hit_times = [m.duration for m in self.metrics if m.result in ['exact_hit', 'semantic_hit'] and m.duration > 0]
        miss_times = [m.duration for m in self.metrics if m.result == 'miss' and m.duration > 0]
        
        cached_avg = sum(hit_times) / len(hit_times) if hit_times else 0.0
        uncached_avg = sum(miss_times) / len(miss_times) if miss_times else 0.0
        
        improvement = 0.0
        if uncached_avg > 0 and cached_avg > 0:
            improvement = ((uncached_avg - cached_avg) / uncached_avg) * 100
        
        return {
            'cached_avg': cached_avg,
            'uncached_avg': uncached_avg,
            'improvement_percent': improvement
        }
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analyze trends in cache performance."""
        if len(self.hit_rate_history) < 2:
            return {'trend': 'insufficient_data'}
        
        # Calculate trend over last hour
        recent_cutoff = datetime.now() - timedelta(hours=1)
        recent_rates = [
            rate for timestamp, rate in self.hit_rate_history
            if timestamp > recent_cutoff
        ]
        
        if len(recent_rates) < 2:
            return {'trend': 'insufficient_recent_data'}
        
        # Simple trend calculation
        first_half = recent_rates[:len(recent_rates)//2]
        second_half = recent_rates[len(recent_rates)//2:]
        
        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)
        
        trend_direction = 'improving' if second_avg > first_avg else 'declining'
        trend_magnitude = abs(second_avg - first_avg)
        
        return {
            'trend': trend_direction,
            'magnitude': trend_magnitude,
            'recent_avg': second_avg,
            'previous_avg': first_avg
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        total_requests = self.hits + self.misses
        uptime = datetime.now() - self.start_time
        
        stats = {
            # Basic metrics
            'hits': self.hits,
            'misses': self.misses,
            'stores': self.stores,
            'evictions': self.evictions,
            'total_requests': total_requests,
            
            # Hit rate metrics
            'hit_rate': self.get_hit_rate(),
            'semantic_hit_rate': self.get_semantic_hit_rate(),
            'exact_hits': self.exact_hits,
            'semantic_hits': self.semantic_hits,
            
            # Performance metrics
            'average_response_time': self.get_average_response_time(),
            'performance_improvement': self.get_performance_improvement(),
            'trend_analysis': self.get_trend_analysis(),
            
            # System metrics
            'cache_type': self.cache_type,
            'uptime_seconds': uptime.total_seconds(),
            'metrics_count': len(self.metrics),
            'start_time': self.start_time.isoformat()
        }
        
        return stats
    
    def export_metrics(self, filepath: str) -> bool:
        """
        Export metrics to file.
        
        Args:
            filepath: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                'cache_type': self.cache_type,
                'export_time': datetime.now().isoformat(),
                'statistics': self.get_stats(),
                'metrics': [metric.to_dict() for metric in self.metrics],
                'hit_rate_history': [
                    {'timestamp': ts.isoformat(), 'hit_rate': rate}
                    for ts, rate in self.hit_rate_history
                ],
                'size_history': [
                    {'timestamp': ts.isoformat(), 'size': size}
                    for ts, size in self.size_history
                ]
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Exported metrics to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False
    
    def reset_stats(self):
        """Reset all statistics."""
        self.hits = 0
        self.misses = 0
        self.stores = 0
        self.evictions = 0
        self.semantic_hits = 0
        self.exact_hits = 0
        self.response_times.clear()
        self.hit_rate_history.clear()
        self.size_history.clear()
        self.metrics.clear()
        self.start_time = datetime.now()
        
        self.logger.info("Cache statistics reset")
    
    def get_hourly_summary(self) -> Dict[str, Any]:
        """Get summary of performance for the last hour."""
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_metrics = [
            m for m in self.metrics
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {'period': 'last_hour', 'data': 'no_data'}
        
        hits = sum(1 for m in recent_metrics if m.result in ['exact_hit', 'semantic_hit'])
        misses = sum(1 for m in recent_metrics if m.result == 'miss')
        stores = sum(1 for m in recent_metrics if m.result == 'store')
        
        return {
            'period': 'last_hour',
            'hits': hits,
            'misses': misses,
            'stores': stores,
            'hit_rate': (hits / (hits + misses)) if (hits + misses) > 0 else 0.0,
            'total_operations': len(recent_metrics),
            'average_duration': sum(m.duration for m in recent_metrics) / len(recent_metrics)
        }
