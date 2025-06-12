"""
Computation Result Cache for Code Execution
Caches results of deterministic code execution operations
"""

import hashlib
import json
import time
import re
import gzip
import platform
import os
from typing import Dict, Any, Optional, Tuple, List, Set
from threading import Lock
from sources.cache.cacheStats import CacheStats

class ComputationCache:
    """
    Intelligent caching system for code execution results.
    Caches deterministic operations while avoiding non-deterministic ones.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize the computation cache.
        
        Args:
            max_size: Maximum number of cached results
            default_ttl: Default TTL in seconds for cached results
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        
        # Cache storage: {cache_key: CacheEntry}
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # Access tracking for LRU eviction
        self.access_times: Dict[str, float] = {}
        self.access_count: Dict[str, int] = {}
        
        # Thread safety
        self.lock = Lock()
        
        # Statistics tracking
        self.stats = CacheStats("ComputationCache")
        
        # Non-deterministic patterns to avoid caching
        self.non_deterministic_patterns = {
            'python': [
                r'random\.',
                r'numpy\.random',
                r'time\.',
                r'datetime\.',
                r'input\(',
                r'raw_input\(',
                r'os\.urandom',
                r'uuid\.',
                r'threading\.',
                r'asyncio\.',
                r'requests\.',
                r'urllib',
                r'socket\.',
                r'subprocess\.',
                r'multiprocessing\.',
            ],
            'java': [
                r'Math\.random',
                r'Random\(',
                r'System\.currentTimeMillis',
                r'new Date',
                r'Scanner\(',
                r'System\.in',
                r'Thread\.',
                r'Timer\(',
                r'HttpURLConnection',
                r'Socket\(',
                r'Runtime\.getRuntime',
            ],
            'go': [
                r'rand\.',
                r'time\.',
                r'fmt\.Scan',
                r'os\.Stdin',
                r'http\.',
                r'net\.',
                r'goroutine',
                r'go func',
                r'runtime\.',
            ],
            'c': [
                r'rand\(',
                r'srand\(',
                r'time\(',
                r'scanf\(',
                r'getchar\(',
                r'system\(',
                r'popen\(',
                r'fork\(',
                r'pthread_',
            ],
            'bash': [
                r'\$RANDOM',
                r'date',
                r'ps\s',
                r'top\s',
                r'who\s',
                r'uptime',
                r'df\s',
                r'free\s',
                r'netstat',
                r'curl\s',
                r'wget\s',
                r'ping\s',
                r'ssh\s',
                r'read\s',
            ]
        }
        
        # Environment fingerprint for cache invalidation
        self.environment_fingerprint = self._generate_environment_fingerprint()
    
    def _generate_environment_fingerprint(self) -> str:
        """Generate a fingerprint of the current environment for cache validation."""
        env_data = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'working_directory': os.getcwd(),
            'path_hash': hashlib.md5(os.environ.get('PATH', '').encode()).hexdigest()[:8]
        }
        
        fingerprint_str = json.dumps(env_data, sort_keys=True)
        return hashlib.md5(fingerprint_str.encode()).hexdigest()[:16]
    
    def _is_deterministic(self, code: str, language: str) -> bool:
        """
        Determine if code execution is likely to be deterministic.
        
        Args:
            code: The code to analyze
            language: Programming language (python, java, go, c, bash)
            
        Returns:
            True if code appears deterministic, False otherwise
        """
        if language not in self.non_deterministic_patterns:
            # Conservative approach: if we don't know the language, don't cache
            return False
        
        patterns = self.non_deterministic_patterns[language]
        
        # Check for non-deterministic patterns
        for pattern in patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return False
        
        # Additional heuristics
        if language == 'python':
            # Check for network operations, file I/O that might be non-deterministic
            non_det_keywords = ['requests', 'urllib', 'socket', 'open(', 'input(']
            for keyword in non_det_keywords:
                if keyword in code.lower():
                    return False
        
        elif language == 'bash':
            # Bash commands that read from system state
            if any(cmd in code.lower() for cmd in ['ps', 'top', 'who', 'df', 'free']):
                return False
        
        return True
    
    def _generate_cache_key(self, code: str, language: str, args: Dict[str, Any] = None) -> str:
        """
        Generate a unique cache key for code execution.
        
        Args:
            code: The code to execute
            language: Programming language
            args: Additional execution arguments
            
        Returns:
            Cache key string
        """
        # Normalize code (remove comments, extra whitespace)
        normalized_code = self._normalize_code(code, language)
        
        key_data = {
            'code': normalized_code,
            'language': language,
            'args': args or {},
            'environment': self.environment_fingerprint
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    def _normalize_code(self, code: str, language: str) -> str:
        """
        Normalize code by removing comments and extra whitespace.
        
        Args:
            code: Raw code string
            language: Programming language
            
        Returns:
            Normalized code string
        """
        if language == 'python':
            # Remove Python comments and docstrings
            lines = []
            in_multiline = False
            for line in code.split('\n'):
                line = line.strip()
                if line.startswith('"""') or line.startswith("'''"):
                    in_multiline = not in_multiline
                    continue
                if not in_multiline and not line.startswith('#') and line:
                    lines.append(line)
            return '\n'.join(lines)
        
        elif language in ['java', 'c', 'go']:
            # Remove C-style comments
            code = re.sub(r'//.*', '', code)  # Single line comments
            code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)  # Multi-line comments
        
        elif language == 'bash':
            # Remove bash comments
            lines = [line.split('#')[0].strip() for line in code.split('\n')]
            code = '\n'.join(line for line in lines if line)
        
        # Remove extra whitespace
        return re.sub(r'\s+', ' ', code).strip()
    
    def _calculate_ttl(self, code: str, language: str, execution_time: float) -> int:
        """
        Calculate appropriate TTL based on code characteristics.
        
        Args:
            code: The executed code
            language: Programming language
            execution_time: Time taken to execute in seconds
            
        Returns:
            TTL in seconds
        """
        base_ttl = self.default_ttl
        
        # Longer TTL for longer execution times
        if execution_time > 10:  # 10+ seconds
            base_ttl *= 4  # 4 hours
        elif execution_time > 5:  # 5+ seconds
            base_ttl *= 2  # 2 hours
        elif execution_time < 0.1:  # Very fast operations
            base_ttl //= 2  # 30 minutes
        
        # Longer TTL for mathematical computations
        math_patterns = ['math.', 'numpy.', 'scipy.', 'calculation', 'compute']
        if any(pattern in code.lower() for pattern in math_patterns):
            base_ttl *= 2
        
        # Shorter TTL for file operations (in case files change)
        file_patterns = ['open(', 'file', 'read', 'write']
        if any(pattern in code.lower() for pattern in file_patterns):
            base_ttl //= 4
        
        return max(300, min(base_ttl, 86400))  # Between 5 minutes and 24 hours
    
    def _evict_if_needed(self):
        """Evict entries if cache size exceeds limit using LRU strategy."""
        if len(self.cache) <= self.max_size:
            return
        
        # Calculate LRU scores (combination of recency and frequency)
        current_time = time.time()
        lru_scores = {}
        
        for key in self.cache:
            recency = current_time - self.access_times.get(key, 0)
            frequency = self.access_count.get(key, 0)
            # Lower score = more likely to be evicted
            lru_scores[key] = frequency / (1 + recency / 3600)  # Frequency per hour
        
        # Sort by LRU score and evict the lowest scoring entries
        sorted_keys = sorted(lru_scores.keys(), key=lambda k: lru_scores[k])
        evict_count = len(self.cache) - self.max_size + 1
        
        for key in sorted_keys[:evict_count]:
            self._remove_entry(key)
            self.stats.increment_evictions()
    
    def _remove_entry(self, key: str):
        """Remove a cache entry and its metadata."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_count.pop(key, None)
    
    def _update_access(self, key: str):
        """Update access tracking for LRU."""
        current_time = time.time()
        self.access_times[key] = current_time
        self.access_count[key] = self.access_count.get(key, 0) + 1
    
    def get(self, code: str, language: str, args: Dict[str, Any] = None) -> Optional[Tuple[str, bool]]:
        """
        Retrieve cached execution result.
        
        Args:
            code: The code to execute
            language: Programming language
            args: Additional execution arguments
            
        Returns:
            Tuple of (result, is_success) if found, None otherwise
        """
        # Don't cache non-deterministic operations
        if not self._is_deterministic(code, language):
            self.stats.increment_misses()
            return None
        
        cache_key = self._generate_cache_key(code, language, args)
        
        with self.lock:
            if cache_key not in self.cache:
                self.stats.increment_misses()
                return None
            
            entry = self.cache[cache_key]
            
            # Check if entry has expired
            if entry['expires_at'] < time.time():
                self._remove_entry(cache_key)
                self.stats.increment_misses()
                return None
            
            # Update access tracking
            self._update_access(cache_key)
            
            # Decompress result if it was compressed
            result = entry['result']
            if entry.get('compressed', False):
                result = gzip.decompress(result.encode()).decode()
            
            self.stats.increment_hits()
            self.stats.add_response_time(0.001)  # Cache access is very fast
            
            return result, entry['is_success']
    
    def put(self, code: str, language: str, result: str, is_success: bool, 
            execution_time: float, args: Dict[str, Any] = None):
        """
        Store execution result in cache.
        
        Args:
            code: The executed code
            language: Programming language
            result: Execution result
            is_success: Whether execution was successful
            execution_time: Time taken to execute
            args: Additional execution arguments
        """
        # Don't cache non-deterministic operations
        if not self._is_deterministic(code, language):
            return
        
        # Don't cache very large results (>1MB)
        if len(result) > 1024 * 1024:
            return
        
        cache_key = self._generate_cache_key(code, language, args)
        ttl = self._calculate_ttl(code, language, execution_time)
        
        # Compress large results
        compressed = False
        stored_result = result
        if len(result) > 1024:  # Compress results larger than 1KB
            try:
                compressed_result = gzip.compress(result.encode())
                if len(compressed_result) < len(result) * 0.8:  # Only if compression saves 20%+
                    stored_result = compressed_result.decode('latin1')
                    compressed = True
            except Exception:
                pass  # Use uncompressed if compression fails
        
        entry = {
            'result': stored_result,
            'is_success': is_success,
            'created_at': time.time(),
            'expires_at': time.time() + ttl,
            'execution_time': execution_time,
            'language': language,
            'code_length': len(code),
            'compressed': compressed,
            'access_count': 1
        }
        
        with self.lock:
            self.cache[cache_key] = entry
            self._update_access(cache_key)
            self._evict_if_needed()
            
            # Update statistics
            self.stats.add_cached_item(len(stored_result))
    
    def invalidate_language(self, language: str):
        """Invalidate all cached results for a specific language."""
        with self.lock:
            keys_to_remove = [
                key for key, entry in self.cache.items()
                if entry['language'] == language
            ]
            
            for key in keys_to_remove:
                self._remove_entry(key)
                self.stats.increment_evictions()
    
    def clear(self):
        """Clear all cached results."""
        with self.lock:
            cleared_count = len(self.cache)
            self.cache.clear()
            self.access_times.clear()
            self.access_count.clear()
            self.stats.add_evictions(cleared_count)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            current_time = time.time()
            
            # Calculate cache characteristics
            total_size = sum(len(entry['result']) for entry in self.cache.values())
            compressed_count = sum(1 for entry in self.cache.values() if entry.get('compressed', False))
            
            # Language distribution
            language_distribution = {}
            execution_time_stats = {}
            
            for entry in self.cache.values():
                lang = entry['language']
                language_distribution[lang] = language_distribution.get(lang, 0) + 1
                
                exec_time = entry['execution_time']
                if lang not in execution_time_stats:
                    execution_time_stats[lang] = []
                execution_time_stats[lang].append(exec_time)
            
            # Calculate average execution times by language
            avg_execution_times = {}
            for lang, times in execution_time_stats.items():
                avg_execution_times[lang] = sum(times) / len(times)
            
            stats = self.stats.get_stats()
            stats.update({
                'cache_size': len(self.cache),
                'max_size': self.max_size,
                'memory_usage_bytes': total_size,
                'compressed_entries': compressed_count,
                'compression_ratio': f"{compressed_count}/{len(self.cache)}" if self.cache else "0/0",
                'language_distribution': language_distribution,
                'avg_execution_times': avg_execution_times,
                'environment_fingerprint': self.environment_fingerprint,
                'oldest_entry_age': min(
                    current_time - entry['created_at'] for entry in self.cache.values()
                ) if self.cache else 0
            })
            
            return stats
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed information about cached entries."""
        with self.lock:
            current_time = time.time()
            
            entries_info = []
            for key, entry in list(self.cache.items())[:10]:  # Show top 10 most recent
                entries_info.append({
                    'cache_key': key[:16] + '...',
                    'language': entry['language'],
                    'code_length': entry['code_length'],
                    'result_size': len(entry['result']),
                    'age_minutes': (current_time - entry['created_at']) / 60,
                    'ttl_remaining_minutes': max(0, (entry['expires_at'] - current_time) / 60),
                    'access_count': self.access_count.get(key, 0),
                    'compressed': entry.get('compressed', False),
                    'is_success': entry['is_success']
                })
            
            return {
                'recent_entries': entries_info,
                'total_entries': len(self.cache),
                'memory_usage_mb': sum(len(entry['result']) for entry in self.cache.values()) / 1024 / 1024
            }
