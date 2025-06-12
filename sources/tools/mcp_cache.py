"""
Advanced MCP Caching System
Provides intelligent caching, result optimization, and performance enhancement for MCP operations
"""

import asyncio
import json
import hashlib
import time
import pickle
import gzip
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
import sqlite3
from contextlib import contextmanager

from sources.utility import pretty_print

class CacheStrategy(Enum):
    """Cache strategy types"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns

class CacheLevel(Enum):
    """Cache levels for hierarchical caching"""
    MEMORY = "memory"     # In-memory cache (fastest)
    DISK = "disk"         # Disk-based cache
    DISTRIBUTED = "distributed"  # Distributed cache (future)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    compressed: bool = False
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access information"""
        self.last_accessed = time.time()
        self.access_count += 1

class MCPMemoryCache:
    """High-performance in-memory cache for MCP results"""
    
    def __init__(self, max_size: int = 1000, strategy: CacheStrategy = CacheStrategy.LRU):
        self.max_size = max_size
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU
        self.frequency: Dict[str, int] = {}  # For LFU
        self.lock = threading.RLock()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def _generate_key(self, server_name: str, tool_name: str, 
                     arguments: Dict[str, Any]) -> str:
        """Generate cache key from parameters"""
        key_data = {
            "server": server_name,
            "tool": tool_name,
            "args": arguments
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, server_name: str, tool_name: str, 
            arguments: Dict[str, Any]) -> Optional[Any]:
        """Get cached result"""
        key = self._generate_key(server_name, tool_name, arguments)
        
        with self.lock:
            entry = self.cache.get(key)
            
            if entry is None:
                self.misses += 1
                return None
            
            if entry.is_expired():
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Update access information
            entry.touch()
            self._update_access_order(key)
            
            self.hits += 1
            
            # Decompress if needed
            if entry.compressed:
                return pickle.loads(gzip.decompress(entry.value))
            
            return entry.value
    
    def put(self, server_name: str, tool_name: str, arguments: Dict[str, Any],
            result: Any, ttl: Optional[float] = None, compress: bool = False) -> bool:
        """Store result in cache"""
        key = self._generate_key(server_name, tool_name, arguments)
        
        with self.lock:
            # Serialize and optionally compress the result
            if compress:
                serialized = gzip.compress(pickle.dumps(result))
                compressed = True
            else:
                serialized = result
                compressed = False
            
            # Calculate size
            size_bytes = len(pickle.dumps(serialized))
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=serialized,
                created_at=time.time(),
                last_accessed=time.time(),
                access_count=1,
                ttl=ttl,
                compressed=compressed,
                size_bytes=size_bytes
            )
            
            # Check if we need to evict entries
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_entries()
            
            # Store entry
            self.cache[key] = entry
            self._update_access_order(key)
            
            return True
    
    def _update_access_order(self, key: str):
        """Update access order for LRU strategy"""
        if self.strategy == CacheStrategy.LRU:
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
        
        elif self.strategy == CacheStrategy.LFU:
            self.frequency[key] = self.frequency.get(key, 0) + 1
    
    def _evict_entries(self):
        """Evict entries based on strategy"""
        if not self.cache:
            return
        
        if self.strategy == CacheStrategy.LRU:
            # Remove least recently used
            oldest_key = self.access_order[0]
            self._remove_entry(oldest_key)
        
        elif self.strategy == CacheStrategy.LFU:
            # Remove least frequently used
            min_freq = min(self.frequency.values())
            lfu_keys = [k for k, f in self.frequency.items() if f == min_freq]
            oldest_key = min(lfu_keys, key=lambda k: self.cache[k].last_accessed)
            self._remove_entry(oldest_key)
        
        elif self.strategy == CacheStrategy.TTL:
            # Remove expired entries first
            expired_keys = [k for k, entry in self.cache.items() if entry.is_expired()]
            if expired_keys:
                for key in expired_keys:
                    self._remove_entry(key)
            else:
                # Fallback to LRU
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k].last_accessed)
                self._remove_entry(oldest_key)
        
        self.evictions += 1
    
    def _remove_entry(self, key: str):
        """Remove entry from cache"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_order:
            self.access_order.remove(key)
        if key in self.frequency:
            del self.frequency[key]
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.frequency.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions,
            "total_entries": len(self.cache),
            "max_size": self.max_size,
            "total_size_bytes": total_size,
            "strategy": self.strategy.value
        }

class MCPDiskCache:
    """Persistent disk-based cache for MCP results"""
    
    def __init__(self, cache_dir: str = "/tmp/agenticseek_mcp_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # SQLite database for metadata
        self.db_path = self.cache_dir / "cache_metadata.db"
        self._init_database()
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
    
    def _init_database(self):
        """Initialize SQLite database for cache metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    key TEXT PRIMARY KEY,
                    server_name TEXT,
                    tool_name TEXT,
                    arguments_hash TEXT,
                    file_path TEXT,
                    created_at REAL,
                    last_accessed REAL,
                    access_count INTEGER,
                    ttl REAL,
                    size_bytes INTEGER,
                    compressed BOOLEAN
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache_entries(created_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed)
            """)
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with automatic cleanup"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _generate_key(self, server_name: str, tool_name: str, 
                     arguments: Dict[str, Any]) -> str:
        """Generate cache key from parameters"""
        key_data = {
            "server": server_name,
            "tool": tool_name,
            "args": arguments
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, server_name: str, tool_name: str, 
            arguments: Dict[str, Any]) -> Optional[Any]:
        """Get cached result from disk"""
        key = self._generate_key(server_name, tool_name, arguments)
        
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM cache_entries WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            
            if row is None:
                self.misses += 1
                return None
            
            # Check if expired
            if row['ttl'] and time.time() - row['created_at'] > row['ttl']:
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Load from file
            file_path = Path(row['file_path'])
            if not file_path.exists():
                self._remove_entry(key)
                self.misses += 1
                return None
            
            try:
                with open(file_path, 'rb') as f:
                    data = f.read()
                
                if row['compressed']:
                    data = gzip.decompress(data)
                
                result = pickle.loads(data)
                
                # Update access information
                conn.execute("""
                    UPDATE cache_entries 
                    SET last_accessed = ?, access_count = access_count + 1
                    WHERE key = ?
                """, (time.time(), key))
                conn.commit()
                
                self.hits += 1
                return result
                
            except Exception as e:
                pretty_print(f"Error reading cache file {file_path}: {e}", color="error")
                self._remove_entry(key)
                self.misses += 1
                return None
    
    def put(self, server_name: str, tool_name: str, arguments: Dict[str, Any],
            result: Any, ttl: Optional[float] = None, compress: bool = True) -> bool:
        """Store result in disk cache"""
        key = self._generate_key(server_name, tool_name, arguments)
        
        try:
            # Serialize result
            data = pickle.dumps(result)
            
            # Optionally compress
            if compress:
                data = gzip.compress(data)
            
            # Save to file
            file_path = self.cache_dir / f"{key}.cache"
            with open(file_path, 'wb') as f:
                f.write(data)
            
            # Store metadata in database
            with self._get_db_connection() as conn:
                args_hash = hashlib.sha256(
                    json.dumps(arguments, sort_keys=True).encode()
                ).hexdigest()
                
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries
                    (key, server_name, tool_name, arguments_hash, file_path,
                     created_at, last_accessed, access_count, ttl, 
                     size_bytes, compressed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    key, server_name, tool_name, args_hash, str(file_path),
                    time.time(), time.time(), 1, ttl,
                    len(data), compress
                ))
                conn.commit()
            
            return True
            
        except Exception as e:
            pretty_print(f"Error storing cache entry: {e}", color="error")
            return False
    
    def _remove_entry(self, key: str):
        """Remove entry from cache"""
        with self._get_db_connection() as conn:
            cursor = conn.execute(
                "SELECT file_path FROM cache_entries WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            
            if row:
                # Remove file
                file_path = Path(row['file_path'])
                if file_path.exists():
                    file_path.unlink()
                
                # Remove from database
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (key,))
                conn.commit()
    
    def cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        
        with self._get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT key, file_path FROM cache_entries 
                WHERE ttl IS NOT NULL AND ? - created_at > ttl
            """, (current_time,))
            
            expired_entries = cursor.fetchall()
            
            for entry in expired_entries:
                # Remove file
                file_path = Path(entry['file_path'])
                if file_path.exists():
                    file_path.unlink()
                
                # Remove from database
                conn.execute("DELETE FROM cache_entries WHERE key = ?", (entry['key'],))
            
            conn.commit()
            
        if expired_entries:
            pretty_print(f"Cleaned up {len(expired_entries)} expired cache entries", color="info")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._get_db_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_entries,
                    SUM(size_bytes) as total_size,
                    AVG(access_count) as avg_access_count,
                    MIN(created_at) as oldest_entry,
                    MAX(last_accessed) as newest_access
                FROM cache_entries
            """)
            row = cursor.fetchone()
            
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "total_entries": row['total_entries'] or 0,
                "total_size_bytes": row['total_size'] or 0,
                "avg_access_count": row['avg_access_count'] or 0,
                "cache_age_days": (time.time() - (row['oldest_entry'] or time.time())) / 86400,
                "cache_dir": str(self.cache_dir)
            }

class MCPHierarchicalCache:
    """Hierarchical cache combining memory and disk caching"""
    
    def __init__(self, memory_size: int = 500, cache_dir: str = None,
                 memory_strategy: CacheStrategy = CacheStrategy.LRU):
        self.memory_cache = MCPMemoryCache(memory_size, memory_strategy)
        self.disk_cache = MCPDiskCache(cache_dir) if cache_dir else None
        
        # Configuration
        self.memory_ttl = 300  # 5 minutes
        self.disk_ttl = 3600   # 1 hour
        self.auto_compress = True
        
        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
    
    def get(self, server_name: str, tool_name: str, 
            arguments: Dict[str, Any]) -> Optional[Any]:
        """Get cached result with hierarchical lookup"""
        # Try memory cache first
        result = self.memory_cache.get(server_name, tool_name, arguments)
        if result is not None:
            return result
        
        # Try disk cache
        if self.disk_cache:
            result = self.disk_cache.get(server_name, tool_name, arguments)
            if result is not None:
                # Store in memory cache for faster future access
                self.memory_cache.put(
                    server_name, tool_name, arguments, result,
                    ttl=self.memory_ttl, compress=False
                )
                return result
        
        return None
    
    def put(self, server_name: str, tool_name: str, arguments: Dict[str, Any],
            result: Any) -> bool:
        """Store result in hierarchical cache"""
        success = True
        
        # Store in memory cache
        memory_success = self.memory_cache.put(
            server_name, tool_name, arguments, result,
            ttl=self.memory_ttl, compress=False
        )
        
        # Store in disk cache
        if self.disk_cache:
            disk_success = self.disk_cache.put(
                server_name, tool_name, arguments, result,
                ttl=self.disk_ttl, compress=self.auto_compress
            )
            success = memory_success and disk_success
        else:
            success = memory_success
        
        return success
    
    def clear(self):
        """Clear all cache levels"""
        self.memory_cache.clear()
        if self.disk_cache:
            # For disk cache, we would need to implement a clear method
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            "memory_cache": self.memory_cache.get_stats()
        }
        
        if self.disk_cache:
            stats["disk_cache"] = self.disk_cache.get_stats()
        
        # Combined statistics
        total_hits = stats["memory_cache"]["hits"]
        total_misses = stats["memory_cache"]["misses"]
        
        if "disk_cache" in stats:
            total_hits += stats["disk_cache"]["hits"]
            total_misses += stats["disk_cache"]["misses"]
        
        total_requests = total_hits + total_misses
        combined_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        stats["combined"] = {
            "total_hits": total_hits,
            "total_misses": total_misses,
            "hit_rate": combined_hit_rate,
            "total_requests": total_requests
        }
        
        return stats
    
    def _cleanup_loop(self):
        """Periodic cleanup of expired entries"""
        while True:
            try:
                if self.disk_cache:
                    self.disk_cache.cleanup_expired()
                time.sleep(3600)  # Run every hour
            except Exception as e:
                pretty_print(f"Error in cache cleanup: {e}", color="error")
                time.sleep(3600)

# Example usage and testing
if __name__ == "__main__":
    def test_hierarchical_cache():
        """Test hierarchical caching functionality"""
        cache = MCPHierarchicalCache(
            memory_size=100,
            cache_dir="/tmp/test_mcp_cache"
        )
        
        # Test data
        server_name = "test_server"
        tool_name = "test_tool"
        arguments = {"param1": "value1", "param2": 42}
        result = {"data": "test_result", "timestamp": time.time()}
        
        # Test put and get
        print("Testing cache put...")
        success = cache.put(server_name, tool_name, arguments, result)
        print(f"Put result: {success}")
        
        print("Testing cache get...")
        cached_result = cache.get(server_name, tool_name, arguments)
        print(f"Get result: {cached_result}")
        
        # Test stats
        print("Cache statistics:")
        stats = cache.get_stats()
        print(json.dumps(stats, indent=2))
    
    # Run test
    test_hierarchical_cache()
