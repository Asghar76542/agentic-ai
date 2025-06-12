"""
LLM Response Cache with Semantic Matching
Intelligent caching system for LLM responses using semantic similarity.
"""

import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import pickle
import os
from pathlib import Path

from sources.logger import Logger
from sources.utility import timer_decorator
from sources.cache.cacheStats import CacheStats


@dataclass
class CacheEntry:
    """Cache entry for LLM responses."""
    query: str
    response: str
    query_hash: str
    timestamp: datetime
    ttl: int  # Time to live in seconds
    metadata: Dict[str, Any]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    confidence_score: float = 1.0
    importance: float = 0.5
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl)
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class LLMCache:
    """
    Intelligent LLM response caching system with semantic similarity matching.
    
    Features:
    - Semantic similarity matching for related queries
    - Adaptive TTL based on response characteristics
    - Memory pressure management with intelligent eviction
    - Performance monitoring and statistics
    """
    
    def __init__(self, 
                 memory_system=None,
                 similarity_threshold: float = 0.85,
                 max_entries: int = 10000,
                 cache_dir: str = "./data/llm_cache",
                 enable_persistence: bool = True):
        """
        Initialize LLM cache.
        
        Args:
            memory_system: Enhanced memory system for semantic search
            similarity_threshold: Minimum similarity for cache hits (0.0-1.0)
            max_entries: Maximum number of cache entries
            cache_dir: Directory for persistent cache storage
            enable_persistence: Whether to persist cache to disk
        """
        self.memory_system = memory_system
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self.cache_dir = Path(cache_dir)
        self.enable_persistence = enable_persistence
        
        # Initialize cache storage
        self.cache_store: Dict[str, CacheEntry] = {}
        self.query_embeddings: Dict[str, Any] = {}
        
        # Statistics and monitoring
        self.hit_stats = CacheStats("llm_cache")
        self.logger = Logger("llm_cache.log")
        
        # Configuration
        self.default_ttl = 3600  # 1 hour
        self.factual_ttl_multiplier = 24  # 24 hours for factual content
        self.min_query_length = 10  # Minimum query length to cache
        
        # Initialize cache
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize cache storage and load existing entries."""
        try:
            if self.enable_persistence:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self._load_cache_from_disk()
            
            self.logger.info(f"LLM cache initialized with {len(self.cache_store)} entries")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM cache: {e}")
            # Continue without persistence
            self.enable_persistence = False
    
    def _load_cache_from_disk(self):
        """Load existing cache entries from disk."""
        try:
            cache_file = self.cache_dir / "cache_entries.json"
            if cache_file.exists():
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for entry_data in data:
                    # Convert timestamp strings back to datetime objects
                    entry_data['timestamp'] = datetime.fromisoformat(entry_data['timestamp'])
                    if entry_data.get('last_accessed'):
                        entry_data['last_accessed'] = datetime.fromisoformat(entry_data['last_accessed'])
                    
                    entry = CacheEntry(**entry_data)
                    
                    # Only load non-expired entries
                    if not entry.is_expired():
                        self.cache_store[entry.query_hash] = entry
                
                self.logger.info(f"Loaded {len(self.cache_store)} cache entries from disk")
                
        except Exception as e:
            self.logger.warning(f"Failed to load cache from disk: {e}")
    
    def _save_cache_to_disk(self):
        """Save cache entries to disk for persistence."""
        if not self.enable_persistence:
            return
            
        try:
            cache_file = self.cache_dir / "cache_entries.json"
            
            # Convert cache entries to serializable format
            cache_data = []
            for entry in self.cache_store.values():
                entry_dict = asdict(entry)
                # Convert datetime objects to ISO strings
                entry_dict['timestamp'] = entry.timestamp.isoformat()
                if entry.last_accessed:
                    entry_dict['last_accessed'] = entry.last_accessed.isoformat()
                else:
                    entry_dict['last_accessed'] = None
                cache_data.append(entry_dict)
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
            self.logger.debug(f"Saved {len(cache_data)} cache entries to disk")
            
        except Exception as e:
            self.logger.error(f"Failed to save cache to disk: {e}")
    
    @timer_decorator
    def get_cached_response(self, query: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get cached response for a query using semantic similarity matching.
        
        Args:
            query: The query to search for
            context: Optional context for the query
            
        Returns:
            Cached response with metadata if found, None otherwise
        """
        try:
            # Skip caching for very short queries
            if len(query.strip()) < self.min_query_length:
                self.hit_stats.record_miss()
                return None
            
            start_time = time.time()
            
            # Generate query hash for exact matching
            query_hash = self._generate_query_hash(query, context)
            
            # First try exact match
            if query_hash in self.cache_store:
                entry = self.cache_store[query_hash]
                if not entry.is_expired():
                    entry.update_access()
                    self.hit_stats.record_hit(is_semantic=False)
                    
                    response_data = {
                        'response': entry.response,
                        'cached': True,
                        'cache_type': 'exact',
                        'confidence': 1.0,
                        'timestamp': entry.timestamp.isoformat(),
                        'access_count': entry.access_count
                    }
                    
                    self.logger.info(f"Exact cache hit for query: {query[:50]}...")
                    return response_data
                else:
                    # Remove expired entry
                    del self.cache_store[query_hash]
            
            # Try semantic similarity matching if memory system is available
            if self.memory_system and hasattr(self.memory_system, 'search_memories'):
                semantic_result = self._semantic_search(query, context)
                if semantic_result:
                    self.hit_stats.record_hit(is_semantic=True)
                    return semantic_result
            
            # No cache hit found
            self.hit_stats.record_miss()
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving cached response: {e}")
            self.hit_stats.record_miss()
            return None
    
    def _semantic_search(self, query: str, context: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Search for semantically similar cached responses."""
        try:
            # Combine query and context for better matching
            search_text = query
            if context:
                search_text = f"{context}\n{query}"
            
            # Search in memory system
            results = self.memory_system.search_memories(
                query=search_text,
                limit=5,
                filters={'role': 'cache_query'},
                include_system=False
            )
            
            # Find the best matching cached response
            for result in results:
                memory_entry = result[0] if isinstance(result, tuple) else result
                similarity_score = result[1] if isinstance(result, tuple) else result.get('score', 0)
                
                if similarity_score >= self.similarity_threshold:
                    # Get the cached response from metadata
                    metadata = memory_entry.metadata if hasattr(memory_entry, 'metadata') else memory_entry.get('metadata', {})
                    cached_response = metadata.get('cached_response')
                    
                    if cached_response:
                        response_data = {
                            'response': cached_response,
                            'cached': True,
                            'cache_type': 'semantic',
                            'confidence': similarity_score,
                            'timestamp': memory_entry.timestamp.isoformat() if hasattr(memory_entry, 'timestamp') else None,
                            'similarity_score': similarity_score
                        }
                        
                        self.logger.info(f"Semantic cache hit (score: {similarity_score:.3f}) for query: {query[:50]}...")
                        return response_data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return None
    
    def cache_response(self, query: str, response: str, context: Optional[str] = None, metadata: Optional[Dict] = None):
        """
        Cache an LLM response with semantic indexing.
        
        Args:
            query: The original query
            response: The LLM response to cache
            context: Optional context for the query
            metadata: Additional metadata to store
        """
        try:
            # Skip caching for very short queries or responses
            if len(query.strip()) < self.min_query_length or len(response.strip()) < 20:
                return
            
            query_hash = self._generate_query_hash(query, context)
            ttl = self._calculate_ttl(response)
            
            # Create cache entry
            cache_entry = CacheEntry(
                query=query,
                response=response,
                query_hash=query_hash,
                timestamp=datetime.now(),
                ttl=ttl,
                metadata=metadata or {},
                confidence_score=1.0,
                importance=self._calculate_importance(query, response)
            )
            
            # Store in local cache
            self.cache_store[query_hash] = cache_entry
            
            # Store in memory system for semantic search
            if self.memory_system and hasattr(self.memory_system, 'store_memory'):
                self._store_in_memory_system(query, response, context, metadata)
            
            # Update statistics
            self.hit_stats.record_cache_store()
            
            # Manage cache size
            self._manage_cache_size()
            
            # Persist to disk
            if self.enable_persistence:
                self._save_cache_to_disk()
            
            self.logger.info(f"Cached response for query: {query[:50]}... (TTL: {ttl}s)")
            
        except Exception as e:
            self.logger.error(f"Error caching response: {e}")
    
    def _store_in_memory_system(self, query: str, response: str, context: Optional[str], metadata: Optional[Dict]):
        """Store cache entry in the enhanced memory system for semantic search."""
        try:
            # Combine query and context for better semantic matching
            search_text = query
            if context:
                search_text = f"{context}\n{query}"
            
            # Prepare metadata for memory storage
            memory_metadata = {
                'cached_response': response,
                'original_query': query,
                'context': context,
                'cache_timestamp': datetime.now().isoformat(),
                **(metadata or {})
            }
            
            # Store in memory system with special role for cache queries
            self.memory_system.store_memory(
                content=search_text,
                role='cache_query',
                metadata=memory_metadata,
                importance=self._calculate_importance(query, response)
            )
            
        except Exception as e:
            self.logger.error(f"Error storing in memory system: {e}")
    
    def _generate_query_hash(self, query: str, context: Optional[str] = None) -> str:
        """Generate a unique hash for query and context combination."""
        combined = query
        if context:
            combined = f"{context}|{query}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def _calculate_ttl(self, response: str) -> int:
        """Calculate adaptive TTL based on response characteristics."""
        base_ttl = self.default_ttl
        
        # Longer TTL for factual content
        if self._is_factual_response(response):
            return base_ttl * self.factual_ttl_multiplier
        
        # Shorter TTL for time-sensitive content
        if self._is_time_sensitive(response):
            return base_ttl // 4  # 15 minutes
        
        # Longer TTL for code and technical content
        if self._is_technical_content(response):
            return base_ttl * 6  # 6 hours
        
        return base_ttl
    
    def _is_factual_response(self, response: str) -> bool:
        """Determine if response contains factual information."""
        factual_indicators = [
            'according to', 'research shows', 'studies indicate',
            'the definition of', 'is defined as', 'historically',
            'the capital of', 'was born in', 'invented in'
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in factual_indicators)
    
    def _is_time_sensitive(self, response: str) -> bool:
        """Determine if response contains time-sensitive information."""
        time_indicators = [
            'current', 'today', 'now', 'recent', 'latest',
            'this year', 'this month', 'as of', 'breaking',
            'stock price', 'exchange rate', 'weather'
        ]
        
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in time_indicators)
    
    def _is_technical_content(self, response: str) -> bool:
        """Determine if response contains technical/code content."""
        technical_indicators = [
            '```', 'def ', 'function', 'import ', 'class ',
            'algorithm', 'implementation', 'syntax',
            '#include', 'public class', 'SELECT ', 'FROM '
        ]
        
        return any(indicator in response for indicator in technical_indicators)
    
    def _calculate_importance(self, query: str, response: str) -> float:
        """Calculate importance score for cache entry."""
        importance = 0.5  # Base importance
        
        # Higher importance for longer, detailed responses
        if len(response) > 1000:
            importance += 0.2
        
        # Higher importance for technical content
        if self._is_technical_content(response):
            importance += 0.2
        
        # Higher importance for factual content
        if self._is_factual_response(response):
            importance += 0.1
        
        # Lower importance for time-sensitive content
        if self._is_time_sensitive(response):
            importance -= 0.1
        
        return min(1.0, max(0.1, importance))
    
    def _manage_cache_size(self):
        """Manage cache size by removing old or low-importance entries."""
        if len(self.cache_store) <= self.max_entries:
            return
        
        # Remove expired entries first
        expired_keys = [
            key for key, entry in self.cache_store.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache_store[key]
            self.hit_stats.record_eviction()
        
        # If still over limit, remove least important/least accessed entries
        if len(self.cache_store) > self.max_entries:
            entries_to_remove = len(self.cache_store) - self.max_entries
            
            # Sort by importance and access count (ascending)
            sorted_entries = sorted(
                self.cache_store.items(),
                key=lambda x: (x[1].importance, x[1].access_count)
            )
            
            for i in range(entries_to_remove):
                key = sorted_entries[i][0]
                del self.cache_store[key]
                self.hit_stats.record_eviction()
    
    def clear_cache(self):
        """Clear all cache entries."""
        self.cache_store.clear()
        self.query_embeddings.clear()
        self.logger.info("Cache cleared")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.hit_stats.get_stats()
        stats.update({
            'cache_size': len(self.cache_store),
            'max_entries': self.max_entries,
            'similarity_threshold': self.similarity_threshold,
            'default_ttl': self.default_ttl
        })
        return stats
    
    def optimize_cache(self):
        """Perform cache optimization and cleanup."""
        try:
            initial_size = len(self.cache_store)
            
            # Remove expired entries
            self._manage_cache_size()
            
            # Save optimized cache to disk
            if self.enable_persistence:
                self._save_cache_to_disk()
            
            final_size = len(self.cache_store)
            self.logger.info(f"Cache optimized: {initial_size} -> {final_size} entries")
            
        except Exception as e:
            self.logger.error(f"Error optimizing cache: {e}")
