"""
Web Content Cache with Intelligent Expiration
Advanced caching system for web content with adaptive expiration policies.
"""

import time
import hashlib
import gzip
import json
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, parse_qs
from pathlib import Path

from sources.logger import Logger
from sources.utility import timer_decorator
from sources.cache.cacheStats import CacheStats


@dataclass
class WebCacheEntry:
    """Cache entry for web content."""
    url: str
    content: bytes
    headers: Dict[str, str]
    content_type: Optional[str]
    timestamp: datetime
    ttl: int  # Time to live in seconds
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size: int = 0
    compressed: bool = False
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl)
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class WebCache:
    """
    Intelligent web content caching system with adaptive expiration policies.
    
    Features:
    - Adaptive TTL based on content type and URL patterns
    - Content compression for large files
    - LRU eviction when cache size limit reached
    - HTTP cache header analysis for better expiration
    - Access pattern tracking for optimization
    """
    
    def __init__(self, 
                 max_size_mb: int = 500,
                 compression_threshold: int = 1024,  # Compress content larger than 1KB
                 enable_compression: bool = True):
        """
        Initialize web cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            compression_threshold: Compress content larger than this size (bytes)
            enable_compression: Whether to enable content compression
        """
        self.max_size = max_size_mb * 1024 * 1024  # Convert MB to bytes
        self.compression_threshold = compression_threshold
        self.enable_compression = enable_compression
        
        # Cache storage
        self.cache_store: Dict[str, WebCacheEntry] = {}
        self.current_size = 0
        
        # Access patterns for optimization
        self.access_patterns: Dict[str, List[datetime]] = {}
        
        # Statistics and monitoring
        self.hit_stats = CacheStats("web_cache")
        self.logger = Logger("web_cache.log")
        
        # Configuration
        self.default_ttl = 600  # 10 minutes default
        self.api_ttl = 300  # 5 minutes for API responses
        self.html_ttl = 1800  # 30 minutes for HTML pages
        self.static_ttl = 86400  # 24 hours for static content
        self.dynamic_ttl = 300  # 5 minutes for dynamic content
        
        self.logger.info(f"Web cache initialized with {max_size_mb}MB limit")
    
    @timer_decorator
    def get_cached_content(self, url: str, headers: Optional[Dict[str, str]] = None) -> Optional[bytes]:
        """
        Get cached content for a URL.
        
        Args:
            url: The URL to retrieve content for
            headers: Optional HTTP headers for cache validation
            
        Returns:
            Cached content if found and fresh, None otherwise
        """
        try:
            start_time = time.time()
            cache_key = self._generate_cache_key(url, headers)
            
            if cache_key in self.cache_store:
                entry = self.cache_store[cache_key]
                
                # Check if content is still fresh
                if self._is_content_fresh(entry, headers):
                    entry.update_access()
                    self._update_access_pattern(cache_key)
                    
                    # Decompress if needed
                    content = self._decompress_content(entry.content, entry.compressed)
                    
                    response_time = time.time() - start_time
                    self.hit_stats.record_hit(response_time=response_time)
                    
                    self.logger.info(f"Cache hit for URL: {url[:100]}...")
                    return content
                else:
                    # Content expired or stale, remove from cache
                    self._remove_entry(cache_key)
                    self.logger.debug(f"Expired content removed for URL: {url[:100]}...")
            
            # Cache miss
            response_time = time.time() - start_time
            self.hit_stats.record_miss(response_time=response_time)
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving cached content: {e}")
            self.hit_stats.record_miss()
            return None
    
    def cache_content(self, 
                     url: str, 
                     content: bytes, 
                     headers: Optional[Dict[str, str]] = None, 
                     content_type: Optional[str] = None,
                     status_code: int = 200):
        """
        Cache web content with intelligent expiration.
        
        Args:
            url: The URL of the content
            content: The content to cache
            headers: HTTP response headers
            content_type: Content type of the response
            status_code: HTTP status code
        """
        try:
            # Only cache successful responses
            if status_code not in [200, 301, 302, 304]:
                return
            
            # Skip caching for very large content (>10MB)
            if len(content) > 10 * 1024 * 1024:
                self.logger.warning(f"Content too large to cache: {len(content)} bytes for {url[:100]}")
                return
            
            cache_key = self._generate_cache_key(url, headers)
            
            # Calculate adaptive TTL
            ttl = self._calculate_adaptive_ttl(url, content, content_type, headers)
            
            # Compress content if needed
            compressed_content, is_compressed = self._compress_if_needed(content)
            
            # Extract cache-relevant headers
            etag = headers.get('etag') or headers.get('ETag') if headers else None
            last_modified = headers.get('last-modified') or headers.get('Last-Modified') if headers else None
            
            # Create cache entry
            cache_entry = WebCacheEntry(
                url=url,
                content=compressed_content,
                headers=headers or {},
                content_type=content_type,
                timestamp=datetime.now(),
                ttl=ttl,
                size=len(compressed_content),
                compressed=is_compressed,
                etag=etag,
                last_modified=last_modified
            )
            
            # Ensure cache space
            self._ensure_cache_space(cache_entry.size)
            
            # Store in cache
            if cache_key in self.cache_store:
                # Update existing entry, adjust current size
                old_entry = self.cache_store[cache_key]
                self.current_size -= old_entry.size
            
            self.cache_store[cache_key] = cache_entry
            self.current_size += cache_entry.size
            
            # Update statistics
            self.hit_stats.record_cache_store()
            self.hit_stats.record_cache_size(len(self.cache_store))
            
            compression_info = f" (compressed {len(content)} -> {len(compressed_content)} bytes)" if is_compressed else ""
            self.logger.info(f"Cached content for URL: {url[:100]}... TTL: {ttl}s{compression_info}")
            
        except Exception as e:
            self.logger.error(f"Error caching content: {e}")
    
    def _generate_cache_key(self, url: str, headers: Optional[Dict[str, str]] = None) -> str:
        """Generate a unique cache key for URL and relevant headers."""
        # Normalize URL by removing some query parameters that don't affect content
        parsed_url = urlparse(url)
        
        # Remove common tracking parameters
        if parsed_url.query:
            query_params = parse_qs(parsed_url.query)
            # Remove tracking parameters
            tracking_params = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 'utm_term', 
                             'gclid', 'fbclid', '_ga', '_gid', 'ref']
            for param in tracking_params:
                query_params.pop(param, None)
            
            # Rebuild query string
            clean_query = '&'.join([f"{k}={v[0]}" for k, v in sorted(query_params.items())])
            clean_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"
            if clean_query:
                clean_url += f"?{clean_query}"
        else:
            clean_url = url
        
        # Include relevant headers in cache key
        header_key = ""
        if headers:
            relevant_headers = ['accept', 'accept-language', 'authorization']
            header_parts = []
            for header in relevant_headers:
                if header in headers:
                    header_parts.append(f"{header}:{headers[header]}")
            header_key = "|".join(header_parts)
        
        combined = f"{clean_url}|{header_key}"
        return hashlib.sha256(combined.encode('utf-8')).hexdigest()
    
    def _calculate_adaptive_ttl(self, 
                               url: str, 
                               content: bytes, 
                               content_type: Optional[str], 
                               headers: Optional[Dict[str, str]] = None) -> int:
        """Calculate adaptive TTL based on content analysis and HTTP headers."""
        
        # Check HTTP cache headers first
        if headers:
            # Check Cache-Control header
            cache_control = headers.get('cache-control') or headers.get('Cache-Control')
            if cache_control:
                max_age_match = re.search(r'max-age=(\d+)', cache_control)
                if max_age_match:
                    max_age = int(max_age_match.group(1))
                    # Use HTTP max-age but with reasonable limits
                    return min(max_age, 7 * 24 * 3600)  # Max 7 days
                
                if 'no-cache' in cache_control or 'no-store' in cache_control:
                    return 60  # Very short TTL for no-cache content
            
            # Check Expires header
            expires = headers.get('expires') or headers.get('Expires')
            if expires:
                try:
                    expires_time = datetime.strptime(expires, '%a, %d %b %Y %H:%M:%S %Z')
                    ttl = int((expires_time - datetime.now()).total_seconds())
                    if ttl > 0:
                        return min(ttl, 7 * 24 * 3600)  # Max 7 days
                except:
                    pass
        
        # Analyze URL patterns
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # API endpoints - short TTL
        if 'api.' in parsed_url.netloc or '/api/' in path or '/rest/' in path:
            return self.api_ttl
        
        # Static content - long TTL
        static_extensions = ['.css', '.js', '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico', 
                           '.woff', '.woff2', '.ttf', '.eot', '.pdf', '.zip']
        if any(path.endswith(ext) for ext in static_extensions):
            return self.static_ttl
        
        # Analyze content type
        if content_type:
            content_type_lower = content_type.lower()
            
            # HTML pages
            if 'text/html' in content_type_lower:
                return self.html_ttl
            
            # Images and media
            if any(t in content_type_lower for t in ['image/', 'video/', 'audio/']):
                return self.static_ttl
            
            # JSON/XML API responses
            if any(t in content_type_lower for t in ['application/json', 'application/xml', 'text/xml']):
                return self.api_ttl
            
            # CSS and JavaScript
            if any(t in content_type_lower for t in ['text/css', 'application/javascript', 'text/javascript']):
                return self.static_ttl
        
        # Analyze content patterns
        try:
            content_str = content.decode('utf-8', errors='ignore')
            
            # Check for dynamic content indicators
            dynamic_indicators = ['<script', 'javascript:', 'ajax', 'fetch(', 'XMLHttpRequest',
                                'real-time', 'live', 'current', 'now()', 'timestamp']
            if any(indicator in content_str.lower() for indicator in dynamic_indicators):
                return self.dynamic_ttl
            
            # Check for static content indicators
            static_indicators = ['<!DOCTYPE html>', '<html', 'cache-friendly', 'static']
            if any(indicator in content_str.lower() for indicator in static_indicators):
                return self.html_ttl
                
        except:
            pass  # If content can't be decoded, use default
        
        # Default TTL
        return self.default_ttl
    
    def _compress_if_needed(self, content: bytes) -> Tuple[bytes, bool]:
        """Compress content if it's large enough and compression is enabled."""
        if not self.enable_compression or len(content) < self.compression_threshold:
            return content, False
        
        try:
            compressed = gzip.compress(content)
            # Only use compression if it actually reduces size significantly
            if len(compressed) < len(content) * 0.9:  # At least 10% reduction
                return compressed, True
            else:
                return content, False
        except Exception as e:
            self.logger.warning(f"Compression failed: {e}")
            return content, False
    
    def _decompress_content(self, content: bytes, is_compressed: bool) -> bytes:
        """Decompress content if it was compressed."""
        if not is_compressed:
            return content
        
        try:
            return gzip.decompress(content)
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            return content  # Return compressed content as fallback
    
    def _is_content_fresh(self, entry: WebCacheEntry, headers: Optional[Dict[str, str]] = None) -> bool:
        """Check if cached content is still fresh."""
        # Check TTL expiration
        if entry.is_expired():
            return False
        
        # Check ETag if available
        if headers and entry.etag:
            if_none_match = headers.get('if-none-match') or headers.get('If-None-Match')
            if if_none_match and if_none_match == entry.etag:
                return True
        
        # Check Last-Modified if available
        if headers and entry.last_modified:
            if_modified_since = headers.get('if-modified-since') or headers.get('If-Modified-Since')
            if if_modified_since and if_modified_since == entry.last_modified:
                return True
        
        return True  # Content is fresh based on TTL
    
    def _update_access_pattern(self, cache_key: str):
        """Update access pattern for the cache key."""
        if cache_key not in self.access_patterns:
            self.access_patterns[cache_key] = []
        
        self.access_patterns[cache_key].append(datetime.now())
        
        # Keep only recent access times (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.access_patterns[cache_key] = [
            access_time for access_time in self.access_patterns[cache_key]
            if access_time > cutoff_time
        ]
    
    def _ensure_cache_space(self, needed_size: int):
        """Ensure there's enough space in cache for new content."""
        # If adding this content would exceed cache size, evict entries
        while self.current_size + needed_size > self.max_size and self.cache_store:
            self._evict_lru_entry()
    
    def _evict_lru_entry(self):
        """Evict the least recently used cache entry."""
        if not self.cache_store:
            return
        
        # Find LRU entry based on last access time and access count
        lru_key = None
        lru_score = float('inf')
        
        for key, entry in self.cache_store.items():
            # Calculate LRU score (lower is more likely to be evicted)
            last_access = entry.last_accessed or entry.timestamp
            time_since_access = (datetime.now() - last_access).total_seconds()
            
            # Consider access frequency and recency
            access_frequency = entry.access_count / max(1, time_since_access / 3600)  # accesses per hour
            score = time_since_access / max(1, access_frequency)
            
            if score < lru_score:
                lru_score = score
                lru_key = key
        
        if lru_key:
            self._remove_entry(lru_key)
            self.hit_stats.record_eviction()
    
    def _remove_entry(self, cache_key: str):
        """Remove an entry from the cache."""
        if cache_key in self.cache_store:
            entry = self.cache_store[cache_key]
            self.current_size -= entry.size
            del self.cache_store[cache_key]
            
            # Clean up access patterns
            self.access_patterns.pop(cache_key, None)
    
    def _is_static_content(self, url: str) -> bool:
        """Determine if URL points to static content."""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        static_patterns = [
            r'\.(css|js|png|jpg|jpeg|gif|svg|ico|woff|woff2|ttf|eot|pdf)$',
            r'/static/',
            r'/assets/',
            r'/media/',
            r'cdn\.',
            r'static\.',
        ]
        
        for pattern in static_patterns:
            if re.search(pattern, url.lower()):
                return True
        
        return False
    
    def clear_cache(self):
        """Clear all cached content."""
        self.cache_store.clear()
        self.access_patterns.clear()
        self.current_size = 0
        self.logger.info("Web cache cleared")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = self.hit_stats.get_stats()
        
        # Calculate compression ratio
        total_original_size = 0
        total_compressed_size = 0
        compressed_entries = 0
        
        for entry in self.cache_store.values():
            if entry.compressed:
                compressed_entries += 1
                # Estimate original size (not perfectly accurate but good enough)
                total_compressed_size += entry.size
                total_original_size += entry.size * 2  # Rough estimate
            else:
                total_original_size += entry.size
                total_compressed_size += entry.size
        
        compression_ratio = (total_compressed_size / total_original_size) if total_original_size > 0 else 1.0
        
        stats.update({
            'cache_size_entries': len(self.cache_store),
            'cache_size_bytes': self.current_size,
            'cache_size_mb': self.current_size / (1024 * 1024),
            'max_size_mb': self.max_size / (1024 * 1024),
            'cache_utilization': (self.current_size / self.max_size) if self.max_size > 0 else 0,
            'compressed_entries': compressed_entries,
            'compression_ratio': compression_ratio,
            'compression_savings_percent': (1 - compression_ratio) * 100,
            'ttl_defaults': {
                'api': self.api_ttl,
                'html': self.html_ttl,
                'static': self.static_ttl,
                'dynamic': self.dynamic_ttl,
                'default': self.default_ttl
            }
        })
        
        return stats
    
    def optimize_cache(self):
        """Perform cache optimization and cleanup."""
        try:
            initial_size = len(self.cache_store)
            initial_bytes = self.current_size
            
            # Remove expired entries
            expired_keys = [
                key for key, entry in self.cache_store.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
            
            # Clean up old access patterns
            cutoff_time = datetime.now() - timedelta(hours=24)
            for key in list(self.access_patterns.keys()):
                if key not in self.cache_store:
                    del self.access_patterns[key]
                else:
                    self.access_patterns[key] = [
                        access_time for access_time in self.access_patterns[key]
                        if access_time > cutoff_time
                    ]
            
            final_size = len(self.cache_store)
            final_bytes = self.current_size
            
            self.logger.info(f"Cache optimized: {initial_size} -> {final_size} entries, "
                           f"{initial_bytes} -> {final_bytes} bytes")
            
        except Exception as e:
            self.logger.error(f"Error optimizing cache: {e}")
    
    def get_access_patterns(self) -> Dict[str, Any]:
        """Get access pattern analysis."""
        if not self.access_patterns:
            return {'total_patterns': 0}
        
        # Analyze access frequencies
        hourly_accesses = {}
        total_accesses = 0
        
        for cache_key, accesses in self.access_patterns.items():
            for access_time in accesses:
                hour = access_time.hour
                hourly_accesses[hour] = hourly_accesses.get(hour, 0) + 1
                total_accesses += 1
        
        # Find peak hours
        peak_hour = max(hourly_accesses, key=hourly_accesses.get) if hourly_accesses else 0
        
        return {
            'total_patterns': len(self.access_patterns),
            'total_accesses_24h': total_accesses,
            'peak_hour': peak_hour,
            'hourly_distribution': hourly_accesses,
            'avg_accesses_per_url': total_accesses / len(self.access_patterns) if self.access_patterns else 0
        }
