"""
LLM Provider Cache Integration
Integrates the LLM response cache with the existing LLM provider system
"""

import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple

# Mock dependencies for testing
try:
    from sources.cache.llmCache import LLMCache
    from sources.cache.unifiedCacheManager import get_cache_manager
    from sources.logger import Logger
except ImportError:
    # Mock classes for testing
    class LLMCache:
        def __init__(self, *args, **kwargs): 
            self.cache = {}
        def get(self, *args): return None
        def put(self, *args): pass
        def get_stats(self): return {'cache_hits': 0, 'cache_misses': 0}
    
    class Logger:
        def __init__(self, name): pass
        def info(self, msg): pass
        def warning(self, msg): pass
        def error(self, msg): pass
    
    def get_cache_manager():
        class MockManager:
            def get_cache(self, name): return LLMCache()
        return MockManager()


class CachedLLMProvider:
    """
    Wrapper for the LLM Provider that adds intelligent caching capabilities.
    Maintains compatibility with the existing Provider interface.
    """
    
    def __init__(self, original_provider, cache: Optional[LLMCache] = None):
        """
        Initialize cached LLM provider wrapper.
        
        Args:
            original_provider: The original Provider instance
            cache: Optional LLMCache instance (gets from unified manager if None)
        """
        self.original_provider = original_provider
        self.logger = Logger("cached_llm_provider.log")
        
        # Get cache instance from unified manager or use provided one
        if cache is None:
            try:
                cache_manager = get_cache_manager()
                self.cache = cache_manager.get_cache('llm')
                if self.cache is None:
                    self.cache = LLMCache()
                    self.logger.warning("Created new LLM cache instance")
            except Exception as e:
                self.logger.warning(f"Failed to get cache from manager: {e}")
                self.cache = LLMCache()
        else:
            self.cache = cache
        
        # Copy attributes from original provider
        for attr in ['provider_name', 'model', 'is_local', 'server_ip', 
                    'server_address', 'api_key', 'logger']:
            if hasattr(original_provider, attr):
                setattr(self, attr, getattr(original_provider, attr))
        
        self.logger.info(f"Cached LLM provider initialized for {self.provider_name}")
    
    def _generate_cache_key(self, history: List[Dict[str, str]], verbose: bool = False) -> str:
        """
        Generate a cache key for the conversation history.
        
        Args:
            history: Conversation history
            verbose: Verbose flag
            
        Returns:
            Cache key string
        """
        # Create a deterministic representation of the history
        cache_data = {
            'provider': self.provider_name,
            'model': self.model,
            'history': history,
            'verbose': verbose
        }
        
        # Convert to JSON string and hash
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()[:32]
    
    def _should_cache_request(self, history: List[Dict[str, str]]) -> bool:
        """
        Determine if a request should be cached based on conversation characteristics.
        
        Args:
            history: Conversation history
            
        Returns:
            True if should cache, False otherwise
        """
        # Don't cache if history is empty
        if not history:
            return False
        
        # Get the last user message
        last_message = None
        for msg in reversed(history):
            if msg.get('role') == 'user':
                last_message = msg.get('content', '')
                break
        
        if not last_message:
            return False
        
        # Don't cache time-sensitive queries
        time_indicators = [
            'current time', 'what time', 'today', 'now', 'latest',
            'recent', 'breaking news', 'live', 'real-time',
            'current weather', 'stock price', 'market data'
        ]
        
        if any(indicator in last_message.lower() for indicator in time_indicators):
            return False
        
        # Don't cache personal/contextual queries
        personal_indicators = [
            'my name', 'where am i', 'my location', 'remember when',
            'our conversation', 'you said earlier', 'continue from'
        ]
        
        if any(indicator in last_message.lower() for indicator in personal_indicators):
            return False
        
        # Don't cache very short queries (likely incomplete)
        if len(last_message.strip()) < 10:
            return False
        
        # Cache factual, educational, and computational queries
        cacheable_indicators = [
            'what is', 'how to', 'explain', 'definition', 'calculate',
            'example', 'tutorial', 'guide', 'algorithm', 'formula',
            'history of', 'compare', 'difference between', 'pros and cons'
        ]
        
        if any(indicator in last_message.lower() for indicator in cacheable_indicators):
            return True
        
        # Cache if conversation is reasonably long (likely educational/informational)
        if len(last_message) > 50:
            return True
        
        return False
    
    def _calculate_response_complexity(self, response: str) -> str:
        """
        Calculate response complexity for TTL determination.
        
        Args:
            response: LLM response
            
        Returns:
            Complexity level: 'simple', 'moderate', 'complex'
        """
        if len(response) < 200:
            return 'simple'
        elif len(response) < 1000:
            return 'moderate'
        else:
            return 'complex'
    
    def respond(self, history: List[Dict[str, str]], verbose: bool = True) -> str:
        """
        Generate response with caching. Maintains same interface as original Provider.
        
        Args:
            history: Conversation history
            verbose: Whether to print verbose output
            
        Returns:
            Generated response
        """
        start_time = time.time()
        
        # Check if we should cache this request
        should_cache = self._should_cache_request(history)
        
        if should_cache:
            # Try to get from cache
            cache_key = self._generate_cache_key(history, verbose)
            cached_response = self.cache.get(cache_key)
            
            if cached_response is not None:
                cache_time = time.time() - start_time
                self.logger.info(f"Cache HIT for {self.provider_name} "
                               f"(history length: {len(history)}) in {cache_time:.4f}s")
                return cached_response
            
            self.logger.info(f"Cache MISS for {self.provider_name} "
                           f"(history length: {len(history)})")
        else:
            self.logger.info(f"Request not cached for {self.provider_name} "
                           f"(non-cacheable content)")
        
        # Generate response using original provider
        try:
            response = self.original_provider.respond(history, verbose)
            generation_time = time.time() - start_time
            
            # Store in cache if appropriate
            if should_cache and response:
                # Determine response type for TTL calculation
                complexity = self._calculate_response_complexity(response)
                
                # Calculate TTL based on response characteristics
                ttl_map = {
                    'simple': 1800,   # 30 minutes
                    'moderate': 3600, # 1 hour
                    'complex': 7200   # 2 hours
                }
                ttl = ttl_map.get(complexity, 3600)
                
                # Store in cache
                self.cache.put(
                    cache_key=cache_key,
                    response=response,
                    provider=self.provider_name,
                    model=self.model,
                    ttl=ttl,
                    metadata={
                        'history_length': len(history),
                        'response_length': len(response),
                        'complexity': complexity,
                        'generation_time': generation_time
                    }
                )
                
                self.logger.info(f"Cached response for {self.provider_name} "
                               f"(TTL: {ttl}s, complexity: {complexity})")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response with {self.provider_name}: {str(e)}")
            raise
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for this LLM provider."""
        stats = self.cache.get_stats()
        stats.update({
            'provider_name': self.provider_name,
            'model': self.model,
            'is_local': getattr(self, 'is_local', False)
        })
        return stats
    
    def clear_cache(self):
        """Clear the LLM cache."""
        self.cache.clear()
        self.logger.info(f"Cleared cache for {self.provider_name}")
    
    def get_model_name(self) -> str:
        """Forward to original provider."""
        return self.original_provider.get_model_name()
    
    def get_api_key(self, provider):
        """Forward to original provider."""
        return self.original_provider.get_api_key(provider)
    
    def __getattr__(self, name):
        """Forward any other attribute access to the original provider."""
        return getattr(self.original_provider, name)


def create_cached_llm_provider(original_provider, cache: Optional[LLMCache] = None) -> CachedLLMProvider:
    """
    Convenience function to create a cached version of an LLM provider.
    
    Args:
        original_provider: Original Provider instance
        cache: Optional LLM cache instance
        
    Returns:
        CachedLLMProvider instance
    """
    return CachedLLMProvider(original_provider, cache)


class LLMProviderCacheManager:
    """
    Manager for LLM provider caching across the system.
    Provides easy integration with existing agent code.
    """
    
    def __init__(self):
        self.cached_providers: Dict[str, CachedLLMProvider] = {}
        self.logger = Logger("llm_provider_cache_manager.log")
    
    def wrap_provider(self, provider, provider_id: Optional[str] = None) -> CachedLLMProvider:
        """
        Wrap a provider with caching capabilities.
        
        Args:
            provider: Original Provider instance
            provider_id: Optional ID for tracking (uses provider_name if None)
            
        Returns:
            CachedLLMProvider instance
        """
        provider_id = provider_id or getattr(provider, 'provider_name', 'unknown')
        
        if provider_id in self.cached_providers:
            self.logger.info(f"Returning existing cached provider: {provider_id}")
            return self.cached_providers[provider_id]
        
        cached_provider = CachedLLMProvider(provider)
        self.cached_providers[provider_id] = cached_provider
        
        self.logger.info(f"Created new cached provider: {provider_id}")
        return cached_provider
    
    def get_provider(self, provider_id: str) -> Optional[CachedLLMProvider]:
        """Get a cached provider by ID."""
        return self.cached_providers.get(provider_id)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all cached providers."""
        return {
            provider_id: provider.get_cache_stats()
            for provider_id, provider in self.cached_providers.items()
        }
    
    def clear_all_caches(self):
        """Clear caches for all providers."""
        for provider_id, provider in self.cached_providers.items():
            provider.clear_cache()
            self.logger.info(f"Cleared cache for provider: {provider_id}")
    
    def clear_provider_cache(self, provider_id: str):
        """Clear cache for a specific provider."""
        if provider_id in self.cached_providers:
            self.cached_providers[provider_id].clear_cache()


# Global manager instance
_global_llm_cache_manager: Optional[LLMProviderCacheManager] = None

def get_llm_cache_manager() -> LLMProviderCacheManager:
    """Get the global LLM provider cache manager."""
    global _global_llm_cache_manager
    if _global_llm_cache_manager is None:
        _global_llm_cache_manager = LLMProviderCacheManager()
    return _global_llm_cache_manager

def enable_llm_provider_caching():
    """
    Enable LLM provider caching globally.
    This can be called during system initialization.
    """
    manager = get_llm_cache_manager()
    logger = Logger("llm_provider_cache_init.log")
    logger.info("LLM provider caching enabled globally")
    return manager
