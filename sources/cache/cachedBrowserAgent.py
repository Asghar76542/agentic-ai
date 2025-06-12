
"""
Cached Browser Agent Integration

This module provides a cached wrapper for the BrowserAgent that integrates
intelligent web content caching to improve performance and reduce redundant
web requests while maintaining full compatibility with the original interface.

Key Features:
- Web search result caching with semantic similarity matching
- Page content caching with URL normalization
- Navigation result caching for instant page retrieval
- Transparent integration maintaining BrowserAgent compatibility
- Intelligent cache invalidation based on content freshness
"""

import hashlib
import time
from typing import List, Tuple, Dict, Optional
from urllib.parse import urlparse, parse_qs, urlencode

from sources.agents.browser_agent import BrowserAgent, Action
from sources.cache.webCache import WebCache
from sources.cache.unifiedCacheManager import UnifiedCacheManager
from sources.utility import pretty_print


class CachedBrowserAgent(BrowserAgent):
    """
    Enhanced BrowserAgent with intelligent web content caching.
    
    Provides transparent caching of web search results, page content,
    and navigation data to improve performance while maintaining
    full compatibility with the original BrowserAgent interface.
    """
    
    def __init__(self, name, prompt_path, provider, verbose=False, browser=None, 
                 cache_manager: Optional[UnifiedCacheManager] = None):
        """
        Initialize cached browser agent.
        
        Args:
            name: Agent name
            prompt_path: Path to agent prompt file
            provider: LLM provider instance
            verbose: Enable verbose logging
            browser: Browser instance
            cache_manager: Optional unified cache manager
        """
        super().__init__(name, prompt_path, provider, verbose, browser)
        
        # Initialize web cache
        self.web_cache = WebCache(
            max_size=1000,  # Cache up to 1000 web pages
            default_ttl=3600  # 1 hour default TTL
        )
        
        # Use provided cache manager or create new one
        self.cache_manager = cache_manager or UnifiedCacheManager()
        
        # Register web cache with unified manager
        self.cache_manager.register_cache("web", self.web_cache)
        
        # Cache hit statistics
        self.cache_stats = {
            "search_hits": 0,
            "search_misses": 0,
            "page_hits": 0,
            "page_misses": 0,
            "navigation_hits": 0,
            "navigation_misses": 0
        }
        
        if verbose:
            pretty_print("Cached Browser Agent initialized with web caching", color="success")
    
    def _normalize_search_query(self, query: str) -> str:
        """
        Normalize search query for cache key generation.
        
        Args:
            query: Raw search query
            
        Returns:
            Normalized query string
        """
        # Remove common stop words and normalize
        normalized = query.lower().strip()
        
        # Remove common search operators that don't change semantic meaning
        normalized = normalized.replace('"', '').replace("'", '')
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _normalize_url(self, url: str) -> str:
        """
        Normalize URL for cache key generation.
        
        Args:
            url: Raw URL
            
        Returns:
            Normalized URL string
        """
        try:
            parsed = urlparse(url)
            
            # Remove tracking parameters
            tracking_params = {
                'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
                'gclid', 'fbclid', 'ref', 'source', 'campaign_id', 'ad_id'
            }
            
            # Parse and filter query parameters
            query_params = parse_qs(parsed.query)
            filtered_params = {
                k: v for k, v in query_params.items() 
                if k.lower() not in tracking_params
            }
            
            # Rebuild URL without tracking parameters
            clean_query = urlencode(filtered_params, doseq=True)
            normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            
            if clean_query:
                normalized_url += f"?{clean_query}"
                
            return normalized_url
            
        except Exception:
            # Fallback to original URL if parsing fails
            return url
    
    def _generate_search_cache_key(self, query: str) -> str:
        """
        Generate cache key for search results.
        
        Args:
            query: Search query
            
        Returns:
            Cache key string
        """
        normalized_query = self._normalize_search_query(query)
        query_hash = hashlib.sha256(normalized_query.encode()).hexdigest()[:16]
        return f"search_{query_hash}"
    
    def _generate_page_cache_key(self, url: str) -> str:
        """
        Generate cache key for page content.
        
        Args:
            url: Page URL
            
        Returns:
            Cache key string
        """
        normalized_url = self._normalize_url(url)
        url_hash = hashlib.sha256(normalized_url.encode()).hexdigest()[:16]
        return f"page_{url_hash}"
    
    def _get_search_ttl(self, query: str) -> int:
        """
        Calculate appropriate TTL for search results based on query type.
        
        Args:
            query: Search query
            
        Returns:
            TTL in seconds
        """
        query_lower = query.lower()
        
        # News/time-sensitive queries: shorter TTL
        if any(keyword in query_lower for keyword in [
            'news', 'today', 'recent', 'latest', 'current', 'breaking',
            '2024', '2025', 'yesterday', 'this week'
        ]):
            return 900  # 15 minutes
            
        # General informational queries: longer TTL
        if any(keyword in query_lower for keyword in [
            'how to', 'what is', 'tutorial', 'guide', 'definition',
            'history', 'biography'
        ]):
            return 7200  # 2 hours
            
        # Default TTL
        return 3600  # 1 hour
    
    def _get_page_ttl(self, url: str, content: str) -> int:
        """
        Calculate appropriate TTL for page content based on URL and content.
        
        Args:
            url: Page URL
            content: Page content
            
        Returns:
            TTL in seconds
        """
        url_lower = url.lower()
        content_lower = content.lower() if content else ""
        
        # News sites: shorter TTL
        if any(domain in url_lower for domain in [
            'news', 'reuters', 'bbc', 'cnn', 'techcrunch', 'wired'
        ]):
            return 1800  # 30 minutes
            
        # Documentation/reference sites: longer TTL
        if any(domain in url_lower for domain in [
            'docs.', 'developer.', 'reference', 'manual', 'wiki'
        ]) or any(keyword in content_lower for keyword in [
            'documentation', 'api reference', 'manual'
        ]):
            return 14400  # 4 hours
            
        # Default TTL
        return 3600  # 1 hour
    
    async def _cached_web_search(self, query: str) -> str:
        """
        Execute web search with caching.
        
        Args:
            query: Search query
            
        Returns:
            Search results string
        """
        cache_key = self._generate_search_cache_key(query)
        
        # Try to get from cache
        cached_result = self.web_cache.get(cache_key)
        if cached_result is not None:
            self.cache_stats["search_hits"] += 1
            if self.verbose:
                pretty_print(f"Cache hit for search: {query[:50]}...", color="success")
            return cached_result
        
        # Cache miss - execute search
        self.cache_stats["search_misses"] += 1
        if self.verbose:
            pretty_print(f"Cache miss for search: {query[:50]}...", color="warning")
        
        # Execute original search
        search_result = self.tools["web_search"].execute([query], False)
        
        # Cache the result with appropriate TTL
        ttl = self._get_search_ttl(query)
        self.web_cache.set(cache_key, search_result, ttl)
        
        return search_result
    
    def _cached_get_page_text(self, limit_to_model_ctx=False) -> str:
        """
        Get page text content with caching.
        
        Args:
            limit_to_model_ctx: Whether to limit text to model context
            
        Returns:
            Page text content
        """
        if not self.current_page:
            return self.get_page_text(limit_to_model_ctx)
        
        cache_key = self._generate_page_cache_key(self.current_page)
        
        # Try to get from cache
        cached_content = self.web_cache.get(cache_key)
        if cached_content is not None:
            self.cache_stats["page_hits"] += 1
            if self.verbose:
                pretty_print(f"Cache hit for page: {self.current_page[:50]}...", color="success")
            
            # Apply model context limiting if requested
            if limit_to_model_ctx:
                return self.memory.trim_text_to_max_ctx(cached_content)
            return cached_content
        
        # Cache miss - get page content
        self.cache_stats["page_misses"] += 1
        if self.verbose:
            pretty_print(f"Cache miss for page: {self.current_page[:50]}...", color="warning")
        
        # Get original page content
        page_text = super().get_page_text(limit_to_model_ctx=False)
        
        if page_text:
            # Cache the content with appropriate TTL
            ttl = self._get_page_ttl(self.current_page, page_text)
            self.web_cache.set(cache_key, page_text, ttl)
        
        # Apply model context limiting if requested
        if limit_to_model_ctx and page_text:
            return self.memory.trim_text_to_max_ctx(page_text)
        
        return page_text
    
    def _cached_navigation(self, url: str) -> Optional[Dict]:
        """
        Check if navigation result is cached.
        
        Args:
            url: Target URL
            
        Returns:
            Cached navigation data if available
        """
        cache_key = f"nav_{self._generate_page_cache_key(url)}"
        
        cached_nav = self.web_cache.get(cache_key)
        if cached_nav is not None:
            self.cache_stats["navigation_hits"] += 1
            if self.verbose:
                pretty_print(f"Cache hit for navigation: {url[:50]}...", color="success")
            return cached_nav
        
        self.cache_stats["navigation_misses"] += 1
        return None
    
    def _cache_navigation_result(self, url: str, success: bool, page_text: str = None):
        """
        Cache navigation result.
        
        Args:
            url: Target URL
            success: Whether navigation was successful
            page_text: Page content if successful
        """
        cache_key = f"nav_{self._generate_page_cache_key(url)}"
        
        nav_data = {
            "success": success,
            "page_text": page_text,
            "timestamp": time.time()
        }
        
        # Cache navigation result for 30 minutes
        self.web_cache.set(cache_key, nav_data, 1800)
    
    # Override parent methods to add caching
    
    def get_page_text(self, limit_to_model_ctx=False) -> str:
        """
        Override to add caching to page text retrieval.
        
        Args:
            limit_to_model_ctx: Whether to limit text to model context
            
        Returns:
            Page text content
        """
        return self._cached_get_page_text(limit_to_model_ctx)
    
    async def process(self, user_prompt: str, speech_module: type) -> Tuple[str, str]:
        """
        Override process method to add search result caching.
        
        Args:
            user_prompt: User's input query
            speech_module: Optional speech output module
            
        Returns:
            Tuple containing final answer and reasoning
        """
        complete = False

        # Generate search prompt (using parent method)
        from sources.utility import animate_thinking
        animate_thinking(f"Thinking...", color="status")
        mem_begin_idx = self.memory.push('user', self.search_prompt(user_prompt))
        ai_prompt, reasoning = await self.llm_request()
        
        if Action.REQUEST_EXIT.value in ai_prompt:
            pretty_print(f"Web agent requested exit.\n{reasoning}\n\n{ai_prompt}", color="failure")
            return ai_prompt, ""
        
        animate_thinking(f"Searching...", color="status")
        self.status_message = "Searching..."
        
        # Use cached web search instead of direct tool execution
        search_result_raw = await self._cached_web_search(ai_prompt)
        search_result = self.jsonify_search_results(search_result_raw)[:16]
        self.show_search_results(search_result)
        
        prompt = self.make_newsearch_prompt(user_prompt, search_result)
        unvisited = [None]
        
        # Continue with original navigation logic
        while not complete and len(unvisited) > 0 and not self.stop:
            self.memory.clear()
            unvisited = self.select_unvisited(search_result)
            answer, reasoning = await self.llm_decide(prompt, show_reasoning=False)
            
            if self.stop:
                pretty_print(f"Requested stop.", color="failure")
                break
                
            if self.last_answer == answer:
                prompt = self.stuck_prompt(user_prompt, unvisited)
                continue
                
            self.last_answer = answer
            pretty_print('▂'*32, color="status")

            extracted_form = self.extract_form(answer)
            if len(extracted_form) > 0:
                self.status_message = "Filling web form..."
                pretty_print(f"Filling inputs form...", color="status")
                fill_success = self.browser.fill_form(extracted_form)
                page_text = self.get_page_text(limit_to_model_ctx=True)
                answer = self.handle_update_prompt(user_prompt, page_text, fill_success)
                answer, reasoning = await self.llm_decide(prompt)

            if Action.FORM_FILLED.value in answer:
                pretty_print(f"Filled form. Handling page update.", color="status")
                page_text = self.get_page_text(limit_to_model_ctx=True)
                self.navigable_links = self.browser.get_navigable()
                prompt = self.make_navigation_prompt(user_prompt, page_text)
                continue

            links = self.parse_answer(answer)
            link = self.select_link(links)
            
            if link == self.current_page:
                pretty_print(f"Already visited {link}. Search callback.", color="status")
                prompt = self.make_newsearch_prompt(user_prompt, unvisited)
                self.search_history.append(link)
                continue

            if Action.REQUEST_EXIT.value in answer:
                self.status_message = "Exiting web browser..."
                pretty_print(f"Agent requested exit.", color="status")
                complete = True
                break

            if (link == None and len(extracted_form) < 3) or Action.GO_BACK.value in answer or link in self.search_history:
                pretty_print(f"Going back to results. Still {len(unvisited)}", color="status")
                self.status_message = "Going back to search results..."
                request_prompt = user_prompt
                if link is None:
                    request_prompt += f"\nYou previously chosen:\n{self.last_answer} but the website is unavailable. Consider other options."
                prompt = self.make_newsearch_prompt(request_prompt, unvisited)
                self.search_history.append(link)
                self.current_page = link
                continue

            # Check for cached navigation result
            cached_nav = self._cached_navigation(link)
            if cached_nav and cached_nav["success"] and cached_nav.get("page_text"):
                # Use cached navigation
                pretty_print(f"Using cached navigation to {link}", color="success")
                self.current_page = link
                self.search_history.append(link)
                self.navigable_links = self.browser.get_navigable() if hasattr(self.browser, 'get_navigable') else []
                prompt = self.make_navigation_prompt(user_prompt, cached_nav["page_text"])
                self.status_message = "Navigating (cached)..."
                continue

            # Perform actual navigation
            animate_thinking(f"Navigating to {link}", color="status")
            if speech_module: 
                speech_module.speak(f"Navigating to {link}")
                
            nav_ok = self.browser.go_to(link)
            self.search_history.append(link)
            
            if not nav_ok:
                pretty_print(f"Failed to navigate to {link}.", color="failure")
                self._cache_navigation_result(link, False)
                prompt = self.make_newsearch_prompt(user_prompt, unvisited)
                continue
                
            self.current_page = link
            page_text = self.get_page_text(limit_to_model_ctx=True)
            
            # Cache successful navigation
            self._cache_navigation_result(link, True, page_text)
            
            self.navigable_links = self.browser.get_navigable()
            prompt = self.make_navigation_prompt(user_prompt, page_text)
            self.status_message = "Navigating..."
            self.browser.screenshot()

        pretty_print("Exited navigation, starting to summarize finding...", color="status")
        prompt = self.conclude_prompt(user_prompt)
        mem_last_idx = self.memory.push('user', prompt)
        self.status_message = "Summarizing findings..."
        answer, reasoning = await self.llm_request()
        pretty_print(answer, color="output")
        self.status_message = "Ready"
        self.last_answer = answer
        
        # Log cache statistics if verbose
        if self.verbose:
            self._log_cache_statistics()
        
        return answer, reasoning
    
    def _log_cache_statistics(self):
        """Log cache performance statistics."""
        total_searches = self.cache_stats["search_hits"] + self.cache_stats["search_misses"]
        total_pages = self.cache_stats["page_hits"] + self.cache_stats["page_misses"]
        total_nav = self.cache_stats["navigation_hits"] + self.cache_stats["navigation_misses"]
        
        search_hit_rate = (self.cache_stats["search_hits"] / total_searches * 100) if total_searches > 0 else 0
        page_hit_rate = (self.cache_stats["page_hits"] / total_pages * 100) if total_pages > 0 else 0
        nav_hit_rate = (self.cache_stats["navigation_hits"] / total_nav * 100) if total_nav > 0 else 0
        
        stats_msg = f"""
        Cache Performance Statistics:
        - Search cache hit rate: {search_hit_rate:.1f}% ({self.cache_stats['search_hits']}/{total_searches})
        - Page cache hit rate: {page_hit_rate:.1f}% ({self.cache_stats['page_hits']}/{total_pages})
        - Navigation cache hit rate: {nav_hit_rate:.1f}% ({self.cache_stats['navigation_hits']}/{total_nav})
        - Web cache size: {len(self.web_cache.cache)} items
        """
        pretty_print(stats_msg, color="info")
    
    def get_cache_stats(self) -> Dict:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary containing cache performance metrics
        """
        web_stats = self.web_cache.get_stats()
        
        return {
            "search_cache": {
                "hits": self.cache_stats["search_hits"],
                "misses": self.cache_stats["search_misses"],
                "hit_rate": (self.cache_stats["search_hits"] / 
                           (self.cache_stats["search_hits"] + self.cache_stats["search_misses"]) * 100) 
                           if (self.cache_stats["search_hits"] + self.cache_stats["search_misses"]) > 0 else 0
            },
            "page_cache": {
                "hits": self.cache_stats["page_hits"],
                "misses": self.cache_stats["page_misses"],
                "hit_rate": (self.cache_stats["page_hits"] / 
                           (self.cache_stats["page_hits"] + self.cache_stats["page_misses"]) * 100) 
                           if (self.cache_stats["page_hits"] + self.cache_stats["page_misses"]) > 0 else 0
            },
            "navigation_cache": {
                "hits": self.cache_stats["navigation_hits"],
                "misses": self.cache_stats["navigation_misses"],
                "hit_rate": (self.cache_stats["navigation_hits"] / 
                           (self.cache_stats["navigation_hits"] + self.cache_stats["navigation_misses"]) * 100) 
                           if (self.cache_stats["navigation_hits"] + self.cache_stats["navigation_misses"]) > 0 else 0
            },
            "web_cache": web_stats
        }
    
    def clear_cache(self):
        """Clear all cached data."""
        self.web_cache.clear()
        
        # Reset statistics
        for key in self.cache_stats:
            self.cache_stats[key] = 0
        
        if self.verbose:
            pretty_print("All browser agent caches cleared", color="info")


# Convenience function to create cached browser agent
def create_cached_browser_agent(name: str, prompt_path: str, provider, 
                               verbose: bool = False, browser=None,
                               cache_manager: Optional[UnifiedCacheManager] = None) -> CachedBrowserAgent:
    """
    Create a cached browser agent instance.
    
    Args:
        name: Agent name
        prompt_path: Path to agent prompt file
        provider: LLM provider instance
        verbose: Enable verbose logging
        browser: Browser instance
        cache_manager: Optional unified cache manager
        
    Returns:
        CachedBrowserAgent instance
    """
    return CachedBrowserAgent(name, prompt_path, provider, verbose, browser, cache_manager)


if __name__ == "__main__":
    # Example usage
    print("Cached Browser Agent - Web Caching Integration")
    print("This module provides intelligent caching for browser agent operations")
