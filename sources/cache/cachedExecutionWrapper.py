"""
Cached Execution Wrapper for Code Interpreters
Provides transparent caching for all code execution operations
"""

import time
from typing import Dict, Any, Optional, Tuple
from sources.cache.computationCache import ComputationCache
from sources.logger import Logger

class CachedExecutionWrapper:
    """
    Wrapper that adds caching capabilities to any code interpreter tool.
    Maintains compatibility with existing Tools interface.
    """
    
    def __init__(self, interpreter, cache: Optional[ComputationCache] = None):
        """
        Initialize cached wrapper.
        
        Args:
            interpreter: The original interpreter tool instance
            cache: ComputationCache instance (creates one if None)
        """
        self.interpreter = interpreter
        self.cache = cache or ComputationCache()
        self.logger = Logger(f"cached_{interpreter.tag}_interpreter.log")
        
        # Copy essential attributes from wrapped interpreter
        self.tag = interpreter.tag
        self.name = f"Cached {interpreter.name}"
        self.description = f"{interpreter.description} (with intelligent caching)"
        
        # Forward other attributes
        for attr in ['work_dir', 'safe_mode', 'allow_language_exec_bash']:
            if hasattr(interpreter, attr):
                setattr(self, attr, getattr(interpreter, attr))
    
    def execute(self, codes, safety=False, **kwargs) -> str:
        """
        Execute code with caching. Maintains same interface as original interpreters.
        
        Args:
            codes: Code to execute (string or list of strings)
            safety: Safety check flag
            **kwargs: Additional arguments passed to original interpreter
            
        Returns:
            Execution result string
        """
        # Normalize code input
        if isinstance(codes, list):
            code_str = '\n'.join(codes)
        else:
            code_str = codes
        
        # Prepare execution arguments for cache key
        exec_args = {
            'safety': safety,
            **kwargs
        }
        
        # Try to get from cache first
        start_time = time.time()
        cached_result = self.cache.get(code_str, self.interpreter.tag, exec_args)
        
        if cached_result is not None:
            result, is_success = cached_result
            cache_time = time.time() - start_time
            
            self.logger.info(f"Cache HIT for {self.interpreter.tag} code (length: {len(code_str)}) "
                           f"in {cache_time:.4f}s")
            
            # If cached result was a failure, we might want to retry occasionally
            if not is_success and self._should_retry_failed_cache(code_str):
                self.logger.info("Retrying failed cached operation")
            else:
                return result
        
        # Cache miss - execute normally
        self.logger.info(f"Cache MISS for {self.interpreter.tag} code (length: {len(code_str)})")
        
        execution_start = time.time()
        
        try:
            # Execute using original interpreter
            result = self.interpreter.execute(codes, safety, **kwargs)
            execution_time = time.time() - execution_start
            
            # Determine if execution was successful
            is_success = not self.interpreter.execution_failure_check(result)
            
            # Store in cache
            self.cache.put(
                code=code_str,
                language=self.interpreter.tag,
                result=result,
                is_success=is_success,
                execution_time=execution_time,
                args=exec_args
            )
            
            self.logger.info(f"Executed {self.interpreter.tag} code in {execution_time:.4f}s "
                           f"(success: {is_success})")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - execution_start
            error_result = f"Execution failed: {str(e)}"
            
            # Cache failures too (with shorter TTL)
            self.cache.put(
                code=code_str,
                language=self.interpreter.tag,
                result=error_result,
                is_success=False,
                execution_time=execution_time,
                args=exec_args
            )
            
            self.logger.error(f"Failed to execute {self.interpreter.tag} code: {str(e)}")
            return error_result
    
    def _should_retry_failed_cache(self, code: str) -> bool:
        """
        Determine if a failed cached result should be retried.
        
        Args:
            code: The code that previously failed
            
        Returns:
            True if should retry, False if should use cached failure
        """
        # Retry import errors occasionally (dependencies might have been installed)
        if any(keyword in code.lower() for keyword in ['import ', 'from ', 'require']):
            return True
        
        # Retry file operations (files might exist now)
        if any(keyword in code.lower() for keyword in ['open(', 'file', 'read', 'write']):
            return True
        
        # Don't retry syntax errors or other fundamental issues
        return False
    
    def interpreter_feedback(self, output: str) -> str:
        """Forward to original interpreter's feedback method."""
        return self.interpreter.interpreter_feedback(output)
    
    def execution_failure_check(self, feedback: str) -> bool:
        """Forward to original interpreter's failure check method."""
        return self.interpreter.execution_failure_check(feedback)
    
    def load_exec_block(self, text: str):
        """Forward to original interpreter's block loading method if it exists."""
        if hasattr(self.interpreter, 'load_exec_block'):
            return self.interpreter.load_exec_block(text)
        # Default implementation for tools that don't have this method
        return [], None
    
    def save_block(self, codes, save_path):
        """Forward to original interpreter's save method if it exists."""
        if hasattr(self.interpreter, 'save_block'):
            return self.interpreter.save_block(codes, save_path)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for this interpreter."""
        stats = self.cache.get_stats()
        stats['interpreter_type'] = self.interpreter.tag
        stats['interpreter_name'] = self.interpreter.name
        return stats
    
    def clear_cache(self):
        """Clear all cached results for this interpreter."""
        self.cache.invalidate_language(self.interpreter.tag)
        self.logger.info(f"Cleared cache for {self.interpreter.tag} interpreter")
    
    def __getattr__(self, name):
        """Forward any other attribute access to the wrapped interpreter."""
        return getattr(self.interpreter, name)


class CachedInterpreterFactory:
    """
    Factory class for creating cached versions of interpreters.
    Manages shared cache instance across all interpreters.
    """
    
    def __init__(self, cache_config: Optional[Dict[str, Any]] = None):
        """
        Initialize factory with shared cache.
        
        Args:
            cache_config: Configuration for the shared cache
        """
        cache_config = cache_config or {}
        self.shared_cache = ComputationCache(
            max_size=cache_config.get('max_size', 1000),
            default_ttl=cache_config.get('default_ttl', 3600)
        )
        self.logger = Logger("cached_interpreter_factory.log")
    
    def wrap_interpreter(self, interpreter):
        """
        Wrap an interpreter with caching capabilities.
        
        Args:
            interpreter: Original interpreter instance
            
        Returns:
            CachedExecutionWrapper instance
        """
        cached_interpreter = CachedExecutionWrapper(interpreter, self.shared_cache)
        self.logger.info(f"Created cached wrapper for {interpreter.tag} interpreter")
        return cached_interpreter
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics for the shared cache."""
        return self.shared_cache.get_stats()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information."""
        return self.shared_cache.get_cache_info()
    
    def clear_all_caches(self):
        """Clear all cached results."""
        self.shared_cache.clear()
        self.logger.info("Cleared all computation caches")
    
    def clear_language_cache(self, language: str):
        """Clear cached results for a specific language."""
        self.shared_cache.invalidate_language(language)
        self.logger.info(f"Cleared cache for {language} language")


# Global factory instance for easy access
_cached_factory = None

def get_cached_factory() -> CachedInterpreterFactory:
    """Get the global cached interpreter factory instance."""
    global _cached_factory
    if _cached_factory is None:
        _cached_factory = CachedInterpreterFactory()
    return _cached_factory

def create_cached_interpreter(interpreter):
    """
    Convenience function to create a cached version of an interpreter.
    
    Args:
        interpreter: Original interpreter instance
        
    Returns:
        CachedExecutionWrapper instance
    """
    return get_cached_factory().wrap_interpreter(interpreter)


# Integration helper functions
def enable_computation_caching():
    """
    Enable computation caching for all interpreters in the system.
    This function can be called during system initialization.
    """
    try:
        # Import all interpreter classes
        from sources.tools.PyInterpreter import PyInterpreter
        from sources.tools.JavaInterpreter import JavaInterpreter
        from sources.tools.GoInterpreter import GoInterpreter
        from sources.tools.C_Interpreter import CInterpreter
        from sources.tools.BashInterpreter import BashInterpreter
        
        factory = get_cached_factory()
        
        # Create a registry of cached interpreters
        cached_interpreters = {
            'python': factory.wrap_interpreter(PyInterpreter()),
            'java': factory.wrap_interpreter(JavaInterpreter()),
            'go': factory.wrap_interpreter(GoInterpreter()),
            'c': factory.wrap_interpreter(CInterpreter()),
            'bash': factory.wrap_interpreter(BashInterpreter())
        }
        
        logger = Logger("computation_cache_init.log")
        logger.info("Computation caching enabled for all interpreters")
        
        return cached_interpreters
        
    except ImportError as e:
        logger = Logger("computation_cache_init.log")
        logger.error(f"Failed to enable computation caching: {str(e)}")
        return {}

def get_computation_cache_stats() -> Dict[str, Any]:
    """Get comprehensive statistics for computation caching."""
    return get_cached_factory().get_cache_stats()

def get_computation_cache_info() -> Dict[str, Any]:
    """Get detailed information about computation cache contents."""
    return get_cached_factory().get_cache_info()
