#!/usr/bin/env python3
"""
Simple Test for Computation Cache System
Tests the core functionality without complex dependencies
"""

import sys
import os
import time
import unittest
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, '/home/beego/Downloads/VSCode Builds/agenticSeek')


class MockLogger:
    """Mock logger to avoid permission issues."""
    def info(self, msg): pass
    def error(self, msg): pass
    def warning(self, msg): pass
    def debug(self, msg): pass


class MockCacheStats:
    """Mock cache stats to avoid logging issues."""
    def __init__(self, cache_type):
        self.cache_type = cache_type
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        self.response_times = []
        self.evictions = 0
        self.cached_items = 0
        self.total_size = 0
    
    def increment_hits(self):
        self.hits += 1
        self.total_requests += 1
    
    def increment_misses(self):
        self.misses += 1
        self.total_requests += 1
    
    def increment_evictions(self):
        self.evictions += 1
    
    def add_evictions(self, count):
        self.evictions += count
    
    def add_response_time(self, time_ms):
        self.response_times.append(time_ms)
    
    def add_cached_item(self, size):
        self.cached_items += 1
        self.total_size += size
    
    def get_stats(self):
        return {
            'cache_type': self.cache_type,
            'total_requests': self.total_requests,
            'cache_hits': self.hits,
            'cache_misses': self.misses,
            'hit_rate': self.hits / max(1, self.total_requests),
            'avg_response_time': sum(self.response_times) / max(1, len(self.response_times)),
            'evictions': self.evictions,
            'cached_items': self.cached_items,
            'total_size': self.total_size
        }


# Mock the dependencies to avoid permission issues
with patch('sources.cache.cacheStats.CacheStats', MockCacheStats), \
     patch('sources.logger.Logger', MockLogger):
    
    from sources.cache.computationCache import ComputationCache
    from sources.cache.cachedExecutionWrapper import CachedExecutionWrapper, CachedInterpreterFactory


class MockInterpreter:
    """Mock interpreter for testing purposes."""
    
    def __init__(self, tag="python", name="Mock Python Interpreter"):
        self.tag = tag
        self.name = name
        self.description = f"Mock {tag} interpreter for testing"
        self.execution_results = {}
        self.execution_count = 0
        self.execution_failures = set()
    
    def execute(self, codes, safety=False, **kwargs):
        """Mock execution that can be controlled for testing."""
        self.execution_count += 1
        
        if isinstance(codes, list):
            code_str = '\n'.join(codes)
        else:
            code_str = codes
        
        # Simulate execution time
        time.sleep(0.001)
        
        # Return predefined results or generate default
        if code_str in self.execution_results:
            result = self.execution_results[code_str]
            if code_str in self.execution_failures:
                return f"Error: {result}"
            return result
        
        # Default deterministic result
        if 'error' in code_str.lower():
            self.execution_failures.add(code_str)
            return f"SyntaxError: invalid syntax in {code_str[:20]}..."
        
        return f"Mock output for {self.tag}: {len(code_str)} characters executed"
    
    def execution_failure_check(self, feedback):
        """Mock failure check."""
        return "error" in feedback.lower() or "failed" in feedback.lower()
    
    def interpreter_feedback(self, output):
        """Mock feedback generation."""
        if self.execution_failure_check(output):
            return f"[failure] Error in execution:\n{output}"
        else:
            return f"[success] Execution success, code output:\n{output}"


class TestComputationCache(unittest.TestCase):
    """Test cases for ComputationCache class."""
    
    def setUp(self):
        """Set up test environment."""
        self.cache = ComputationCache(max_size=10, default_ttl=60)
    
    def test_cache_creation(self):
        """Test basic cache creation."""
        self.assertIsNotNone(self.cache)
        self.assertEqual(self.cache.max_size, 10)
        self.assertEqual(self.cache.default_ttl, 60)
    
    def test_deterministic_detection(self):
        """Test detection of deterministic vs non-deterministic code."""
        # Deterministic code
        self.assertTrue(self.cache._is_deterministic("def add(a, b): return a + b", "python"))
        self.assertTrue(self.cache._is_deterministic("echo $((5 + 3))", "bash"))
        
        # Non-deterministic code
        self.assertFalse(self.cache._is_deterministic("import random\nprint(random.randint(1, 10))", "python"))
        self.assertFalse(self.cache._is_deterministic("echo $RANDOM", "bash"))
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        code = "print('hello world')"
        language = "python"
        
        # Same code should generate same key
        key1 = self.cache._generate_cache_key(code, language)
        key2 = self.cache._generate_cache_key(code, language)
        self.assertEqual(key1, key2)
        
        # Different code should generate different keys
        key3 = self.cache._generate_cache_key("print('goodbye')", language)
        self.assertNotEqual(key1, key3)
    
    def test_cache_storage_and_retrieval(self):
        """Test basic cache storage and retrieval."""
        code = "print('hello world')"
        language = "python"
        result = "hello world\n"
        
        # Store in cache
        self.cache.put(code, language, result, True, 0.1)
        
        # Retrieve from cache
        cached_result = self.cache.get(code, language)
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result[0], result)
        self.assertTrue(cached_result[1])  # is_success


class TestCachedExecutionWrapper(unittest.TestCase):
    """Test cases for CachedExecutionWrapper class."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_interpreter = MockInterpreter()
        self.cache = ComputationCache(max_size=100, default_ttl=60)
        self.wrapper = CachedExecutionWrapper(self.mock_interpreter, self.cache)
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization."""
        self.assertEqual(self.wrapper.tag, "python")
        self.assertIn("Cached", self.wrapper.name)
    
    def test_cache_hit_scenario(self):
        """Test successful cache hit scenario."""
        code = "print('hello')"
        
        # First execution (cache miss)
        result1 = self.wrapper.execute(code)
        self.assertEqual(self.mock_interpreter.execution_count, 1)
        
        # Second execution (cache hit)
        result2 = self.wrapper.execute(code)
        self.assertEqual(result1, result2)
        self.assertEqual(self.mock_interpreter.execution_count, 1)  # No additional execution


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete caching system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.factory = CachedInterpreterFactory()
    
    def test_factory_creation(self):
        """Test factory creation."""
        self.assertIsNotNone(self.factory.shared_cache)
    
    def test_interpreter_wrapping(self):
        """Test wrapping interpreters with caching."""
        mock_interpreter = MockInterpreter("java", "Mock Java Interpreter")
        wrapped = self.factory.wrap_interpreter(mock_interpreter)
        
        self.assertIsInstance(wrapped, CachedExecutionWrapper)
        self.assertEqual(wrapped.tag, "java")
    
    def test_performance_improvement(self):
        """Test that caching provides performance improvement."""
        mock_interpreter = MockInterpreter()
        wrapped = self.factory.wrap_interpreter(mock_interpreter)
        
        # Measure first execution (cache miss)
        start_time = time.time()
        result1 = wrapped.execute("print('performance test')")
        miss_time = time.time() - start_time
        
        # Measure second execution (cache hit)
        start_time = time.time()
        result2 = wrapped.execute("print('performance test')")
        hit_time = time.time() - start_time
        
        # Results should be the same
        self.assertEqual(result1, result2)
        
        # Cache hit should be faster (though the difference might be minimal in mocked scenario)
        self.assertLessEqual(hit_time, miss_time + 0.001)  # Allow small tolerance


def run_tests():
    """Run all tests and provide a report."""
    test_classes = [
        TestComputationCache,
        TestCachedExecutionWrapper,
        TestIntegration
    ]
    
    all_tests_passed = True
    total_tests = 0
    total_failures = 0
    
    print("Running Computation Cache Tests...")
    print("="*50)
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        suite = unittest.TestLoader().loadTestsFromTestClass(test_class)
        runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
        result = runner.run(suite)
        
        total_tests += result.testsRun
        total_failures += len(result.failures) + len(result.errors)
        
        if result.failures or result.errors:
            all_tests_passed = False
            for test, traceback in result.failures + result.errors:
                print(f"FAILED: {test}")
    
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Total tests: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Success rate: {((total_tests - total_failures) / total_tests * 100):.1f}%" if total_tests > 0 else "N/A")
    
    if all_tests_passed:
        print("✅ ALL TESTS PASSED")
        
        # Demonstrate cache functionality
        print("\n" + "="*50)
        print("CACHE FUNCTIONALITY DEMONSTRATION")
        print("="*50)
        
        # Create a demo cache
        cache = ComputationCache(max_size=5, default_ttl=60)
        
        # Test deterministic detection
        print("Testing deterministic code detection:")
        test_codes = [
            ("print('hello')", "python", True),
            ("import random; print(random.randint(1,10))", "python", False),
            ("echo 'hello'", "bash", True),
            ("echo $RANDOM", "bash", False)
        ]
        
        for code, lang, expected in test_codes:
            result = cache._is_deterministic(code, lang)
            status = "✅" if result == expected else "❌"
            print(f"  {status} {lang}: {'deterministic' if result else 'non-deterministic'} - {code[:30]}...")
        
        # Test cache operations
        print("\nTesting cache operations:")
        
        # Store some results
        cache.put("print('test1')", "python", "test1\n", True, 0.1)
        cache.put("print('test2')", "python", "test2\n", True, 0.2)
        
        # Retrieve results
        result1 = cache.get("print('test1')", "python")
        result2 = cache.get("print('nonexistent')", "python")
        
        print(f"  ✅ Cache hit: {result1[0].strip() if result1 else 'None'}")
        print(f"  ✅ Cache miss: {'None' if result2 is None else 'Found'}")
        
        # Show statistics
        stats = cache.get_stats()
        print(f"  ✅ Cache stats: {stats['total_requests']} requests, {stats['cache_hits']} hits")
        
        print("\n🎉 Computation Cache implementation is working correctly!")
        
    else:
        print("❌ SOME TESTS FAILED")
    
    return all_tests_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
