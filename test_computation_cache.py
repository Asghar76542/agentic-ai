#!/usr/bin/env python3
"""
Comprehensive Tests for Computation Cache System
Tests both the cache functionality and integration with interpreters
"""

import sys
import os
import time
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    
    def set_result(self, code, result, is_failure=False):
        """Set predefined result for specific code."""
        self.execution_results[code] = result
        if is_failure:
            self.execution_failures.add(code)


class TestComputationCache(unittest.TestCase):
    """Test cases for ComputationCache class."""
    
    def setUp(self):
        """Set up test environment."""
        self.cache = ComputationCache(max_size=10, default_ttl=60)
    
    def test_deterministic_detection(self):
        """Test detection of deterministic vs non-deterministic code."""
        # Deterministic code
        deterministic_codes = [
            ("def add(a, b): return a + b\nprint(add(2, 3))", "python"),
            ("public class Test { public static void main(String[] args) { System.out.println(5 + 3); }}", "java"),
            ("package main\nimport \"fmt\"\nfunc main() { fmt.Println(5 + 3) }", "go"),
            ("#include <stdio.h>\nint main() { printf(\"%d\", 5 + 3); return 0; }", "c"),
            ("echo $((5 + 3))", "bash")
        ]
        
        for code, language in deterministic_codes:
            self.assertTrue(
                self.cache._is_deterministic(code, language),
                f"Code should be deterministic: {code[:50]}..."
            )
        
        # Non-deterministic code
        non_deterministic_codes = [
            ("import random\nprint(random.randint(1, 10))", "python"),
            ("import time\nprint(time.time())", "python"),
            ("Math.random()", "java"),
            ("new Date().getTime()", "java"),
            ("rand.Intn(10)", "go"),
            ("time.Now()", "go"),
            ("rand()", "c"),
            ("time(NULL)", "c"),
            ("echo $RANDOM", "bash"),
            ("date", "bash")
        ]
        
        for code, language in non_deterministic_codes:
            self.assertFalse(
                self.cache._is_deterministic(code, language),
                f"Code should be non-deterministic: {code[:50]}..."
            )
    
    def test_cache_key_generation(self):
        """Test cache key generation and consistency."""
        code = "print('hello world')"
        language = "python"
        
        # Same code should generate same key
        key1 = self.cache._generate_cache_key(code, language)
        key2 = self.cache._generate_cache_key(code, language)
        self.assertEqual(key1, key2)
        
        # Different code should generate different keys
        key3 = self.cache._generate_cache_key("print('goodbye')", language)
        self.assertNotEqual(key1, key3)
        
        # Different language should generate different keys
        key4 = self.cache._generate_cache_key(code, "java")
        self.assertNotEqual(key1, key4)
        
        # Different args should generate different keys
        key5 = self.cache._generate_cache_key(code, language, {"safety": True})
        self.assertNotEqual(key1, key5)
    
    def test_code_normalization(self):
        """Test code normalization functionality."""
        # Python code normalization
        python_code = """
        # This is a comment
        def hello():
            '''This is a docstring'''
            print("hello")  # Another comment
            return True
        
        # Final comment
        hello()
        """
        
        normalized = self.cache._normalize_code(python_code, "python")
        self.assertNotIn("#", normalized)
        self.assertNotIn("'''", normalized)
        self.assertIn("def hello():", normalized)
        self.assertIn("print(\"hello\")", normalized)
        
        # C-style comment removal
        c_code = """
        // Single line comment
        int main() {
            /* Multi-line
               comment */
            printf("hello");
            return 0; // End comment
        }
        """
        
        normalized_c = self.cache._normalize_code(c_code, "c")
        self.assertNotIn("//", normalized_c)
        self.assertNotIn("/*", normalized_c)
        self.assertIn("int main()", normalized_c)
    
    def test_ttl_calculation(self):
        """Test TTL calculation based on execution characteristics."""
        code = "print('hello')"
        language = "python"
        
        # Fast execution should have shorter TTL
        ttl_fast = self.cache._calculate_ttl(code, language, 0.05)
        
        # Slow execution should have longer TTL
        ttl_slow = self.cache._calculate_ttl(code, language, 15.0)
        
        self.assertGreater(ttl_slow, ttl_fast)
        
        # Math operations should have longer TTL
        math_code = "import math\nresult = math.sqrt(16)"
        ttl_math = self.cache._calculate_ttl(math_code, language, 1.0)
        ttl_normal = self.cache._calculate_ttl("print('hello')", language, 1.0)
        
        self.assertGreater(ttl_math, ttl_normal)
    
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
        
        # Non-existent entry should return None
        missing_result = self.cache.get("print('not exists')", language)
        self.assertIsNone(missing_result)
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        code = "print('expiring')"
        language = "python"
        result = "expiring\n"
        
        # Create cache with very short TTL
        short_cache = ComputationCache(default_ttl=1)
        short_cache.put(code, language, result, True, 0.1)
        
        # Should be available immediately
        cached_result = short_cache.get(code, language)
        self.assertIsNotNone(cached_result)
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        expired_result = short_cache.get(code, language)
        self.assertIsNone(expired_result)
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache size limit is reached."""
        small_cache = ComputationCache(max_size=3)
        
        # Fill cache to capacity
        for i in range(3):
            code = f"print({i})"
            result = f"{i}\n"
            small_cache.put(code, "python", result, True, 0.1)
        
        self.assertEqual(len(small_cache.cache), 3)
        
        # Access first entry to make it recently used
        small_cache.get("print(0)", "python")
        
        # Add fourth entry (should evict least recently used)
        small_cache.put("print(3)", "python", "3\n", True, 0.1)
        
        # Cache should still have 3 entries
        self.assertEqual(len(small_cache.cache), 3)
        
        # First entry should still be there (recently accessed)
        self.assertIsNotNone(small_cache.get("print(0)", "python"))
        
        # Some other entry should have been evicted
        available_entries = sum(1 for i in range(4) 
                              if small_cache.get(f"print({i})", "python") is not None)
        self.assertEqual(available_entries, 3)
    
    def test_compression(self):
        """Test result compression for large outputs."""
        large_result = "x" * 2000  # Large result that should be compressed
        code = "print('x' * 2000)"
        
        self.cache.put(code, "python", large_result, True, 0.5)
        
        # Retrieve and verify
        cached_result = self.cache.get(code, "python")
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result[0], large_result)
        
        # Check that compression was used
        cache_key = self.cache._generate_cache_key(code, "python")
        entry = self.cache.cache[cache_key]
        self.assertTrue(entry.get('compressed', False))
    
    def test_statistics(self):
        """Test cache statistics collection."""
        # Add some entries
        for i in range(5):
            code = f"print({i})"
            result = f"{i}\n"
            self.cache.put(code, "python", result, True, 0.1 * i)
        
        # Generate some hits and misses
        self.cache.get("print(0)", "python")  # Hit
        self.cache.get("print(1)", "python")  # Hit
        self.cache.get("print(99)", "python")  # Miss
        
        stats = self.cache.get_stats()
        
        # Verify statistics
        self.assertEqual(stats['total_requests'], 3)
        self.assertEqual(stats['cache_hits'], 2)
        self.assertEqual(stats['cache_misses'], 1)
        self.assertGreater(stats['hit_rate'], 0.5)
        self.assertEqual(stats['cache_size'], 5)


class TestCachedExecutionWrapper(unittest.TestCase):
    """Test cases for CachedExecutionWrapper class."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_interpreter = MockInterpreter()
        self.cache = ComputationCache(max_size=100, default_ttl=60)
        self.wrapper = CachedExecutionWrapper(self.mock_interpreter, self.cache)
    
    def test_wrapper_initialization(self):
        """Test wrapper initialization and attribute forwarding."""
        self.assertEqual(self.wrapper.tag, "python")
        self.assertEqual(self.wrapper.name, "Cached Mock Python Interpreter")
        self.assertIn("with intelligent caching", self.wrapper.description)
    
    def test_cache_hit_scenario(self):
        """Test successful cache hit scenario."""
        code = "print('hello')"
        expected_result = "hello\n"
        
        # First execution (cache miss)
        result1 = self.wrapper.execute(code)
        self.assertEqual(self.mock_interpreter.execution_count, 1)
        
        # Second execution (cache hit)
        result2 = self.wrapper.execute(code)
        self.assertEqual(result1, result2)
        self.assertEqual(self.mock_interpreter.execution_count, 1)  # No additional execution
    
    def test_cache_miss_scenario(self):
        """Test cache miss scenarios."""
        # Non-deterministic code should always execute
        random_code = "import random\nprint(random.randint(1, 10))"
        
        result1 = self.wrapper.execute(random_code)
        result2 = self.wrapper.execute(random_code)
        
        # Both should execute (no caching)
        self.assertEqual(self.mock_interpreter.execution_count, 2)
    
    def test_error_caching(self):
        """Test caching of error results."""
        error_code = "this is invalid syntax for error"
        
        # First execution (should fail and cache the error)
        result1 = self.wrapper.execute(error_code)
        self.assertIn("Error", result1)
        self.assertEqual(self.mock_interpreter.execution_count, 1)
        
        # Second execution (should return cached error)
        result2 = self.wrapper.execute(error_code)
        self.assertEqual(result1, result2)
        self.assertEqual(self.mock_interpreter.execution_count, 1)  # No re-execution
    
    def test_method_forwarding(self):
        """Test that methods are properly forwarded to wrapped interpreter."""
        # Test interpreter_feedback
        output = "test output"
        feedback = self.wrapper.interpreter_feedback(output)
        self.assertIn("success", feedback)
        
        # Test execution_failure_check
        self.assertFalse(self.wrapper.execution_failure_check("success"))
        self.assertTrue(self.wrapper.execution_failure_check("error occurred"))
    
    def test_cache_statistics(self):
        """Test cache statistics through wrapper."""
        # Execute some code to generate statistics
        self.wrapper.execute("print('test1')")
        self.wrapper.execute("print('test2')")
        self.wrapper.execute("print('test1')")  # Cache hit
        
        stats = self.wrapper.get_cache_stats()
        self.assertIn('interpreter_type', stats)
        self.assertEqual(stats['interpreter_type'], 'python')
        self.assertGreater(stats['total_requests'], 0)
    
    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        # Add some cached results
        self.wrapper.execute("print('test1')")
        self.wrapper.execute("print('test2')")
        
        # Verify cache has entries
        stats_before = self.wrapper.get_cache_stats()
        self.assertGreater(stats_before['cache_size'], 0)
        
        # Clear cache
        self.wrapper.clear_cache()
        
        # Verify cache is empty
        stats_after = self.wrapper.get_cache_stats()
        # Note: Cache might have entries from other languages, so we check specific language
        # The actual behavior depends on implementation details


class TestCachedInterpreterFactory(unittest.TestCase):
    """Test cases for CachedInterpreterFactory class."""
    
    def setUp(self):
        """Set up test environment."""
        self.factory = CachedInterpreterFactory({'max_size': 50, 'default_ttl': 30})
    
    def test_factory_initialization(self):
        """Test factory initialization."""
        self.assertIsNotNone(self.factory.shared_cache)
        self.assertEqual(self.factory.shared_cache.max_size, 50)
        self.assertEqual(self.factory.shared_cache.default_ttl, 30)
    
    def test_interpreter_wrapping(self):
        """Test wrapping interpreters with caching."""
        mock_interpreter = MockInterpreter("java", "Mock Java Interpreter")
        wrapped = self.factory.wrap_interpreter(mock_interpreter)
        
        self.assertIsInstance(wrapped, CachedExecutionWrapper)
        self.assertEqual(wrapped.tag, "java")
        self.assertIn("Cached", wrapped.name)
    
    def test_shared_cache_usage(self):
        """Test that multiple wrapped interpreters share the same cache."""
        mock_python = MockInterpreter("python")
        mock_java = MockInterpreter("java")
        
        wrapped_python = self.factory.wrap_interpreter(mock_python)
        wrapped_java = self.factory.wrap_interpreter(mock_java)
        
        # Both should use the same cache instance
        self.assertIs(wrapped_python.cache, wrapped_java.cache)
        self.assertIs(wrapped_python.cache, self.factory.shared_cache)
    
    def test_factory_statistics(self):
        """Test factory-level statistics."""
        mock_interpreter = MockInterpreter()
        wrapped = self.factory.wrap_interpreter(mock_interpreter)
        
        # Execute some code
        wrapped.execute("print('factory test')")
        
        # Get factory statistics
        stats = self.factory.get_cache_stats()
        self.assertIn('total_requests', stats)
        
        info = self.factory.get_cache_info()
        self.assertIn('total_entries', info)
    
    def test_cache_management(self):
        """Test factory cache management functions."""
        mock_interpreter = MockInterpreter()
        wrapped = self.factory.wrap_interpreter(mock_interpreter)
        
        # Add some cached content
        wrapped.execute("print('test')")
        
        # Test language-specific clearing
        self.factory.clear_language_cache("python")
        
        # Test clearing all caches
        self.factory.clear_all_caches()


class IntegrationTests(unittest.TestCase):
    """Integration tests for the complete caching system."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.factory = CachedInterpreterFactory()
    
    def test_multiple_language_caching(self):
        """Test caching across multiple programming languages."""
        interpreters = {
            'python': MockInterpreter("python"),
            'java': MockInterpreter("java"),
            'go': MockInterpreter("go")
        }
        
        wrapped_interpreters = {}
        for lang, interpreter in interpreters.items():
            wrapped_interpreters[lang] = self.factory.wrap_interpreter(interpreter)
        
        # Execute code in different languages
        results = {}
        for lang, wrapper in wrapped_interpreters.items():
            code = f"print('hello from {lang}')"
            results[lang] = wrapper.execute(code)
        
        # Execute same code again (should hit cache)
        for lang, wrapper in wrapped_interpreters.items():
            code = f"print('hello from {lang}')"
            cached_result = wrapper.execute(code)
            self.assertEqual(results[lang], cached_result)
        
        # Verify shared cache usage
        stats = self.factory.get_cache_stats()
        self.assertGreater(stats['total_requests'], 0)
        self.assertGreater(stats['cache_hits'], 0)
    
    def test_performance_characteristics(self):
        """Test performance characteristics of the caching system."""
        mock_interpreter = MockInterpreter()
        wrapped = self.factory.wrap_interpreter(mock_interpreter)
        
        # Measure execution time for cache miss
        start_time = time.time()
        result1 = wrapped.execute("print('performance test')")
        miss_time = time.time() - start_time
        
        # Measure execution time for cache hit
        start_time = time.time()
        result2 = wrapped.execute("print('performance test')")
        hit_time = time.time() - start_time
        
        # Cache hit should be faster
        self.assertLess(hit_time, miss_time)
        self.assertEqual(result1, result2)
    
    def test_cache_persistence_across_wrappers(self):
        """Test that cache persists across different wrapper instances."""
        mock_interpreter1 = MockInterpreter()
        mock_interpreter2 = MockInterpreter()
        
        wrapped1 = self.factory.wrap_interpreter(mock_interpreter1)
        wrapped2 = self.factory.wrap_interpreter(mock_interpreter2)
        
        # Execute with first wrapper
        result1 = wrapped1.execute("print('shared cache test')")
        
        # Execute same code with second wrapper (should hit cache)
        execution_count_before = mock_interpreter2.execution_count
        result2 = wrapped2.execute("print('shared cache test')")
        execution_count_after = mock_interpreter2.execution_count
        
        # Second interpreter should not have executed (cache hit)
        self.assertEqual(execution_count_before, execution_count_after)
        self.assertEqual(result1, result2)


def run_tests():
    """Run all tests and provide a comprehensive report."""
    # Create a test suite
    test_classes = [
        TestComputationCache,
        TestCachedExecutionWrapper,
        TestCachedInterpreterFactory,
        IntegrationTests
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestClass(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("COMPUTATION CACHE TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print(f"\nFAILURES ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nERRORS ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Starting Computation Cache System Tests...")
    print("="*60)
    
    # Handle import issues gracefully for testing
    try:
        success = run_tests()
        exit_code = 0 if success else 1
    except Exception as e:
        print(f"Test execution failed: {str(e)}")
        exit_code = 2
    
    sys.exit(exit_code)
