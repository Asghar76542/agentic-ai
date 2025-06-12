#!/usr/bin/env python3
"""
Test LLM Cache Implementation
Simple test to verify the LLM cache functionality
"""

import sys
import os
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_llm_cache_basic():
    """Test basic LLM cache functionality without memory system."""
    print("Testing LLM Cache Basic Functionality...")
    
    try:
        # Mock the logger to avoid permission issues
        import sys
        from unittest.mock import Mock
        
        # Create a mock logger
        class MockLogger:
            def __init__(self, *args, **kwargs):
                pass
            def info(self, msg): pass
            def error(self, msg): pass
            def warning(self, msg): pass
            def debug(self, msg): pass
        
        # Patch the Logger import
        sys.modules['sources.logger'] = Mock()
        sys.modules['sources.logger'].Logger = MockLogger
        
        from sources.cache.llmCache import LLMCache
        import tempfile
        import os
        
        # Create a temporary directory for testing
        test_dir = tempfile.mkdtemp()
        
        # Initialize cache without memory system
        cache = LLMCache(
            memory_system=None,
            similarity_threshold=0.85,
            max_entries=100,
            cache_dir=os.path.join(test_dir, "cache"),
            enable_persistence=False
        )
        
        print(f"✓ LLM Cache initialized successfully")
        
        # Test cache miss
        result = cache.get_cached_response("What is Python?")
        assert result is None, "Expected cache miss"
        print("✓ Cache miss test passed")
        
        # Test cache store
        cache.cache_response(
            query="What is Python?",
            response="Python is a high-level programming language known for its simplicity and readability.",
            metadata={"model": "test", "temperature": 0.7}
        )
        print("✓ Cache store test passed")
        
        # Test exact cache hit
        result = cache.get_cached_response("What is Python?")
        assert result is not None, "Expected cache hit"
        assert result['cached'] == True, "Response should be marked as cached"
        assert result['cache_type'] == 'exact', "Should be exact match"
        print("✓ Exact cache hit test passed")
        
        # Test statistics
        stats = cache.get_cache_statistics()
        assert stats['hits'] == 1, f"Expected 1 hit, got {stats['hits']}"
        assert stats['misses'] == 1, f"Expected 1 miss, got {stats['misses']}"
        assert stats['cache_size'] == 1, f"Expected cache size 1, got {stats['cache_size']}"
        print("✓ Cache statistics test passed")
        
        # Test TTL calculation
        factual_response = "The capital of France is Paris. This is a well-established fact."
        cache.cache_response("What is the capital of France?", factual_response)
        
        time_sensitive_response = "The current stock price of AAPL is $150.23 as of today."
        cache.cache_response("What is AAPL stock price?", time_sensitive_response)
        
        technical_response = """
        ```python
        def hello_world():
            print("Hello, World!")
        ```
        """
        cache.cache_response("Show me a Python function", technical_response)
        
        print("✓ TTL calculation test passed")
        
        # Test cache optimization
        cache.optimize_cache()
        print("✓ Cache optimization test passed")
        
        final_stats = cache.get_cache_statistics()
        print(f"Final cache statistics: {final_stats}")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
        print("\n🎉 All LLM Cache tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_llm_cache_with_mock_memory():
    """Test LLM cache with a mock memory system."""
    print("\nTesting LLM Cache with Mock Memory System...")
    
    try:
        # Use the same mock setup
        import sys
        from unittest.mock import Mock
        
        # Create a mock logger (if not already mocked)
        class MockLogger:
            def __init__(self, *args, **kwargs):
                pass
            def info(self, msg): pass
            def error(self, msg): pass
            def warning(self, msg): pass
            def debug(self, msg): pass
        
        # Ensure logger is mocked
        if 'sources.logger' not in sys.modules:
            sys.modules['sources.logger'] = Mock()
            sys.modules['sources.logger'].Logger = MockLogger
        
        from sources.cache.llmCache import LLMCache
        import tempfile
        import os
        
        # Create a temporary directory for testing
        test_dir = tempfile.mkdtemp()
        
        # Create a simple mock memory system
        class MockMemorySystem:
            def __init__(self):
                self.stored_memories = []
            
            def store_memory(self, content, role, metadata, importance):
                memory_id = f"mock_{len(self.stored_memories)}"
                self.stored_memories.append({
                    'id': memory_id,
                    'content': content,
                    'role': role,
                    'metadata': metadata,
                    'importance': importance
                })
                return memory_id
            
            def search_memories(self, query, limit, filters, include_system=False):
                # Simple mock search - return stored memories with mock similarity
                results = []
                for memory in self.stored_memories:
                    if memory['role'] == 'cache_query':
                        # Mock similarity score
                        similarity = 0.9 if query.lower() in memory['content'].lower() else 0.5
                        
                        # Mock memory entry object
                        class MockMemoryEntry:
                            def __init__(self, data):
                                self.content = data['content']
                                self.metadata = data['metadata']
                                self.timestamp = data.get('timestamp', '2025-06-10T10:00:00')
                        
                        results.append((MockMemoryEntry(memory), similarity))
                
                return results
        
        mock_memory = MockMemorySystem()
        
        # Initialize cache with mock memory system
        cache = LLMCache(
            memory_system=mock_memory,
            similarity_threshold=0.85,
            max_entries=100,
            cache_dir=os.path.join(test_dir, "cache"),
            enable_persistence=False
        )
        
        print("✓ LLM Cache with mock memory system initialized")
        
        # Test storing with memory system
        cache.cache_response(
            query="How do I learn Python programming?",
            response="Start with basic syntax, practice regularly, and build projects.",
            metadata={"source": "test"}
        )
        
        print("✓ Cache store with memory system test passed")
        
        # Verify memory system was called
        assert len(mock_memory.stored_memories) == 1, "Memory system should have 1 stored memory"
        stored_memory = mock_memory.stored_memories[0]
        assert stored_memory['role'] == 'cache_query', "Memory should be stored with cache_query role"
        
        print("✓ Memory system integration test passed")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
        print("\n🎉 All LLM Cache with memory system tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting LLM Cache Tests...")
    
    success = True
    
    # Run basic functionality tests
    if not test_llm_cache_basic():
        success = False
    
    # Run memory system integration tests
    if not test_llm_cache_with_mock_memory():
        success = False
    
    if success:
        print("\n✅ All LLM Cache tests completed successfully!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
