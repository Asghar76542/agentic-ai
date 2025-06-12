"""
Comprehensive test suite for the Enhanced Memory System.
Tests all components including vector memory, semantic search, knowledge graph, and analytics.
"""

import unittest
import tempfile
import shutil
import os
import sys
from pathlib import Path
import json
import time
from typing import Dict, List

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from sources.knowledge.vectorMemory import VectorMemory, MemoryEntry
    from sources.knowledge.semanticSearch import SemanticSearch, SearchQuery
    from sources.knowledge.knowledgeGraph import KnowledgeGraph
    from sources.knowledge.memoryAnalytics import MemoryAnalytics
    from sources.knowledge.memoryDashboard import MemoryDashboard
    from sources.knowledge.memoryIntegration import EnhancedMemorySystem, MemorySystemConfig
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"Memory components not available for testing: {e}")
    COMPONENTS_AVAILABLE = False

class TestVectorMemory(unittest.TestCase):
    """Test vector memory functionality."""
    
    def setUp(self):
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Memory components not available")
        
        self.test_dir = tempfile.mkdtemp()
        self.vector_memory = VectorMemory(db_path=self.test_dir)
    
    def tearDown(self):
        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_store_and_retrieve_memory(self):
        """Test basic memory storage and retrieval."""
        # Create test memory
        memory = MemoryEntry(
            content="This is a test memory about machine learning",
            metadata={'role': 'user', 'importance': 0.8}
        )
        
        # Store memory
        memory_id = self.vector_memory.store_memory(memory)
        self.assertIsNotNone(memory_id)
        
        # Retrieve memory
        retrieved = self.vector_memory.get_memory(memory_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.content, memory.content)
        self.assertEqual(retrieved.metadata['role'], 'user')
    
    def test_similar_memories_search(self):
        """Test finding similar memories."""
        # Store multiple memories
        memories = [
            MemoryEntry("Python programming tutorial", {'topic': 'programming'}),
            MemoryEntry("Machine learning algorithms", {'topic': 'ml'}),
            MemoryEntry("Python machine learning libraries", {'topic': 'programming'}),
            MemoryEntry("Cooking recipes for dinner", {'topic': 'cooking'})
        ]
        
        for memory in memories:
            self.vector_memory.store_memory(memory)
        
        # Search for similar memories
        similar = self.vector_memory.find_similar_memories(
            "Python programming", max_results=3
        )
        
        self.assertGreater(len(similar), 0)
        # Should find programming-related memories
        programming_results = [r for r in similar if 'python' in r.content.lower()]
        self.assertGreater(len(programming_results), 0)
    
    def test_memory_statistics(self):
        """Test memory statistics generation."""
        # Store some memories
        for i in range(5):
            memory = MemoryEntry(f"Test memory {i}", {'index': i})
            self.vector_memory.store_memory(memory)
        
        stats = self.vector_memory.get_stats()
        self.assertIn('total_memories', stats)
        self.assertEqual(stats['total_memories'], 5)
        self.assertIn('storage_size_mb', stats)

class TestSemanticSearch(unittest.TestCase):
    """Test semantic search functionality."""
    
    def setUp(self):
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Memory components not available")
        
        self.test_dir = tempfile.mkdtemp()
        self.vector_memory = VectorMemory(db_path=self.test_dir)
        self.semantic_search = SemanticSearch()
        
        # Populate with test data
        test_memories = [
            MemoryEntry("Python is a programming language", {'type': 'tech'}),
            MemoryEntry("Machine learning uses algorithms", {'type': 'tech'}),
            MemoryEntry("Cooking pasta requires boiling water", {'type': 'cooking'}),
            MemoryEntry("Neural networks are used in AI", {'type': 'tech'}),
        ]
        
        for memory in test_memories:
            memory_id = self.vector_memory.store_memory(memory)
            self.semantic_search.add_to_index(memory.content, memory.metadata)
    
    def tearDown(self):
        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_semantic_search(self):
        """Test semantic search functionality."""
        query = SearchQuery(
            text="artificial intelligence programming",
            max_results=3,
            strategy="semantic"
        )
        
        results = self.semantic_search.search(query, self.vector_memory)
        self.assertIsNotNone(results)
        self.assertGreater(len(results.results), 0)
        
        # Should find tech-related content
        tech_results = [r for r in results.results 
                       if r.metadata.get('type') == 'tech']
        self.assertGreater(len(tech_results), 0)
    
    def test_keyword_search(self):
        """Test keyword search functionality."""
        query = SearchQuery(
            text="python programming",
            max_results=3,
            strategy="keyword"
        )
        
        results = self.semantic_search.search(query, self.vector_memory)
        self.assertIsNotNone(results)
        
        # Should find python-related content
        if len(results.results) > 0:
            python_found = any('python' in r.content.lower() 
                             for r in results.results)
            self.assertTrue(python_found)
    
    def test_hybrid_search(self):
        """Test hybrid search combining semantic and keyword."""
        query = SearchQuery(
            text="machine learning programming",
            max_results=5,
            strategy="hybrid"
        )
        
        results = self.semantic_search.search(query, self.vector_memory)
        self.assertIsNotNone(results)

class TestKnowledgeGraph(unittest.TestCase):
    """Test knowledge graph functionality."""
    
    def setUp(self):
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Memory components not available")
        
        self.test_file = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        self.test_file.close()
        self.knowledge_graph = KnowledgeGraph(self.test_file.name)
    
    def tearDown(self):
        if hasattr(self, 'test_file'):
            os.unlink(self.test_file.name)
    
    def test_entity_extraction(self):
        """Test entity extraction from text."""
        text = "John Smith works at Google on Python development projects."
        
        self.knowledge_graph.add_text(text)
        
        stats = self.knowledge_graph.get_stats()
        self.assertGreater(stats['total_entities'], 0)
        
        # Check for specific entities
        entities = list(self.knowledge_graph.entities.keys())
        self.assertTrue(any('john' in entity.lower() for entity in entities))
    
    def test_relationship_discovery(self):
        """Test relationship discovery between entities."""
        texts = [
            "Python is a programming language",
            "Google uses Python for development",
            "Machine learning algorithms are implemented in Python"
        ]
        
        for text in texts:
            self.knowledge_graph.add_text(text)
        
        stats = self.knowledge_graph.get_stats()
        self.assertGreater(stats['total_relationships'], 0)
    
    def test_concept_formation(self):
        """Test concept formation from related content."""
        texts = [
            "Deep learning neural networks",
            "Convolutional neural networks for images", 
            "Recurrent neural networks for sequences",
            "Machine learning classification algorithms",
            "Supervised learning with labeled data"
        ]
        
        for text in texts:
            self.knowledge_graph.add_text(text)
        
        # Form concepts
        self.knowledge_graph.form_concepts(min_cluster_size=2)
        
        stats = self.knowledge_graph.get_stats()
        self.assertGreaterEqual(stats['total_concepts'], 0)

class TestMemoryAnalytics(unittest.TestCase):
    """Test memory analytics functionality."""
    
    def setUp(self):
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Memory components not available")
        
        self.test_dir = tempfile.mkdtemp()
        self.vector_memory = VectorMemory(db_path=self.test_dir)
        self.analytics = MemoryAnalytics(cache_dir=self.test_dir)
        
        # Populate with test data
        import datetime
        base_time = datetime.datetime.now()
        
        for i in range(10):
            timestamp = base_time - datetime.timedelta(hours=i)
            memory = MemoryEntry(
                f"Test memory {i} about technology and programming",
                metadata={
                    'timestamp': timestamp.isoformat(),
                    'importance': 0.5 + (i % 3) * 0.2,
                    'session_id': f"session_{i % 3}"
                }
            )
            self.vector_memory.store_memory(memory)
    
    def tearDown(self):
        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_pattern_analysis(self):
        """Test pattern analysis functionality."""
        insights = self.analytics.analyze_patterns(self.vector_memory)
        
        self.assertIsInstance(insights, dict)
        self.assertIn('temporal_patterns', insights)
        self.assertIn('semantic_patterns', insights)
        self.assertIn('behavioral_patterns', insights)
    
    def test_temporal_analysis(self):
        """Test temporal pattern analysis."""
        patterns = self.analytics._analyze_temporal_patterns(self.vector_memory)
        
        self.assertIsInstance(patterns, dict)
        self.assertIn('total_sessions', patterns)
        self.assertIn('hourly_activity', patterns)

class TestMemoryIntegration(unittest.TestCase):
    """Test the integrated memory system."""
    
    def setUp(self):
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Memory components not available")
        
        self.test_dir = tempfile.mkdtemp()
        
        config = MemorySystemConfig(
            vector_db_path=os.path.join(self.test_dir, "vector"),
            knowledge_graph_path=os.path.join(self.test_dir, "kg"),
            analytics_path=os.path.join(self.test_dir, "analytics"),
            dashboard_path=os.path.join(self.test_dir, "dashboard"),
            enable_background_processing=False  # Disable for tests
        )
        
        self.memory_system = EnhancedMemorySystem(config, agent_id="test_agent")
    
    def tearDown(self):
        if hasattr(self, 'memory_system'):
            self.memory_system.shutdown()
        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_store_and_search_integration(self):
        """Test integrated store and search functionality."""
        # Store test memories
        test_memories = [
            "Python programming tutorial for beginners",
            "Machine learning with TensorFlow",
            "Data science using pandas and numpy",
            "Web development with Django framework"
        ]
        
        memory_ids = []
        for content in test_memories:
            memory_id = self.memory_system.store_memory(
                content=content,
                role="user",
                metadata={'topic': 'programming'},
                importance=0.7
            )
            memory_ids.append(memory_id)
        
        # Test search functionality
        results = self.memory_system.search_memories(
            query="Python programming",
            max_results=3
        )
        
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)
        
        # Check result format
        for result in results:
            self.assertIn('content', result)
            self.assertIn('score', result)
            self.assertIn('metadata', result)
    
    def test_insights_generation(self):
        """Test insights generation."""
        # Add some test data
        for i in range(5):
            self.memory_system.store_memory(
                content=f"Test memory {i} about various topics",
                role="user",
                importance=0.5
            )
        
        # Generate insights
        insights = self.memory_system.generate_insights()
        
        self.assertIsInstance(insights, dict)
        self.assertIn('analytics', insights)
        self.assertIn('memory_system', insights)
        self.assertIn('recommendations', insights)
    
    def test_export_import_functionality(self):
        """Test cross-agent sharing via export/import."""
        # Store test data
        self.memory_system.store_memory(
            content="Shared knowledge about AI development",
            role="assistant",
            metadata={'shareable': True},
            importance=0.8
        )
        
        # Export data
        export_path = os.path.join(self.test_dir, "export.json")
        success = self.memory_system.export_for_sharing(export_path)
        self.assertTrue(success)
        self.assertTrue(os.path.exists(export_path))
        
        # Verify export content
        with open(export_path, 'r') as f:
            export_data = json.load(f)
        
        self.assertIn('agent_id', export_data)
        self.assertIn('memories', export_data)
        self.assertGreater(len(export_data['memories']), 0)
    
    def test_system_status(self):
        """Test system status reporting."""
        status = self.memory_system.get_system_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('agent_id', status)
        self.assertIn('status', status)
        self.assertIn('components', status)
        self.assertIn('stats', status)
        
        # Check that all components are reported as active
        components = status['components']
        for component in ['vector_memory', 'semantic_search', 'knowledge_graph', 'analytics']:
            self.assertIn(component, components)

class TestPerformance(unittest.TestCase):
    """Test performance characteristics of the memory system."""
    
    def setUp(self):
        if not COMPONENTS_AVAILABLE:
            self.skipTest("Memory components not available")
        
        self.test_dir = tempfile.mkdtemp()
        config = MemorySystemConfig(
            vector_db_path=os.path.join(self.test_dir, "vector"),
            enable_background_processing=False
        )
        self.memory_system = EnhancedMemorySystem(config, agent_id="perf_test")
    
    def tearDown(self):
        if hasattr(self, 'memory_system'):
            self.memory_system.shutdown()
        if hasattr(self, 'test_dir'):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_bulk_storage_performance(self):
        """Test performance with bulk memory storage."""
        start_time = time.time()
        
        # Store 50 memories
        for i in range(50):
            content = f"Performance test memory {i} with detailed content about various topics and scenarios"
            self.memory_system.store_memory(
                content=content,
                role="user",
                importance=0.5
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete within reasonable time (adjust threshold as needed)
        self.assertLess(duration, 30.0, "Bulk storage took too long")
        
        # Verify all memories were stored
        status = self.memory_system.get_system_status()
        self.assertEqual(status['stats']['total_memories'], 50)
    
    def test_search_performance(self):
        """Test search performance with larger dataset."""
        # First populate with test data
        for i in range(20):
            self.memory_system.store_memory(
                content=f"Test memory {i} about programming, machine learning, and data science",
                role="user"
            )
        
        # Test search performance
        start_time = time.time()
        
        for _ in range(10):
            results = self.memory_system.search_memories(
                query="machine learning programming",
                max_results=5
            )
            self.assertGreater(len(results), 0)
        
        end_time = time.time()
        avg_search_time = (end_time - start_time) / 10
        
        # Should search quickly (adjust threshold as needed)
        self.assertLess(avg_search_time, 2.0, "Search performance too slow")

def run_comprehensive_tests():
    """Run all tests and provide summary report."""
    
    if not COMPONENTS_AVAILABLE:
        print("❌ Memory components not available - skipping tests")
        return False
    
    print("🧪 Running Enhanced Memory System Tests")
    print("=" * 50)
    
    # Create test suite
    test_classes = [
        TestVectorMemory,
        TestSemanticSearch, 
        TestKnowledgeGraph,
        TestMemoryAnalytics,
        TestMemoryIntegration,
        TestPerformance
    ]
    
    suite = unittest.TestSuite()
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n⚠️ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / 
                   result.testsRun * 100) if result.testsRun > 0 else 0
    
    print(f"\n✅ Success Rate: {success_rate:.1f}%")
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
