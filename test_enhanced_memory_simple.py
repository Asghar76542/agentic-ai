"""
Simple test for Enhanced Memory System without logging dependencies.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import uuid

# Add project root to path
sys.path.append(os.getcwd())

def test_memory_components():
    """Test individual memory components."""
    print("🧠 Testing Enhanced Memory System Components")
    print("=" * 50)
    
    # Test MemoryEntry
    try:
        from sources.knowledge.vectorMemory import MemoryEntry
        
        entry = MemoryEntry(
            id=str(uuid.uuid4()),
            content='This is a test memory for the enhanced system',
            role='user',
            timestamp=datetime.now(),
            session_id='test_session',
            agent_type='test_agent',
            importance=0.8,
            tags=['test', 'enhanced'],
            metadata={'source': 'test', 'category': 'demo'}
        )
        print(f"✅ MemoryEntry: Created successfully")
        print(f"   Content: {entry.content[:50]}...")
        print(f"   Role: {entry.role}, Importance: {entry.importance}")
        
    except Exception as e:
        print(f"❌ MemoryEntry: Failed - {e}")
        return False
    
    # Test SearchQuery and SearchResult
    try:
        from sources.knowledge.semanticSearch import SearchQuery, SearchResult
        
        query = SearchQuery(
            text='test memory enhancement',
            search_type='semantic',
            limit=10,
            filters={'role': 'user'},
            include_metadata=True
        )
        print(f"✅ SearchQuery: Created successfully")
        print(f"   Query: {query.text}, Type: {query.search_type}")
        
        result = SearchResult(
            entry=entry,
            score=0.95,
            rank=1,
            match_type='semantic',
            explanation='High semantic similarity'
        )
        print(f"✅ SearchResult: Created successfully")
        print(f"   Score: {result.score}, Rank: {result.rank}")
        
    except Exception as e:
        print(f"❌ SearchQuery/SearchResult: Failed - {e}")
        return False
    
    # Test KnowledgeGraph entities
    try:
        from sources.knowledge.knowledgeGraph import Entity, Relationship, Concept
        
        entity = Entity(
            name="Enhanced Memory System",
            entity_type="TECHNOLOGY",
            frequency=5,
            metadata={'description': 'Advanced AI memory system'}
        )
        print(f"✅ Entity: Created successfully")
        print(f"   Name: {entity.name}, Type: {entity.entity_type}")
        
        relationship = Relationship(
            source="Memory System",
            target="Vector Database",
            relationship_type="USES",
            strength=0.9,
            metadata={'connection': 'storage'}
        )
        print(f"✅ Relationship: Created successfully")
        print(f"   {relationship.source} --{relationship.relationship_type}--> {relationship.target}")
        
    except Exception as e:
        print(f"❌ KnowledgeGraph components: Failed - {e}")
        return False
    
    # Test PatternInsight
    try:
        from sources.knowledge.memoryAnalytics import PatternInsight
        
        insight = PatternInsight(
            pattern_type="temporal",
            description="Increased activity in evening hours",
            confidence=0.85,
            supporting_data={'peak_hours': [18, 19, 20]},
            recommendations=["Schedule important tasks during peak hours"]
        )
        print(f"✅ PatternInsight: Created successfully")
        print(f"   Type: {insight.pattern_type}, Confidence: {insight.confidence}")
        
    except Exception as e:
        print(f"❌ PatternInsight: Failed - {e}")
        return False
    
    print("\n✨ All memory components tested successfully!")
    return True

def test_memory_integration_lite():
    """Test memory integration without full system initialization."""
    print("\n🔗 Testing Memory Integration (Lite Mode)")
    print("=" * 50)
    
    try:
        # Test configuration
        from sources.knowledge.memoryIntegration import MemorySystemConfig
        
        config = MemorySystemConfig(
            vector_db_path='./test_data/vector_memory',
            knowledge_graph_path='./test_data/knowledge_graph',
            analytics_path='./test_data/analytics',
            dashboard_path='./test_data/dashboard',
            enable_background_processing=False,  # Disable for testing
            auto_cleanup_enabled=False  # Disable for testing
        )
        print("✅ MemorySystemConfig: Created successfully")
        print(f"   Vector DB Path: {config.vector_db_path}")
        print(f"   Background Processing: {config.enable_background_processing}")
        
        # Test directory creation
        for path in [config.vector_db_path, config.knowledge_graph_path,
                    config.analytics_path, config.dashboard_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
        print("✅ Test directories: Created successfully")
        
    except Exception as e:
        print(f"❌ Memory integration test: Failed - {e}")
        return False
    
    print("\n✨ Memory integration (lite) tested successfully!")
    return True

def test_dashboard_component():
    """Test dashboard component creation."""
    print("\n📊 Testing Dashboard Component")
    print("=" * 50)
    
    try:
        from sources.knowledge.memoryDashboard import MemoryDashboard, DashboardConfig
        
        dashboard_config = DashboardConfig(
            host="localhost",
            port=5001,
            debug=False,
            auto_refresh_interval=30
        )
        print("✅ DashboardConfig: Created successfully")
        print(f"   Host: {dashboard_config.host}:{dashboard_config.port}")
        
        dashboard = MemoryDashboard(dashboard_config)
        print("✅ MemoryDashboard: Created successfully")
        print(f"   Dashboard URL: {dashboard.get_dashboard_url()}")
        
    except Exception as e:
        print(f"❌ Dashboard component test: Failed - {e}")
        return False
    
    print("\n✨ Dashboard component tested successfully!")
    return True

def main():
    """Run all enhanced memory system tests."""
    print("🚀 Enhanced Memory System Test Suite")
    print("=" * 60)
    print(f"Test time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    success_count = 0
    total_tests = 3
    
    # Run tests
    if test_memory_components():
        success_count += 1
    
    if test_memory_integration_lite():
        success_count += 1
        
    if test_dashboard_component():
        success_count += 1
    
    # Final results
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {success_count}/{total_tests} tests")
    print(f"❌ Failed: {total_tests - success_count}/{total_tests} tests")
    
    if success_count == total_tests:
        print("\n🎉 ALL TESTS PASSED! Enhanced Memory System is ready.")
        return True
    else:
        print(f"\n⚠️  {total_tests - success_count} tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
