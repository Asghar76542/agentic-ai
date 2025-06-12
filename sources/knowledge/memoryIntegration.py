"""
Enhanced Memory System Integration
Coordinates all memory components and provides unified interface.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time
from dataclasses import dataclass, asdict

# Import all memory components
from sources.knowledge.vectorMemory import VectorMemory, MemoryEntry
from sources.knowledge.semanticSearch import SemanticSearch, SearchQuery, SearchResult
from sources.knowledge.knowledgeGraph import KnowledgeGraph
from sources.knowledge.memoryAnalytics import MemoryAnalytics
from sources.knowledge.memoryDashboard import MemoryDashboard

@dataclass
class MemorySystemConfig:
    """Configuration for the enhanced memory system."""
    vector_db_path: str = "./data/vector_memory"
    knowledge_graph_path: str = "./data/knowledge_graph"
    analytics_path: str = "./data/analytics"
    dashboard_path: str = "./data/dashboard"
    enable_background_processing: bool = True
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 24
    memory_retention_days: int = 365
    importance_threshold: float = 0.3
    max_memory_entries: int = 100000
    enable_cross_agent_sharing: bool = True
    sharing_export_interval_hours: int = 6

class EnhancedMemorySystem:
    """
    Unified enhanced memory system that coordinates all memory components.
    Provides high-level interface for memory operations across agents.
    """
    
    def __init__(self, config: Optional[MemorySystemConfig] = None, 
                 agent_id: str = "default"):
        self.config = config or MemorySystemConfig()
        self.agent_id = agent_id
        self.logger = logging.getLogger(f"memory_system_{agent_id}")
        
        # Initialize components
        self._initialize_directories()
        self._initialize_components()
        
        # Background processing
        self._shutdown_event = threading.Event()
        self._background_thread = None
        
        if self.config.enable_background_processing:
            self._start_background_processing()
    
    def _initialize_directories(self):
        """Create necessary directories."""
        for path in [self.config.vector_db_path, self.config.knowledge_graph_path,
                    self.config.analytics_path, self.config.dashboard_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize all memory system components."""
        try:
            # Vector memory
            self.vector_memory = VectorMemory(
                memory_path=self.config.vector_db_path,
                session_id=f"agent_{self.agent_id}"
            )
            
            # Initialize semantic search with proper parameters
            self.semantic_search = SemanticSearch(
                vector_memory=self.vector_memory,
                enable_keyword_search=True,
                enable_query_expansion=True
            )
            
            # Knowledge graph
            kg_path = Path(self.config.knowledge_graph_path) / f"{self.agent_id}_kg.json"
            self.knowledge_graph = KnowledgeGraph(str(kg_path))
            
            # Analytics
            analytics_path = Path(self.config.analytics_path) / f"{self.agent_id}_analytics"
            self.memory_analytics = MemoryAnalytics(
                vector_memory=self.vector_memory,
                semantic_search=self.semantic_search,
                knowledge_graph=self.knowledge_graph,
                analytics_path=str(analytics_path)
            )
            
            # Dashboard
            dashboard_path = Path(self.config.dashboard_path) / self.agent_id
            self.dashboard = MemoryDashboard()
            
            self.logger.info("Enhanced memory system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory components: {e}")
            raise
    
    def store_memory(self, content: str, role: str = "user", 
                    metadata: Optional[Dict] = None, 
                    importance: float = 0.5) -> str:
        """
        Store a memory entry across all systems.
        
        Args:
            content: Memory content
            role: Role (user, assistant, system)
            metadata: Additional metadata
            importance: Importance score (0.0-1.0)
            
        Returns:
            Memory ID
        """
        try:
            # Create memory entry
            entry_metadata = {
                'role': role,
                'agent_id': self.agent_id,
                'timestamp': datetime.now().isoformat(),
                'importance': importance,
                **(metadata or {})
            }
            
            memory_entry = MemoryEntry(
                content=content,
                metadata=entry_metadata
            )
            
            # Store in vector memory
            memory_id = self.vector_memory.store_memory(memory_entry)
            
            # Update knowledge graph
            self.knowledge_graph.add_text(content)
            
            # Update search index
            self.semantic_search.add_to_index(content, entry_metadata)
            
            self.logger.info(f"Memory stored with ID: {memory_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            raise
    
    def search_memories(self, query: str, max_results: int = 10,
                       filters: Optional[Dict] = None,
                       search_type: str = "hybrid") -> List[Dict]:
        """
        Search memories using advanced semantic search.
        
        Args:
            query: Search query
            max_results: Maximum results to return
            filters: Optional filters (role, date range, etc.)
            search_type: Search strategy (semantic, keyword, hybrid)
            
        Returns:
            List of relevant memory entries
        """
        try:
            search_query = SearchQuery(
                text=query,
                max_results=max_results,
                filters=filters or {},
                strategy=search_type
            )
            
            results = self.semantic_search.search(search_query, self.vector_memory)
            
            # Format results
            formatted_results = []
            for result in results.results:
                formatted_results.append({
                    'content': result.content,
                    'score': result.score,
                    'metadata': result.metadata,
                    'timestamp': result.metadata.get('timestamp'),
                    'importance': result.metadata.get('importance', 0.5)
                })
            
            self.logger.info(f"Search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Memory search failed: {e}")
            return []
    
    def get_related_memories(self, memory_id: str, max_results: int = 5) -> List[Dict]:
        """Get memories related to a specific memory entry."""
        try:
            # Get the original memory
            memory = self.vector_memory.get_memory(memory_id)
            if not memory:
                return []
            
            # Search for similar memories
            return self.search_memories(
                query=memory.content,
                max_results=max_results + 1,  # +1 to exclude self
                search_type="semantic"
            )[1:]  # Remove the original memory
            
        except Exception as e:
            self.logger.error(f"Failed to get related memories: {e}")
            return []
    
    def get_conversation_context(self, session_id: str, 
                               max_entries: int = 20) -> List[Dict]:
        """Get conversation context for a specific session."""
        try:
            filters = {'session_id': session_id}
            
            # Get recent memories from this session
            recent_memories = self.search_memories(
                query="",  # Empty query to get all
                max_results=max_entries,
                filters=filters
            )
            
            # Sort by timestamp
            recent_memories.sort(
                key=lambda x: x.get('timestamp', ''), 
                reverse=True
            )
            
            return recent_memories
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation context: {e}")
            return []
    
    def get_entity_information(self, entity_name: str) -> Dict:
        """Get comprehensive information about an entity."""
        try:
            # Get entity from knowledge graph
            entity_info = self.knowledge_graph.get_entity_info(entity_name)
            
            # Get related memories
            related_memories = self.search_memories(
                query=entity_name,
                max_results=10,
                search_type="hybrid"
            )
            
            return {
                'entity': entity_info,
                'related_memories': related_memories,
                'relationships': self.knowledge_graph.get_entity_relationships(entity_name)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get entity information: {e}")
            return {}
    
    def generate_insights(self) -> Dict:
        """Generate comprehensive memory insights."""
        try:
            # Get analytics insights
            insights = self.memory_analytics.analyze_patterns(self.vector_memory)
            
            # Get knowledge graph statistics
            kg_stats = self.knowledge_graph.get_stats()
            
            # Get memory statistics
            memory_stats = self.vector_memory.get_stats()
            
            # Combine insights
            comprehensive_insights = {
                'analytics': insights,
                'knowledge_graph': kg_stats,
                'memory_system': memory_stats,
                'recommendations': self._generate_recommendations(insights, memory_stats)
            }
            
            return comprehensive_insights
            
        except Exception as e:
            self.logger.error(f"Failed to generate insights: {e}")
            return {}
    
    def _generate_recommendations(self, insights: Dict, memory_stats: Dict) -> List[str]:
        """Generate actionable recommendations based on insights."""
        recommendations = []
        
        try:
            total_memories = memory_stats.get('total_memories', 0)
            
            # Storage recommendations
            if total_memories > self.config.max_memory_entries * 0.8:
                recommendations.append(
                    "Consider running memory cleanup - approaching storage limit"
                )
            
            # Performance recommendations
            avg_similarity = memory_stats.get('average_similarity', 0)
            if avg_similarity < 0.3:
                recommendations.append(
                    "Low memory similarity detected - consider improving content diversity"
                )
            
            # Usage pattern recommendations
            temporal_patterns = insights.get('temporal_patterns', {})
            if temporal_patterns:
                peak_hour = temporal_patterns.get('peak_hour')
                if peak_hour:
                    recommendations.append(
                        f"Peak usage at hour {peak_hour} - consider optimizing for this time"
                    )
            
            # Knowledge graph recommendations
            kg_entities = insights.get('knowledge_graph', {}).get('total_entities', 0)
            if kg_entities < total_memories * 0.1:
                recommendations.append(
                    "Low entity extraction rate - consider improving text processing"
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to generate some recommendations: {e}")
        
        return recommendations
    
    def export_for_sharing(self, export_path: str, 
                          include_private: bool = False) -> bool:
        """Export memories for cross-agent sharing."""
        try:
            if not self.config.enable_cross_agent_sharing:
                self.logger.warning("Cross-agent sharing is disabled")
                return False
            
            # Prepare export data
            export_data = {
                'agent_id': self.agent_id,
                'export_timestamp': datetime.now().isoformat(),
                'memories': [],
                'knowledge_graph': self.knowledge_graph.export_graph(),
                'metadata': {
                    'total_memories': self.vector_memory.get_stats().get('total_memories', 0),
                    'include_private': include_private
                }
            }
            
            # Export memories (filtered if not including private)
            all_memories = self.vector_memory.get_all_memories()
            for memory in all_memories:
                if include_private or not memory.metadata.get('private', False):
                    # Remove sensitive metadata
                    clean_metadata = {k: v for k, v in memory.metadata.items() 
                                    if k not in ['session_id', 'private']}
                    
                    export_data['memories'].append({
                        'content': memory.content,
                        'metadata': clean_metadata
                    })
            
            # Save export file
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported {len(export_data['memories'])} memories to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory export failed: {e}")
            return False
    
    def import_from_sharing(self, import_path: str, 
                          merge_strategy: str = "append") -> bool:
        """Import memories from another agent."""
        try:
            if not self.config.enable_cross_agent_sharing:
                self.logger.warning("Cross-agent sharing is disabled")
                return False
            
            # Load import data
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            source_agent = import_data.get('agent_id', 'unknown')
            memories = import_data.get('memories', [])
            
            # Import memories
            imported_count = 0
            for memory_data in memories:
                try:
                    # Add source agent info to metadata
                    metadata = memory_data.get('metadata', {})
                    metadata['source_agent'] = source_agent
                    metadata['imported_at'] = datetime.now().isoformat()
                    
                    # Store memory
                    self.store_memory(
                        content=memory_data['content'],
                        metadata=metadata,
                        importance=metadata.get('importance', 0.3)  # Lower importance for imported
                    )
                    imported_count += 1
                    
                except Exception as e:
                    self.logger.warning(f"Failed to import individual memory: {e}")
            
            # Import knowledge graph
            if 'knowledge_graph' in import_data:
                self.knowledge_graph.import_graph(import_data['knowledge_graph'])
            
            self.logger.info(f"Imported {imported_count} memories from {source_agent}")
            return True
            
        except Exception as e:
            self.logger.error(f"Memory import failed: {e}")
            return False
    
    def cleanup_old_memories(self, force: bool = False) -> int:
        """Clean up old, low-importance memories."""
        try:
            if not (self.config.auto_cleanup_enabled or force):
                return 0
            
            cleanup_count = self.vector_memory.cleanup_memories(
                days_threshold=self.config.memory_retention_days,
                importance_threshold=self.config.importance_threshold
            )
            
            # Also cleanup knowledge graph
            self.knowledge_graph.cleanup_unused_entities()
            
            self.logger.info(f"Cleaned up {cleanup_count} old memories")
            return cleanup_count
            
        except Exception as e:
            self.logger.error(f"Memory cleanup failed: {e}")
            return 0
    
    def generate_dashboard(self) -> str:
        """Generate analytics dashboard."""
        try:
            dashboard_path = self.dashboard.generate_dashboard(
                self.memory_analytics,
                self.vector_memory, 
                self.knowledge_graph
            )
            
            self.logger.info(f"Dashboard generated at {dashboard_path}")
            return dashboard_path
            
        except Exception as e:
            self.logger.error(f"Dashboard generation failed: {e}")
            return ""
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status."""
        try:
            return {
                'agent_id': self.agent_id,
                'status': 'healthy',
                'components': {
                    'vector_memory': 'active',
                    'semantic_search': 'active', 
                    'knowledge_graph': 'active',
                    'analytics': 'active',
                    'dashboard': 'active'
                },
                'stats': {
                    'total_memories': self.vector_memory.get_stats().get('total_memories', 0),
                    'total_entities': self.knowledge_graph.get_stats().get('total_entities', 0),
                    'storage_mb': self.vector_memory.get_stats().get('storage_size_mb', 0)
                },
                'config': {
                    'auto_cleanup': self.config.auto_cleanup_enabled,
                    'cross_agent_sharing': self.config.enable_cross_agent_sharing,
                    'background_processing': self.config.enable_background_processing
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _start_background_processing(self):
        """Start background processing thread."""
        if self._background_thread is not None:
            return
        
        self._background_thread = threading.Thread(
            target=self._background_worker,
            daemon=True
        )
        self._background_thread.start()
        self.logger.info("Background processing started")
    
    def _background_worker(self):
        """Background worker for maintenance tasks."""
        while not self._shutdown_event.is_set():
            try:
                # Auto cleanup
                if self.config.auto_cleanup_enabled:
                    self.cleanup_old_memories()
                
                # Auto export for sharing
                if self.config.enable_cross_agent_sharing:
                    export_path = Path(self.config.dashboard_path) / f"{self.agent_id}_export.json"
                    self.export_for_sharing(str(export_path))
                
                # Update analytics
                self.memory_analytics.update_cache(self.vector_memory)
                
                # Sleep until next cycle
                self._shutdown_event.wait(self.config.cleanup_interval_hours * 3600)
                
            except Exception as e:
                self.logger.error(f"Background worker error: {e}")
                self._shutdown_event.wait(300)  # Wait 5 minutes on error
    
    def shutdown(self):
        """Shutdown the memory system."""
        try:
            # Stop background processing
            if self._background_thread is not None:
                self._shutdown_event.set()
                self._background_thread.join(timeout=10)
            
            # Save final state
            self.knowledge_graph.save_graph()
            
            self.logger.info("Enhanced memory system shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
