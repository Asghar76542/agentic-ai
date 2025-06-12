"""
Memory Analytics Implementation for AgenticSeek
Advanced analytics and insights for memory usage, patterns, and knowledge discovery.
"""

import json
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import statistics

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx

from sources.logger import Logger
from sources.knowledge.vectorMemory import VectorMemory, MemoryEntry
from sources.knowledge.semanticSearch import SemanticSearch, SearchQuery
from sources.knowledge.knowledgeGraph import KnowledgeGraph


@dataclass
class MemoryPattern:
    """Represents a discovered pattern in memory usage."""
    pattern_id: str
    pattern_type: str  # temporal, semantic, behavioral, structural
    description: str
    confidence: float
    frequency: int
    entities_involved: List[str]
    time_range: Tuple[datetime, datetime]
    metadata: Dict[str, Any]


@dataclass
class InsightReport:
    """Comprehensive insight report about memory and knowledge."""
    report_id: str
    generated_at: datetime
    time_range: Tuple[datetime, datetime]
    summary: str
    key_insights: List[str]
    patterns_discovered: List[MemoryPattern]
    recommendations: List[str]
    statistics: Dict[str, Any]
    visualizations: List[str]  # Paths to generated charts


class MemoryAnalytics:
    """
    Advanced analytics engine for memory and knowledge analysis.
    Provides insights, pattern discovery, and recommendations for optimization.
    """
    
    def __init__(self, 
                 vector_memory: VectorMemory,
                 semantic_search: SemanticSearch,
                 knowledge_graph: KnowledgeGraph,
                 analytics_path: str = "analytics"):
        
        self.logger = Logger("memory_analytics.log")
        self.vector_memory = vector_memory
        self.semantic_search = semantic_search
        self.knowledge_graph = knowledge_graph
        self.analytics_path = analytics_path
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_expiry = timedelta(hours=6)
        
        # Pattern detection thresholds
        self.pattern_thresholds = {
            "temporal_pattern_min_frequency": 3,
            "semantic_cluster_min_size": 5,
            "behavioral_pattern_confidence": 0.7,
            "entity_co_occurrence_threshold": 0.3
        }
        
        # Visualization settings
        self.viz_settings = {
            "figure_size": (12, 8),
            "color_palette": "viridis",
            "dpi": 300
        }
        
        # Analytics statistics
        self.analytics_stats = {
            "analyses_performed": 0,
            "patterns_discovered": 0,
            "insights_generated": 0,
            "reports_created": 0
        }
        
        # Initialize analytics environment
        self._initialize_analytics()
    
    def _initialize_analytics(self):
        """Initialize analytics environment and create directories."""
        try:
            import os
            os.makedirs(self.analytics_path, exist_ok=True)
            os.makedirs(f"{self.analytics_path}/visualizations", exist_ok=True)
            os.makedirs(f"{self.analytics_path}/reports", exist_ok=True)
            
            # Set matplotlib backend for headless environments
            try:
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
            except ImportError:
                self.logger.warning("Matplotlib not available, visualizations will be disabled")
            
            self.logger.info("Memory analytics initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing analytics: {e}")
    
    def generate_comprehensive_report(self, 
                                    time_range: Optional[Tuple[datetime, datetime]] = None,
                                    include_visualizations: bool = True) -> InsightReport:
        """Generate a comprehensive analytics report."""
        try:
            start_time = datetime.now()
            
            # Set default time range if not provided
            if time_range is None:
                end_time = datetime.now()
                start_time_range = end_time - timedelta(days=30)
                time_range = (start_time_range, end_time)
            
            self.logger.info(f"Generating comprehensive report for period {time_range[0]} to {time_range[1]}")
            
            # Collect basic statistics
            stats = self._collect_comprehensive_statistics(time_range)
            
            # Discover patterns
            patterns = self._discover_all_patterns(time_range)
            
            # Generate insights
            insights = self._generate_insights(stats, patterns)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(stats, patterns, insights)
            
            # Create visualizations
            visualizations = []
            if include_visualizations:
                visualizations = self._create_visualizations(stats, patterns, time_range)
            
            # Create summary
            summary = self._create_report_summary(stats, patterns, insights)
            
            # Create report
            report = InsightReport(
                report_id=f"report_{int(datetime.now().timestamp())}",
                generated_at=datetime.now(),
                time_range=time_range,
                summary=summary,
                key_insights=insights,
                patterns_discovered=patterns,
                recommendations=recommendations,
                statistics=stats,
                visualizations=visualizations
            )
            
            # Save report
            self._save_report(report)
            
            # Update statistics
            self.analytics_stats["reports_created"] += 1
            self.analytics_stats["analyses_performed"] += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Comprehensive report generated in {processing_time:.2f} seconds")
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            raise
    
    def _collect_comprehensive_statistics(self, time_range: Tuple[datetime, datetime]) -> Dict[str, Any]:
        """Collect comprehensive statistics for the given time range."""
        try:
            stats = {}
            
            # Filter memories by time range
            filtered_memories = self._filter_memories_by_time(time_range)
            
            # Basic memory statistics
            stats["memory_stats"] = {
                "total_memories": len(filtered_memories),
                "total_characters": sum(len(m.content) for m in filtered_memories),
                "average_memory_length": statistics.mean([len(m.content) for m in filtered_memories]) if filtered_memories else 0,
                "median_memory_length": statistics.median([len(m.content) for m in filtered_memories]) if filtered_memories else 0
            }
            
            # Role distribution
            role_counts = Counter(m.role for m in filtered_memories)
            stats["role_distribution"] = dict(role_counts)
            
            # Agent type distribution
            agent_type_counts = Counter(m.agent_type for m in filtered_memories)
            stats["agent_type_distribution"] = dict(agent_type_counts)
            
            # Importance distribution
            importance_values = [m.importance for m in filtered_memories]
            if importance_values:
                stats["importance_stats"] = {
                    "mean": statistics.mean(importance_values),
                    "median": statistics.median(importance_values),
                    "std_dev": statistics.stdev(importance_values) if len(importance_values) > 1 else 0,
                    "min": min(importance_values),
                    "max": max(importance_values)
                }
            
            # Temporal statistics
            stats["temporal_stats"] = self._analyze_temporal_patterns(filtered_memories)
            
            # Knowledge graph statistics
            stats["knowledge_graph_stats"] = self.knowledge_graph.get_graph_statistics()
            
            # Search statistics
            stats["search_stats"] = self.semantic_search.get_search_statistics()
            
            # Memory access patterns
            stats["access_patterns"] = self._analyze_access_patterns(filtered_memories)
            
            # Content analysis
            stats["content_analysis"] = self._analyze_content_characteristics(filtered_memories)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error collecting statistics: {e}")
            return {}
    
    def _filter_memories_by_time(self, time_range: Tuple[datetime, datetime]) -> List[MemoryEntry]:
        """Filter memories by time range."""
        start_time, end_time = time_range
        filtered_memories = []
        
        for memory in self.vector_memory.memory_entries.values():
            if start_time <= memory.timestamp <= end_time:
                filtered_memories.append(memory)
        
        return filtered_memories
    
    def _analyze_temporal_patterns(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Analyze temporal patterns in memory creation."""
        try:
            if not memories:
                return {}
            
            # Group by hour of day
            hourly_counts = defaultdict(int)
            daily_counts = defaultdict(int)
            weekly_counts = defaultdict(int)
            
            for memory in memories:
                hour = memory.timestamp.hour
                day = memory.timestamp.strftime('%Y-%m-%d')
                week = memory.timestamp.strftime('%Y-W%W')
                
                hourly_counts[hour] += 1
                daily_counts[day] += 1
                weekly_counts[week] += 1
            
            # Find peak hours
            if hourly_counts:
                peak_hour = max(hourly_counts.items(), key=lambda x: x[1])
                avg_hourly = statistics.mean(hourly_counts.values())
            else:
                peak_hour = (0, 0)
                avg_hourly = 0
            
            # Calculate activity patterns
            timestamps = [m.timestamp for m in memories]
            time_diffs = []
            for i in range(1, len(timestamps)):
                diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # Hours
                time_diffs.append(diff)
            
            return {
                "memory_count": len(memories),
                "time_span_days": (max(timestamps) - min(timestamps)).days if len(timestamps) > 1 else 0,
                "peak_hour": peak_hour[0],
                "peak_hour_count": peak_hour[1],
                "average_hourly_activity": avg_hourly,
                "average_interval_hours": statistics.mean(time_diffs) if time_diffs else 0,
                "daily_activity": dict(daily_counts),
                "hourly_distribution": dict(hourly_counts)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal patterns: {e}")
            return {}
    
    def _analyze_access_patterns(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Analyze memory access patterns."""
        try:
            access_stats = {
                "total_accesses": sum(m.access_count for m in memories),
                "accessed_memories": len([m for m in memories if m.access_count > 0]),
                "never_accessed": len([m for m in memories if m.access_count == 0])
            }
            
            if access_stats["total_accesses"] > 0:
                access_counts = [m.access_count for m in memories if m.access_count > 0]
                access_stats.update({
                    "average_access_count": statistics.mean(access_counts),
                    "median_access_count": statistics.median(access_counts),
                    "max_access_count": max(access_counts),
                    "most_accessed_memory": max(memories, key=lambda m: m.access_count).id
                })
                
                # Recent access analysis
                recent_accesses = [m for m in memories if m.last_accessed and 
                                 (datetime.now() - m.last_accessed).days < 7]
                access_stats["recently_accessed_count"] = len(recent_accesses)
            
            return access_stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing access patterns: {e}")
            return {}
    
    def _analyze_content_characteristics(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Analyze content characteristics of memories."""
        try:
            if not memories:
                return {}
            
            # Tag analysis
            all_tags = []
            for memory in memories:
                all_tags.extend(memory.tags)
            
            tag_counts = Counter(all_tags)
            
            # Content length analysis
            lengths = [len(m.content) for m in memories]
            
            # Session analysis
            session_counts = Counter(m.session_id for m in memories)
            
            return {
                "unique_tags": len(tag_counts),
                "most_common_tags": dict(tag_counts.most_common(10)),
                "content_length_stats": {
                    "min": min(lengths),
                    "max": max(lengths),
                    "mean": statistics.mean(lengths),
                    "median": statistics.median(lengths)
                },
                "unique_sessions": len(session_counts),
                "session_distribution": dict(session_counts.most_common(5))
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing content characteristics: {e}")
            return {}
    
    def _discover_all_patterns(self, time_range: Tuple[datetime, datetime]) -> List[MemoryPattern]:
        """Discover all types of patterns in the specified time range."""
        try:
            all_patterns = []
            
            # Discover temporal patterns
            temporal_patterns = self._discover_temporal_patterns(time_range)
            all_patterns.extend(temporal_patterns)
            
            # Discover semantic patterns
            semantic_patterns = self._discover_semantic_patterns(time_range)
            all_patterns.extend(semantic_patterns)
            
            # Discover behavioral patterns
            behavioral_patterns = self._discover_behavioral_patterns(time_range)
            all_patterns.extend(behavioral_patterns)
            
            # Discover structural patterns
            structural_patterns = self._discover_structural_patterns(time_range)
            all_patterns.extend(structural_patterns)
            
            self.analytics_stats["patterns_discovered"] += len(all_patterns)
            
            return all_patterns
            
        except Exception as e:
            self.logger.error(f"Error discovering patterns: {e}")
            return []
    
    def _discover_temporal_patterns(self, time_range: Tuple[datetime, datetime]) -> List[MemoryPattern]:
        """Discover temporal patterns in memory creation and access."""
        try:
            patterns = []
            filtered_memories = self._filter_memories_by_time(time_range)
            
            if len(filtered_memories) < self.pattern_thresholds["temporal_pattern_min_frequency"]:
                return patterns
            
            # Group memories by hour
            hourly_groups = defaultdict(list)
            for memory in filtered_memories:
                hour = memory.timestamp.hour
                hourly_groups[hour].append(memory)
            
            # Find peak activity hours
            hour_counts = {hour: len(memories) for hour, memories in hourly_groups.items()}
            if hour_counts:
                max_count = max(hour_counts.values())
                avg_count = statistics.mean(hour_counts.values())
                
                # Identify peak hours (significantly above average)
                peak_hours = [hour for hour, count in hour_counts.items() 
                             if count > avg_count * 1.5 and count >= self.pattern_thresholds["temporal_pattern_min_frequency"]]
                
                if peak_hours:
                    pattern = MemoryPattern(
                        pattern_id=f"temporal_peak_{int(datetime.now().timestamp())}",
                        pattern_type="temporal",
                        description=f"Peak activity during hours: {', '.join(map(str, peak_hours))}",
                        confidence=min(1.0, max_count / (avg_count + 1)),
                        frequency=sum(hour_counts[hour] for hour in peak_hours),
                        entities_involved=[],
                        time_range=time_range,
                        metadata={"peak_hours": peak_hours, "hour_counts": hour_counts}
                    )
                    patterns.append(pattern)
            
            # Detect regular intervals
            timestamps = sorted([m.timestamp for m in filtered_memories])
            if len(timestamps) > 3:
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # Hours
                    intervals.append(interval)
                
                # Check for regular patterns
                if len(intervals) > 5:
                    interval_std = statistics.stdev(intervals)
                    interval_mean = statistics.mean(intervals)
                    
                    if interval_std < interval_mean * 0.3:  # Low variance indicates regularity
                        pattern = MemoryPattern(
                            pattern_id=f"temporal_regular_{int(datetime.now().timestamp())}",
                            pattern_type="temporal",
                            description=f"Regular activity pattern with {interval_mean:.1f}h intervals",
                            confidence=1.0 - (interval_std / interval_mean),
                            frequency=len(intervals),
                            entities_involved=[],
                            time_range=time_range,
                            metadata={"average_interval": interval_mean, "std_deviation": interval_std}
                        )
                        patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error discovering temporal patterns: {e}")
            return []
    
    def _discover_semantic_patterns(self, time_range: Tuple[datetime, datetime]) -> List[MemoryPattern]:
        """Discover semantic patterns using clustering and similarity analysis."""
        try:
            patterns = []
            filtered_memories = self._filter_memories_by_time(time_range)
            
            if len(filtered_memories) < self.pattern_thresholds["semantic_cluster_min_size"]:
                return patterns
            
            # Extract embeddings
            embeddings = []
            memory_ids = []
            
            for memory in filtered_memories:
                if memory.embeddings is not None:
                    embeddings.append(memory.embeddings)
                    memory_ids.append(memory.id)
            
            if len(embeddings) < self.pattern_thresholds["semantic_cluster_min_size"]:
                return patterns
            
            # Perform clustering
            embeddings_array = np.array(embeddings)
            
            # Use DBSCAN for density-based clustering
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings_array)
            
            clustering = DBSCAN(eps=0.3, min_samples=3)
            cluster_labels = clustering.fit_predict(embeddings_scaled)
            
            # Analyze clusters
            unique_labels = set(cluster_labels)
            unique_labels.discard(-1)  # Remove noise cluster
            
            for cluster_id in unique_labels:
                cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_indices) >= self.pattern_thresholds["semantic_cluster_min_size"]:
                    cluster_memories = [filtered_memories[memory_ids.index(memory_ids[i])] 
                                      for i in cluster_indices]
                    
                    # Extract common themes
                    common_words = self._extract_common_themes(cluster_memories)
                    
                    pattern = MemoryPattern(
                        pattern_id=f"semantic_cluster_{cluster_id}_{int(datetime.now().timestamp())}",
                        pattern_type="semantic",
                        description=f"Semantic cluster around themes: {', '.join(common_words[:5])}",
                        confidence=len(cluster_indices) / len(filtered_memories),
                        frequency=len(cluster_indices),
                        entities_involved=[m.id for m in cluster_memories],
                        time_range=time_range,
                        metadata={"cluster_id": cluster_id, "common_themes": common_words}
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error discovering semantic patterns: {e}")
            return []
    
    def _discover_behavioral_patterns(self, time_range: Tuple[datetime, datetime]) -> List[MemoryPattern]:
        """Discover behavioral patterns in user interactions."""
        try:
            patterns = []
            filtered_memories = self._filter_memories_by_time(time_range)
            
            # Group by role to analyze interaction patterns
            role_sequences = []
            current_sequence = []
            
            for memory in sorted(filtered_memories, key=lambda m: m.timestamp):
                if memory.role != 'system':  # Focus on user-assistant interactions
                    current_sequence.append(memory.role)
                    
                    # If sequence gets too long, split it
                    if len(current_sequence) > 20:
                        role_sequences.append(current_sequence[:])
                        current_sequence = current_sequence[-5:]  # Keep some overlap
            
            if current_sequence:
                role_sequences.append(current_sequence)
            
            # Find common interaction patterns
            pattern_counts = defaultdict(int)
            for sequence in role_sequences:
                for i in range(len(sequence) - 2):
                    pattern = tuple(sequence[i:i+3])
                    pattern_counts[pattern] += 1
            
            # Identify significant patterns
            total_patterns = sum(pattern_counts.values())
            for pattern, count in pattern_counts.items():
                frequency = count / total_patterns if total_patterns > 0 else 0
                
                if (count >= self.pattern_thresholds["temporal_pattern_min_frequency"] and 
                    frequency >= self.pattern_thresholds["behavioral_pattern_confidence"]):
                    
                    pattern_obj = MemoryPattern(
                        pattern_id=f"behavioral_{'-'.join(pattern)}_{int(datetime.now().timestamp())}",
                        pattern_type="behavioral",
                        description=f"Common interaction pattern: {' → '.join(pattern)}",
                        confidence=frequency,
                        frequency=count,
                        entities_involved=[],
                        time_range=time_range,
                        metadata={"interaction_pattern": pattern, "frequency_ratio": frequency}
                    )
                    patterns.append(pattern_obj)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error discovering behavioral patterns: {e}")
            return []
    
    def _discover_structural_patterns(self, time_range: Tuple[datetime, datetime]) -> List[MemoryPattern]:
        """Discover structural patterns in the knowledge graph."""
        try:
            patterns = []
            
            # Analyze entity co-occurrence patterns
            entity_pairs = defaultdict(int)
            filtered_memories = self._filter_memories_by_time(time_range)
            
            for memory in filtered_memories:
                # Get entities mentioned in this memory
                memory_entities = []
                for entity_id, entity in self.knowledge_graph.entities.items():
                    if entity.name.lower() in memory.content.lower():
                        memory_entities.append(entity_id)
                
                # Count co-occurrences
                for i, entity1 in enumerate(memory_entities):
                    for entity2 in memory_entities[i+1:]:
                        pair = tuple(sorted([entity1, entity2]))
                        entity_pairs[pair] += 1
            
            # Find significant co-occurrence patterns
            total_pairs = sum(entity_pairs.values())
            for (entity1, entity2), count in entity_pairs.items():
                frequency = count / total_pairs if total_pairs > 0 else 0
                
                if (count >= self.pattern_thresholds["temporal_pattern_min_frequency"] and 
                    frequency >= self.pattern_thresholds["entity_co_occurrence_threshold"]):
                    
                    entity1_name = self.knowledge_graph.entities.get(entity1, {}).name or entity1
                    entity2_name = self.knowledge_graph.entities.get(entity2, {}).name or entity2
                    
                    pattern = MemoryPattern(
                        pattern_id=f"structural_cooccur_{entity1}_{entity2}_{int(datetime.now().timestamp())}",
                        pattern_type="structural",
                        description=f"Strong co-occurrence: {entity1_name} ↔ {entity2_name}",
                        confidence=frequency,
                        frequency=count,
                        entities_involved=[entity1, entity2],
                        time_range=time_range,
                        metadata={"entity_pair": (entity1, entity2), "co_occurrence_count": count}
                    )
                    patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error discovering structural patterns: {e}")
            return []
    
    def _extract_common_themes(self, memories: List[MemoryEntry]) -> List[str]:
        """Extract common themes from a group of memories."""
        try:
            # Combine all content
            combined_content = ' '.join([m.content for m in memories])
            
            # Simple word frequency analysis
            words = combined_content.lower().split()
            word_counts = Counter(words)
            
            # Filter out common stop words and short words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            filtered_words = [word for word, count in word_counts.items() 
                            if len(word) > 3 and word not in stop_words and count > 1]
            
            return filtered_words[:10]
            
        except Exception as e:
            self.logger.error(f"Error extracting common themes: {e}")
            return []
    
    def _generate_insights(self, stats: Dict[str, Any], patterns: List[MemoryPattern]) -> List[str]:
        """Generate insights from statistics and patterns."""
        try:
            insights = []
            
            # Memory usage insights
            memory_stats = stats.get("memory_stats", {})
            if memory_stats.get("total_memories", 0) > 0:
                avg_length = memory_stats.get("average_memory_length", 0)
                if avg_length > 500:
                    insights.append("📝 Your memories tend to be quite detailed, which is great for context retention.")
                elif avg_length < 100:
                    insights.append("📝 Your memories are concise. Consider adding more context for better retrieval.")
            
            # Temporal insights
            temporal_stats = stats.get("temporal_stats", {})
            peak_hour = temporal_stats.get("peak_hour")
            if peak_hour is not None:
                if 9 <= peak_hour <= 17:
                    insights.append(f"⏰ You're most active during business hours (peak at {peak_hour}:00).")
                elif 18 <= peak_hour <= 23:
                    insights.append(f"🌆 You're a night owl! Peak activity at {peak_hour}:00.")
                else:
                    insights.append(f"🌅 Early bird detected! Peak activity at {peak_hour}:00.")
            
            # Knowledge graph insights
            kg_stats = stats.get("knowledge_graph_stats", {})
            entity_count = kg_stats.get("total_entities", 0)
            relationship_count = kg_stats.get("total_relationships", 0)
            
            if entity_count > 50:
                insights.append(f"🕸️ Rich knowledge graph with {entity_count} entities and {relationship_count} relationships.")
            
            # Pattern insights
            if patterns:
                semantic_patterns = [p for p in patterns if p.pattern_type == "semantic"]
                temporal_patterns = [p for p in patterns if p.pattern_type == "temporal"]
                
                if semantic_patterns:
                    insights.append(f"🎯 Discovered {len(semantic_patterns)} semantic themes in your conversations.")
                
                if temporal_patterns:
                    insights.append(f"⏱️ Found {len(temporal_patterns)} temporal patterns in your activity.")
            
            # Access pattern insights
            access_stats = stats.get("access_patterns", {})
            never_accessed = access_stats.get("never_accessed", 0)
            total_memories = memory_stats.get("total_memories", 1)
            
            if never_accessed / total_memories > 0.5:
                insights.append("💡 Many memories are never accessed. Consider regular memory reviews.")
            
            self.analytics_stats["insights_generated"] += len(insights)
            return insights
            
        except Exception as e:
            self.logger.error(f"Error generating insights: {e}")
            return ["⚠️ Unable to generate insights due to analysis error."]
    
    def _generate_recommendations(self, 
                                stats: Dict[str, Any], 
                                patterns: List[MemoryPattern], 
                                insights: List[str]) -> List[str]:
        """Generate actionable recommendations."""
        try:
            recommendations = []
            
            # Memory management recommendations
            memory_stats = stats.get("memory_stats", {})
            total_memories = memory_stats.get("total_memories", 0)
            
            if total_memories > 1000:
                recommendations.append("🗂️ Consider implementing memory archival for older, less-accessed content.")
            
            # Search optimization recommendations
            search_stats = stats.get("search_stats", {})
            avg_results = search_stats.get("average_results", 0)
            
            if avg_results < 3:
                recommendations.append("🔍 Search queries might be too specific. Try broader terms for better results.")
            elif avg_results > 20:
                recommendations.append("🎯 Search results are numerous. Use more specific queries or filters.")
            
            # Knowledge graph recommendations
            kg_stats = stats.get("knowledge_graph_stats", {})
            entity_count = kg_stats.get("total_entities", 0)
            relationship_count = kg_stats.get("total_relationships", 0)
            
            if entity_count > 0 and relationship_count / entity_count < 0.5:
                recommendations.append("🔗 Knowledge graph could benefit from more relationship discovery.")
            
            # Pattern-based recommendations
            temporal_patterns = [p for p in patterns if p.pattern_type == "temporal"]
            if temporal_patterns:
                recommendations.append("⏰ Use discovered temporal patterns to schedule automated memory reviews.")
            
            # Access pattern recommendations
            access_stats = stats.get("access_patterns", {})
            recently_accessed = access_stats.get("recently_accessed_count", 0)
            
            if recently_accessed < total_memories * 0.1:
                recommendations.append("📚 Consider implementing memory suggestions to surface relevant old content.")
            
            # Content quality recommendations
            content_stats = stats.get("content_analysis", {})
            unique_tags = content_stats.get("unique_tags", 0)
            
            if unique_tags < total_memories * 0.1:
                recommendations.append("🏷️ Add more diverse tags to improve memory categorization and retrieval.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["⚠️ Unable to generate recommendations due to analysis error."]
    
    def _create_report_summary(self, 
                             stats: Dict[str, Any], 
                             patterns: List[MemoryPattern], 
                             insights: List[str]) -> str:
        """Create a concise summary of the analysis."""
        try:
            memory_count = stats.get("memory_stats", {}).get("total_memories", 0)
            pattern_count = len(patterns)
            insight_count = len(insights)
            
            summary = f"""
            Memory Analytics Summary:
            
            📊 Analyzed {memory_count} memories across the specified time period
            🔍 Discovered {pattern_count} significant patterns in your data
            💡 Generated {insight_count} key insights about your memory usage
            
            The analysis reveals interesting patterns in your interaction behavior,
            content themes, and memory access patterns. These insights can help
            optimize your AI assistant experience and improve knowledge retention.
            """
            
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"Error creating report summary: {e}")
            return "Summary generation failed due to analysis error."
    
    def _create_visualizations(self, 
                             stats: Dict[str, Any], 
                             patterns: List[MemoryPattern], 
                             time_range: Tuple[datetime, datetime]) -> List[str]:
        """Create visualizations for the analytics report."""
        try:
            viz_paths = []
            
            # Only create visualizations if matplotlib is available
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
            except ImportError:
                self.logger.warning("Matplotlib/Seaborn not available, skipping visualizations")
                return viz_paths
            
            # Set style
            plt.style.use('default')
            sns.set_palette(self.viz_settings["color_palette"])
            
            # 1. Memory creation timeline
            timeline_path = self._create_timeline_visualization(stats, time_range)
            if timeline_path:
                viz_paths.append(timeline_path)
            
            # 2. Role distribution pie chart
            role_dist_path = self._create_role_distribution_chart(stats)
            if role_dist_path:
                viz_paths.append(role_dist_path)
            
            # 3. Hourly activity heatmap
            activity_path = self._create_activity_heatmap(stats)
            if activity_path:
                viz_paths.append(activity_path)
            
            # 4. Knowledge graph visualization
            kg_path = self._create_knowledge_graph_visualization()
            if kg_path:
                viz_paths.append(kg_path)
            
            # 5. Pattern discovery chart
            pattern_path = self._create_pattern_visualization(patterns)
            if pattern_path:
                viz_paths.append(pattern_path)
            
            return viz_paths
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            return []
    
    def _create_timeline_visualization(self, stats: Dict[str, Any], time_range: Tuple[datetime, datetime]) -> Optional[str]:
        """Create a timeline visualization of memory creation."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            temporal_stats = stats.get("temporal_stats", {})
            daily_activity = temporal_stats.get("daily_activity", {})
            
            if not daily_activity:
                return None
            
            # Prepare data
            dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in daily_activity.keys()]
            counts = list(daily_activity.values())
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.viz_settings["figure_size"])
            ax.plot(dates, counts, marker='o', linewidth=2, markersize=6)
            
            ax.set_title('Memory Creation Timeline', fontsize=16, fontweight='bold')
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Number of Memories', fontsize=12)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            viz_path = f"{self.analytics_path}/visualizations/timeline_{int(datetime.now().timestamp())}.png"
            plt.savefig(viz_path, dpi=self.viz_settings["dpi"], bbox_inches='tight')
            plt.close()
            
            return viz_path
            
        except Exception as e:
            self.logger.error(f"Error creating timeline visualization: {e}")
            return None
    
    def _create_role_distribution_chart(self, stats: Dict[str, Any]) -> Optional[str]:
        """Create a pie chart of role distribution."""
        try:
            import matplotlib.pyplot as plt
            
            role_dist = stats.get("role_distribution", {})
            if not role_dist:
                return None
            
            # Prepare data
            roles = list(role_dist.keys())
            counts = list(role_dist.values())
            
            # Create plot
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(counts, labels=roles, autopct='%1.1f%%', startangle=90)
            
            ax.set_title('Role Distribution in Memories', fontsize=16, fontweight='bold')
            
            # Beautify text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            
            # Save plot
            viz_path = f"{self.analytics_path}/visualizations/role_distribution_{int(datetime.now().timestamp())}.png"
            plt.savefig(viz_path, dpi=self.viz_settings["dpi"], bbox_inches='tight')
            plt.close()
            
            return viz_path
            
        except Exception as e:
            self.logger.error(f"Error creating role distribution chart: {e}")
            return None
    
    def _create_activity_heatmap(self, stats: Dict[str, Any]) -> Optional[str]:
        """Create an hourly activity heatmap."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            
            temporal_stats = stats.get("temporal_stats", {})
            hourly_dist = temporal_stats.get("hourly_distribution", {})
            
            if not hourly_dist:
                return None
            
            # Prepare data for heatmap (24 hours x 7 days)
            activity_matrix = np.zeros((7, 24))
            
            # Fill with available data (simplified for demo)
            for hour, count in hourly_dist.items():
                activity_matrix[:, hour] = count / 7  # Distribute across days
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.viz_settings["figure_size"])
            
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            hours = [f'{h:02d}:00' for h in range(24)]
            
            sns.heatmap(activity_matrix, 
                       xticklabels=hours[::2],  # Show every 2 hours
                       yticklabels=days,
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Activity Level'},
                       ax=ax)
            
            ax.set_title('Activity Heatmap by Hour and Day', fontsize=16, fontweight='bold')
            ax.set_xlabel('Hour of Day', fontsize=12)
            ax.set_ylabel('Day of Week', fontsize=12)
            
            plt.tight_layout()
            
            # Save plot
            viz_path = f"{self.analytics_path}/visualizations/activity_heatmap_{int(datetime.now().timestamp())}.png"
            plt.savefig(viz_path, dpi=self.viz_settings["dpi"], bbox_inches='tight')
            plt.close()
            
            return viz_path
            
        except Exception as e:
            self.logger.error(f"Error creating activity heatmap: {e}")
            return None
    
    def _create_knowledge_graph_visualization(self) -> Optional[str]:
        """Create a visualization of the knowledge graph."""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            if self.knowledge_graph.graph.number_of_nodes() == 0:
                return None
            
            # Create a subgraph with most important nodes
            top_entities = sorted(
                self.knowledge_graph.entities.items(),
                key=lambda x: self.knowledge_graph.calculate_entity_importance(x[0]),
                reverse=True
            )[:20]  # Top 20 entities
            
            subgraph = self.knowledge_graph.graph.subgraph([entity_id for entity_id, _ in top_entities])
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.viz_settings["figure_size"])
            
            # Layout
            pos = nx.spring_layout(subgraph, k=3, iterations=50)
            
            # Draw nodes
            node_sizes = [self.knowledge_graph.calculate_entity_importance(node) * 1000 + 100 
                         for node in subgraph.nodes()]
            nx.draw_networkx_nodes(subgraph, pos, node_size=node_sizes, 
                                 node_color='lightblue', alpha=0.7, ax=ax)
            
            # Draw edges
            nx.draw_networkx_edges(subgraph, pos, alpha=0.5, ax=ax)
            
            # Draw labels
            labels = {node: self.knowledge_graph.entities[node].name[:10] 
                     for node in subgraph.nodes() if node in self.knowledge_graph.entities}
            nx.draw_networkx_labels(subgraph, pos, labels, font_size=8, ax=ax)
            
            ax.set_title('Knowledge Graph Structure (Top Entities)', fontsize=16, fontweight='bold')
            ax.axis('off')
            
            plt.tight_layout()
            
            # Save plot
            viz_path = f"{self.analytics_path}/visualizations/knowledge_graph_{int(datetime.now().timestamp())}.png"
            plt.savefig(viz_path, dpi=self.viz_settings["dpi"], bbox_inches='tight')
            plt.close()
            
            return viz_path
            
        except Exception as e:
            self.logger.error(f"Error creating knowledge graph visualization: {e}")
            return None
    
    def _create_pattern_visualization(self, patterns: List[MemoryPattern]) -> Optional[str]:
        """Create a visualization of discovered patterns."""
        try:
            import matplotlib.pyplot as plt
            
            if not patterns:
                return None
            
            # Group patterns by type
            pattern_types = defaultdict(list)
            for pattern in patterns:
                pattern_types[pattern.pattern_type].append(pattern)
            
            # Create plot
            fig, ax = plt.subplots(figsize=self.viz_settings["figure_size"])
            
            types = list(pattern_types.keys())
            counts = [len(patterns) for patterns in pattern_types.values()]
            confidences = [statistics.mean([p.confidence for p in patterns]) 
                          for patterns in pattern_types.values()]
            
            # Create bar chart
            x = range(len(types))
            bars = ax.bar(x, counts, alpha=0.7)
            
            # Color bars by confidence
            for bar, confidence in zip(bars, confidences):
                bar.set_color(plt.cm.viridis(confidence))
            
            ax.set_title('Discovered Patterns by Type', fontsize=16, fontweight='bold')
            ax.set_xlabel('Pattern Type', fontsize=12)
            ax.set_ylabel('Number of Patterns', fontsize=12)
            ax.set_xticks(x)
            ax.set_xticklabels(types)
            
            # Add confidence values on bars
            for i, (count, confidence) in enumerate(zip(counts, confidences)):
                ax.text(i, count + 0.1, f'{confidence:.2f}', 
                       ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            viz_path = f"{self.analytics_path}/visualizations/patterns_{int(datetime.now().timestamp())}.png"
            plt.savefig(viz_path, dpi=self.viz_settings["dpi"], bbox_inches='tight')
            plt.close()
            
            return viz_path
            
        except Exception as e:
            self.logger.error(f"Error creating pattern visualization: {e}")
            return None
    
    def _save_report(self, report: InsightReport):
        """Save the analytics report to disk."""
        try:
            report_path = f"{self.analytics_path}/reports/report_{report.report_id}.json"
            
            # Convert report to serializable format
            report_dict = asdict(report)
            
            # Convert datetime objects to strings
            report_dict['generated_at'] = report.generated_at.isoformat()
            report_dict['time_range'] = [t.isoformat() for t in report.time_range]
            
            # Convert pattern time ranges
            for pattern_dict in report_dict['patterns_discovered']:
                if 'time_range' in pattern_dict and pattern_dict['time_range']:
                    pattern_dict['time_range'] = [t.isoformat() for t in pattern_dict['time_range']]
            
            with open(report_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            
            self.logger.info(f"Analytics report saved: {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")
    
    def get_analytics_statistics(self) -> Dict[str, Any]:
        """Get comprehensive analytics statistics."""
        return {
            **self.analytics_stats,
            "cache_size": len(self.analysis_cache),
            "pattern_thresholds": self.pattern_thresholds,
            "last_analysis": datetime.now().isoformat()
        }
    
    def update_cache(self, vector_memory: VectorMemory):
        """
        Update the analytics cache with new memory data.
        
        Args:
            vector_memory: The vector memory instance to analyze
        """
        try:
            cache_key = f"memory_stats_{datetime.now().strftime('%Y%m%d_%H')}"
            
            # Check if cache entry already exists and is recent
            if cache_key in self.analysis_cache:
                cache_time = self.analysis_cache[cache_key].get('timestamp')
                if cache_time and datetime.fromisoformat(cache_time) > datetime.now() - self.cache_expiry:
                    self.logger.debug("Cache is still fresh, skipping update")
                    return
            
            # Generate fresh analytics data
            current_stats = {
                'timestamp': datetime.now().isoformat(),
                'total_memories': len(vector_memory.memory_entries),
                'memory_distribution': {},
                'importance_stats': {},
                'temporal_distribution': {},
                'session_stats': {}
            }
            
            # Analyze memory distribution by role
            role_counts = defaultdict(int)
            importance_values = []
            temporal_data = defaultdict(int)
            session_data = defaultdict(int)
            
            for memory in vector_memory.memory_entries.values():
                # Role distribution
                role_counts[memory.role] += 1
                
                # Importance statistics
                importance_values.append(memory.importance)
                
                # Temporal distribution (by day)
                day_key = memory.timestamp.strftime('%Y-%m-%d')
                temporal_data[day_key] += 1
                
                # Session statistics
                session_data[memory.session_id] += 1
            
            # Update cache with computed statistics
            current_stats['memory_distribution'] = dict(role_counts)
            
            if importance_values:
                current_stats['importance_stats'] = {
                    'mean': statistics.mean(importance_values),
                    'median': statistics.median(importance_values),
                    'std_dev': statistics.stdev(importance_values) if len(importance_values) > 1 else 0,
                    'min': min(importance_values),
                    'max': max(importance_values)
                }
            
            current_stats['temporal_distribution'] = dict(temporal_data)
            current_stats['session_stats'] = {
                'total_sessions': len(session_data),
                'avg_memories_per_session': statistics.mean(session_data.values()) if session_data else 0
            }
            
            # Store in cache
            self.analysis_cache[cache_key] = current_stats
            
            # Clean up old cache entries
            cutoff_time = datetime.now() - timedelta(days=7)
            keys_to_remove = []
            for key, data in self.analysis_cache.items():
                if 'timestamp' in data:
                    try:
                        entry_time = datetime.fromisoformat(data['timestamp'])
                        if entry_time < cutoff_time:
                            keys_to_remove.append(key)
                    except (ValueError, TypeError):
                        keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.analysis_cache[key]
            
            self.logger.info(f"Analytics cache updated with {len(vector_memory.memory_entries)} memories")
            
        except Exception as e:
            self.logger.error(f"Error updating analytics cache: {e}")
    
    def get_cached_insights(self, max_age_hours: int = 6) -> Optional[Dict[str, Any]]:
        """
        Get cached insights if available and fresh.
        
        Args:
            max_age_hours: Maximum age of cache in hours
            
        Returns:
            Cached insights or None if not available/stale
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
            
            for cache_key, data in self.analysis_cache.items():
                if 'timestamp' in data:
                    try:
                        cache_time = datetime.fromisoformat(data['timestamp'])
                        if cache_time > cutoff_time:
                            return data
                    except (ValueError, TypeError):
                        continue
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving cached insights: {e}")
            return None
    
    def clear_cache(self):
        """Clear the analytics cache."""
        try:
            cache_size = len(self.analysis_cache)
            self.analysis_cache.clear()
            self.logger.info(f"Cleared analytics cache ({cache_size} entries)")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
