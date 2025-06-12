"""
Knowledge Graph Implementation for AgenticSeek
Advanced knowledge representation with entity relationships, concept mapping, and graph-based reasoning.
"""

import json
import uuid
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import re

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from sources.logger import Logger
from sources.knowledge.vectorMemory import MemoryEntry


@dataclass
class Entity:
    """Knowledge graph entity with properties and relationships."""
    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any]
    confidence: float = 1.0
    first_seen: datetime = None
    last_updated: datetime = None
    mention_count: int = 0
    
    def __post_init__(self):
        if self.first_seen is None:
            self.first_seen = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()


@dataclass
class Relationship:
    """Knowledge graph relationship between entities."""
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    properties: Dict[str, Any]
    confidence: float = 1.0
    evidence: List[str] = None  # Memory IDs supporting this relationship
    created: datetime = None
    last_confirmed: datetime = None
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []
        if self.created is None:
            self.created = datetime.now()
        if self.last_confirmed is None:
            self.last_confirmed = datetime.now()


@dataclass
class Concept:
    """Abstract concept in the knowledge graph."""
    id: str
    name: str
    description: str
    keywords: List[str]
    related_entities: List[str]
    abstraction_level: int = 1  # 1=concrete, higher=more abstract
    confidence: float = 1.0
    created: datetime = None
    
    def __post_init__(self):
        if self.created is None:
            self.created = datetime.now()


class KnowledgeGraph:
    """
    Advanced knowledge graph system for organizing and reasoning about information.
    Supports entity extraction, relationship discovery, and concept formation.
    """
    
    def __init__(self, storage_path: str = "knowledge_graph"):
        self.logger = Logger("knowledge_graph.log")
        self.storage_path = storage_path
        
        # Core graph structure
        self.graph = nx.MultiDiGraph()  # Directed graph allowing multiple edges
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.concepts: Dict[str, Concept] = {}
        
        # Entity recognition patterns
        self.entity_patterns = self._initialize_entity_patterns()
        
        # Relationship extraction patterns
        self.relationship_patterns = self._initialize_relationship_patterns()
        
        # Clustering for concept discovery
        self.concept_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.concept_clusters = {}
        
        # Graph analysis cache
        self.centrality_cache = {}
        self.path_cache = {}
        self.cache_timestamp = datetime.now()
        
        # Statistics
        self.stats = {
            "entities_extracted": 0,
            "relationships_discovered": 0,
            "concepts_formed": 0,
            "graph_updates": 0,
            "queries_processed": 0
        }
        
        # Load existing graph if available
        self._load_graph()
    
    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for entity recognition."""
        return {
            "PERSON": [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # John Smith
                r'\b(?:Mr|Mrs|Ms|Dr|Prof)\.? [A-Z][a-z]+\b',  # Dr. Johnson
            ],
            "ORGANIZATION": [
                r'\b[A-Z][a-z]+ (?:Inc|Corp|LLC|Ltd|Company|Organization)\b',
                r'\b(?:Microsoft|Google|Apple|Amazon|Facebook|Tesla|IBM)\b',
            ],
            "TECHNOLOGY": [
                r'\b(?:Python|JavaScript|Java|C\+\+|React|Node\.js|Docker|Kubernetes)\b',
                r'\b(?:API|REST|GraphQL|JSON|XML|HTML|CSS|SQL|NoSQL)\b',
            ],
            "CONCEPT": [
                r'\b(?:machine learning|artificial intelligence|deep learning|neural network)\b',
                r'\b(?:algorithm|data structure|design pattern|architecture)\b',
            ],
            "TOOL": [
                r'\b(?:Visual Studio Code|PyCharm|IntelliJ|Eclipse|Sublime Text)\b',
                r'\b(?:Git|GitHub|GitLab|Docker|Jenkins|Kubernetes)\b',
            ],
            "FILE": [
                r'\b\w+\.[a-zA-Z]{2,4}\b',  # filename.ext
                r'\b/[\w/.-]+\b',  # Unix path
                r'\b[A-Z]:\\[\w\\.-]+\b',  # Windows path
            ]
        }
    
    def _initialize_relationship_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for relationship extraction."""
        return {
            "USES": [
                r'(\w+)\s+(?:uses|utilizes|employs)\s+(\w+)',
                r'(\w+)\s+is built with\s+(\w+)',
                r'(\w+)\s+depends on\s+(\w+)',
            ],
            "CREATES": [
                r'(\w+)\s+(?:creates|generates|produces)\s+(\w+)',
                r'(\w+)\s+(?:developed|built|designed)\s+(\w+)',
            ],
            "CONTAINS": [
                r'(\w+)\s+(?:contains|includes|has)\s+(\w+)',
                r'(\w+)\s+consists of\s+(\w+)',
            ],
            "IMPLEMENTS": [
                r'(\w+)\s+implements\s+(\w+)',
                r'(\w+)\s+is an implementation of\s+(\w+)',
            ],
            "INHERITS": [
                r'(\w+)\s+(?:inherits from|extends)\s+(\w+)',
                r'(\w+)\s+is a (?:subclass|child) of\s+(\w+)',
            ],
            "SOLVES": [
                r'(\w+)\s+solves\s+(\w+)',
                r'(\w+)\s+is a solution (?:to|for)\s+(\w+)',
            ]
        }
    
    def extract_entities_from_memory(self, memory_entry: MemoryEntry) -> List[Entity]:
        """Extract entities from a memory entry."""
        try:
            extracted_entities = []
            content = memory_entry.content
            
            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        entity_text = match.group().strip()
                        if len(entity_text) < 2:
                            continue
                        
                        # Create or update entity
                        entity_id = self._generate_entity_id(entity_text, entity_type)
                        
                        if entity_id in self.entities:
                            # Update existing entity
                            entity = self.entities[entity_id]
                            entity.mention_count += 1
                            entity.last_updated = datetime.now()
                        else:
                            # Create new entity
                            entity = Entity(
                                id=entity_id,
                                name=entity_text,
                                entity_type=entity_type,
                                properties={
                                    "source_memory": memory_entry.id,
                                    "context": content[max(0, match.start()-50):match.end()+50]
                                },
                                mention_count=1
                            )
                            self.entities[entity_id] = entity
                            extracted_entities.append(entity)
                            self.stats["entities_extracted"] += 1
            
            return extracted_entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {e}")
            return []
    
    def extract_relationships_from_memory(self, memory_entry: MemoryEntry, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships from a memory entry."""
        try:
            extracted_relationships = []
            content = memory_entry.content
            
            # Extract explicit relationships using patterns
            for rel_type, patterns in self.relationship_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if len(match.groups()) >= 2:
                            source_text = match.group(1).strip()
                            target_text = match.group(2).strip()
                            
                            # Find corresponding entities
                            source_entity = self._find_entity_by_name(source_text)
                            target_entity = self._find_entity_by_name(target_text)
                            
                            if source_entity and target_entity:
                                rel_id = f"{source_entity.id}_{rel_type}_{target_entity.id}"
                                
                                if rel_id not in self.relationships:
                                    relationship = Relationship(
                                        id=rel_id,
                                        source_entity_id=source_entity.id,
                                        target_entity_id=target_entity.id,
                                        relationship_type=rel_type,
                                        properties={
                                            "source_memory": memory_entry.id,
                                            "context": match.group()
                                        },
                                        evidence=[memory_entry.id]
                                    )
                                    self.relationships[rel_id] = relationship
                                    extracted_relationships.append(relationship)
                                    self.stats["relationships_discovered"] += 1
                                else:
                                    # Update existing relationship
                                    existing_rel = self.relationships[rel_id]
                                    if memory_entry.id not in existing_rel.evidence:
                                        existing_rel.evidence.append(memory_entry.id)
                                        existing_rel.last_confirmed = datetime.now()
                                        existing_rel.confidence = min(1.0, existing_rel.confidence + 0.1)
            
            # Extract co-occurrence relationships
            co_occurrence_rels = self._extract_co_occurrence_relationships(memory_entry, entities)
            extracted_relationships.extend(co_occurrence_rels)
            
            return extracted_relationships
            
        except Exception as e:
            self.logger.error(f"Error extracting relationships: {e}")
            return []
    
    def _extract_co_occurrence_relationships(self, memory_entry: MemoryEntry, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships based on entity co-occurrence."""
        try:
            co_occurrence_rels = []
            
            # Look for entities that appear in the same context
            for i, entity1 in enumerate(entities):
                for j, entity2 in enumerate(entities[i+1:], i+1):
                    # Check if entities appear close to each other
                    content = memory_entry.content.lower()
                    pos1 = content.find(entity1.name.lower())
                    pos2 = content.find(entity2.name.lower())
                    
                    if pos1 != -1 and pos2 != -1:
                        distance = abs(pos1 - pos2)
                        
                        # If entities are close, create a co-occurrence relationship
                        if distance < 100:  # Within 100 characters
                            rel_id = f"{entity1.id}_CO_OCCURS_{entity2.id}"
                            
                            if rel_id not in self.relationships:
                                confidence = max(0.1, 1.0 - distance / 100.0)
                                relationship = Relationship(
                                    id=rel_id,
                                    source_entity_id=entity1.id,
                                    target_entity_id=entity2.id,
                                    relationship_type="CO_OCCURS",
                                    properties={
                                        "distance": distance,
                                        "source_memory": memory_entry.id
                                    },
                                    confidence=confidence,
                                    evidence=[memory_entry.id]
                                )
                                self.relationships[rel_id] = relationship
                                co_occurrence_rels.append(relationship)
            
            return co_occurrence_rels
            
        except Exception as e:
            self.logger.error(f"Error extracting co-occurrence relationships: {e}")
            return []
    
    def _generate_entity_id(self, name: str, entity_type: str) -> str:
        """Generate a unique ID for an entity."""
        # Normalize name for consistent IDs
        normalized_name = re.sub(r'[^\w\s]', '', name.lower().strip())
        normalized_name = '_'.join(normalized_name.split())
        return f"{entity_type}_{normalized_name}"
    
    def _find_entity_by_name(self, name: str) -> Optional[Entity]:
        """Find an entity by its name (case-insensitive)."""
        name_lower = name.lower()
        for entity in self.entities.values():
            if entity.name.lower() == name_lower:
                return entity
        return None
    
    def update_graph_structure(self):
        """Update the NetworkX graph structure with current entities and relationships."""
        try:
            self.graph.clear()
            
            # Add entity nodes
            for entity in self.entities.values():
                self.graph.add_node(
                    entity.id,
                    name=entity.name,
                    type=entity.entity_type,
                    confidence=entity.confidence,
                    mention_count=entity.mention_count
                )
            
            # Add relationship edges
            for relationship in self.relationships.values():
                self.graph.add_edge(
                    relationship.source_entity_id,
                    relationship.target_entity_id,
                    id=relationship.id,
                    type=relationship.relationship_type,
                    confidence=relationship.confidence,
                    evidence_count=len(relationship.evidence)
                )
            
            # Clear caches after structure update
            self.centrality_cache.clear()
            self.path_cache.clear()
            self.cache_timestamp = datetime.now()
            
            self.stats["graph_updates"] += 1
            self.logger.info(f"Graph structure updated: {len(self.entities)} entities, {len(self.relationships)} relationships")
            
        except Exception as e:
            self.logger.error(f"Error updating graph structure: {e}")
    
    def find_paths_between_entities(self, source_entity_id: str, target_entity_id: str, max_length: int = 3) -> List[List[str]]:
        """Find paths between two entities in the knowledge graph."""
        try:
            # Check cache first
            cache_key = f"{source_entity_id}->{target_entity_id}:{max_length}"
            if cache_key in self.path_cache:
                return self.path_cache[cache_key]
            
            if source_entity_id not in self.graph or target_entity_id not in self.graph:
                return []
            
            # Find all simple paths up to max_length
            try:
                paths = list(nx.all_simple_paths(
                    self.graph, 
                    source_entity_id, 
                    target_entity_id, 
                    cutoff=max_length
                ))
                
                # Cache the result
                self.path_cache[cache_key] = paths
                return paths
                
            except nx.NetworkXNoPath:
                self.path_cache[cache_key] = []
                return []
            
        except Exception as e:
            self.logger.error(f"Error finding paths between entities: {e}")
            return []
    
    def get_entity_neighbors(self, entity_id: str, relationship_types: List[str] = None) -> List[Tuple[str, str, float]]:
        """Get neighboring entities and their relationship types."""
        try:
            if entity_id not in self.graph:
                return []
            
            neighbors = []
            
            # Get outgoing edges
            for neighbor_id in self.graph.successors(entity_id):
                for edge_data in self.graph[entity_id][neighbor_id].values():
                    rel_type = edge_data.get('type', 'UNKNOWN')
                    confidence = edge_data.get('confidence', 0.0)
                    
                    if relationship_types is None or rel_type in relationship_types:
                        neighbors.append((neighbor_id, rel_type, confidence))
            
            # Get incoming edges
            for neighbor_id in self.graph.predecessors(entity_id):
                for edge_data in self.graph[neighbor_id][entity_id].values():
                    rel_type = edge_data.get('type', 'UNKNOWN')
                    confidence = edge_data.get('confidence', 0.0)
                    
                    if relationship_types is None or rel_type in relationship_types:
                        neighbors.append((neighbor_id, f"INVERSE_{rel_type}", confidence))
            
            return neighbors
            
        except Exception as e:
            self.logger.error(f"Error getting entity neighbors: {e}")
            return []
    
    def calculate_entity_importance(self, entity_id: str) -> float:
        """Calculate the importance of an entity based on graph metrics."""
        try:
            if entity_id not in self.graph:
                return 0.0
            
            # Use cached centrality if available and recent
            if (self.centrality_cache and 
                datetime.now() - self.cache_timestamp < timedelta(hours=1)):
                return self.centrality_cache.get(entity_id, 0.0)
            
            # Calculate various centrality measures
            degree_centrality = nx.degree_centrality(self.graph).get(entity_id, 0.0)
            
            try:
                betweenness_centrality = nx.betweenness_centrality(self.graph).get(entity_id, 0.0)
            except:
                betweenness_centrality = 0.0
            
            try:
                eigenvector_centrality = nx.eigenvector_centrality(self.graph, max_iter=1000).get(entity_id, 0.0)
            except:
                eigenvector_centrality = 0.0
            
            # Combine centrality measures with entity-specific factors
            entity = self.entities.get(entity_id)
            mention_factor = min(1.0, entity.mention_count / 10.0) if entity else 0.0
            
            # Weighted combination
            importance = (
                0.3 * degree_centrality +
                0.3 * betweenness_centrality +
                0.2 * eigenvector_centrality +
                0.2 * mention_factor
            )
            
            # Cache the result
            self.centrality_cache[entity_id] = importance
            
            return importance
            
        except Exception as e:
            self.logger.error(f"Error calculating entity importance: {e}")
            return 0.0
    
    def discover_concepts(self, min_cluster_size: int = 3) -> List[Concept]:
        """Discover abstract concepts by clustering related entities."""
        try:
            if len(self.entities) < min_cluster_size * 2:
                return []
            
            # Prepare entity descriptions for clustering
            entity_descriptions = []
            entity_ids = []
            
            for entity in self.entities.values():
                # Create description from entity properties and relationships
                description_parts = [entity.name, entity.entity_type]
                
                # Add context from properties
                if 'context' in entity.properties:
                    description_parts.append(entity.properties['context'])
                
                # Add related entity names
                neighbors = self.get_entity_neighbors(entity.id)
                neighbor_names = [self.entities[neighbor_id].name 
                                for neighbor_id, _, _ in neighbors 
                                if neighbor_id in self.entities]
                description_parts.extend(neighbor_names[:5])  # Limit to 5 neighbors
                
                entity_descriptions.append(' '.join(description_parts))
                entity_ids.append(entity.id)
            
            # Vectorize descriptions
            try:
                tfidf_matrix = self.concept_vectorizer.fit_transform(entity_descriptions)
            except ValueError:
                # Not enough data for TF-IDF
                return []
            
            # Cluster entities
            n_clusters = min(10, len(entity_ids) // min_cluster_size)
            if n_clusters < 2:
                return []
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Create concepts from clusters
            discovered_concepts = []
            for cluster_id in range(n_clusters):
                cluster_entities = [entity_ids[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_entities) >= min_cluster_size:
                    concept = self._create_concept_from_cluster(cluster_entities, cluster_id)
                    if concept:
                        discovered_concepts.append(concept)
                        self.concepts[concept.id] = concept
                        self.stats["concepts_formed"] += 1
            
            self.logger.info(f"Discovered {len(discovered_concepts)} new concepts")
            return discovered_concepts
            
        except Exception as e:
            self.logger.error(f"Error discovering concepts: {e}")
            return []
    
    def _create_concept_from_cluster(self, entity_ids: List[str], cluster_id: int) -> Optional[Concept]:
        """Create a concept from a cluster of related entities."""
        try:
            cluster_entities = [self.entities[entity_id] for entity_id in entity_ids if entity_id in self.entities]
            
            if not cluster_entities:
                return None
            
            # Analyze entity types in cluster
            entity_types = [entity.entity_type for entity in cluster_entities]
            most_common_type = max(set(entity_types), key=entity_types.count)
            
            # Generate concept name
            entity_names = [entity.name for entity in cluster_entities]
            
            # Try to find common words in entity names
            all_words = []
            for name in entity_names:
                words = re.findall(r'\w+', name.lower())
                all_words.extend(words)
            
            word_counts = defaultdict(int)
            for word in all_words:
                if len(word) > 2:  # Ignore very short words
                    word_counts[word] += 1
            
            # Use most common word as concept name base
            if word_counts:
                common_word = max(word_counts.items(), key=lambda x: x[1])[0]
                concept_name = f"{common_word.title()} Concept"
            else:
                concept_name = f"{most_common_type} Cluster {cluster_id}"
            
            # Generate description
            description = f"Concept representing {len(cluster_entities)} related {most_common_type.lower()} entities"
            
            # Extract keywords
            keywords = list(word_counts.keys())[:10]
            
            # Calculate confidence based on cluster cohesion
            confidence = min(1.0, len(cluster_entities) / 10.0)
            
            concept = Concept(
                id=f"concept_{cluster_id}_{int(datetime.now().timestamp())}",
                name=concept_name,
                description=description,
                keywords=keywords,
                related_entities=entity_ids,
                abstraction_level=2,  # Derived concepts have higher abstraction
                confidence=confidence
            )
            
            return concept
            
        except Exception as e:
            self.logger.error(f"Error creating concept from cluster: {e}")
            return None
    
    def query_knowledge_graph(self, query: str, query_type: str = "entity") -> Dict[str, Any]:
        """Query the knowledge graph for information."""
        try:
            self.stats["queries_processed"] += 1
            results = {"query": query, "type": query_type, "results": []}
            
            if query_type == "entity":
                # Search for entities by name
                query_lower = query.lower()
                for entity in self.entities.values():
                    if query_lower in entity.name.lower():
                        entity_info = {
                            "entity": asdict(entity),
                            "importance": self.calculate_entity_importance(entity.id),
                            "neighbors": self.get_entity_neighbors(entity.id)[:5]
                        }
                        results["results"].append(entity_info)
            
            elif query_type == "relationship":
                # Search for relationships by type
                query_upper = query.upper()
                for rel in self.relationships.values():
                    if query_upper in rel.relationship_type:
                        rel_info = {
                            "relationship": asdict(rel),
                            "source_entity": self.entities.get(rel.source_entity_id),
                            "target_entity": self.entities.get(rel.target_entity_id)
                        }
                        results["results"].append(rel_info)
            
            elif query_type == "concept":
                # Search for concepts
                query_lower = query.lower()
                for concept in self.concepts.values():
                    if (query_lower in concept.name.lower() or 
                        query_lower in concept.description.lower() or
                        any(query_lower in keyword for keyword in concept.keywords)):
                        results["results"].append(asdict(concept))
            
            elif query_type == "path":
                # Find paths between entities mentioned in query
                words = query.split()
                if len(words) >= 2:
                    source_entity = self._find_entity_by_name(words[0])
                    target_entity = self._find_entity_by_name(words[-1])
                    
                    if source_entity and target_entity:
                        paths = self.find_paths_between_entities(source_entity.id, target_entity.id)
                        results["results"] = [{
                            "source": source_entity.name,
                            "target": target_entity.name,
                            "paths": paths
                        }]
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error querying knowledge graph: {e}")
            return {"query": query, "type": query_type, "results": [], "error": str(e)}
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive knowledge graph statistics."""
        try:
            stats = self.stats.copy()
            
            # Graph structure statistics
            stats.update({
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
                "total_concepts": len(self.concepts),
                "graph_nodes": self.graph.number_of_nodes(),
                "graph_edges": self.graph.number_of_edges(),
                "graph_density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0.0
            })
            
            # Entity type distribution
            entity_types = defaultdict(int)
            for entity in self.entities.values():
                entity_types[entity.entity_type] += 1
            stats["entity_type_distribution"] = dict(entity_types)
            
            # Relationship type distribution
            rel_types = defaultdict(int)
            for rel in self.relationships.values():
                rel_types[rel.relationship_type] += 1
            stats["relationship_type_distribution"] = dict(rel_types)
            
            # Most important entities
            if self.entities:
                entity_importance = [(entity.id, self.calculate_entity_importance(entity.id)) 
                                   for entity in self.entities.values()]
                entity_importance.sort(key=lambda x: x[1], reverse=True)
                stats["most_important_entities"] = entity_importance[:5]
            
            # Graph connectivity
            if self.graph.number_of_nodes() > 0:
                stats["connected_components"] = nx.number_weakly_connected_components(self.graph)
                stats["average_clustering"] = nx.average_clustering(self.graph.to_undirected())
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting graph statistics: {e}")
            return self.stats
    
    def save_graph(self):
        """Save the knowledge graph to disk."""
        try:
            import os
            os.makedirs(self.storage_path, exist_ok=True)
            
            # Save entities
            entities_data = {entity_id: asdict(entity) for entity_id, entity in self.entities.items()}
            with open(f"{self.storage_path}/entities.json", 'w') as f:
                json.dump(entities_data, f, indent=2, default=str)
            
            # Save relationships
            relationships_data = {rel_id: asdict(rel) for rel_id, rel in self.relationships.items()}
            with open(f"{self.storage_path}/relationships.json", 'w') as f:
                json.dump(relationships_data, f, indent=2, default=str)
            
            # Save concepts
            concepts_data = {concept_id: asdict(concept) for concept_id, concept in self.concepts.items()}
            with open(f"{self.storage_path}/concepts.json", 'w') as f:
                json.dump(concepts_data, f, indent=2, default=str)
            
            # Save NetworkX graph
            nx.write_gpickle(self.graph, f"{self.storage_path}/graph.gpickle")
            
            # Save statistics
            with open(f"{self.storage_path}/statistics.json", 'w') as f:
                json.dump(self.get_graph_statistics(), f, indent=2, default=str)
            
            self.logger.info("Knowledge graph saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving knowledge graph: {e}")
    
    def _load_graph(self):
        """Load the knowledge graph from disk."""
        try:
            import os
            
            if not os.path.exists(self.storage_path):
                return
            
            # Load entities
            entities_path = f"{self.storage_path}/entities.json"
            if os.path.exists(entities_path):
                with open(entities_path, 'r') as f:
                    entities_data = json.load(f)
                    for entity_id, entity_dict in entities_data.items():
                        # Convert datetime strings back to datetime objects
                        if isinstance(entity_dict.get('first_seen'), str):
                            entity_dict['first_seen'] = datetime.fromisoformat(entity_dict['first_seen'])
                        if isinstance(entity_dict.get('last_updated'), str):
                            entity_dict['last_updated'] = datetime.fromisoformat(entity_dict['last_updated'])
                        
                        self.entities[entity_id] = Entity(**entity_dict)
            
            # Load relationships
            relationships_path = f"{self.storage_path}/relationships.json"
            if os.path.exists(relationships_path):
                with open(relationships_path, 'r') as f:
                    relationships_data = json.load(f)
                    for rel_id, rel_dict in relationships_data.items():
                        # Convert datetime strings back to datetime objects
                        if isinstance(rel_dict.get('created'), str):
                            rel_dict['created'] = datetime.fromisoformat(rel_dict['created'])
                        if isinstance(rel_dict.get('last_confirmed'), str):
                            rel_dict['last_confirmed'] = datetime.fromisoformat(rel_dict['last_confirmed'])
                        
                        self.relationships[rel_id] = Relationship(**rel_dict)
            
            # Load concepts
            concepts_path = f"{self.storage_path}/concepts.json"
            if os.path.exists(concepts_path):
                with open(concepts_path, 'r') as f:
                    concepts_data = json.load(f)
                    for concept_id, concept_dict in concepts_data.items():
                        # Convert datetime strings back to datetime objects
                        if isinstance(concept_dict.get('created'), str):
                            concept_dict['created'] = datetime.fromisoformat(concept_dict['created'])
                        
                        self.concepts[concept_id] = Concept(**concept_dict)
            
            # Load NetworkX graph
            graph_path = f"{self.storage_path}/graph.gpickle"
            if os.path.exists(graph_path):
                self.graph = nx.read_gpickle(graph_path)
            else:
                # Rebuild graph structure
                self.update_graph_structure()
            
            self.logger.info(f"Knowledge graph loaded: {len(self.entities)} entities, {len(self.relationships)} relationships, {len(self.concepts)} concepts")
            
        except Exception as e:
            self.logger.error(f"Error loading knowledge graph: {e}")
    
    def process_memory_entry(self, memory_entry: MemoryEntry) -> Dict[str, Any]:
        """Process a memory entry to extract knowledge and update the graph."""
        try:
            # Extract entities
            entities = self.extract_entities_from_memory(memory_entry)
            
            # Extract relationships
            relationships = self.extract_relationships_from_memory(memory_entry, entities)
            
            # Update graph structure
            self.update_graph_structure()
            
            # Try to discover new concepts periodically
            if len(self.entities) % 20 == 0:  # Every 20 new entities
                self.discover_concepts()
            
            return {
                "memory_id": memory_entry.id,
                "entities_extracted": len(entities),
                "relationships_extracted": len(relationships),
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing memory entry: {e}")
            return {"error": str(e)}
    
    def export_graph(self) -> Dict[str, Any]:
        """
        Export the knowledge graph to a serializable format.
        
        Returns:
            Dictionary containing graph data
        """
        try:
            export_data = {
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_entities": len(self.entities),
                    "total_relationships": len(self.relationships),
                    "total_concepts": len(self.concepts),
                    "graph_nodes": self.graph.number_of_nodes(),
                    "graph_edges": self.graph.number_of_edges()
                },
                "entities": [],
                "relationships": [],
                "concepts": [],
                "graph_structure": {
                    "nodes": list(self.graph.nodes(data=True)),
                    "edges": list(self.graph.edges(data=True))
                }
            }
            
            # Export entities
            for entity in self.entities.values():
                entity_dict = asdict(entity)
                entity_dict['first_seen'] = entity.first_seen.isoformat() if entity.first_seen else None
                entity_dict['last_updated'] = entity.last_updated.isoformat() if entity.last_updated else None
                export_data["entities"].append(entity_dict)
            
            # Export relationships
            for relationship in self.relationships.values():
                rel_dict = asdict(relationship)
                rel_dict['created'] = relationship.created.isoformat() if relationship.created else None
                rel_dict['last_confirmed'] = relationship.last_confirmed.isoformat() if relationship.last_confirmed else None
                export_data["relationships"].append(rel_dict)
            
            # Export concepts
            for concept in self.concepts.values():
                concept_dict = asdict(concept)
                concept_dict['created'] = concept.created.isoformat() if concept.created else None
                export_data["concepts"].append(concept_dict)
            
            self.logger.info(f"Exported knowledge graph with {len(self.entities)} entities, {len(self.relationships)} relationships, {len(self.concepts)} concepts")
            return export_data
            
        except Exception as e:
            self.logger.error(f"Error exporting knowledge graph: {e}")
            return {
                "error": str(e),
                "metadata": {
                    "export_timestamp": datetime.now().isoformat(),
                    "total_entities": 0,
                    "total_relationships": 0,
                    "total_concepts": 0
                }
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        try:
            return {
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
                "total_concepts": len(self.concepts),
                "graph_nodes": self.graph.number_of_nodes(),
                "graph_edges": self.graph.number_of_edges(),
                "entity_types": len(self.entity_patterns),
                "relationship_types": len(self.relationship_patterns)
            }
        except Exception as e:
            self.logger.error(f"Error getting knowledge graph stats: {e}")
            return {"error": str(e)}
    
    def cleanup_unused_entities(self) -> int:
        """Clean up entities that are no longer referenced."""
        try:
            # Find entities with no relationships and low mention count
            to_remove = []
            for entity_id, entity in self.entities.items():
                # Check if entity has any relationships
                has_relationships = any(
                    rel.source_entity_id == entity_id or rel.target_entity_id == entity_id
                    for rel in self.relationships.values()
                )
                
                # Remove if no relationships and low mention count
                if not has_relationships and entity.mention_count <= 1:
                    to_remove.append(entity_id)
            
            # Remove unused entities
            for entity_id in to_remove:
                del self.entities[entity_id]
                if entity_id in self.graph:
                    self.graph.remove_node(entity_id)
            
            cleanup_count = len(to_remove)
            if cleanup_count > 0:
                self.logger.info(f"Cleaned up {cleanup_count} unused entities")
            
            return cleanup_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up unused entities: {e}")
            return 0
