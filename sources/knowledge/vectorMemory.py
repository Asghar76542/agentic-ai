"""
Vector-based Memory Implementation for AgenticSeek
Enhanced memory system with vector embeddings, semantic search, and intelligent retrieval.
"""

import os
import json
import uuid
import time
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
import asyncio
from pathlib import Path

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

from sources.logger import Logger
from sources.utility import timer_decorator


@dataclass
class MemoryEntry:
    """Structured memory entry with metadata."""
    id: str
    content: str
    role: str
    timestamp: datetime
    session_id: str
    agent_type: str
    importance: float = 0.5
    tags: List[str] = None
    embeddings: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if isinstance(self.last_accessed, str) and self.last_accessed:
            self.last_accessed = datetime.fromisoformat(self.last_accessed)


class VectorMemory:
    """
    Advanced vector-based memory system with semantic search, hierarchical storage,
    and intelligent retrieval mechanisms.
    """
    
    def __init__(self, 
                 memory_path: str = "memory_vectors",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 max_memory_size: int = 10000,
                 similarity_threshold: float = 0.7,
                 importance_decay: float = 0.95,
                 session_id: Optional[str] = None):
        
        self.logger = Logger("vector_memory.log")
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(exist_ok=True)
        
        # Configuration
        self.max_memory_size = max_memory_size
        self.similarity_threshold = similarity_threshold
        self.importance_decay = importance_decay
        self.session_id = session_id or str(uuid.uuid4())
        
        # Initialize embedding model
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        self.tokenizer = None
        
        # Initialize vector database
        self.chroma_client = None
        self.collection = None
        
        # In-memory storage for fast access
        self.memory_entries: Dict[str, MemoryEntry] = {}
        self.faiss_index = None
        self.id_to_index = {}
        self.index_to_id = {}
        
        # Statistics and analytics
        self.stats = {
            "total_memories": 0,
            "queries_performed": 0,
            "cache_hits": 0,
            "average_retrieval_time": 0.0,
            "memory_usage_mb": 0.0
        }
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all memory system components."""
        try:
            self._load_embedding_model()
            self._initialize_chroma()
            self._initialize_faiss()
            self._load_existing_memories()
            self.logger.info("Vector memory system initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing vector memory: {e}")
            raise
    
    def _load_embedding_model(self):
        """Load the sentence transformer model for embeddings."""
        try:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.logger.info(f"Loaded embedding model: {self.embedding_model_name}")
        except Exception as e:
            self.logger.error(f"Error loading embedding model: {e}")
            raise
    
    def _initialize_chroma(self):
        """Initialize ChromaDB for persistent vector storage."""
        try:
            chroma_path = self.memory_path / "chroma_db"
            self.chroma_client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    is_persistent=True
                )
            )
            
            collection_name = "agenticseek_memories"
            try:
                self.collection = self.chroma_client.get_collection(collection_name)
            except:
                self.collection = self.chroma_client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            
            self.logger.info("ChromaDB initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def _initialize_faiss(self):
        """Initialize FAISS index for fast similarity search."""
        try:
            faiss_path = self.memory_path / "faiss_index.bin"
            
            if faiss_path.exists():
                self.faiss_index = faiss.read_index(str(faiss_path))
                self.logger.info("Loaded existing FAISS index")
            else:
                # Create new FAISS index
                self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
                self.logger.info("Created new FAISS index")
            
            # Load ID mappings
            mapping_path = self.memory_path / "id_mappings.json"
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    mappings = json.load(f)
                    self.id_to_index = mappings.get("id_to_index", {})
                    self.index_to_id = mappings.get("index_to_id", {})
        except Exception as e:
            self.logger.error(f"Error initializing FAISS: {e}")
            raise
    
    def _load_existing_memories(self):
        """Load existing memories from disk."""
        try:
            memories_path = self.memory_path / "memories.json"
            if memories_path.exists():
                with open(memories_path, 'r') as f:
                    data = json.load(f)
                    for entry_data in data:
                        entry = MemoryEntry(**entry_data)
                        self.memory_entries[entry.id] = entry
                
                self.stats["total_memories"] = len(self.memory_entries)
                self.logger.info(f"Loaded {len(self.memory_entries)} existing memories")
        except Exception as e:
            self.logger.error(f"Error loading existing memories: {e}")
    
    @timer_decorator
    def add_memory(self, 
                   content: str, 
                   role: str, 
                   agent_type: str = "default",
                   importance: float = 0.5,
                   tags: List[str] = None,
                   metadata: Dict[str, Any] = None) -> str:
        """
        Add a new memory entry with vector embedding.
        
        Args:
            content: The content to store
            role: Role of the speaker (user, assistant, system)
            agent_type: Type of agent creating the memory
            importance: Importance score (0.0 to 1.0)
            tags: List of tags for categorization
            metadata: Additional metadata
            
        Returns:
            Memory entry ID
        """
        try:
            # Generate embedding
            embedding = self._generate_embedding(content)
            
            # Create memory entry
            memory_id = str(uuid.uuid4())
            entry = MemoryEntry(
                id=memory_id,
                content=content,
                role=role,
                timestamp=datetime.now(),
                session_id=self.session_id,
                agent_type=agent_type,
                importance=importance,
                tags=tags or [],
                embeddings=embedding,
                metadata=metadata or {}
            )
            
            # Store in memory
            self.memory_entries[memory_id] = entry
            
            # Add to vector databases
            self._add_to_chroma(entry)
            self._add_to_faiss(entry)
            
            # Update statistics
            self.stats["total_memories"] += 1
            
            # Check memory size and cleanup if necessary
            if len(self.memory_entries) > self.max_memory_size:
                self._cleanup_old_memories()
            
            self.logger.info(f"Added memory entry: {memory_id}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Error adding memory: {e}")
            raise
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using the sentence transformer."""
        try:
            # Truncate text if too long
            max_tokens = 512
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                text = self.tokenizer.decode(tokens)
            
            embedding = self.embedding_model.encode(text, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            raise
    
    def _add_to_chroma(self, entry: MemoryEntry):
        """Add memory entry to ChromaDB."""
        try:
            self.collection.add(
                embeddings=[entry.embeddings.tolist()],
                documents=[entry.content],
                metadatas=[{
                    "role": entry.role,
                    "timestamp": entry.timestamp.isoformat(),
                    "session_id": entry.session_id,
                    "agent_type": entry.agent_type,
                    "importance": entry.importance,
                    "tags": ",".join(entry.tags)
                }],
                ids=[entry.id]
            )
        except Exception as e:
            self.logger.error(f"Error adding to ChromaDB: {e}")
            raise
    
    def _add_to_faiss(self, entry: MemoryEntry):
        """Add memory entry to FAISS index."""
        try:
            # Add to FAISS index
            embedding = entry.embeddings.reshape(1, -1)
            new_index = self.faiss_index.ntotal
            
            self.faiss_index.add(embedding)
            
            # Update mappings
            self.id_to_index[entry.id] = new_index
            self.index_to_id[str(new_index)] = entry.id
            
        except Exception as e:
            self.logger.error(f"Error adding to FAISS: {e}")
            raise
    
    @timer_decorator
    def search_memories(self, 
                       query: str, 
                       limit: int = 10,
                       similarity_threshold: float = None,
                       filters: Dict[str, Any] = None,
                       include_system: bool = False) -> List[Tuple[MemoryEntry, float]]:
        """
        Search for relevant memories using semantic similarity.
        
        Args:
            query: Search query
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Additional filters (role, agent_type, tags, etc.)
            include_system: Whether to include system messages
            
        Returns:
            List of (MemoryEntry, similarity_score) tuples
        """
        try:
            start_time = time.time()
            threshold = similarity_threshold or self.similarity_threshold
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            # Search using FAISS for fast retrieval
            similarities, indices = self.faiss_index.search(
                query_embedding.reshape(1, -1), 
                min(limit * 3, self.faiss_index.ntotal)  # Get more candidates for filtering
            )
            
            results = []
            for similarity, index in zip(similarities[0], indices[0]):
                if index == -1:  # FAISS returns -1 for invalid indices
                    continue
                    
                memory_id = self.index_to_id.get(str(index))
                if not memory_id or memory_id not in self.memory_entries:
                    continue
                
                entry = self.memory_entries[memory_id]
                
                # Apply filters
                if not self._passes_filters(entry, filters, include_system):
                    continue
                
                # Check similarity threshold
                if similarity >= threshold:
                    # Update access statistics
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                    
                    results.append((entry, float(similarity)))
            
            # Sort by similarity and limit results
            results.sort(key=lambda x: x[1], reverse=True)
            results = results[:limit]
            
            # Update statistics
            self.stats["queries_performed"] += 1
            self.stats["average_retrieval_time"] = (
                (self.stats["average_retrieval_time"] * (self.stats["queries_performed"] - 1) + 
                 (time.time() - start_time)) / self.stats["queries_performed"]
            )
            
            self.logger.info(f"Memory search completed: {len(results)} results for query '{query[:50]}...'")
            return results
            
        except Exception as e:
            self.logger.error(f"Error searching memories: {e}")
            return []
    
    def _passes_filters(self, entry: MemoryEntry, filters: Dict[str, Any], include_system: bool) -> bool:
        """Check if memory entry passes the specified filters."""
        if not include_system and entry.role == "system":
            return False
        
        if filters:
            if "role" in filters and entry.role != filters["role"]:
                return False
            if "agent_type" in filters and entry.agent_type != filters["agent_type"]:
                return False
            if "tags" in filters:
                filter_tags = filters["tags"] if isinstance(filters["tags"], list) else [filters["tags"]]
                if not any(tag in entry.tags for tag in filter_tags):
                    return False
            if "min_importance" in filters and entry.importance < filters["min_importance"]:
                return False
            if "max_age_days" in filters:
                age_limit = datetime.now() - timedelta(days=filters["max_age_days"])
                if entry.timestamp < age_limit:
                    return False
        
        return True
    
    def get_memory_by_id(self, memory_id: str) -> Optional[MemoryEntry]:
        """Retrieve a specific memory by ID."""
        entry = self.memory_entries.get(memory_id)
        if entry:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
        return entry
    
    def update_memory_importance(self, memory_id: str, importance: float):
        """Update the importance score of a memory."""
        if memory_id in self.memory_entries:
            self.memory_entries[memory_id].importance = importance
            self.logger.info(f"Updated importance for memory {memory_id}: {importance}")
    
    def add_tags_to_memory(self, memory_id: str, tags: List[str]):
        """Add tags to an existing memory."""
        if memory_id in self.memory_entries:
            existing_tags = set(self.memory_entries[memory_id].tags)
            new_tags = set(tags)
            self.memory_entries[memory_id].tags = list(existing_tags.union(new_tags))
            self.logger.info(f"Added tags to memory {memory_id}: {tags}")
    
    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        try:
            if memory_id not in self.memory_entries:
                return False
            
            # Remove from ChromaDB
            self.collection.delete(ids=[memory_id])
            
            # Remove from in-memory storage
            del self.memory_entries[memory_id]
            
            # Note: FAISS doesn't support deletion, so we'll rebuild index during save
            self.stats["total_memories"] -= 1
            
            self.logger.info(f"Deleted memory: {memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting memory {memory_id}: {e}")
            return False
    
    def _cleanup_old_memories(self):
        """Remove old or less important memories to stay within size limits."""
        try:
            if len(self.memory_entries) <= self.max_memory_size:
                return
            
            # Calculate cleanup target
            target_size = int(self.max_memory_size * 0.8)  # Remove 20% to avoid frequent cleanups
            num_to_remove = len(self.memory_entries) - target_size
            
            # Score memories for removal (lower score = more likely to be removed)
            memory_scores = []
            for memory_id, entry in self.memory_entries.items():
                age_days = (datetime.now() - entry.timestamp).days
                access_recency = (datetime.now() - (entry.last_accessed or entry.timestamp)).days
                
                # Scoring formula: importance * recency_factor * access_factor
                recency_factor = max(0.1, 1.0 / (1.0 + age_days * 0.1))
                access_factor = max(0.1, entry.access_count / max(1, age_days))
                
                score = entry.importance * recency_factor * access_factor
                memory_scores.append((memory_id, score))
            
            # Sort by score and remove lowest scoring memories
            memory_scores.sort(key=lambda x: x[1])
            to_remove = memory_scores[:num_to_remove]
            
            for memory_id, _ in to_remove:
                self.delete_memory(memory_id)
            
            self.logger.info(f"Cleaned up {len(to_remove)} old memories")
            
        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")
    
    def cleanup_memories(self, days_threshold: int = 30, importance_threshold: float = 0.3) -> int:
        """
        Public method to clean up old, low-importance memories.
        
        Args:
            days_threshold: Remove memories older than this many days
            importance_threshold: Remove memories with importance below this threshold
            
        Returns:
            Number of memories cleaned up
        """
        try:
            initial_count = len(self.memory_entries)
            cutoff_date = datetime.now() - timedelta(days=days_threshold)
            
            # Find memories to remove based on criteria
            to_remove = []
            for memory_id, entry in self.memory_entries.items():
                should_remove = (
                    entry.timestamp < cutoff_date and 
                    entry.importance < importance_threshold
                )
                if should_remove:
                    to_remove.append(memory_id)
            
            # Remove the identified memories
            for memory_id in to_remove:
                self.delete_memory(memory_id)
            
            cleanup_count = len(to_remove)
            self.logger.info(f"Cleaned up {cleanup_count} memories (older than {days_threshold} days with importance < {importance_threshold})")
            
            return cleanup_count
            
        except Exception as e:
            self.logger.error(f"Error during memory cleanup: {e}")
            return 0
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        try:
            # Calculate memory usage
            total_size = 0
            for entry in self.memory_entries.values():
                total_size += len(entry.content.encode('utf-8'))
                if entry.embeddings is not None:
                    total_size += entry.embeddings.nbytes
            
            self.stats["memory_usage_mb"] = total_size / (1024 * 1024)
            
            # Add more detailed statistics
            stats = self.stats.copy()
            stats.update({
                "memory_entries_count": len(self.memory_entries),
                "average_content_length": np.mean([len(e.content) for e in self.memory_entries.values()]) if self.memory_entries else 0,
                "role_distribution": self._get_role_distribution(),
                "agent_type_distribution": self._get_agent_type_distribution(),
                "importance_distribution": self._get_importance_distribution(),
                "session_count": len(set(e.session_id for e in self.memory_entries.values())),
                "oldest_memory": min(e.timestamp for e in self.memory_entries.values()) if self.memory_entries else None,
                "newest_memory": max(e.timestamp for e in self.memory_entries.values()) if self.memory_entries else None
            })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting memory statistics: {e}")
            return self.stats
    
    def _get_role_distribution(self) -> Dict[str, int]:
        """Get distribution of memory entries by role."""
        distribution = {}
        for entry in self.memory_entries.values():
            distribution[entry.role] = distribution.get(entry.role, 0) + 1
        return distribution
    
    def _get_agent_type_distribution(self) -> Dict[str, int]:
        """Get distribution of memory entries by agent type."""
        distribution = {}
        for entry in self.memory_entries.values():
            distribution[entry.agent_type] = distribution.get(entry.agent_type, 0) + 1
        return distribution
    
    def _get_importance_distribution(self) -> Dict[str, int]:
        """Get distribution of memory entries by importance ranges."""
        distribution = {"low": 0, "medium": 0, "high": 0}
        for entry in self.memory_entries.values():
            if entry.importance < 0.33:
                distribution["low"] += 1
            elif entry.importance < 0.67:
                distribution["medium"] += 1
            else:
                distribution["high"] += 1
        return distribution
    
    def save_memory_state(self):
        """Save the current memory state to disk."""
        try:
            # Save memory entries
            memories_path = self.memory_path / "memories.json"
            memory_data = []
            for entry in self.memory_entries.values():
                entry_dict = asdict(entry)
                # Convert numpy array to list for JSON serialization
                if entry_dict['embeddings'] is not None:
                    entry_dict['embeddings'] = entry.embeddings.tolist()
                # Convert datetime to ISO string
                entry_dict['timestamp'] = entry.timestamp.isoformat()
                if entry_dict['last_accessed']:
                    entry_dict['last_accessed'] = entry.last_accessed.isoformat()
                memory_data.append(entry_dict)
            
            with open(memories_path, 'w') as f:
                json.dump(memory_data, f, indent=2)
            
            # Save FAISS index
            if self.faiss_index and self.faiss_index.ntotal > 0:
                faiss_path = self.memory_path / "faiss_index.bin"
                faiss.write_index(self.faiss_index, str(faiss_path))
            
            # Save ID mappings
            mapping_path = self.memory_path / "id_mappings.json"
            mappings = {
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id
            }
            with open(mapping_path, 'w') as f:
                json.dump(mappings, f, indent=2)
            
            # Save statistics
            stats_path = self.memory_path / "statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(self.get_memory_statistics(), f, indent=2, default=str)
            
            self.logger.info("Memory state saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving memory state: {e}")
            raise
    
    def export_memories(self, 
                       export_path: str, 
                       format: str = "json",
                       filters: Dict[str, Any] = None) -> bool:
        """
        Export memories to external format.
        
        Args:
            export_path: Path to export file
            format: Export format (json, csv, txt)
            filters: Optional filters to apply
            
        Returns:
            Success status
        """
        try:
            memories_to_export = []
            for entry in self.memory_entries.values():
                if self._passes_filters(entry, filters, True):
                    memories_to_export.append(entry)
            
            if format.lower() == "json":
                export_data = []
                for entry in memories_to_export:
                    entry_dict = asdict(entry)
                    entry_dict['embeddings'] = None  # Don't export embeddings
                    entry_dict['timestamp'] = entry.timestamp.isoformat()
                    if entry_dict['last_accessed']:
                        entry_dict['last_accessed'] = entry.last_accessed.isoformat()
                    export_data.append(entry_dict)
                
                with open(export_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
            
            elif format.lower() == "txt":
                with open(export_path, 'w') as f:
                    for entry in memories_to_export:
                        f.write(f"[{entry.timestamp}] {entry.role}: {entry.content}\n")
                        f.write(f"Tags: {', '.join(entry.tags)}\n")
                        f.write(f"Importance: {entry.importance}\n\n")
            
            self.logger.info(f"Exported {len(memories_to_export)} memories to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting memories: {e}")
            return False
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        try:
            self.save_memory_state()
        except:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the vector memory system.
        
        Returns:
            Dictionary containing memory statistics
        """
        try:
            total_memories = len(self.memory_entries)
            
            # Calculate average importance
            avg_importance = 0.0
            if total_memories > 0:
                avg_importance = sum(entry.importance for entry in self.memory_entries.values()) / total_memories
            
            # Calculate memory by role
            role_counts = {}
            for entry in self.memory_entries.values():
                role_counts[entry.role] = role_counts.get(entry.role, 0) + 1
            
            # Calculate average access count
            avg_access_count = 0.0
            if total_memories > 0:
                avg_access_count = sum(entry.access_count for entry in self.memory_entries.values()) / total_memories
            
            # Calculate storage size estimate (in MB)
            storage_size_mb = 0.0
            for entry in self.memory_entries.values():
                content_size = len(entry.content.encode('utf-8'))
                embedding_size = entry.embeddings.nbytes if entry.embeddings is not None else 0
                storage_size_mb += (content_size + embedding_size) / (1024 * 1024)
            
            return {
                'total_memories': total_memories,
                'average_importance': round(avg_importance, 3),
                'average_access_count': round(avg_access_count, 2),
                'role_distribution': role_counts,
                'storage_size_mb': round(storage_size_mb, 2),
                'max_memory_size': self.max_memory_size,
                'similarity_threshold': self.similarity_threshold,
                'embedding_model': self.embedding_model_name,
                'faiss_index_size': self.faiss_index.ntotal if self.faiss_index else 0,
                'session_id': self.session_id,
                'stats_snapshot': self.stats.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stats: {e}")
            return {
                'total_memories': 0,
                'error': str(e)
            }
