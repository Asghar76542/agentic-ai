"""
Semantic Search Implementation for AgenticSeek
Advanced semantic search capabilities with intelligent query processing and result ranking.
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textstat

from sources.logger import Logger
from sources.knowledge.vectorMemory import VectorMemory, MemoryEntry


@dataclass
class SearchQuery:
    """Structured search query with metadata and processing options."""
    text: str
    filters: Dict[str, Any] = None
    search_type: str = "semantic"  # semantic, keyword, hybrid
    boost_factors: Dict[str, float] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    importance_threshold: float = 0.0
    max_results: int = 10
    include_system: bool = False
    expand_query: bool = True
    
    def __post_init__(self):
        if self.filters is None:
            self.filters = {}
        if self.boost_factors is None:
            self.boost_factors = {}


@dataclass
class SearchResult:
    """Enhanced search result with detailed scoring and metadata."""
    memory_entry: MemoryEntry
    relevance_score: float
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    context_score: float = 0.0
    explanation: str = ""
    matched_keywords: List[str] = None
    
    def __post_init__(self):
        if self.matched_keywords is None:
            self.matched_keywords = []


class SemanticSearch:
    """
    Advanced semantic search engine with multiple search strategies,
    intelligent query processing, and sophisticated result ranking.
    """
    
    def __init__(self, 
                 vector_memory: VectorMemory,
                 enable_keyword_search: bool = True,
                 enable_query_expansion: bool = True,
                 max_query_tokens: int = 512):
        
        self.logger = Logger("semantic_search.log")
        self.vector_memory = vector_memory
        self.enable_keyword_search = enable_keyword_search
        self.enable_query_expansion = enable_query_expansion
        self.max_query_tokens = max_query_tokens
        
        # Initialize components
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.tfidf_vectorizer = None
        self.keyword_index = defaultdict(set)
        
        # Query expansion vocabulary
        self.expansion_vocab = self._build_expansion_vocabulary()
        
        # Search statistics
        self.search_stats = {
            "total_searches": 0,
            "semantic_searches": 0,
            "keyword_searches": 0,
            "hybrid_searches": 0,
            "average_results": 0.0,
            "average_processing_time": 0.0
        }
        
        # Initialize keyword search if enabled
        if self.enable_keyword_search:
            self._initialize_keyword_search()
    
    def _build_expansion_vocabulary(self) -> Dict[str, List[str]]:
        """Build vocabulary for query expansion."""
        return {
            # Programming terms
            "code": ["programming", "development", "script", "function", "algorithm"],
            "bug": ["error", "issue", "problem", "exception", "fault"],
            "fix": ["solve", "repair", "correct", "resolve", "debug"],
            
            # General terms
            "help": ["assist", "support", "aid", "guide", "explain"],
            "learn": ["understand", "study", "comprehend", "grasp", "master"],
            "create": ["make", "build", "generate", "produce", "develop"],
            
            # Technical terms
            "database": ["db", "storage", "data", "table", "query"],
            "api": ["interface", "endpoint", "service", "web service", "rest"],
            "server": ["backend", "service", "host", "machine", "system"]
        }
    
    def _initialize_keyword_search(self):
        """Initialize keyword search components."""
        try:
            # Build keyword index from existing memories
            self._build_keyword_index()
            
            # Initialize TF-IDF vectorizer
            if len(self.vector_memory.memory_entries) > 0:
                documents = [entry.content for entry in self.vector_memory.memory_entries.values()]
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=10000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    lowercase=True
                )
                self.tfidf_vectorizer.fit(documents)
            
            self.logger.info("Keyword search initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing keyword search: {e}")
    
    def _build_keyword_index(self):
        """Build inverted index for keyword search."""
        try:
            self.keyword_index.clear()
            
            for memory_id, entry in self.vector_memory.memory_entries.items():
                # Extract keywords from content
                keywords = self._extract_keywords(entry.content)
                
                # Add to inverted index
                for keyword in keywords:
                    self.keyword_index[keyword.lower()].add(memory_id)
                
                # Also index tags
                for tag in entry.tags:
                    self.keyword_index[tag.lower()].add(memory_id)
            
            self.logger.info(f"Built keyword index with {len(self.keyword_index)} terms")
            
        except Exception as e:
            self.logger.error(f"Error building keyword index: {e}")
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from text."""
        # Simple keyword extraction using regex
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'these', 'those', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should', 'can', 'may', 'might'}
        
        keywords = [word for word in words if word not in stop_words]
        return keywords
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Perform comprehensive search using multiple strategies.
        
        Args:
            query: SearchQuery object with search parameters
            
        Returns:
            List of SearchResult objects ranked by relevance
        """
        try:
            start_time = datetime.now()
            
            # Preprocess query
            processed_query = self._preprocess_query(query)
            
            # Perform search based on type
            if query.search_type == "semantic":
                results = self._semantic_search(processed_query)
            elif query.search_type == "keyword":
                results = self._keyword_search(processed_query)
            elif query.search_type == "hybrid":
                results = self._hybrid_search(processed_query)
            else:
                self.logger.warning(f"Unknown search type: {query.search_type}, defaulting to semantic")
                results = self._semantic_search(processed_query)
            
            # Post-process results
            results = self._post_process_results(results, query)
            
            # Update statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_search_stats(query.search_type, len(results), processing_time)
            
            self.logger.info(f"Search completed: {len(results)} results for '{query.text[:50]}...' in {processing_time:.3f}s")
            return results
            
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return []
    
    def _preprocess_query(self, query: SearchQuery) -> SearchQuery:
        """Preprocess and enhance the search query."""
        try:
            # Clean and normalize query text
            cleaned_text = re.sub(r'[^\w\s]', ' ', query.text)
            cleaned_text = ' '.join(cleaned_text.split())
            
            # Expand query if enabled
            if query.expand_query and self.enable_query_expansion:
                expanded_text = self._expand_query(cleaned_text)
                if expanded_text != cleaned_text:
                    self.logger.info(f"Query expanded: '{cleaned_text}' -> '{expanded_text}'")
                    cleaned_text = expanded_text
            
            # Truncate if too long
            tokens = self.tokenizer.encode(cleaned_text)
            if len(tokens) > self.max_query_tokens:
                tokens = tokens[:self.max_query_tokens]
                cleaned_text = self.tokenizer.decode(tokens)
            
            # Create processed query
            processed_query = SearchQuery(
                text=cleaned_text,
                filters=query.filters,
                search_type=query.search_type,
                boost_factors=query.boost_factors,
                time_range=query.time_range,
                importance_threshold=query.importance_threshold,
                max_results=query.max_results,
                include_system=query.include_system,
                expand_query=False  # Already expanded
            )
            
            return processed_query
            
        except Exception as e:
            self.logger.error(f"Error preprocessing query: {e}")
            return query
    
    def _expand_query(self, query_text: str) -> str:
        """Expand query with related terms."""
        try:
            words = query_text.lower().split()
            expanded_words = words.copy()
            
            for word in words:
                if word in self.expansion_vocab:
                    # Add related terms (limit to avoid over-expansion)
                    related_terms = self.expansion_vocab[word][:2]
                    expanded_words.extend(related_terms)
            
            return ' '.join(expanded_words)
            
        except Exception as e:
            self.logger.error(f"Error expanding query: {e}")
            return query_text
    
    def _semantic_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform semantic search using vector similarity."""
        try:
            # Use vector memory's search function
            memory_results = self.vector_memory.search_memories(
                query=query.text,
                limit=query.max_results * 2,  # Get more candidates for re-ranking
                similarity_threshold=0.0,  # We'll filter later
                filters=query.filters,
                include_system=query.include_system
            )
            
            # Convert to SearchResult objects
            results = []
            for memory_entry, similarity_score in memory_results:
                result = SearchResult(
                    memory_entry=memory_entry,
                    relevance_score=similarity_score,
                    semantic_score=similarity_score,
                    explanation=f"Semantic similarity: {similarity_score:.3f}"
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []
    
    def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform keyword-based search."""
        try:
            if not self.enable_keyword_search:
                return []
            
            query_keywords = self._extract_keywords(query.text)
            if not query_keywords:
                return []
            
            # Find memories containing query keywords
            candidate_memory_ids = set()
            keyword_matches = defaultdict(list)
            
            for keyword in query_keywords:
                matching_ids = self.keyword_index.get(keyword.lower(), set())
                candidate_memory_ids.update(matching_ids)
                for memory_id in matching_ids:
                    keyword_matches[memory_id].append(keyword)
            
            # Score and rank results
            results = []
            for memory_id in candidate_memory_ids:
                if memory_id not in self.vector_memory.memory_entries:
                    continue
                
                memory_entry = self.vector_memory.memory_entries[memory_id]
                
                # Apply filters
                if not self.vector_memory._passes_filters(memory_entry, query.filters, query.include_system):
                    continue
                
                # Calculate keyword score
                matched_keywords = keyword_matches[memory_id]
                keyword_score = self._calculate_keyword_score(
                    memory_entry.content, 
                    query_keywords, 
                    matched_keywords
                )
                
                if keyword_score > 0:
                    result = SearchResult(
                        memory_entry=memory_entry,
                        relevance_score=keyword_score,
                        keyword_score=keyword_score,
                        matched_keywords=matched_keywords,
                        explanation=f"Keyword matches: {', '.join(matched_keywords)}"
                    )
                    results.append(result)
            
            # Sort by keyword score
            results.sort(key=lambda x: x.keyword_score, reverse=True)
            return results[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"Error in keyword search: {e}")
            return []
    
    def _calculate_keyword_score(self, content: str, query_keywords: List[str], matched_keywords: List[str]) -> float:
        """Calculate keyword-based relevance score."""
        try:
            content_lower = content.lower()
            total_score = 0.0
            
            for keyword in matched_keywords:
                # Count occurrences
                occurrences = content_lower.count(keyword.lower())
                
                # Calculate TF score
                tf_score = occurrences / len(content.split())
                
                # Boost for exact matches
                exact_match_boost = 1.5 if keyword in query_keywords else 1.0
                
                # Position boost (early occurrence gets higher score)
                position_boost = 1.0
                first_occurrence = content_lower.find(keyword.lower())
                if first_occurrence != -1:
                    position_boost = 1.0 + (1.0 - first_occurrence / len(content))
                
                keyword_score = tf_score * exact_match_boost * position_boost
                total_score += keyword_score
            
            # Normalize by number of query keywords
            return total_score / len(query_keywords) if query_keywords else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating keyword score: {e}")
            return 0.0
    
    def _hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform hybrid search combining semantic and keyword approaches."""
        try:
            # Perform both searches
            semantic_query = SearchQuery(
                text=query.text,
                filters=query.filters,
                search_type="semantic",
                max_results=query.max_results * 2,
                include_system=query.include_system,
                expand_query=False
            )
            
            keyword_query = SearchQuery(
                text=query.text,
                filters=query.filters,
                search_type="keyword",
                max_results=query.max_results * 2,
                include_system=query.include_system,
                expand_query=False
            )
            
            semantic_results = self._semantic_search(semantic_query)
            keyword_results = self._keyword_search(keyword_query)
            
            # Combine and merge results
            combined_results = {}
            
            # Add semantic results
            for result in semantic_results:
                memory_id = result.memory_entry.id
                combined_results[memory_id] = result
            
            # Merge keyword results
            for result in keyword_results:
                memory_id = result.memory_entry.id
                if memory_id in combined_results:
                    # Combine scores
                    existing = combined_results[memory_id]
                    existing.keyword_score = result.keyword_score
                    existing.matched_keywords = result.matched_keywords
                    
                    # Hybrid scoring
                    semantic_weight = 0.7
                    keyword_weight = 0.3
                    existing.relevance_score = (
                        semantic_weight * existing.semantic_score +
                        keyword_weight * existing.keyword_score
                    )
                    
                    existing.explanation = f"Hybrid: semantic={existing.semantic_score:.3f}, keyword={existing.keyword_score:.3f}"
                else:
                    # Add as keyword-only result
                    result.relevance_score = result.keyword_score * 0.3  # Lower weight for keyword-only
                    combined_results[memory_id] = result
            
            # Sort by combined score
            results = list(combined_results.values())
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return results[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            return []
    
    def _post_process_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Post-process search results with additional scoring and filtering."""
        try:
            # Apply importance threshold
            if query.importance_threshold > 0:
                results = [r for r in results if r.memory_entry.importance >= query.importance_threshold]
            
            # Apply time range filter
            if query.time_range:
                start_time, end_time = query.time_range
                results = [r for r in results if start_time <= r.memory_entry.timestamp <= end_time]
            
            # Calculate additional scores
            for result in results:
                result.recency_score = self._calculate_recency_score(result.memory_entry)
                result.importance_score = result.memory_entry.importance
                result.context_score = self._calculate_context_score(result.memory_entry, query)
                
                # Apply boost factors
                if query.boost_factors:
                    boost = 1.0
                    if result.memory_entry.role in query.boost_factors:
                        boost *= query.boost_factors[result.memory_entry.role]
                    if result.memory_entry.agent_type in query.boost_factors:
                        boost *= query.boost_factors[result.memory_entry.agent_type]
                    
                    result.relevance_score *= boost
            
            # Re-rank with all factors
            for result in results:
                final_score = (
                    0.4 * result.relevance_score +
                    0.2 * result.recency_score +
                    0.2 * result.importance_score +
                    0.2 * result.context_score
                )
                result.relevance_score = final_score
            
            # Final sort and limit
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            return results[:query.max_results]
            
        except Exception as e:
            self.logger.error(f"Error post-processing results: {e}")
            return results
    
    def _calculate_recency_score(self, memory_entry: MemoryEntry) -> float:
        """Calculate recency score based on timestamp."""
        try:
            now = datetime.now()
            age = (now - memory_entry.timestamp).total_seconds()
            
            # Exponential decay with 7-day half-life
            half_life = 7 * 24 * 3600  # 7 days in seconds
            recency_score = math.exp(-age / half_life)
            
            return recency_score
            
        except Exception as e:
            self.logger.error(f"Error calculating recency score: {e}")
            return 0.0
    
    def _calculate_context_score(self, memory_entry: MemoryEntry, query: SearchQuery) -> float:
        """Calculate contextual relevance score."""
        try:
            score = 0.0
            
            # Boost for matching session
            if memory_entry.session_id == self.vector_memory.session_id:
                score += 0.3
            
            # Boost for recent access
            if memory_entry.last_accessed:
                hours_since_access = (datetime.now() - memory_entry.last_accessed).total_seconds() / 3600
                if hours_since_access < 24:
                    score += 0.2 * (1.0 - hours_since_access / 24)
            
            # Boost for high access count
            if memory_entry.access_count > 0:
                score += min(0.2, memory_entry.access_count * 0.05)
            
            # Content quality indicators
            content_quality = self._assess_content_quality(memory_entry.content)
            score += 0.3 * content_quality
            
            return min(1.0, score)
            
        except Exception as e:
            self.logger.error(f"Error calculating context score: {e}")
            return 0.0
    
    def _assess_content_quality(self, content: str) -> float:
        """Assess the quality of content for ranking purposes."""
        try:
            # Length-based quality (not too short, not too long)
            length_score = 1.0
            if len(content) < 20:
                length_score = 0.3
            elif len(content) > 2000:
                length_score = 0.7
            
            # Readability score
            try:
                reading_ease = textstat.flesch_reading_ease(content)
                readability_score = max(0.0, min(1.0, reading_ease / 100))
            except:
                readability_score = 0.5
            
            # Information density (longer sentences generally contain more info)
            sentences = len([s for s in content.split('.') if s.strip()])
            words = len(content.split())
            density_score = min(1.0, (words / max(1, sentences)) / 20)
            
            # Combine scores
            quality = (0.4 * length_score + 0.3 * readability_score + 0.3 * density_score)
            return quality
            
        except Exception as e:
            self.logger.error(f"Error assessing content quality: {e}")
            return 0.5
    
    def _update_search_stats(self, search_type: str, result_count: int, processing_time: float):
        """Update search statistics."""
        self.search_stats["total_searches"] += 1
        
        if search_type == "semantic":
            self.search_stats["semantic_searches"] += 1
        elif search_type == "keyword":
            self.search_stats["keyword_searches"] += 1
        elif search_type == "hybrid":
            self.search_stats["hybrid_searches"] += 1
        
        # Update averages
        total = self.search_stats["total_searches"]
        self.search_stats["average_results"] = (
            (self.search_stats["average_results"] * (total - 1) + result_count) / total
        )
        self.search_stats["average_processing_time"] = (
            (self.search_stats["average_processing_time"] * (total - 1) + processing_time) / total
        )
    
    def suggest_queries(self, partial_query: str, limit: int = 5) -> List[str]:
        """Suggest query completions based on memory content."""
        try:
            if len(partial_query) < 2:
                return []
            
            # Extract frequent terms from memory content
            term_counts = defaultdict(int)
            partial_lower = partial_query.lower()
            
            for entry in self.vector_memory.memory_entries.values():
                words = self._extract_keywords(entry.content)
                for word in words:
                    if word.startswith(partial_lower):
                        term_counts[word] += 1
                
                # Also check tags
                for tag in entry.tags:
                    if tag.lower().startswith(partial_lower):
                        term_counts[tag.lower()] += 1
            
            # Sort by frequency and return top suggestions
            suggestions = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)
            return [term for term, count in suggestions[:limit]]
            
        except Exception as e:
            self.logger.error(f"Error generating query suggestions: {e}")
            return []
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get comprehensive search statistics."""
        stats = self.search_stats.copy()
        stats.update({
            "keyword_index_size": len(self.keyword_index),
            "total_indexed_terms": sum(len(ids) for ids in self.keyword_index.values()),
            "expansion_vocabulary_size": len(self.expansion_vocab)
        })
        return stats
    
    def rebuild_indexes(self):
        """Rebuild search indexes (call after bulk memory updates)."""
        try:
            if self.enable_keyword_search:
                self._build_keyword_index()
                
                # Rebuild TF-IDF if we have enough documents
                if len(self.vector_memory.memory_entries) > 0:
                    documents = [entry.content for entry in self.vector_memory.memory_entries.values()]
                    self.tfidf_vectorizer = TfidfVectorizer(
                        max_features=10000,
                        stop_words='english',
                        ngram_range=(1, 2),
                        lowercase=True
                    )
                    self.tfidf_vectorizer.fit(documents)
            
            self.logger.info("Search indexes rebuilt successfully")
            
        except Exception as e:
            self.logger.error(f"Error rebuilding indexes: {e}")
