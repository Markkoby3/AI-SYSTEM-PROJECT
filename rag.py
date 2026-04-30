"""
RAG (Retrieval-Augmented Generation) Pipeline for Music Recommendation.

This module implements semantic search and retrieval over the song database using:
- Sentence embeddings for semantic similarity
- FAISS vector index for efficient nearest-neighbor search
- Query expansion and relevance ranking

Usage:
    rag = RAGPipeline(songs)
    rag.build_index()
    retrieved = rag.retrieve("energetic pop songs for workout", top_k=200)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import os


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for semantic song search."""

    def __init__(self, songs: List[Dict], model_name: str = "all-MiniLM-L6-v2"):
        """Initialize RAG with songs and embedding model.
        
        Args:
            songs: List of song dictionaries with metadata (title, artist, genre, mood, etc.)
            model_name: Hugging Face model for embeddings. 'all-MiniLM-L6-v2' is fast and accurate.
        """
        self.songs = songs
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.song_texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None

    def _create_song_text(self, song: Dict) -> str:
        """Create a rich text representation of a song for embedding.
        
        This combines title, artist, genre, mood, and other attributes
        into a single string for semantic search.
        """
        parts = [
            song["title"],
            song["artist"],
            f"genre {song['genre']}",
            f"mood {song['mood']}",
            f"energy level {song['energy']:.1f}",
            f"danceability {song['danceability']:.1f}",
            f"acousticness {song['acousticness']:.1f}",
        ]
        return " ".join(parts)

    def build_index(self, force_rebuild: bool = False) -> None:
        """Build FAISS vector index from songs.
        
        Args:
            force_rebuild: If True, rebuild even if cached embeddings exist.
        """
        cache_file = "song_embeddings.npy"
        
        # Try to load cached embeddings
        if os.path.exists(cache_file) and not force_rebuild:
            print(f"Loading cached embeddings from {cache_file}")
            self.embeddings = np.load(cache_file)
        else:
            print(f"Generating embeddings for {len(self.songs)} songs...")
            self.song_texts = [self._create_song_text(song) for song in self.songs]
            self.embeddings = self.embedding_model.encode(
                self.song_texts, 
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            # Cache for future runs
            np.save(cache_file, self.embeddings)
            print(f"Embeddings cached to {cache_file}")
        
        # Build FAISS index
        print("Building FAISS index...")
        embedding_dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.index.add(self.embeddings.astype(np.float32))
        print(f"FAISS index built with {len(self.songs)} songs")

    def retrieve(
        self, 
        query: str, 
        top_k: int = 200,
        rerank: bool = True
    ) -> List[Tuple[Dict, float]]:
        """Retrieve top-k songs semantically similar to query.
        
        Args:
            query: Natural language query (e.g., "energetic workout pop songs")
            top_k: Number of candidates to retrieve (typically 200 for ensembling)
            rerank: If True, rerank by combining semantic similarity with metadata matching
            
        Returns:
            List of (song_dict, relevance_score) tuples sorted by score descending.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        # Search FAISS
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.songs)))
        distances = distances[0]  # Get the first (and only) query result
        indices = indices[0]
        
        # Convert L2 distance to similarity score (higher is better)
        # L2 distance range: 0 (identical) to ~sqrt(dim) (very different)
        similarities = 1.0 / (1.0 + distances)  # Convert distance to similarity
        
        results = []
        for idx, similarity in zip(indices, similarities):
            song = self.songs[idx]
            results.append((song, float(similarity)))
        
        if rerank:
            results = self._rerank(query, results)
        
        return results[:top_k]

    def _rerank(self, query: str, candidates: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
        """Rerank candidates by combining semantic similarity with metadata matching.
        
        This helps prioritize songs where the genre/mood appear in the query text.
        """
        query_lower = query.lower()
        reranked = []
        
        for song, similarity in candidates:
            boost = 1.0
            
            # Boost if genre or mood appears in query
            if song["genre"].lower() in query_lower:
                boost *= 1.3
            if song["mood"].lower() in query_lower:
                boost *= 1.2
            
            reranked.append((song, similarity * boost))
        
        # Re-sort by boosted score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 200
    ) -> List[List[Tuple[Dict, float]]]:
        """Retrieve candidates for multiple queries in batch.
        
        Args:
            queries: List of natural language queries
            top_k: Number of candidates per query
            
        Returns:
            List of retrieval results (one list per query)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Batch encode all queries
        query_embeddings = self.embedding_model.encode(
            queries,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True
        ).astype(np.float32)
        
        # Batch search
        distances, indices = self.index.search(query_embeddings, min(top_k, len(self.songs)))
        
        all_results = []
        for dist_row, idx_row in zip(distances, indices):
            similarities = 1.0 / (1.0 + dist_row)
            results = [
                (self.songs[idx], float(sim))
                for idx, sim in zip(idx_row, similarities)
            ]
            all_results.append(results)
        
        return all_results

    def explain_retrieval(self, song: Dict, query: str) -> str:
        """Generate a natural language explanation for why a song was retrieved.
        
        Args:
            song: The retrieved song
            query: The original query
            
        Returns:
            Human-readable explanation
        """
        query_lower = query.lower()
        reasons = []
        
        # Check for genre match
        if song["genre"].lower() in query_lower:
            reasons.append(f"matches '{song['genre']}' genre in your search")
        
        # Check for mood match
        if song["mood"].lower() in query_lower:
            reasons.append(f"matches '{song['mood']}' mood in your search")
        
        # Check for energy/intensity keywords
        if any(word in query_lower for word in ["high energy", "intense", "energetic", "workout"]):
            if song["energy"] > 0.75:
                reasons.append("high energy matches your query")
        
        if any(word in query_lower for word in ["chill", "relaxed", "calm", "peaceful"]):
            if song["energy"] < 0.45:
                reasons.append("low energy matches your query")
        
        # Check for acoustic preference
        if "acoustic" in query_lower and song["acousticness"] > 0.6:
            reasons.append("has acoustic qualities you mentioned")
        
        if "electronic" in query_lower and song["acousticness"] < 0.4:
            reasons.append("has electronic sound you mentioned")
        
        if not reasons:
            reasons.append("semantically similar to your search query")
        
        return "Retrieved because it " + " and ".join(reasons) + "."


def create_rag_pipeline(songs: List[Dict]) -> RAGPipeline:
    """Factory function to create and initialize a RAG pipeline."""
    rag = RAGPipeline(songs)
    rag.build_index()
    return rag
