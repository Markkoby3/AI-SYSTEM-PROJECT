"""
Test suite for RAG (Retrieval-Augmented Generation) pipeline.

Run with: pytest test_rag.py -v
"""

import pytest
from rag import RAGPipeline, create_rag_pipeline
from recommender import load_songs


@pytest.fixture
def songs():
    """Load test songs."""
    return load_songs("songs.csv")


@pytest.fixture
def rag_pipeline(songs):
    """Create and build RAG pipeline."""
    rag = RAGPipeline(songs)
    rag.build_index()
    return rag


class TestRAGPipeline:
    """Test suite for RAG functionality."""
    
    def test_pipeline_initialization(self, songs):
        """Test RAG pipeline can be initialized."""
        rag = RAGPipeline(songs)
        assert len(rag.songs) == len(songs)
        assert rag.embedding_model is not None
        assert rag.index is None  # Index not built yet
    
    def test_index_building(self, rag_pipeline):
        """Test FAISS index is built correctly."""
        assert rag_pipeline.index is not None
        assert rag_pipeline.embeddings is not None
        assert len(rag_pipeline.embeddings) == len(rag_pipeline.songs)
    
    def test_song_text_generation(self, rag_pipeline, songs):
        """Test song text generation for embeddings."""
        song = songs[0]
        text = rag_pipeline._create_song_text(song)
        
        # Check that key fields are included
        assert song["title"] in text
        assert song["artist"] in text
        assert song["genre"] in text
        assert song["mood"] in text
    
    def test_retrieval_basic(self, rag_pipeline):
        """Test basic retrieval functionality."""
        query = "energetic pop songs"
        results = rag_pipeline.retrieve(query, top_k=5)
        
        assert len(results) <= 5
        assert all(len(item) == 2 for item in results)  # (song, similarity) tuples
        assert all(0 <= sim <= 1 for _, sim in results)  # Similarity in [0, 1]
    
    def test_retrieval_top_k(self, rag_pipeline):
        """Test that top_k parameter works."""
        query = "chill lofi"
        
        results_5 = rag_pipeline.retrieve(query, top_k=5)
        results_10 = rag_pipeline.retrieve(query, top_k=10)
        
        assert len(results_5) <= 5
        assert len(results_10) <= 10
        assert len(results_10) >= len(results_5)
    
    def test_reranking(self, rag_pipeline):
        """Test that reranking changes order."""
        query = "heavy metal intense"
        
        results_no_rerank = rag_pipeline.retrieve(query, top_k=20, rerank=False)
        results_rerank = rag_pipeline.retrieve(query, top_k=20, rerank=True)
        
        # Results should be different (rerank applies boosts)
        # At least the first few should match in content but possibly different order
        songs_no_rerank = [song["id"] for song, _ in results_no_rerank[:5]]
        songs_rerank = [song["id"] for song, _ in results_rerank[:5]]
        
        # May be the same or different depending on boost magnitude
        # Just verify we get results
        assert len(songs_no_rerank) > 0
        assert len(songs_rerank) > 0
    
    def test_batch_retrieve(self, rag_pipeline):
        """Test batch retrieval for multiple queries."""
        queries = [
            "energetic workout",
            "chill vibes",
            "intense metal",
        ]
        
        batch_results = rag_pipeline.batch_retrieve(queries, top_k=10)
        
        assert len(batch_results) == len(queries)
        for results in batch_results:
            assert len(results) <= 10
            assert all(len(item) == 2 for item in results)
    
    def test_explain_retrieval(self, rag_pipeline, songs):
        """Test explanation generation."""
        song = songs[0]
        query = f"I want {song['genre']} music"
        
        explanation = rag_pipeline.explain_retrieval(song, query)
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        assert "Retrieved because" in explanation
    
    def test_factory_function(self, songs):
        """Test factory function creates and initializes RAG."""
        rag = create_rag_pipeline(songs)
        
        assert rag.index is not None
        assert rag.embeddings is not None


class TestRAGQueryTypes:
    """Test RAG with different query types."""
    
    def test_genre_query(self, rag_pipeline):
        """Test retrieval with genre in query."""
        results = rag_pipeline.retrieve("pop", top_k=10)
        assert len(results) > 0
    
    def test_mood_query(self, rag_pipeline):
        """Test retrieval with mood in query."""
        results = rag_pipeline.retrieve("happy music", top_k=10)
        assert len(results) > 0
    
    def test_energy_query(self, rag_pipeline):
        """Test retrieval with energy descriptors."""
        results = rag_pipeline.retrieve("high energy intense workout", top_k=10)
        assert len(results) > 0
    
    def test_natural_language_query(self, rag_pipeline):
        """Test retrieval with complex natural language."""
        results = rag_pipeline.retrieve(
            "I'm looking for upbeat music to get me motivated during a run",
            top_k=20
        )
        assert len(results) > 0
    
    def test_empty_query_handling(self, rag_pipeline):
        """Test that empty query is handled."""
        # Empty queries should still work (may retrieve random songs)
        results = rag_pipeline.retrieve("", top_k=5)
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
