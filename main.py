"""
Command line runner for the Music Recommender Simulation.

Run from the project root with:
    python main.py

Features:
- Rule-based scoring system for baseline recommendations
- RAG (Retrieval-Augmented Generation) pipeline for semantic search
- Ensemble ranking combining RAG retrieval with feature-based scoring
"""

from recommender import load_songs, recommend_songs, DEFAULT_WEIGHTS
from rag import create_rag_pipeline

# ---------------------------------------------------------------------------
# User profiles for stress-testing
# ---------------------------------------------------------------------------

PROFILES = {
    "High-Energy Pop": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.80,
        "valence": 0.85,
    },
    "Chill Lofi": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.38,
        "valence": 0.58,
    },
    "Deep Intense Rock": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.92,
        "valence": 0.45,
    },
    # Adversarial: conflicting preferences — very high energy but a sad, low-valence mood.
    # Tests whether the system handles contradictory signals gracefully.
    "Adversarial (High-Energy + Sad)": {
        "genre": "metal",
        "mood": "sad",
        "energy": 0.90,
        "valence": 0.20,
    },
}

# Experiment weights: genre halved (1.0), energy doubled (4.0)
EXPERIMENT_WEIGHTS = {
    "genre": 1.0,
    "mood": 1.0,
    "energy": 4.0,
    "valence": 1.0,
}

# Sample natural language queries for RAG testing
RAG_QUERIES = [
    "I want energetic pop songs for working out",
    "Looking for chill lofi beats to focus",
    "Give me intense metal tracks for the gym",
    "Peaceful acoustic songs for relaxation",
    "Fun and dancy tracks for a party",
]


def print_recommendations(label: str, user_prefs: dict, songs: list, weights=None) -> None:
    """Print a formatted recommendation block for one user profile."""
    w = weights or DEFAULT_WEIGHTS
    max_score = w["genre"] + w["mood"] + w["energy"] + w["valence"]

    print(f"\n{'=' * 60}")
    print(f"  Profile: {label}")
    print(f"  genre={user_prefs['genre']}  mood={user_prefs['mood']}  "
          f"energy={user_prefs['energy']}  valence={user_prefs['valence']}")
    if weights:
        print(f"  [EXPERIMENT weights: genre={w['genre']} mood={w['mood']} "
              f"energy={w['energy']} valence={w['valence']}]")
    print(f"{'=' * 60}\n")

    recs = recommend_songs(user_prefs, songs, k=5, weights=w)
    for rank, (song, score, reasons) in enumerate(recs, start=1):
        print(f"  #{rank}  {song['title']} by {song['artist']}")
        print(f"       Genre: {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']}")
        print(f"       Score: {score:.2f} / {max_score:.1f}")
        for reason in reasons:
            print(f"         * {reason}")
        print()


def demo_rag_pipeline(songs: list) -> None:
    """Demonstrate the RAG (Retrieval-Augmented Generation) pipeline.
    
    This shows how semantic search can retrieve relevant songs from
    natural language queries, forming the first stage of the ensemble pipeline.
    """
    print("\n" + "=" * 80)
    print("  RAG (RETRIEVAL-AUGMENTED GENERATION) PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    print("\nInitializing RAG pipeline with semantic embeddings...")
    rag = create_rag_pipeline(songs)
    
    print(f"\nRAG Index built successfully with {len(songs)} songs\n")
    
    # Test each query
    for query in RAG_QUERIES:
        print(f"\n{'-' * 80}")
        print(f"Query: \"{query}\"")
        print(f"{'-' * 80}\n")
        
        # Retrieve top 5 semantically similar songs
        results = rag.retrieve(query, top_k=5, rerank=True)
        
        for rank, (song, similarity) in enumerate(results, start=1):
            print(f"  #{rank} [{similarity:.3f}]  {song['title']} - {song['artist']}")
            print(f"         Genre: {song['genre']} | Mood: {song['mood']} | Energy: {song['energy']:.2f}")
            explanation = rag.explain_retrieval(song, query)
            print(f"         {explanation}\n")


def demo_rag_ensemble(songs: list) -> None:
    """Demonstrate RAG-powered ensemble recommendations.
    
    This combines RAG retrieval with the feature-based scoring system
    for improved recommendations.
    """
    print("\n" + "=" * 80)
    print("  ENSEMBLE RECOMMENDATIONS (RAG + FEATURE-BASED SCORING)")
    print("=" * 80)
    
    rag = create_rag_pipeline(songs)
    
    # Example: Use a query to get RAG candidates, then score them
    query = "energetic workout music with high energy and intensity"
    print(f"\nQuery: \"{query}\"")
    
    # Step 1: RAG retrieves top candidates
    rag_results = rag.retrieve(query, top_k=50, rerank=True)
    rag_candidates = [song for song, _ in rag_results]
    
    print(f"\nRAG retrieved {len(rag_candidates)} candidate songs")
    
    # Step 2: Score candidates with feature-based system
    user_prefs = {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.90,
        "valence": 0.50,
    }
    
    print(f"\nUser preferences: {user_prefs}")
    print(f"\nTop 5 recommendations after ensemble scoring:\n")
    
    # Score only the RAG candidates (not the full database)
    from recommender import score_song
    scored = [(song, score_song(user_prefs, song, DEFAULT_WEIGHTS)) for song in rag_candidates]
    scored.sort(key=lambda x: x[1][0], reverse=True)
    
    max_score = sum(DEFAULT_WEIGHTS.values())
    for rank, (song, (score, reasons)) in enumerate(scored[:5], start=1):
        print(f"  #{rank}  {song['title']} by {song['artist']}")
        print(f"       Score: {score:.2f} / {max_score:.1f}")
        print(f"       Genre: {song['genre']} | Mood: {song['mood']} | Energy: {song['energy']:.2f}")
        for reason in reasons:
            print(f"         * {reason}")
        print()


def main() -> None:
    songs = load_songs("songs.csv")
    print(f"Loaded {len(songs)} songs from database")

    # --- Section 1: Baseline Rule-Based Recommendations ---
    print("\n" + "#" * 80)
    print("  SECTION 1: BASELINE RULE-BASED RECOMMENDATIONS")
    print("#" * 80)
    
    for label, prefs in list(PROFILES.items())[:2]:  # Show 2 profiles for brevity
        print_recommendations(label, prefs, songs)

    # --- Section 2: RAG Retrieval Pipeline ---
    print("\n" + "#" * 80)
    print("  SECTION 2: SEMANTIC SEARCH WITH RAG PIPELINE")
    print("#" * 80)
    
    try:
        demo_rag_pipeline(songs)
    except Exception as e:
        print(f"\nNote: RAG demo requires sentence-transformers and faiss.")
        print(f"Error: {e}")
        print("Install with: pip install -r requirements.txt")

    # --- Section 3: Ensemble Recommendations ---
    print("\n" + "#" * 80)
    print("  SECTION 3: ENSEMBLE RECOMMENDATIONS (RAG + FEATURE SCORING)")
    print("#" * 80)
    
    try:
        demo_rag_ensemble(songs)
    except Exception as e:
        print(f"\nRAG ensemble demo skipped: {e}")

    # --- Section 4: Experiment with Weights ---
    print("\n" + "#" * 80)
    print("  SECTION 4: WEIGHT SENSITIVITY ANALYSIS")
    print("  Testing: genre weight 2.0→1.0 | energy weight 2.0→4.0")
    print("#" * 80)
    
    profile = PROFILES["Deep Intense Rock"]
    print_recommendations("DEFAULT weights", profile, songs)
    print_recommendations("EXPERIMENT weights", profile, songs, weights=EXPERIMENT_WEIGHTS)


if __name__ == "__main__":
    main()
