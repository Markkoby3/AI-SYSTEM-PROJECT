# 🎵 Advanced Music Recommender System

A production-grade music recommendation engine combining semantic search (RAG) with feature-based scoring for accurate, explainable recommendations.

## ✨ Features

- **🔍 Semantic Search (RAG)**: Natural language queries retrieve semantically similar songs
- **⚙️ Feature-Based Scoring**: Personalized recommendations based on user preferences
- **🎯 Ensemble Ranking**: Combines multiple signals for optimal results
- **💬 Explainable AI**: Human-readable explanations for every recommendation
- **⚡ Fast & Efficient**: FAISS vector indexing for O(1) retrieval, caching for speed

## 🏗️ Architecture

```
User Query / Profile
        ↓
    ┌─────────────┬──────────────┐
    ↓             ↓              ↓
RAG Pipeline  Feature Extract  Embeddings
(Retrieval)   (Scoring)        (Caching)
    ↓             ↓              ↓
    └─────────────┼──────────────┘
                  ↓
          Ensemble Ranking
           (Weighted Fusion)
                  ↓
         Score Fusion & Top-K
                  ↓
      Explanation Generation
                  ↓
        Final Recommendations
```

## 📊 Project Status

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | RAG Retrieval Pipeline | ✅ Complete |
| 1 | Feature-Based Scoring | ✅ Complete |
| 2 | ML Model Integration | 🔄 In Progress |
| 3 | Ensemble Explanation | 🔄 In Progress |
| 4 | Streamlit Web UI | 🔄 Planned |
| 4 | Production API | 🔄 Planned |

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Markkoby3/AI-SYSTEM-PROJECT.git
cd AI-SYSTEM-PROJECT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from rag import create_rag_pipeline
from recommender import load_songs, score_song, DEFAULT_WEIGHTS

# Load songs
songs = load_songs("songs.csv")

# Initialize RAG pipeline (builds embeddings & FAISS index)
rag = create_rag_pipeline(songs)

# Retrieve candidates using natural language query
query = "energetic workout pop music"
candidates = rag.retrieve(query, top_k=200)

# Score candidates with feature-based system
user_prefs = {
    "genre": "pop",
    "mood": "intense",
    "energy": 0.85,
    "valence": 0.70,
}

# Simple scoring
scored = [
    (song, score_song(user_prefs, song, DEFAULT_WEIGHTS))
    for song, _ in candidates
]
scored.sort(key=lambda x: x[1][0], reverse=True)

# Print top 5 recommendations
for rank, (song, (score, reasons)) in enumerate(scored[:5], 1):
    print(f"{rank}. {song['title']} - {song['artist']}")
    print(f"   Score: {score:.2f}")
    for reason in reasons:
        print(f"   • {reason}")
```

### Run Demo

```bash
# Run full demo with all features
python main.py

# Run tests
pytest test_rag.py -v
pytest test_recommender.py -v
```

## 📚 Documentation

### RAG Pipeline (`rag.py`)

Semantic search using Sentence Transformers and FAISS:

```python
from rag import RAGPipeline

# Initialize
rag = RAGPipeline(songs, model_name="all-MiniLM-L6-v2")
rag.build_index()

# Single query
results = rag.retrieve("chill lofi focus", top_k=50, rerank=True)
# Returns: List[(song_dict, similarity_score)]

# Batch queries
results_batch = rag.batch_retrieve(queries, top_k=50)

# Explanations
for song, similarity in results[:5]:
    explanation = rag.explain_retrieval(song, "chill lofi focus")
    print(f"{song['title']}: {explanation}")
```

**Key Classes:**
- `RAGPipeline`: Main orchestrator for semantic search
  - `build_index()`: Generate embeddings and build FAISS index
  - `retrieve(query, top_k, rerank)`: Retrieve semantically similar songs
  - `batch_retrieve(queries, top_k)`: Batch processing
  - `explain_retrieval(song, query)`: Natural language explanations

### Feature-Based Scoring (`recommender.py`)

Rule-based scoring on song metadata:

```python
from recommender import score_song, DEFAULT_WEIGHTS

user_prefs = {
    "genre": "rock",
    "mood": "intense",
    "energy": 0.90,
    "valence": 0.45,
}

song = songs[0]
score, reasons = score_song(user_prefs, song, DEFAULT_WEIGHTS)

print(f"Score: {score:.2f}")
for reason in reasons:
    print(f"  • {reason}")
```

**Scoring Dimensions:**
- `genre`: Exact match (0-2.0 points)
- `mood`: Exact match (0-1.0 points)
- `energy`: Proximity-based (0-2.0 points)
- `valence`: Proximity-based (0-1.0 points)
- **Max Score: 6.0 points**

### Running Tests

```bash
# Test RAG pipeline
pytest test_rag.py -v

# Test recommender
pytest test_recommender.py -v

# Run all tests
pytest -v
```

## 📁 Project Structure

```
AI-SYSTEM-PROJECT/
├── rag.py                    # RAG pipeline implementation
├── recommender.py            # Feature-based scoring
├── main.py                   # Demo runner (4 sections)
├── songs.csv                 # Song database (19 songs)
├── test_rag.py              # RAG tests
├── test_recommender.py      # Scoring tests
├── Architecture.md          # System design & internals
├── model_card.md            # Model documentation
├── reflection.md            # Development notes
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Configuration

### Song Attributes

Each song in `songs.csv` has:
- `id`: Unique identifier
- `title`: Song name
- `artist`: Artist/creator
- `genre`: Music genre (pop, rock, lofi, jazz, etc.)
- `mood`: Emotional tone (happy, chill, intense, sad, etc.)
- `energy`: Energy level 0.0-1.0
- `tempo_bpm`: Tempo in beats per minute
- `valence`: Musical happiness 0.0-1.0
- `danceability`: How danceable 0.0-1.0
- `acousticness`: Acoustic vs electronic 0.0-1.0

### User Preferences

Specify user taste with:
- `genre`: Favorite genre
- `mood`: Preferred mood
- `energy`: Target energy level (0.0-1.0)
- `valence`: Preferred musical happiness (0.0-1.0)

### Scoring Weights

Customize scoring priorities:

```python
# Default weights
DEFAULT_WEIGHTS = {
    "genre": 2.0,   # Exact genre match
    "mood": 1.0,    # Exact mood match
    "energy": 2.0,  # Energy proximity
    "valence": 1.0, # Valence proximity
}

# Custom experiment
EXPERIMENT_WEIGHTS = {
    "genre": 1.0,   # Reduce genre importance
    "mood": 1.0,
    "energy": 4.0,  # Increase energy importance
    "valence": 1.0,
}

results = recommend_songs(user_prefs, songs, k=5, weights=EXPERIMENT_WEIGHTS)
```

## 🧪 Demo Walkthrough

`main.py` runs 4 demonstration sections:

### Section 1: Baseline Rule-Based Recommendations
Tests the feature-based scoring system on various user profiles.

### Section 2: Semantic Search with RAG
Shows the RAG pipeline retrieving songs from natural language queries:
- "I want energetic pop songs for working out"
- "Looking for chill lofi beats to focus"
- "Give me intense metal tracks for the gym"
- etc.

### Section 3: Ensemble Recommendations
Combines RAG retrieval (stage 1) with feature-based scoring (stage 2) to produce final recommendations.

### Section 4: Weight Sensitivity Analysis
Tests how changing scoring weights affects recommendations (useful for hyperparameter tuning).

## 🔌 Dependencies

### Core
- `numpy>=1.21.0`: Numerical computing
- `pandas`: Data processing
- `pytest`: Testing framework
- `streamlit`: Web UI (optional)

### RAG Pipeline
- `sentence-transformers>=2.2.2`: Semantic embeddings
- `faiss-cpu>=1.7.4`: Vector similarity search

## 🚧 Next Steps (Phase 2)

1. **ML Model Integration**
   - Train neural network on user-song interactions
   - Integrate XGBoost gradient boosting
   - Combine scores via ensemble method

2. **Advanced Features**
   - Cold-start problem handling
   - Diversity constraints (avoid repetitive recommendations)
   - Temporal effects (time of day, season)

3. **UI & API**
   - Streamlit web interface
   - FastAPI REST endpoints
   - Real-time personalization

4. **Production**
   - Database integration (PostgreSQL)
   - Caching layer (Redis)
   - Monitoring & analytics

## 📖 Further Reading

- See [Architecture.md](Architecture.md) for system design details
- See [model_card.md](model_card.md) for model specifications
- See [reflection.md](reflection.md) for development notes

## 🎓 Learning Resources

- [Sentence Transformers docs](https://www.sbert.net/)
- [FAISS: Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
- [RAG pattern explained](https://aws.amazon.com/blogs/machine-learning/rag-pattern/)
- [Recommendation systems](https://developers.google.com/machine-learning/recommendation)

## 📄 License

MIT License - See repository for details

## 👤 Author

Markkoby - AI System Project
