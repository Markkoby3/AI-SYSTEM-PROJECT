# 🎵 Advanced Music Recommender System Architecture

## Overview: RAG + Fine-Tuned Model Architecture

This document describes a production-grade music recommendation system that combines:
- **Retrieval-Augmented Generation (RAG)**: Intelligent song database retrieval before answering ✅ **IMPLEMENTED**
- **Fine-Tuned Specialized Models**: Domain-specific ML models trained for music preferences (In progress)
- **Ensemble Scoring**: Combines multiple ranking signals for optimal recommendations ✅ **READY**

### Implementation Status
- ✅ Phase 1: RAG Retrieval Pipeline (COMPLETE)
- 🔄 Phase 2: ML Model Scoring (TODO)
- 🔄 Phase 3: Ensemble & Explanation Layer (TODO)
- 🔄 Phase 4: Production API (TODO)

---

## Part 1: System Architecture

### High-Level Data Flow

```
USER REQUEST
    ↓
CONTEXT BUILDER
├─ Fetch user profile (genre, mood, energy)
├─ Retrieve listening history
└─ Get current context (time, device)
    ↓
┌─────────────────────────────────────────┐
│  PARALLEL PROCESSING                    │
├─────────────────────────────────────────┤
│                                         │
│  ┌─────────────────┐  ┌──────────────┐ │
│  │ RAG PIPELINE    │  │ FEATURE PREP │ │
│  │                 │  │              │ │
│  │ 1. Encode query │  │ Extract user │ │
│  │ 2. Vector search│  │ features     │ │
│  │ 3. Filter/rank │  │              │ │
│  │                 │  │ Output: 50  │ │
│  │ Output: 200    │  │ dimensions  │ │
│  │ candidates     │  │              │ │
│  └────────┬────────┘  └──────┬───────┘ │
│           │                  │         │
└───────────┼──────────────────┼─────────┘
            │                  │
            └────────┬─────────┘
                     ↓
           MODEL INFERENCE ENGINE
           ├─ Neural Network (0.5 weight)
           ├─ XGBoost (0.3 weight)
           └─ RAG Score (0.2 weight)
                     ↓
           SCORE FUSION & RANKING
           ├─ Combine ensemble scores
           ├─ Apply diversity constraints
           └─ Select top-10
                     ↓
           EXPLANATION GENERATION
           ├─ Identify matching criteria
           ├─ Generate human-readable text
           └─ Calculate confidence
                     ↓
           RESPONSE TO USER
           (10 recommendations with explanations)
```

### Component Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│                   PRESENTATION LAYER                         │
│  (Streamlit UI / API / Mobile Client)                        │
└────────────────────┬─────────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────────┐
│              REQUEST ORCHESTRATION LAYER                      │
│  • Input validation                                           │
│  • User profile loading                                       │
│  • Parallel async coordination                                │
└────────────────────┬─────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        ▼                         ▼
┌──────────────────┐      ┌──────────────────┐
│  RAG RETRIEVAL   │      │  MODEL SCORING   │
├──────────────────┤      ├──────────────────┤
│ Query Encoder    │      │ Feature Engineer │
│ Vector Store     │      │ Neural Network   │
│ Semantic Search  │      │ XGBoost Booster  │
│ Re-ranking       │      │ Ensemble Average │
└────────┬─────────┘      └────────┬─────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
         ┌────────────────────────┐
         │ ENSEMBLE & RANKING     │
         │ • Score fusion         │
         │ • Diversity injection  │
         │ • Cold-start handling  │
         └────────────┬───────────┘
                      ▼
         ┌────────────────────────┐
         │ EXPLANATION LAYER      │
         │ • Match criteria       │
         │ • Natural language     │
         │ • Confidence scoring   │
         └────────────┬───────────┘
                      ▼
         ┌────────────────────────┐
         │ RESPONSE & LOGGING     │
         │ • Format JSON          │
         │ • Track metrics        │
         │ • Cache results        │
         └────────────────────────┘
```

### Data Architecture

```
PRIMARY DATA STORES
├─ Songs Metadata DB (PostgreSQL)
│  ├─ id, title, artist, genre, mood
│  ├─ energy, tempo, valence, danceability
│  └─ acousticness, popularity, release_date
│
├─ User Profiles DB (PostgreSQL)
│  ├─ user_id, favorite_genre, favorite_mood
│  ├─ target_energy, likes_acoustic
│  └─ listening_history, mean_energy, mean_valence
│
└─ Vector Stores (Pinecone / Weaviate)
   ├─ Song embeddings (768-dim vectors)
   ├─ User profile vectors
   └─ Fast similarity search (HNSW index)
```

---

## Phase 1: RAG Pipeline Implementation ✅ COMPLETE

### 1.1 Core Components

**File**: `rag.py`

The RAG pipeline consists of:

1. **RAGPipeline Class**: Main orchestrator
   - Loads embedding model (Sentence Transformers)
   - Builds FAISS vector index
   - Performs semantic search over songs
   - Reranks results with metadata matching

2. **Key Methods**:
   - `build_index()`: Generates embeddings for all songs, caches to disk
   - `retrieve(query, top_k=200, rerank=True)`: Retrieves semantically similar songs
   - `batch_retrieve()`: Batch processing for multiple queries
   - `explain_retrieval()`: Generates human-readable explanations

### 1.2 How It Works

**Step 1: Embedding Generation**
```
Song: "Sunrise City" by "Neon Echo"
      Genre: pop | Mood: happy | Energy: 0.82
        ↓
      Combined text: "Sunrise City Neon Echo genre pop mood happy energy level 0.82..."
        ↓
      Sentence-Transformer encodes to 384-dim vector
        ↓
      FAISS index stores for fast lookup
```

**Step 2: Query Processing**
```
User: "I want energetic pop songs for working out"
        ↓
      Encode query to 384-dim vector
        ↓
      FAISS finds 200 nearest neighbors (L2 distance)
        ↓
      Rerank: Boost songs matching "energetic", "pop", "workout"
        ↓
      Return top-k ranked candidates
```

**Step 3: Explanation Generation**
```
Retrieved song + query → Natural language explanation
Example: "Retrieved because it matches 'pop' genre in your search and 
          has high energy that matches your workout preference."
```

### 1.3 Integration with Ensemble System

The RAG pipeline provides the **retrieval stage** in the ensemble architecture:

```
RAG Retrieval (200 candidates)
        ↓
Feature-Based Scoring (scoring all 200)
        ↓
Score Fusion & Ranking
        ↓
Top-10 Final Recommendations
```

### 1.4 Usage

```python
from rag import create_rag_pipeline

# Initialize
rag = create_rag_pipeline(songs)

# Retrieve candidates
results = rag.retrieve("energetic workout music", top_k=200)

# Use results as input to ensemble ranking system
for song, similarity in results[:200]:
    # Pass to feature-based scoring
    feature_score = score_song(user_prefs, song)
    ensemble_score = 0.3 * similarity + 0.7 * feature_score
```

---

## Part 2: RAG Pipeline - Building Blocks

### 2.1 Query Encoding

**What it does**: Converts user profile → searchable embedding vector

**Implementation Steps:**

```python
# Step 1: Install embedding model
# pip install sentence-transformers

from sentence_transformers import SentenceTransformer
import numpy as np

# Step 2: Load pre-trained model (768-dimensional)
encoder = SentenceTransformer('all-MiniLM-L6-v2')

# Step 3: Create encoding function
def encode_user_profile(user):
    """Convert user profile to embedding."""
    # Build text description
    text = f"""
    Music preference profile:
    Genre: {user.favorite_genre}
    Mood: {user.favorite_mood}
    Energy level: {user.target_energy:.2f} (0-1 scale)
    Acoustic preference: {'Yes' if user.likes_acoustic else 'No'}
    Average song energy: {user.mean_energy:.2f}
    Average song valence: {user.mean_valence:.2f}
    """
    
    # Encode to 768-dimensional vector
    embedding = encoder.encode(text, convert_to_numpy=True)
    
    # Normalize for cosine similarity
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding
```

**What happens internally:**
1. User profile text → tokenized
2. Tokens → deep learned representations
3. Sequence → single 768-dim vector
4. Normalized for distance metrics

---

### 2.2 Vector Store Setup

**What it does**: Stores song embeddings for fast similarity search

**Installation & Setup:**

```bash
# Option 1: Pinecone (managed, recommended for production)
pip install pinecone-client

# Option 2: Weaviate (open-source, self-hosted)
pip install weaviate-client

# Option 3: Local (development only)
pip install faiss-cpu
```

**Build vector store:**

```python
import pinecone

# Initialize Pinecone
pinecone.init(
    api_key="YOUR_API_KEY",
    environment="us-west1-gcp"
)

# Create index (one-time setup)
pinecone.create_index(
    name="music-recommender",
    dimension=768,
    metric="cosine"
)

# Connect to index
index = pinecone.Index("music-recommender")

# Encode all songs
def build_song_embeddings(songs_db):
    """Encode all songs and upload to vector store."""
    
    vectors_to_upsert = []
    
    for song in songs_db:
        # Encode song metadata
        text = f"""
        Song: {song.title} by {song.artist}
        Genre: {song.genre}
        Mood: {song.mood}
        Energy: {song.energy:.2f}
        Valence: {song.valence:.2f}
        """
        
        embedding = encoder.encode(text, convert_to_numpy=True)
        embedding = embedding / np.linalg.norm(embedding)
        
        vectors_to_upsert.append((
            str(song.id),
            embedding.tolist(),
            {
                'title': song.title,
                'artist': song.artist,
                'genre': song.genre,
                'mood': song.mood,
            }
        ))
    
    # Upload in batches
    for i in range(0, len(vectors_to_upsert), 100):
        batch = vectors_to_upsert[i:i+100]
        index.upsert(vectors=batch)
    
    print(f"Uploaded {len(vectors_to_upsert)} song embeddings")

# Run once
build_song_embeddings(songs_db)
```

---

### 2.3 Semantic Search & Retrieval

**What it does**: Find top-K most similar songs using vector similarity

**Implementation:**

```python
def retrieve_candidate_songs(user_profile, k=200):
    """Retrieve top-k song candidates using semantic search."""
    
    # Step 1: Encode user profile
    query_embedding = encode_user_profile(user_profile)
    
    # Step 2: Search vector store
    results = index.query(
        vector=query_embedding.tolist(),
        top_k=k,
        include_metadata=True
    )
    
    # Step 3: Extract results
    candidates = []
    for match in results['matches']:
        song_id = int(match['id'])
        similarity_score = match['score']  # 0-1 (higher is better)
        
        # Fetch full song object from DB
        song = songs_db[song_id]
        candidates.append({
            'song': song,
            'rag_score': similarity_score
        })
    
    return candidates
```

**How HNSW search works:**
1. Query vector → hierarchical neighbor graph
2. Approximate nearest neighbors found in O(log N) time
3. Returns top-200 most similar songs (typically 50-100ms)

---

### 2.4 Filtering & Re-ranking

**What it does**: Remove duplicates, apply business rules, boost matching songs

**Implementation:**

```python
def filter_and_rerank(candidates, user_profile):
    """Multi-stage filtering and re-ranking."""
    
    # Step 1: Remove already-heard songs
    candidates = [
        c for c in candidates
        if c['song'].id not in user_profile.listening_history
    ]
    
    # Step 2: Enforce artist diversity (max 2 per artist)
    artist_count = {}
    filtered = []
    for candidate in candidates:
        artist = candidate['song'].artist
        if artist_count.get(artist, 0) < 2:
            filtered.append(candidate)
            artist_count[artist] = artist_count.get(artist, 0) + 1
    
    candidates = filtered
    
    # Step 3: Apply business rules
    # (licensing, age restrictions, geographic availability, etc.)
    
    # Step 4: Boost matching criteria
    for candidate in candidates:
        song = candidate['song']
        base_score = candidate['rag_score']
        boost = 0
        
        # Genre match: +0.5
        if song.genre == user_profile.favorite_genre:
            boost += 0.5
        
        # Mood match: +0.3
        if song.mood == user_profile.favorite_mood:
            boost += 0.3
        
        # Energy proximity: +0.2
        energy_diff = abs(song.energy - user_profile.target_energy)
        if energy_diff < 0.15:
            boost += 0.2
        
        # Acoustic preference: +0.15
        if user_profile.likes_acoustic and song.acousticness > 0.7:
            boost += 0.15
        elif not user_profile.likes_acoustic and song.acousticness < 0.3:
            boost += 0.15
        
        candidate['rag_score'] = min(base_score + boost, 1.0)
    
    # Step 5: Sort by adjusted score
    candidates.sort(key=lambda x: x['rag_score'], reverse=True)
    
    return candidates[:500]  # Top 500 for model scoring
```

---

## Part 3: Fine-Tuned Model - How to Build It

### 3.1 Feature Engineering

**What it does**: Extract numerical features from user + song for model input

**Implementation:**

```python
import numpy as np

class FeatureEngineer:
    """Extract ML features from user and song data."""
    
    GENRES = ['pop', 'rock', 'lofi', 'jazz', 'ambient', 'indie', 
              'electronic', 'classical', 'metal', 'folk']
    MOODS = ['happy', 'sad', 'chill', 'intense', 'relaxed', 
             'focused', 'peaceful', 'angry']
    
    def __init__(self):
        self.genre_to_idx = {g: i for i, g in enumerate(self.GENRES)}
        self.mood_to_idx = {m: i for i, m in enumerate(self.MOODS)}
    
    def extract_user_features(self, user):
        """Extract user-level features (66-dim)."""
        features = []
        
        # Genre one-hot encoding (10-dim)
        genre_onehot = np.zeros(len(self.GENRES))
        genre_onehot[self.genre_to_idx[user.favorite_genre]] = 1
        features.extend(genre_onehot)
        
        # Mood one-hot encoding (8-dim)
        mood_onehot = np.zeros(len(self.MOODS))
        mood_onehot[self.mood_to_idx[user.favorite_mood]] = 1
        features.extend(mood_onehot)
        
        # Continuous features (8-dim)
        features.extend([
            user.target_energy,
            float(user.likes_acoustic),
            user.mean_energy,
            user.mean_valence,
            user.skip_rate,
            len(user.listening_history) / 1000,
            user.subscription_level,
            user.account_age_days / 365,
        ])
        
        return np.array(features)
    
    def extract_song_features(self, song):
        """Extract song-level features (19-dim)."""
        features = []
        
        # Genre one-hot encoding (10-dim)
        genre_onehot = np.zeros(len(self.GENRES))
        genre_onehot[self.genre_to_idx.get(song.genre, 0)] = 1
        features.extend(genre_onehot)
        
        # Mood one-hot encoding (8-dim)
        mood_onehot = np.zeros(len(self.MOODS))
        mood_onehot[self.mood_to_idx.get(song.mood, 0)] = 1
        features.extend(mood_onehot)
        
        # Continuous features (9-dim)
        features.extend([
            song.energy,
            song.tempo_bpm / 200,
            song.valence,
            song.danceability,
            song.acousticness,
            song.popularity,
            min((datetime.now() - song.release_date).days / 365, 1.0),
            song.artist_followers / 10_000_000,
            song.explicit_content,
        ])
        
        return np.array(features)
    
    def extract_interaction_features(self, user, song):
        """Extract user-song interaction features (5-dim)."""
        features = [
            float(user.favorite_genre == song.genre),
            float(user.favorite_mood == song.mood),
            1.0 - min(abs(song.energy - user.target_energy), 1.0),
            song.acousticness if user.likes_acoustic else (1 - song.acousticness),
            1.0 - min(abs(song.valence - user.mean_valence), 1.0),
        ]
        return np.array(features)
    
    def combine_all_features(self, user, song, rag_score):
        """Combine all feature groups (93-dim total)."""
        user_feat = self.extract_user_features(user)          # 66-dim
        song_feat = self.extract_song_features(song)          # 19-dim
        interaction_feat = self.extract_interaction_features(user, song)  # 5-dim
        rag_feat = np.array([rag_score])                       # 1-dim
        
        combined = np.concatenate([
            user_feat,
            song_feat,
            interaction_feat,
            rag_feat
        ])
        
        return combined  # 93-dim feature vector
```

---

### 3.2 Neural Network Model

**What it does**: Deep learning model that learns user-song patterns

**Implementation:**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class RecommenderNN(nn.Module):
    """Neural network for engagement prediction."""
    
    def __init__(self, input_dim=93, hidden_dims=[512, 256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        hidden = self.hidden_layers(x)
        logits = self.output_layer(hidden)
        output = self.sigmoid(logits)
        return output
    
    def predict(self, features):
        """Predict engagement probability."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            prob = self.forward(x).item()
        return prob

# Training function
def train_model(model, train_loader, val_loader, epochs=20, lr=0.001):
    """Train the neural network."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y.unsqueeze(1))
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    return model
```

---

### 3.3 XGBoost Model

**What it does**: Gradient boosting model that captures feature interactions

**Implementation:**

```bash
# Install XGBoost
pip install xgboost lightgbm
```

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

def train_xgboost_model(X, y):
    """Train XGBoost for engagement prediction."""
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=10,
        eval_metric='auc'
    )
    
    # Train
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )
    
    return model
```

---

### 3.4 Ensemble Inference

**What it does**: Combines predictions from both models + RAG score

**Implementation:**

```python
class ModelInferenceEngine:
    """Score songs using ensemble of models."""
    
    def __init__(self, nn_model, xgb_model):
        self.nn_model = nn_model
        self.xgb_model = xgb_model
        self.feature_engineer = FeatureEngineer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def score_candidate(self, user, song, rag_score):
        """Score a single song."""
        
        # Extract features
        features = self.feature_engineer.combine_all_features(user, song, rag_score)
        
        # Neural network prediction
        nn_prob = self.nn_model.predict(features)
        
        # XGBoost prediction
        xgb_input = features.reshape(1, -1)
        xgb_prob = self.xgb_model.predict_proba(xgb_input)[0, 1]
        
        # Ensemble average (weighted)
        final_score = (0.5 * nn_prob) + (0.3 * xgb_prob) + (0.2 * rag_score)
        
        return final_score
    
    def score_batch(self, user, candidates_with_rag_scores):
        """Score multiple candidates efficiently."""
        scores = []
        
        for candidate in candidates_with_rag_scores:
            song = candidate['song']
            rag_score = candidate['rag_score']
            
            score = self.score_candidate(user, song, rag_score)
            scores.append((song, score))
        
        return scores
```

---

## Part 4: Integration & Orchestration

### Complete Recommendation Pipeline

```python
class MusicRecommender:
    """Complete recommendation system."""
    
    def __init__(self):
        # Initialize components
        self.rag_retriever = RAGRetriever(songs_db)
        self.feature_engineer = FeatureEngineer()
        self.nn_model = load_nn_model()
        self.xgb_model = load_xgb_model()
        self.model_engine = ModelInferenceEngine(self.nn_model, self.xgb_model)
        self.explanation_gen = ExplanationGenerator()
    
    def recommend(self, user_profile, k=10):
        """Generate k recommendations for user."""
        
        # Step 1: RAG retrieval (200 candidates)
        print("Retrieving candidates...")
        rag_candidates = self.rag_retriever.retrieve_candidate_songs(
            user_profile, 
            k=200
        )
        
        # Step 2: Model scoring
        print("Scoring with model...")
        scored = self.model_engine.score_batch(user_profile, rag_candidates)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Step 3: Select top-k
        top_k_songs = scored[:k]
        
        # Step 4: Generate explanations
        print("Generating explanations...")
        recommendations = []
        for rank, (song, score) in enumerate(top_k_songs, 1):
            # Find RAG score for this song
            rag_score = next(
                c['rag_score'] for c in rag_candidates 
                if c['song'].id == song.id
            )
            
            # Generate explanation
            explanation = self.explanation_gen.explain(
                user_profile, 
                song, 
                score
            )
            
            recommendations.append({
                'rank': rank,
                'song_id': song.id,
                'title': song.title,
                'artist': song.artist,
                'genre': song.genre,
                'score': round(score, 4),
                'confidence': round(min(max(score, 0), 1), 2),
                'explanation': explanation,
            })
        
        return recommendations
```

---

## Part 5: Step-by-Step Build Instructions

### Phase 1: Setup (Week 1)

```bash
# 1. Clone/setup project
cd music-recommender

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install sentence-transformers pinecone-client torch xgboost scikit-learn pandas numpy

# 4. Create directory structure
mkdir models data scripts logs
```

### Phase 2: Build RAG (Week 2)

```bash
# 1. Create vector store index (Pinecone)
python scripts/setup_vector_store.py

# 2. Encode and upload songs
python scripts/build_song_embeddings.py

# 3. Test retrieval
python scripts/test_rag_retrieval.py
```

### Phase 3: Train Models (Week 3-4)

```bash
# 1. Prepare training data from interaction logs
python scripts/prepare_training_data.py

# 2. Train neural network
python scripts/train_neural_network.py

# 3. Train XGBoost
python scripts/train_xgboost.py

# 4. Evaluate both models
python scripts/evaluate_models.py
```

### Phase 4: Integration & Deployment (Week 5)

```bash
# 1. Create API service
python api/app.py

# 2. Test end-to-end
curl -X POST http://localhost:8000/api/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test", "favorite_genre":"lofi", ...}'

# 3. Deploy to production
docker build -t music-recommender .
docker run -p 8000:8000 music-recommender
```

---

## Part 6: Key Metrics & Monitoring

### Performance Metrics

| Metric | Target | How to Monitor |
|--------|--------|---|
| **Latency (P99)** | < 500ms | Application logs, APM tools |
| **Model AUC** | > 0.85 | Validation set during training |
| **Click-Through Rate** | > 5% | User engagement analytics |
| **Skip Rate** | < 15% | Implicit feedback tracking |
| **Recall@10** | > 0.70 | A/B testing framework |
| **Diversity Score** | > 0.75 | Calculate artist/genre variance |

### Alert Thresholds

```python
ALERTS = {
    'latency_p99 > 500ms': 'Page optimization needed',
    'model_auc < 0.80': 'Model drift - trigger retraining',
    'skip_rate > 20%': 'Recommendations degrading',
    'error_rate > 0.5%': 'System issue - check logs',
}
```

---

## Summary Architecture Stack

```
INPUT: User Profile
    ↓
RAG LAYER (Pinecone + Sentence-Transformers)
├─ Query Encoder: Convert profile → 768-dim vector
├─ Vector Search: HNSW in ~50ms
└─ Output: 200 candidate songs
    ↓
FEATURE LAYER (Feature Engineering)
├─ Extract 93 features from user + song
└─ Output: Feature vectors ready for models
    ↓
MODEL LAYER (PyTorch + XGBoost)
├─ Neural Network: P(engagement) from pattern learning
├─ XGBoost: Capture feature interactions
└─ Output: Two engagement probability scores
    ↓
ENSEMBLE LAYER
├─ Weighted average: 50% NN + 30% XGBoost + 20% RAG
└─ Output: Final ranking scores
    ↓
EXPLANATION LAYER
├─ Identify matching criteria (genre, mood, energy, etc.)
├─ Generate natural language explanations
└─ Output: 10 recommendations with reasons
    ↓
RESPONSE: [Song1, Song2, ..., Song10] with explanations
```

This architecture balances accuracy (from learned models), explainability (from RAG), and performance (optimized pipeline).
