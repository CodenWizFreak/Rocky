# KG + GNN Movie Recommender with Machine Unlearning

Build a full-stack movie recommendation system using a knowledge graph, sharded LightGCN, and machine unlearning on the MovieLens 100K dataset. FastAPI backend + Next.js frontend, targeting Apple M4 Air (MPS backend).

## User Review Required

> [!IMPORTANT]
> **Dataset location**: MovieLens 100K at `ml-100k/`. Will be read as `data/raw/` symlink or direct path reference.
> 
> **Franchise mapping**: The MovieLens 100K dataset (1995-1998 era movies) does NOT contain Conjuring/modern horror franchises. I'll create franchise mappings for the actual movie series present (e.g., Star Wars, Star Trek, Indiana Jones, Die Hard, Alien, etc.) and use a horror-adjacent franchise for demo purposes.
>
> **MPS vs CPU**: Will default to CPU with MPS fallback since PyG+MPS can have compatibility issues. Training is fast enough (~2-5 min per shard on CPU for 100K).

---

## Proposed Changes

### Phase 1 — Project Setup & Data Pipeline

#### [NEW] [pyproject.toml](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/pyproject.toml)
Root-level Python project config managed with `uv`. Dependencies: `fastapi`, `uvicorn`, `torch`, `torch_geometric`, `networkx`, `pandas`, `numpy`, `scikit-learn`, `tqdm`.

#### [NEW] [graph/build_kg.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/graph/build_kg.py)
- Load [ml-100k/u.data](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/ml-100k/u.data), [ml-100k/u.item](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/ml-100k/u.item), [ml-100k/u.genre](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/ml-100k/u.genre)
- Parse 943 users, 1682 movies, 19 genres
- Build NetworkX DiGraph with node types (user, movie, genre, franchise)
- Create edges: `user→watched→movie` (ratings ≥ 3.5), `movie→genre→genre`, `movie→franchise→franchise`
- Convert to PyG HeteroData, save as `data/processed/graph.pt`
- Create a hardcoded franchise mapping for real ML-100K movie series

#### [NEW] [graph/partition.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/graph/partition.py)
- Split 943 users into 5 shards (random, seed=42)
- For each shard: extract all edges involving its users
- Save shard edge indices as `data/processed/shards/shard_0.pt` ... `shard_4.pt`
- Save `shard_user_map.json` mapping user_id → shard_index

---

### Phase 2 — LightGCN Model & Training

#### [NEW] [models/lightgcn.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/models/lightgcn.py)
- LightGCN class with user/movie embeddings (dim=64), 3 LGConv layers
- BPR loss function
- Forward pass: propagate embeddings through graph layers
- `recommend()` method: dot product scoring

#### [NEW] [train.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/train.py)
- Load processed graph and shard data
- For each shard: instantiate LightGCN, train with BPR loss, 50 epochs
- Save model weights to `models/shards/shard_0.pt` ... `shard_4.pt`
- Device: CPU (with MPS option)

#### [NEW] [evaluate.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/evaluate.py)
- Compute Hit Rate@10, NDCG@10, Precision@10 on held-out test set
- Compute unlearning-specific metrics (forgetting score, recommendation shift)

---

### Phase 3 — Recommendation & Unlearning Engine

#### [NEW] [recommend/inference.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/recommend/inference.py)
- Load all 5 shard models
- For a given user: get scores from user's shard model, aggregate
- Apply dislike penalties, return top-K

#### [NEW] [recommend/explain.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/recommend/explain.py)
- Genre overlap, franchise overlap explanations
- Collaborative filtering explanations (similar users)
- Post-unlearning exclusion explanations

#### [NEW] [recommend/penalize.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/recommend/penalize.py)
- DISLIKE_PENALTY = 100.0
- Penalize individual movies and full franchises
- Optional embedding-level subtraction

#### [NEW] [unlearn/edge_deletion.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/unlearn/edge_deletion.py)
- Remove user→movie edge from graph
- Remove from shard graph
- Persist changes

#### [NEW] [unlearn/embedding_update.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/unlearn/embedding_update.py)
- Level 1 (fast): recompute user embedding from remaining watched movies
- Store dislike for movie and franchise

#### [NEW] [unlearn/shard_retrain.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/unlearn/shard_retrain.py)
- Level 2 (strong): identify affected shard, retrain only that shard
- Save updated weights

---

### Phase 4 — FastAPI Backend

#### [NEW] [backend/main.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/backend/main.py)
- FastAPI app with CORS (allow localhost:3000)
- Include routers for recommend, unlearn, history
- Startup event: load graph, models, state

#### [NEW] [backend/routers/recommend.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/backend/routers/recommend.py)
- `GET /recommend/{user_id}?k=10` — top-K recommendations
- `GET /explain/{user_id}/{movie_id}` — why recommended

#### [NEW] [backend/routers/unlearn.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/backend/routers/unlearn.py)
- `POST /unlearn` — trigger unlearning (body: user_id, movie_id, scope)

#### [NEW] [backend/routers/history.py](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/backend/routers/history.py)
- `GET /users` — list users with names
- `GET /history/{user_id}` — watch history
- `GET /graph/{user_id}` — subgraph data for D3

#### [NEW] [backend/services/](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/backend/services/)
- `graph_service.py` — KG operations
- `model_service.py` — LightGCN inference wrapper
- `unlearn_service.py` — unlearning orchestration
- `explain_service.py` — explanation generation

---

### Phase 5 — Next.js Frontend

#### [NEW] [frontend/](file:///Users/arnabsengupta/Desktop/code/final%20year%20project/rocky/frontend/)
- Next.js 14 App Router with TypeScript + Tailwind
- Premium dark theme with glassmorphism, gradients, micro-animations
- Framer Motion for diff animations (red fade-out / green slide-in)

Key components:
- `UserSelector` — avatar + name picker (sample users)
- `WatchHistory` — movie grid of watched titles
- `RecommendList` — before/after recommendation cards with animated diff
- `UnlearnPanel` — dislike input with movie/franchise scope toggle
- `ExplainCard` — explainability tooltip per recommendation
- `GraphViz` — D3 force-directed KG subgraph visualizer

---

## Verification Plan

### Automated Tests

1. **Knowledge Graph build**: `python3 graph/build_kg.py` — verify `data/processed/graph.pt` is created with correct node/edge counts
2. **Shard partitioning**: `python3 graph/partition.py` — verify 5 shard files + user map are created, all users assigned
3. **Training**: `python3 train.py` — verify 5 model weight files created, loss decreases per shard
4. **Evaluation**: `python3 evaluate.py` — verify metrics are printed correctly
5. **Backend**: `cd backend && uv run uvicorn main:app --port 8000` then test endpoints with curl:
   - `curl http://localhost:8000/users`
   - `curl http://localhost:8000/history/1`
   - `curl http://localhost:8000/recommend/1?k=10`
   - `curl -X POST http://localhost:8000/unlearn -H 'Content-Type: application/json' -d '{"user_id":1,"movie_id":50,"scope":"movie"}'`
6. **Frontend**: `cd frontend && npm run dev` — verify page loads at localhost:3000

### Manual Verification
- Open http://localhost:3000 in browser
- Select a user → verify watch history loads
- Click "Get Recommendations" → verify movie cards appear
- Type a movie to dislike → Apply Unlearning → verify diff animation shows removed/new movies
- Hover recommendations → verify explanation tooltips
