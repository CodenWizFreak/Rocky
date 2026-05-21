# AGENTS.md — KG + GNN Movie Recommender with Machine Unlearning

## Project Overview

A movie recommendation system built on a knowledge graph, trained with a sharded GNN (LightGCN), and capable of machine unlearning — so when a user says "never recommend Conjuring movies again", the system forgets that preference and behaves accordingly.

**Target hardware:** Apple M4 Air (MPS backend, low compute)
**Dataset:** MovieLens 100K
**Core stack:** PyTorch Geometric, NetworkX, FastAPI (Python backend) + Next.js (frontend)
**Demo context:** Academic / research presentation

---

## Architecture at a Glance

```
MovieLens Data
     │
     ▼
Knowledge Graph (NetworkX / PyG)
[User] ──watched──► [Movie] ──genre──► [Genre]
                        └──franchise──► [Series]
                        └──actor──────► [Actor]
     │
     ▼
Shard Partitioner (split by user groups)
[Shard 1]  [Shard 2]  [Shard 3]  [Shard 4]  [Shard 5]
     │
     ▼
LightGCN per shard (trains independently)
     │
     ▼
Aggregated Recommendation Score
final_score = weighted average of shard outputs
     │
     ▼
Unlearning Request Handler
→ delete edge → recompute user embedding → retrain only affected shard
     │
     ▼
Recommendation Output (with explanation)
```

---

## Directory Structure

```
kg-gnn-unlearn/
├── AGENTS.md                  ← this file
├── data/
│   ├── raw/                   ← MovieLens 100K files
│   └── processed/
│       ├── graph.pt           ← PyG graph object
│       └── shards/            ← shard edge lists
│           ├── shard_0.pt
│           └── ...
├── graph/
│   ├── build_kg.py            ← knowledge graph construction
│   └── partition.py           ← shard partitioning logic
├── models/
│   ├── lightgcn.py            ← LightGCN model definition
│   └── shards/                ← saved shard model weights
│       ├── shard_0.pt
│       └── ...
├── unlearn/
│   ├── edge_deletion.py       ← remove user-movie edge
│   ├── embedding_update.py    ← recompute user embedding cheaply
│   └── shard_retrain.py       ← retrain only affected shard
├── recommend/
│   ├── inference.py           ← score aggregation across shards
│   ├── explain.py             ← why was this recommended
│   └── penalize.py            ← dislike penalty / preference subtraction
├── chatbot/
│   └── bot.py                 ← optional CLI chatbot interface
├── train.py                   ← main training entry point
├── evaluate.py                ← hit rate, NDCG metrics
└── requirements.txt
```

---

## Phase-by-Phase Build Plan

### Phase 1 — Data + Knowledge Graph

**Goal:** Load MovieLens 100K and build a graph.

**Nodes:**
- `User` — from `u.user`
- `Movie` — from `u.item`
- `Genre` — extracted from genre flags
- `Franchise` — manually mapped (e.g., Conjuring, Halloween, Saw)

**Edges:**
- `User → watched → Movie` (from `u.data` ratings ≥ 3.5)
- `Movie → genre → Genre`
- `Movie → franchise → Franchise` (custom mapping)

**File:** `graph/build_kg.py`

```python
# Key libraries
import networkx as nx
from torch_geometric.utils import from_networkx

# Steps
# 1. Load u.data (ratings), u.item (movies), u.genre (genres)
# 2. Create nx.DiGraph()
# 3. Add user nodes, movie nodes, genre nodes
# 4. Add edges with type attributes
# 5. Convert to PyG HeteroData
# 6. Save to data/processed/graph.pt
```

**Output:** `HeteroData` object with node types and edge types.

---

### Phase 2 — Shard Partitioning

**Goal:** Split users into N shards (start with 5).

**Strategy:** Random user partitioning (simplest, works well for demo).

**File:** `graph/partition.py`

```python
N_SHARDS = 5

# 1. Get all user IDs
# 2. Shuffle randomly (fix seed for reproducibility)
# 3. Split into N equal groups
# 4. For each shard:
#    - Extract all edges involving those users
#    - Save as shard_k.pt
# 5. Store shard_user_map: {user_id: shard_index}
```

**Why this matters for unlearning:**
When user X requests unlearning → only retrain shard that contains user X. Not the whole model.

---

### Phase 3 — LightGCN Model

**Goal:** Train one LightGCN per shard.

**Why LightGCN:**
- No feature transformation (just embedding propagation)
- Very fast on small graphs
- Perfect for M4 Air

**File:** `models/lightgcn.py`

```python
import torch
from torch_geometric.nn import LGConv

class LightGCN(torch.nn.Module):
    def __init__(self, num_users, num_movies, emb_dim=64, n_layers=3):
        super().__init__()
        self.user_emb = torch.nn.Embedding(num_users, emb_dim)
        self.movie_emb = torch.nn.Embedding(num_movies, emb_dim)
        self.convs = torch.nn.ModuleList([LGConv() for _ in range(n_layers)])

    def forward(self, edge_index):
        # Propagate embeddings through graph layers
        # Return final user and movie embeddings
        ...

    def recommend(self, user_id, top_k=10):
        # Dot product between user embedding and all movie embeddings
        # Return top K movies
        ...
```

**Hyperparameters (M4 Air safe):**
```
emb_dim    = 64
n_layers   = 3
lr         = 0.001
epochs     = 20
batch_size = 1024
device     = mps  (Apple Silicon)
```

**Training loop:** BPR loss (Bayesian Personalised Ranking) — standard for implicit feedback.

**File:** `train.py`

```python
device = torch.device("mps")  # Apple Silicon GPU

for shard_id in range(N_SHARDS):
    model = LightGCN(...).to(device)
    shard_graph = load_shard(shard_id)
    train_shard(model, shard_graph)
    torch.save(model.state_dict(), f"models/shards/shard_{shard_id}.pt")
```

**Expected training time on M4 Air:** ~2–5 minutes per shard.

---

### Phase 4 — Recommendation Inference

**Goal:** Given a user, aggregate scores from all shard models.

**File:** `recommend/inference.py`

```python
def recommend(user_id, top_k=10):
    scores = []
    for shard_id in range(N_SHARDS):
        model = load_shard_model(shard_id)
        shard_scores = model.recommend(user_id)  # returns movie_id → score dict
        scores.append(shard_scores)

    # Weighted average
    final_scores = aggregate(scores, weights=[0.2, 0.2, 0.2, 0.2, 0.2])

    # Apply dislike penalties
    final_scores = penalize_dislikes(user_id, final_scores)

    return top_k_movies(final_scores, k=top_k)
```

---

### Phase 5 — Machine Unlearning

This is the core differentiator of the project.

**Two-level unlearning strategy:**

#### Level 1 — Fast (zero retraining)

When user says "I hate Conjuring 2":

```python
# Step 1: Delete the edge
graph.remove_edge(user_id, conjuring_2_id)

# Step 2: Recompute user embedding from scratch
watched_movies = get_remaining_watched(user_id)
user_embedding = mean(movie_embeddings[watched_movies])

# Step 3: Store dislike
dislikes[user_id].append(conjuring_2_id)
dislikes[user_id].append(franchise["Conjuring"])  # penalise whole franchise

# No retraining. Done in milliseconds.
```

**File:** `unlearn/embedding_update.py`

#### Level 2 — True unlearning (shard retrain)

For stronger guarantees:

```python
# Step 1: Find which shard contains this user
shard_id = shard_user_map[user_id]

# Step 2: Remove edge from that shard's graph
shard_graphs[shard_id].remove_edge(user_id, movie_id)

# Step 3: Retrain only that shard
retrain_shard(shard_id)

# Other 4 shards untouched.
```

**File:** `unlearn/shard_retrain.py`

**Expected retrain time on M4 Air:** < 1 minute per shard.

---

### Phase 6 — Dislike Penalty + Preference Subtraction

**File:** `recommend/penalize.py`

```python
DISLIKE_PENALTY = 100.0

def penalize_dislikes(user_id, scores):
    disliked = get_dislikes(user_id)          # e.g., [conjuring_2, conjuring_3]
    disliked_franchises = get_disliked_franchises(user_id)  # e.g., ["Conjuring"]

    for movie_id, score in scores.items():
        if movie_id in disliked:
            scores[movie_id] -= DISLIKE_PENALTY

        if get_franchise(movie_id) in disliked_franchises:
            scores[movie_id] -= DISLIKE_PENALTY  # kills the whole franchise

    return scores
```

**Embedding-level subtraction (optional, stronger):**

```python
# Subtract disliked movie vector from user vector
user_vec = user_embedding[user_id]
for disliked_movie in dislikes[user_id]:
    user_vec = user_vec - alpha * movie_embedding[disliked_movie]
```

---

### Phase 7 — Explainability

**File:** `recommend/explain.py`

```python
def explain(user_id, recommended_movie_id):
    watched = get_watched(user_id)
    common_genres = genre_overlap(watched, recommended_movie_id)
    common_actors = actor_overlap(watched, recommended_movie_id)
    similar_users = get_similar_users(user_id)

    explanation = f"Recommended because:"
    if common_genres:
        explanation += f"\n- Same genre ({', '.join(common_genres)}) as movies you watched"
    if common_actors:
        explanation += f"\n- Stars {common_actors[0]}, who appeared in your watch history"
    if similar_users:
        explanation += f"\n- Users with similar taste also liked this"

    return explanation
```

**After unlearning, also explain:**
```
"Conjuring: The Devil Made Me Do It" was excluded from your recommendations
because you removed Conjuring 2 from your watch history.
```

---

### Phase 8 — Chatbot Interface (optional but cool)

**File:** `chatbot/bot.py`

```python
# Simple CLI chatbot that wraps the whole system

# Commands:
# "recommend me something"
# "i hate Conjuring 2"
# "never show me horror again"
# "why did you recommend this?"
# "show my watch history"

# Parse intent → call appropriate function → print result
```

Parse intents with simple keyword matching or plug in an LLM (Ollama locally).

---

## Compute Budget (M4 Air)

| Task | Time estimate |
|---|---|
| Build knowledge graph | ~30 seconds |
| Partition into 5 shards | ~10 seconds |
| Train 1 shard (LightGCN) | ~2–5 min |
| Train all 5 shards | ~15–25 min total |
| Inference (recommend) | < 1 second |
| Level 1 unlearning | milliseconds |
| Level 2 shard retrain | ~2–5 min |

**Memory estimate:** < 2GB RAM for full system with MovieLens 100K.

---

## Demo Architecture (Next.js + FastAPI)

### Full Stack Layout

```
kg-gnn-unlearn/
├── backend/                   ← Python, managed with uv
│   ├── pyproject.toml
│   ├── main.py                ← FastAPI app entry point
│   ├── routers/
│   │   ├── recommend.py       ← GET /recommend/{user_id}
│   │   ├── unlearn.py         ← POST /unlearn
│   │   └── history.py         ← GET /history/{user_id}
│   ├── services/
│   │   ├── graph.py           ← KG operations
│   │   ├── model.py           ← LightGCN inference
│   │   ├── unlearn.py         ← edge deletion + embedding update
│   │   └── explain.py         ← explanation generation
│   └── state/
│       ├── dislikes.json      ← persisted user dislikes
│       └── embeddings.pt      ← cached user embeddings
│
└── frontend/                  ← Next.js 14 (App Router)
    ├── app/
    │   ├── page.tsx           ← main demo page
    │   ├── layout.tsx
    │   └── globals.css
    ├── components/
    │   ├── UserSelector.tsx   ← pick a dummy user
    │   ├── WatchHistory.tsx   ← show what they watched
    │   ├── RecommendList.tsx  ← before/after cards
    │   ├── UnlearnPanel.tsx   ← the "I hate this" input
    │   ├── ExplainCard.tsx    ← why was this recommended
    │   └── GraphViz.tsx       ← optional: D3 KG visualiser
    └── lib/
        └── api.ts             ← typed fetch wrappers
```

---

### Backend API (FastAPI + uv)

**Setup:**
```bash
cd backend
uv init
uv add fastapi uvicorn torch torch-geometric networkx pandas
uv run uvicorn main:app --reload --port 8000
```

**Endpoints:**

```
GET  /users                          → list of dummy users
GET  /history/{user_id}              → watch history for user
GET  /recommend/{user_id}?k=10       → top-K recommendations
POST /unlearn                        → trigger unlearning
GET  /explain/{user_id}/{movie_id}   → why this was recommended
GET  /graph/{user_id}                → subgraph data for visualisation
```

**`POST /unlearn` body:**
```json
{
  "user_id": 42,
  "movie_id": 318,
  "scope": "franchise"
}
```

**Response shape for `/recommend`:**
```json
{
  "user_id": 42,
  "recommendations": [
    {
      "movie_id": 101,
      "title": "Sinister",
      "genre": ["Horror", "Mystery"],
      "franchise": null,
      "score": 0.94,
      "reason": "Same genre as movies you watched"
    }
  ],
  "unlearn_applied": true,
  "excluded_franchises": ["Conjuring"]
}
```

**`main.py` skeleton:**
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import recommend, unlearn, history

app = FastAPI(title="GNN Recommender with Unlearning")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(recommend.router)
app.include_router(unlearn.router)
app.include_router(history.router)
```

---

### Frontend Layout (Next.js 14, App Router)

**Setup:**
```bash
cd frontend
npx create-next-app@latest . --typescript --tailwind --app
npm install framer-motion d3 @radix-ui/react-tooltip
```

**Main demo page — `app/page.tsx` — four panel layout:**

```
┌─────────────────────────────────────────────────────┐
│  🎬 GNN Movie Recommender + Machine Unlearning Demo  │
├───────────────┬─────────────────────────────────────┤
│               │  BEFORE          │  AFTER            │
│  User Panel   │  ─────────────   │  ─────────────    │
│               │  1. Conjuring 3  │  1. Sinister      │
│  HorrorNerd   │  2. Annabelle 2  │  2. Hereditary    │
│  [avatar]     │  3. The Nun      │  3. Talk to Me    │
│               │  4. Sinister     │  4. The Black..   │
│  Watch History│  5. Hereditary   │  5. Midsommar     │
│  ──────────── │                  │                   │
│  • Conjuring  │  [Explain ▼]     │  [Explain ▼]      │
│  • Conjuring 2│                  │                   │
│  • Annabelle  ├──────────────────┴───────────────────┤
│  • Insidious  │  💬 Unlearn Panel                     │
│  • Hereditary │  "I hate: [Conjuring 2        ] [✕]"  │
│               │  Scope: ○ Movie  ● Franchise          │
│  [Change User]│  [Apply Unlearning]                   │
└───────────────┴─────────────────────────────────────┘
```

**Key UX moments for presentation:**

1. Select a user → watch history loads on the left
2. Click "Get Recommendations" → before-list populates
3. Type or click a movie to dislike → "Apply Unlearning"
4. Loading state shows: *"Recomputing embeddings... Retraining shard 3..."*
5. After-list populates → Conjuring movies visually struck through and gone
6. Hover any recommendation → explainability tooltip appears
7. Optional: graph panel shows KG with the deleted edge highlighted in red

**`components/UnlearnPanel.tsx` key behaviour:**
```tsx
// On submit:
// 1. POST /unlearn → get confirmation
// 2. Re-fetch /recommend/{user_id}
// 3. Animate the diff between before and after lists
//    - removed movies slide out with red fade
//    - new movies slide in with green fade
```

**`components/RecommendList.tsx` diff animation:**
```tsx
// Use framer-motion to animate:
// - movies that disappeared → red, strikethrough, slide left out
// - movies that appeared    → green, slide in from right
// This visually proves the unlearning worked — very presentation-friendly
```

---

### Dev Workflow

```bash
# Terminal 1 — backend
cd backend && uv run uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd frontend && npm run dev  # runs on localhost:3000
```

**Environment — `frontend/.env.local`:**
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

### Presentation Flow (5-minute demo script)

```
1. Open browser → localhost:3000

2. "This is HorrorNerd_007 — they've watched every Conjuring movie."
   → Show watch history panel

3. "The GNN recommends the next Conjuring and Annabelle films."
   → Show before-recommendations

4. "Now they tell the system: I hate Conjuring 2, forget I ever watched it."
   → Type Conjuring 2 in the unlearn panel, scope = Franchise, click Apply

5. "The system deletes that edge from the knowledge graph,
    recomputes the user embedding, and retrains only the affected shard."
   → Watch loading animation

6. "The entire Conjuring franchise is gone from recommendations."
   → Show after-recommendations, red fade-out animation

7. "Hover any card to see why it was recommended."
   → Show explainability tooltip

8. "This is GDPR-compliant selective forgetting, running
    entirely on a MacBook Air with no cloud compute."
```

---



**Backend (`backend/pyproject.toml` via uv):**
```toml
[project]
name = "kg-gnn-backend"
requires-python = ">=3.11"
dependencies = [
    "fastapi",
    "uvicorn[standard]",
    "torch>=2.2.0",
    "torch_geometric",
    "networkx",
    "pandas",
    "numpy",
    "scikit-learn",
    "tqdm",
]
```

```bash
cd backend
uv init && uv sync

# PyG for Apple Silicon (run once after uv sync)
uv run pip install pyg_lib torch_scatter torch_sparse \
  -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
```

**Frontend:**
```bash
cd frontend
npx create-next-app@latest . --typescript --tailwind --app
npm install framer-motion d3 @radix-ui/react-tooltip
```

---

## Dataset Setup

```bash
# Download MovieLens 100K
wget https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip -d data/raw/

# Key files:
# data/raw/ml-100k/u.data    — ratings (user, movie, rating, timestamp)
# data/raw/ml-100k/u.item    — movie info (title, genres)
# data/raw/ml-100k/u.user    — user demographics
```

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| Graph library | PyTorch Geometric | MPS support, research-grade |
| GNN model | LightGCN | Lightest model, still SOTA for recs |
| Embedding size | 64 | Low memory, still expressive |
| Shard count | 5 | Cheap retraining, good coverage |
| Shard strategy | Random user split | Simple, effective for demo |
| Unlearning default | Level 1 (embedding update) | Instant, zero retraining |
| Unlearning strong | Level 2 (shard retrain) | Research-grade, ~5 min cost |
| Dislike scope | Movie + franchise | Prevents franchise bleed-through |
| Device | `torch.device("mps")` | Apple Silicon GPU |
| Backend | FastAPI + uv | Fast, typed, async-ready |
| Frontend | Next.js 14 App Router | Research-grade presentation quality |
| Animations | framer-motion | Visual proof of unlearning working |

---

## Evaluation Metrics

```python
# Standard recommender metrics
# Computed on held-out 20% of ratings

Hit Rate @ 10     — was the true movie in top 10?
NDCG @ 10         — ranking quality
Precision @ 10    — accuracy of top 10

# Unlearning-specific metrics
Forgetting Score  — does unlearned movie appear in top-K?
Recommendation Shift — how much did the list change after unlearning?
```

---

## Things NOT to Do

- Do not use the full MovieLens 25M dataset on M4 Air
- Do not use embedding size > 128 (unnecessary, slow)
- Do not run all 5 shard retrains for a single unlearning request — only retrain the affected shard
- Do not use GAT or GraphSAGE as first choice — heavier than LightGCN with no benefit here
- Do not implement perfect cryptographic unlearning — approximate is fine and is standard in research

---

## Research Angle (if writing a paper or report)

This system sits at the intersection of three active research areas:

1. **Graph-based recommendation** — using KG structure for better item understanding
2. **Machine unlearning on graphs** — edge/node deletion with retraining efficiency
3. **Right to be forgotten** — GDPR-compliant recommendation systems

Related papers to cite:
- *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation* (He et al., 2020)
- *GraphEraser: Defending Graph Neural Networks Against Backdoor Attacks via Subgraph-Based Model Resets* (2022)
- *Making Recommender Systems Forget: Learning to Unlearn and Certifiably Forget* (Chen et al., 2022)

---

## Demo Flow (what to show)

```
1. User "HorrorNerd_007" has watched:
   Conjuring, Conjuring 2, Annabelle, The Nun, Insidious, Hereditary

2. System recommends:
   → Conjuring 3, Annabelle 2, Sinister, Midsommar, The Black Phone

3. User says: "fuck Conjuring 2, remove it, I hate that franchise"

4. System:
   - Deletes: HorrorNerd_007 → watched → Conjuring 2
   - Marks dislike: franchise = Conjuring
   - Recomputes embedding
   - Applies franchise penalty

5. New recommendations:
   → Sinister, Hereditary 2, Midsommar, The Black Phone, Talk to Me

6. Conjuring 3 is GONE. Annabelle is GONE. The Nun is GONE.

7. Explanation shown:
   "Conjuring franchise excluded based on your preference update."
```

---

*Build it phase by phase. Phase 1–3 alone is already a working recommender. Phases 4–5 make it research-grade.*
