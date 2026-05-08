# Implementation status report

This report tracks progress against [`implementation_plan.md`](implementation_plan.md) (KG + GNN movie recommender with machine unlearning on MovieLens 100K). Status reflects the repository as of the last update to this document.

## Summary

| Phase | Topic | Status |
|-------|--------|--------|
| 1 | Project setup, data pipeline, KG build, sharding | **Done** (artifacts present) |
| 2 | LightGCN, training, evaluation | **Done** (code + trained shard weights) |
| 3 | Recommendation, explainability, penalization, unlearning | **Done** (Python modules) |
| 4 | FastAPI backend | **Done** (routers; logic wired to `recommend/` / `unlearn/`) |
| 5 | Next.js frontend | **Done** (App Router UI + API client) |
| Verification | Automated tests / full manual demo | **Partial** (scripts exist; formal test suite not listed in repo) |

---

## Phase 1 — Project setup and data pipeline

**Completed**

- Root [`pyproject.toml`](pyproject.toml): `uv`-managed project with FastAPI, Uvicorn, PyTorch, PyTorch Geometric, NetworkX, pandas, NumPy, scikit-learn, tqdm.
- [`graph/build_kg.py`](graph/build_kg.py): builds the knowledge graph from MovieLens inputs and exports PyG-style processed data.
- [`graph/partition.py`](graph/partition.py): partitions users into five shards and writes shard data plus user→shard mapping.

**Artifacts (under `data/processed/`)**

- `graph.pt`, `kg_networkx.gpickle`, supporting tensors (`all_ratings.pt`, `movie_info.pt`, `user_info.pt`, `franchise_data.pt`, `shard_assignments.pt`).
- `shards/shard_0.pt` … `shard_4.pt` (per-shard training data).
- `shard_user_map.json` (user id → shard index).

**Plan alignment**

- Dataset is expected at project-relative paths (e.g. `ml-100k/` or as documented in your environment); the plan’s note on franchise mappings for era-appropriate series applies to demo narratives.

---

## Phase 2 — LightGCN model and training

**Completed**

- [`models/lightgcn.py`](models/lightgcn.py): LightGCN-style model with BPR-oriented training hooks (as used by [`train.py`](train.py)).
- [`train.py`](train.py): trains each shard independently (default 50 epochs, embedding dim 64, three layers, BPR with negative sampling); optional `--shard` for single-shard retrain.
- [`evaluate.py`](evaluate.py): Hit Rate@10, NDCG@10, Precision@10 on held-out interactions per shard.

**Artifacts**

- Trained weights: [`models/shards/shard_0.pt`](models/shards/shard_0.pt) … [`shard_4.pt`](models/shards/shard_4.pt).

---

## Phase 3 — Recommendation and unlearning engine

**Completed**

- [`recommend/inference.py`](recommend/inference.py): loads shard models and serves recommendations (with engine lifecycle used by the API).
- [`recommend/explain.py`](recommend/explain.py): explanations (genres, franchises, similar users, post-unlearn messaging).
- [`recommend/penalize.py`](recommend/penalize.py): score penalties for disliked movies and franchises.
- [`unlearn/edge_deletion.py`](unlearn/edge_deletion.py): removes user–movie edges from persisted graph/shard state.
- [`unlearn/embedding_update.py`](unlearn/embedding_update.py): fast (level 1) path: embedding-style updates and dislike persistence.
- [`unlearn/shard_retrain.py`](unlearn/shard_retrain.py): strong (level 2) path: retrain only the affected shard.

**Persistent state**

- [`backend/state/dislikes.json`](backend/state/dislikes.json): stored dislikes / preferences for the running demo.

---

## Phase 4 — FastAPI backend

**Completed**

- [`backend/main.py`](backend/main.py): FastAPI app, CORS for local frontend, startup preload of the recommendation engine, `/health`.
- [`backend/routers/recommend.py`](backend/routers/recommend.py): recommendations and explanations.
- [`backend/routers/unlearn.py`](backend/routers/unlearn.py): unlearning API.
- [`backend/routers/history.py`](backend/routers/history.py): users, watch history, graph payload for visualization, movie search.

**Deviation from the plan**

- The plan listed a `backend/services/` package (`graph_service`, `model_service`, etc.). The current codebase implements behavior by importing **`recommend/`** and **`unlearn/`** directly from routers and the inference engine, without a separate `services` layer. Functionality is still covered; only the folder layout differs.

---

## Phase 5 — Next.js frontend

**Completed**

- [`frontend/`](frontend/): App Router UI with TypeScript and Tailwind.
- Components: `UserSelector`, `WatchHistory`, `RecommendList`, `UnlearnPanel`, `ExplainCard`, `GraphViz`; [`frontend/lib/api.ts`](frontend/lib/api.ts) wraps backend calls; [`frontend/app/page.tsx`](frontend/app/page.tsx) composes the demo.

**Deviation from the plan**

- Dependencies use **Next.js 16** (and React 19) per `package.json`, not “Next.js 14” as in the plan. Behavior matches the same demo goals (recommendations, unlearn flow, explainability, optional graph view).

---

## Verification (from the implementation plan)

| Check | Notes |
|-------|--------|
| `python graph/build_kg.py` | Should (re)generate `data/processed/graph.pt` and related files. |
| `python graph/partition.py` | Should (re)generate shards and `shard_user_map.json`. |
| `python train.py` | Should refresh `models/shards/shard_*.pt`. |
| `python evaluate.py` | Should print per-shard metrics. |
| Backend curl examples | Valid if the engine loads (trained shards + processed data on disk). |
| Frontend `npm run dev` | Serves the demo against `NEXT_PUBLIC_API_URL` (typically `http://localhost:8000`). |

A dedicated automated test suite (e.g. pytest) is **not** described in the plan as a deliverable; treating “verification” as **manual / script-based** unless you add tests later.

---

## Suggested next steps (optional)

1. Add a minimal `pytest` or smoke script that asserts files exist and `/health` returns 200.
2. Document exact MovieLens path and any symlink (`data/raw` ↔ `ml-100k`) in the README for new clones.
3. If the academic write-up requires it, capture baseline `evaluate.py` numbers in this report or a separate results file.

---

*This report is derived from [`implementation_plan.md`](implementation_plan.md) and the current repository layout and artifacts.*
