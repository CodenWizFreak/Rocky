# KG + GNN Movie Recommender with Machine Unlearning

A research-oriented demo that recommends movies using a **knowledge graph** and **sharded LightGCN** models trained on **MovieLens 100K**, with **machine unlearning**: users can remove a watched movie (and optionally penalize a whole franchise) so recommendations and explanations update accordingly.

**Stack:** Python 3.11+, [uv](https://github.com/astral-sh/uv), PyTorch / PyTorch Geometric, NetworkX · **FastAPI** backend · **Next.js** (App Router) frontend.

For a line-by-line comparison of what was planned versus what exists in the repo, see [`REPORT.md`](REPORT.md). The original build specification lives in [`implementation_plan.md`](implementation_plan.md). High-level architecture notes are in [`CLAUDE.md`](CLAUDE.md).

---

## What this project does

1. **Build a KG** from ratings (implicit positive edges), genres, and franchise groupings, then export a heterogeneous graph for training.
2. **Split users into five shards** so each LightGCN trains on a subgraph; at inference time the user’s shard drives scores, with aggregation and penalties as implemented in `recommend/`.
3. **Unlearn** via API: delete a user→movie edge, persist dislikes, optionally recompute embeddings quickly or retrain a single shard (`unlearn/`).

---

## Repository layout

| Path | Role |
|------|------|
| `graph/` | `build_kg.py`, `partition.py` — data → graph and shards |
| `data/processed/` | `graph.pt`, `shards/`, `shard_user_map.json`, etc. |
| `models/lightgcn.py` | LightGCN model and BPR-style training helpers |
| `models/shards/` | Trained weights `shard_0.pt` … `shard_4.pt` |
| `train.py`, `evaluate.py` | Train all or one shard; evaluation metrics |
| `recommend/` | Inference, explanations, dislike penalties |
| `unlearn/` | Edge removal, fast embedding update, shard retrain |
| `backend/` | FastAPI app and routers |
| `frontend/` | Next.js demo UI |

---

## Prerequisites

- **Python** ≥ 3.11 and **uv** (recommended).
- **Node.js** and npm (for the frontend).
- **MovieLens 100K** unpacked so paths in `graph/build_kg.py` resolve (often `ml-100k/` at the project root or via `data/raw` — adjust to match your checkout).

---

## Python environment

From the project root:

```bash
uv sync
```

If you need PyG wheels that match your Torch build, follow the hints in [`pyproject.toml`](pyproject.toml) (`find-links` for PyG) or the PyG install docs for your platform.

---

## Data pipeline and training

```bash
# 1) Build the knowledge graph and processed tensors
uv run python graph/build_kg.py

# 2) Partition into five shards
uv run python graph/partition.py

# 3) Train all shard models (or: uv run python train.py --shard N)
uv run python train.py

# 4) Optional: evaluation metrics
uv run python evaluate.py
```

---

## Run the API

From the project root (so imports like `recommend` and `backend` resolve):

```bash
uv run uvicorn backend.main:app --reload --port 8000
```

Try `GET http://localhost:8000/health`. Other routes are documented in [`backend/main.py`](backend/main.py) and the routers under `backend/routers/`.

---

## Run the frontend

```bash
cd frontend
npm install
npm run dev
```

Set `NEXT_PUBLIC_API_URL` (e.g. in `frontend/.env.local`) to your API base URL, typically `http://localhost:8000`.

---

## Configuration and state

- **Dislikes / unlearn preferences:** persisted under `backend/state/` (e.g. `dislikes.json`).
- **Device:** training defaults favor CPU with optional MPS where supported; see `train.py` for device selection.

---

## License and citations

Use this project in line with your institution’s rules for coursework and research. If you publish related work, cite LightGCN (He et al., 2020) and relevant graph unlearning / recommender unlearning literature; pointers appear in `CLAUDE.md`.

---

## Documentation index

| File | Purpose |
|------|---------|
| [`README.md`](README.md) | This overview and run instructions |
| [`REPORT.md`](REPORT.md) | What is implemented vs the plan |
| [`implementation_plan.md`](implementation_plan.md) | Phased specification |
| [`CLAUDE.md`](CLAUDE.md) | Architecture and design context for contributors |
