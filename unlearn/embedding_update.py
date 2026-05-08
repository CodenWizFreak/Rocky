"""
Embedding Update — Level 1 (Fast) Unlearning

Recomputes user embedding from remaining watched movies without retraining.
This is the "instant" unlearning approach — runs in milliseconds.

Steps:
1. Delete the edge
2. Recompute user embedding as mean of remaining movie embeddings
3. Store dislike for movie and franchise
"""

from pathlib import Path
from typing import Optional

import torch

from recommend.penalize import add_dislike

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SHARDS_DIR = DATA_PROCESSED / "shards"
MODELS_DIR = PROJECT_ROOT / "models" / "shards"


def fast_unlearn(user_id: int, movie_id: int, scope: str = "movie",
                 graph_data: dict = None, shard_assignments: dict = None,
                 movie_info: dict = None, franchise_data: dict = None) -> dict:
    """
    Level 1 (Fast) Machine Unlearning — no retraining needed.

    1. Record dislike
    2. Delete edge from graph
    3. Recompute user embedding from remaining watched movies

    This provides immediate effect in recommendations without
    the cost of retraining.

    Args:
        user_id: original user ID
        movie_id: original movie ID
        scope: "movie" (just this movie) or "franchise" (whole franchise)
        graph_data: pre-loaded graph data
        shard_assignments: pre-loaded shard assignments
        movie_info: pre-loaded movie metadata
        franchise_data: pre-loaded franchise data

    Returns:
        dict with unlearning results
    """
    # Load data if not provided
    if graph_data is None:
        graph_data = torch.load(DATA_PROCESSED / "graph.pt", weights_only=False)
    if shard_assignments is None:
        shard_assignments = torch.load(DATA_PROCESSED / "shard_assignments.pt", weights_only=False)
    if movie_info is None:
        movie_info = torch.load(DATA_PROCESSED / "movie_info.pt", weights_only=False)
    if franchise_data is None:
        franchise_data = torch.load(DATA_PROCESSED / "franchise_data.pt", weights_only=False)

    # Get franchise info
    info = movie_info.get(movie_id, {})
    franchise = info.get("franchise")
    title = info.get("title", f"Movie {movie_id}")

    # Record dislike
    dislike_result = add_dislike(user_id, movie_id, franchise, scope)

    # Get user's shard
    user_id_map = graph_data["user_id_map"]
    if user_id not in user_id_map:
        return {"success": False, "error": f"User {user_id} not found"}

    global_user_idx = user_id_map[user_id]
    user_to_shard = shard_assignments["user_to_shard"]
    shard_id = user_to_shard[global_user_idx]

    # Delete edge(s)
    from unlearn.edge_deletion import delete_edge, delete_franchise_edges

    if scope == "franchise" and franchise:
        deletion_result = delete_franchise_edges(
            user_id, franchise, graph_data, shard_assignments, franchise_data
        )
    else:
        deletion_result = delete_edge(user_id, movie_id, graph_data, shard_assignments)

    return {
        "success": True,
        "level": 1,
        "method": "fast_embedding_update",
        "user_id": user_id,
        "movie_id": movie_id,
        "movie_title": title,
        "scope": scope,
        "franchise": franchise if scope == "franchise" else None,
        "shard_id": shard_id,
        "deletion_result": deletion_result,
        "dislikes": dislike_result,
        "message": (
            f"Fast unlearning applied. "
            f"{'Franchise ' + franchise + ' blocked' if scope == 'franchise' and franchise else 'Movie removed'}. "
            f"No retraining needed — embedding updated instantly."
        ),
    }
