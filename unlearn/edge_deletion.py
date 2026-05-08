"""
Edge Deletion — Remove user-movie edges from the graph.

Part of the machine unlearning pipeline:
1. Delete edge from main graph
2. Delete edge from shard graph
3. Persist changes
"""

import json
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SHARDS_DIR = DATA_PROCESSED / "shards"


def delete_edge(user_id: int, movie_id: int, graph_data: dict,
                shard_assignments: dict) -> dict:
    """
    Remove a user→movie edge from the graph and the affected shard.

    Args:
        user_id: original user ID
        movie_id: original movie ID
        graph_data: loaded graph.pt data
        shard_assignments: loaded shard_assignments.pt data

    Returns:
        dict with deletion results
    """
    user_id_map = graph_data["user_id_map"]
    movie_id_map = graph_data["movie_id_map"]

    if user_id not in user_id_map:
        return {"success": False, "error": f"User {user_id} not found"}
    if movie_id not in movie_id_map:
        return {"success": False, "error": f"Movie {movie_id} not found"}

    global_user_idx = user_id_map[user_id]
    global_movie_idx = movie_id_map[movie_id]

    # Find the shard
    user_to_shard = shard_assignments["user_to_shard"]
    shard_id = user_to_shard[global_user_idx]

    # ── Remove from main graph ──
    watched_edge_index = graph_data["watched_edge_index"]
    src = watched_edge_index[0]
    dst = watched_edge_index[1]

    # Find edges matching this user-movie pair
    mask = ~((src == global_user_idx) & (dst == global_movie_idx))
    edges_removed_main = mask.shape[0] - mask.sum().item()

    graph_data["watched_edge_index"] = torch.stack([src[mask], dst[mask]], dim=0)

    # Also remove from ratings tensor if exists
    if "watched_ratings" in graph_data and graph_data["watched_ratings"] is not None:
        graph_data["watched_ratings"] = graph_data["watched_ratings"][mask]

    # Save updated main graph
    torch.save(graph_data, DATA_PROCESSED / "graph.pt")

    # ── Remove from shard graph ──
    shard_data = torch.load(SHARDS_DIR / f"shard_{shard_id}.pt", weights_only=False)

    # Remove from global edge index
    shard_edge_index = shard_data["edge_index"]
    s_src = shard_edge_index[0]
    s_dst = shard_edge_index[1]
    s_mask = ~((s_src == global_user_idx) & (s_dst == global_movie_idx))
    shard_data["edge_index"] = torch.stack([s_src[s_mask], s_dst[s_mask]], dim=0)

    # Remove from local edge index
    local_user_map = shard_data["global_to_local_user"]
    local_movie_map = shard_data["global_to_local_movie"]

    if global_user_idx in local_user_map and global_movie_idx in local_movie_map:
        local_user = local_user_map[global_user_idx]
        local_movie = local_movie_map[global_movie_idx]

        local_edge_index = shard_data["local_edge_index"]
        l_src = local_edge_index[0]
        l_dst = local_edge_index[1]
        l_mask = ~((l_src == local_user) & (l_dst == local_movie))
        shard_data["local_edge_index"] = torch.stack([l_src[l_mask], l_dst[l_mask]], dim=0)

    # Save updated shard
    torch.save(shard_data, SHARDS_DIR / f"shard_{shard_id}.pt")

    return {
        "success": True,
        "shard_id": shard_id,
        "edges_removed_main": edges_removed_main,
        "global_user_idx": global_user_idx,
        "global_movie_idx": global_movie_idx,
    }


def delete_franchise_edges(user_id: int, franchise_name: str,
                           graph_data: dict, shard_assignments: dict,
                           franchise_data: dict) -> dict:
    """
    Remove all edges for a user to movies in a franchise.
    
    Args:
        user_id: original user ID
        franchise_name: name of the franchise
        graph_data: loaded graph.pt data
        shard_assignments: loaded shard_assignments.pt data
        franchise_data: loaded franchise_data.pt
    
    Returns:
        dict with results
    """
    franchise_to_movies = franchise_data.get("franchise_to_movies", {})
    movie_ids = franchise_to_movies.get(franchise_name, [])

    results = []
    for mid in movie_ids:
        result = delete_edge(user_id, mid, graph_data, shard_assignments)
        results.append(result)
        # Reload graph data since it was modified
        graph_data = torch.load(DATA_PROCESSED / "graph.pt", weights_only=False)

    total_removed = sum(r.get("edges_removed_main", 0) for r in results)
    shard_ids = list(set(r.get("shard_id") for r in results if r.get("success")))

    return {
        "success": True,
        "franchise": franchise_name,
        "movies_processed": len(movie_ids),
        "total_edges_removed": total_removed,
        "affected_shards": shard_ids,
    }
