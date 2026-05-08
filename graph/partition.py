"""
Shard Partitioner

Splits users into N_SHARDS groups and creates per-shard subgraphs.
Each shard contains all edges for its assigned users.

Outputs:
- data/processed/shards/shard_0.pt ... shard_4.pt
- data/processed/shard_user_map.json
"""

import json
import random
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SHARDS_DIR = DATA_PROCESSED / "shards"

N_SHARDS = 5
SEED = 42


def partition_graph():
    """Partition users into shards and create per-shard subgraphs."""

    print("=" * 60)
    print(f"Partitioning into {N_SHARDS} shards")
    print("=" * 60)

    # Load processed graph
    print("\n[1/4] Loading processed graph...")
    graph_data = torch.load(DATA_PROCESSED / "graph.pt", weights_only=False)

    num_users = graph_data["num_users"]
    num_movies = graph_data["num_movies"]
    watched_edge_index = graph_data["watched_edge_index"]  # [2, E]
    user_id_map = graph_data["user_id_map"]           # original → 0-indexed
    reverse_user_map = graph_data["reverse_user_map"]  # 0-indexed → original

    print(f"  Users: {num_users}, Total watched edges: {watched_edge_index.shape[1]:,}")

    # Shuffle and split user indices
    print(f"\n[2/4] Shuffling users (seed={SEED}) and splitting into {N_SHARDS} shards...")
    all_user_indices = list(range(num_users))
    random.seed(SEED)
    random.shuffle(all_user_indices)

    shard_size = num_users // N_SHARDS
    shard_users = {}
    for i in range(N_SHARDS):
        start = i * shard_size
        if i == N_SHARDS - 1:
            # Last shard gets remainder
            shard_users[i] = all_user_indices[start:]
        else:
            shard_users[i] = all_user_indices[start:start + shard_size]

    # Build user → shard mapping
    user_to_shard = {}
    for shard_id, users in shard_users.items():
        for u in users:
            user_to_shard[u] = shard_id

    # Verify all users assigned
    assert len(user_to_shard) == num_users, f"Not all users assigned: {len(user_to_shard)} vs {num_users}"

    print("  Shard sizes:")
    for sid, users in shard_users.items():
        print(f"    Shard {sid}: {len(users)} users")

    # Build per-shard edge indices
    print(f"\n[3/4] Extracting per-shard edges...")
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)

    src_nodes = watched_edge_index[0].numpy()
    dst_nodes = watched_edge_index[1].numpy()

    for shard_id in range(N_SHARDS):
        shard_user_set = set(shard_users[shard_id])

        # Find edges where the user (src) belongs to this shard
        mask = np.array([s in shard_user_set for s in src_nodes])
        shard_src = src_nodes[mask]
        shard_dst = dst_nodes[mask]

        shard_edge_index = torch.tensor(
            np.stack([shard_src, shard_dst]),
            dtype=torch.long
        )

        # Create local user mapping for this shard
        # Map global 0-indexed user IDs → local shard IDs (0..shard_size-1)
        sorted_shard_users = sorted(shard_users[shard_id])
        global_to_local_user = {g: l for l, g in enumerate(sorted_shard_users)}

        # Get all unique movies in this shard
        unique_movies = sorted(set(shard_dst.tolist()))
        global_to_local_movie = {g: l for l, g in enumerate(unique_movies)}

        # Create local edge index
        local_src = [global_to_local_user[s] for s in shard_src]
        local_dst = [global_to_local_movie[d] for d in shard_dst]
        local_edge_index = torch.tensor([local_src, local_dst], dtype=torch.long)

        shard_data = {
            "shard_id": shard_id,
            "num_users": len(sorted_shard_users),
            "num_movies": len(unique_movies),
            "num_movies_global": num_movies,
            "edge_index": shard_edge_index,           # global IDs
            "local_edge_index": local_edge_index,     # local IDs
            "global_user_ids": sorted_shard_users,    # global 0-indexed user IDs
            "global_movie_ids": unique_movies,         # global 0-indexed movie IDs
            "global_to_local_user": global_to_local_user,
            "global_to_local_movie": global_to_local_movie,
            "local_to_global_user": {v: k for k, v in global_to_local_user.items()},
            "local_to_global_movie": {v: k for k, v in global_to_local_movie.items()},
        }

        torch.save(shard_data, SHARDS_DIR / f"shard_{shard_id}.pt")
        print(f"    Shard {shard_id}: {len(sorted_shard_users)} users, "
              f"{len(unique_movies)} movies, {shard_edge_index.shape[1]:,} edges")

    # Save shard-user map (using original user IDs)
    print(f"\n[4/4] Saving shard user map...")
    shard_user_map_original = {}
    for global_idx, shard_id in user_to_shard.items():
        original_uid = reverse_user_map[global_idx]
        shard_user_map_original[str(original_uid)] = shard_id

    with open(DATA_PROCESSED / "shard_user_map.json", "w") as f:
        json.dump(shard_user_map_original, f, indent=2)

    # Also save the internal mapping
    torch.save({
        "user_to_shard": user_to_shard,             # 0-indexed → shard_id
        "shard_users": shard_users,                  # shard_id → [0-indexed user ids]
        "n_shards": N_SHARDS,
    }, DATA_PROCESSED / "shard_assignments.pt")

    print(f"\n{'=' * 60}")
    print("Partitioning complete!")
    print(f"  Shard files: {SHARDS_DIR}")
    print(f"  User map:    {DATA_PROCESSED / 'shard_user_map.json'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    partition_graph()
