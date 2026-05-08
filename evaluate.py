"""
Evaluation Script — Compute recommendation metrics

Metrics:
- Hit Rate @ K: was the true held-out movie in top K?
- NDCG @ K: ranking quality (Normalized Discounted Cumulative Gain)
- Precision @ K: fraction of relevant items in top K

Usage:
    python evaluate.py
"""

import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.lightgcn import LightGCN
from train import load_shard_model

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SHARDS_DIR = DATA_PROCESSED / "shards"

TOP_K = 10


def compute_metrics(model, shard_data, device, top_k=TOP_K):
    """Compute Hit Rate, NDCG, and Precision for a shard."""
    model.eval()
    local_edge_index = shard_data["local_edge_index"].to(device)

    # Build user → item interactions
    src = local_edge_index[0].cpu().numpy()
    dst = local_edge_index[1].cpu().numpy()

    user_items = {}
    for u, i in zip(src, dst):
        user_items.setdefault(int(u), []).append(int(i))

    hits = 0
    ndcg_sum = 0.0
    precision_sum = 0.0
    total_users = 0

    with torch.no_grad():
        for user_idx, items in user_items.items():
            if len(items) < 2:
                continue  # Need at least 1 train + 1 test

            # Hold out last item as test
            train_items = set(items[:-1])
            test_item = items[-1]

            # Get scores
            scores = model.get_all_scores(local_edge_index, user_idx)

            # Exclude training items
            for train_item in train_items:
                scores[train_item] = float('-inf')

            # Get top-K
            _, top_indices = torch.topk(scores, min(top_k, len(scores)))
            top_list = top_indices.cpu().tolist()

            # Hit Rate
            if test_item in top_list:
                hits += 1
                # NDCG
                rank = top_list.index(test_item) + 1
                ndcg_sum += 1.0 / math.log2(rank + 1)
                # Precision
                precision_sum += 1.0 / top_k
            
            total_users += 1

    if total_users == 0:
        return {"hit_rate": 0.0, "ndcg": 0.0, "precision": 0.0, "users_evaluated": 0}

    return {
        "hit_rate": hits / total_users,
        "ndcg": ndcg_sum / total_users,
        "precision": precision_sum / total_users,
        "users_evaluated": total_users,
    }


def evaluate_all_shards(device=None):
    """Evaluate all shard models and print aggregate metrics."""
    if device is None:
        device = torch.device("cpu")

    shard_assignments = torch.load(DATA_PROCESSED / "shard_assignments.pt", weights_only=False)
    n_shards = shard_assignments["n_shards"]

    print("=" * 60)
    print(f"Evaluating {n_shards} shard models (Top-{TOP_K})")
    print("=" * 60)

    all_metrics = []
    total_users = 0

    for shard_id in range(n_shards):
        model, shard_data = load_shard_model(shard_id, device)
        metrics = compute_metrics(model, shard_data, device)
        all_metrics.append(metrics)
        total_users += metrics["users_evaluated"]

        print(f"\nShard {shard_id}:")
        print(f"  Hit Rate @{TOP_K}:  {metrics['hit_rate']:.4f}")
        print(f"  NDCG @{TOP_K}:      {metrics['ndcg']:.4f}")
        print(f"  Precision @{TOP_K}: {metrics['precision']:.4f}")
        print(f"  Users evaluated:   {metrics['users_evaluated']}")

    # Weighted average across shards
    if total_users > 0:
        avg_hr = sum(m["hit_rate"] * m["users_evaluated"] for m in all_metrics) / total_users
        avg_ndcg = sum(m["ndcg"] * m["users_evaluated"] for m in all_metrics) / total_users
        avg_prec = sum(m["precision"] * m["users_evaluated"] for m in all_metrics) / total_users

        print(f"\n{'=' * 60}")
        print(f"Aggregate (weighted by shard size):")
        print(f"  Hit Rate @{TOP_K}:  {avg_hr:.4f}")
        print(f"  NDCG @{TOP_K}:      {avg_ndcg:.4f}")
        print(f"  Precision @{TOP_K}: {avg_prec:.4f}")
        print(f"  Total users:       {total_users}")
        print(f"{'=' * 60}")

    return all_metrics


if __name__ == "__main__":
    evaluate_all_shards()
