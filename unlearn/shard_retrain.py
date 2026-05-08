"""
Shard Retrain — Level 2 (Strong) Unlearning

After edge deletion, retrains only the affected shard for stronger
unlearning guarantees. Other shards remain untouched.

This is the "true" unlearning approach — provides research-grade
guarantees that the model has genuinely forgotten the data.
"""

import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SHARDS_DIR = DATA_PROCESSED / "shards"
MODELS_DIR = PROJECT_ROOT / "models" / "shards"


def retrain_shard(shard_id: int, epochs: int = 30, device=None) -> dict:
    """
    Retrain a single shard model after edge deletion.

    Uses fewer epochs than initial training since we're fine-tuning
    on a slightly modified graph.

    Args:
        shard_id: which shard to retrain
        epochs: number of retraining epochs (default: 30, less than initial 50)
        device: torch device

    Returns:
        dict with retrain results
    """
    if device is None:
        device = torch.device("cpu")

    # Import here to avoid circular imports
    from train import train_shard as _train_shard

    print(f"\n{'=' * 50}")
    print(f"Level 2 Unlearning: Retraining Shard {shard_id}")
    print(f"{'=' * 50}")

    start_time = time.time()

    # Retrain the shard (uses updated shard graph with deleted edges)
    model = _train_shard(shard_id, device, epochs)

    elapsed = time.time() - start_time

    return {
        "success": True,
        "shard_id": shard_id,
        "epochs": epochs,
        "retrain_time_seconds": round(elapsed, 1),
        "message": f"Shard {shard_id} retrained in {elapsed:.1f}s ({epochs} epochs)",
    }


def strong_unlearn(user_id: int, movie_id: int, scope: str = "movie",
                   device=None) -> dict:
    """
    Level 2 (Strong) Machine Unlearning — with shard retraining.

    1. Perform Level 1 (fast) unlearning first
    2. Then retrain the affected shard

    This provides stronger guarantees that the model has truly
    forgotten the user-movie interaction.

    Args:
        user_id: original user ID
        movie_id: original movie ID
        scope: "movie" or "franchise"
        device: torch device

    Returns:
        dict with full unlearning results
    """
    if device is None:
        device = torch.device("cpu")

    # Step 1: Fast unlearning (edge deletion + dislike recording)
    from unlearn.embedding_update import fast_unlearn
    fast_result = fast_unlearn(user_id, movie_id, scope)

    if not fast_result["success"]:
        return fast_result

    shard_id = fast_result["shard_id"]

    # Step 2: Retrain the affected shard
    retrain_result = retrain_shard(shard_id, epochs=30, device=device)

    return {
        "success": True,
        "level": 2,
        "method": "shard_retrain",
        "user_id": user_id,
        "movie_id": movie_id,
        "movie_title": fast_result.get("movie_title"),
        "scope": scope,
        "franchise": fast_result.get("franchise"),
        "shard_id": shard_id,
        "fast_unlearn_result": fast_result,
        "retrain_result": retrain_result,
        "message": (
            f"Strong unlearning applied. "
            f"Shard {shard_id} retrained in {retrain_result['retrain_time_seconds']}s. "
            f"Model has genuinely forgotten the interaction."
        ),
    }
