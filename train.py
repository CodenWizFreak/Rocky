"""
Training Script — Train one LightGCN model per shard

Uses BPR (Bayesian Personalised Ranking) loss with negative sampling.
Trains each shard independently and saves model weights.

Usage:
    python train.py              # train all shards
    python train.py --shard 2    # retrain only shard 2
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.lightgcn import LightGCN, bpr_loss

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SHARDS_DIR = DATA_PROCESSED / "shards"
MODELS_DIR = PROJECT_ROOT / "models" / "shards"

# ──────────────────────────────────────────────
# Hyperparameters (M4 Air safe)
# ──────────────────────────────────────────────
EMB_DIM = 64
N_LAYERS = 3
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 1024
REG_WEIGHT = 1e-4
SEED = 42


def set_seed(seed: int):
    """Set reproducibility seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        try:
            # Test if MPS actually works with basic ops
            t = torch.zeros(1, device="mps")
            _ = t + 1
            return torch.device("mps")
        except Exception:
            pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sample_negative(num_items: int, positive_set: set) -> int:
    """Sample a negative item not in the positive set."""
    while True:
        neg = random.randint(0, num_items - 1)
        if neg not in positive_set:
            return neg


def train_shard(shard_id: int, device: torch.device, epochs: int = EPOCHS):
    """Train LightGCN on a single shard."""

    print(f"\n{'─' * 50}")
    print(f"Training Shard {shard_id}")
    print(f"{'─' * 50}")

    # Load shard data
    shard_data = torch.load(SHARDS_DIR / f"shard_{shard_id}.pt", weights_only=False)
    num_users = shard_data["num_users"]
    num_movies = shard_data["num_movies"]
    local_edge_index = shard_data["local_edge_index"].to(device)

    print(f"  Users: {num_users}, Movies: {num_movies}, Edges: {local_edge_index.shape[1]:,}")

    # Build user → positive items mapping
    user_pos_items = {}
    src = local_edge_index[0].cpu().numpy()
    dst = local_edge_index[1].cpu().numpy()
    for u, i in zip(src, dst):
        user_pos_items.setdefault(int(u), set()).add(int(i))

    # Initialize model
    model = LightGCN(num_users, num_movies, EMB_DIM, N_LAYERS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Build training pairs
    all_edges = list(zip(src.tolist(), dst.tolist()))

    best_loss = float('inf')
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        random.shuffle(all_edges)

        total_loss = 0.0
        num_batches = 0

        for batch_start in range(0, len(all_edges), BATCH_SIZE):
            batch_edges = all_edges[batch_start:batch_start + BATCH_SIZE]

            users = []
            pos_items = []
            neg_items = []

            for u, pos_i in batch_edges:
                neg_i = sample_negative(num_movies, user_pos_items.get(u, set()))
                users.append(u)
                pos_items.append(pos_i)
                neg_items.append(neg_i)

            users_t = torch.tensor(users, dtype=torch.long, device=device)
            pos_t = torch.tensor(pos_items, dtype=torch.long, device=device)
            neg_t = torch.tensor(neg_items, dtype=torch.long, device=device)

            user_emb, pos_emb, neg_emb, u0, p0, n0 = model(local_edge_index, users_t, pos_t, neg_t)
            loss = bpr_loss(user_emb, pos_emb, neg_emb, u0, p0, n0, REG_WEIGHT)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # Save model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODELS_DIR / f"shard_{shard_id}.pt"

    model_state = {
        "model_state_dict": model.cpu().state_dict(),
        "num_users": num_users,
        "num_movies": num_movies,
        "emb_dim": EMB_DIM,
        "n_layers": N_LAYERS,
        "shard_id": shard_id,
        "best_loss": best_loss,
        "epochs_trained": epochs,
    }
    torch.save(model_state, save_path)

    elapsed = time.time() - start_time
    print(f"  ✓ Shard {shard_id} trained in {elapsed:.1f}s | Best loss: {best_loss:.4f}")
    print(f"  ✓ Saved to {save_path}")

    return model


def load_shard_model(shard_id: int, device: torch.device = None) -> tuple:
    """Load a trained shard model and its data."""
    if device is None:
        device = torch.device("cpu")

    model_state = torch.load(MODELS_DIR / f"shard_{shard_id}.pt", weights_only=False)
    shard_data = torch.load(SHARDS_DIR / f"shard_{shard_id}.pt", weights_only=False)

    model = LightGCN(
        model_state["num_users"],
        model_state["num_movies"],
        model_state["emb_dim"],
        model_state["n_layers"],
    ).to(device)
    model.load_state_dict(model_state["model_state_dict"])
    model.eval()

    return model, shard_data


def main():
    parser = argparse.ArgumentParser(description="Train LightGCN shard models")
    parser.add_argument("--shard", type=int, default=None,
                        help="Train only this shard (0-4). Default: train all.")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu, mps, cuda. Default: auto-detect.")
    args = parser.parse_args()

    set_seed(SEED)

    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device()

    print(f"Device: {device}")

    # Load shard assignments to know how many shards
    shard_assignments = torch.load(DATA_PROCESSED / "shard_assignments.pt", weights_only=False)
    n_shards = shard_assignments["n_shards"]

    start_total = time.time()

    if args.shard is not None:
        assert 0 <= args.shard < n_shards, f"Shard must be 0-{n_shards - 1}"
        train_shard(args.shard, device, args.epochs)
    else:
        print(f"\nTraining all {n_shards} shards...")
        for shard_id in range(n_shards):
            train_shard(shard_id, device, args.epochs)

    total_time = time.time() - start_total
    print(f"\n{'=' * 50}")
    print(f"Total training time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
