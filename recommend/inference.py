"""
Recommendation Inference

Aggregates scores from shard models to produce final recommendations.
Handles both single-shard and multi-shard aggregation.
"""

import json
from pathlib import Path
from typing import Optional

import torch

from models.lightgcn import LightGCN
from recommend.penalize import penalize_dislikes, load_dislikes

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SHARDS_DIR = DATA_PROCESSED / "shards"
MODELS_DIR = PROJECT_ROOT / "models" / "shards"


class RecommendationEngine:
    """
    Aggregated recommendation engine across all shards.
    
    For a given user:
    1. Find which shard contains the user
    2. Get LightGCN scores from that shard model
    3. Map local item IDs back to global movie IDs
    4. Apply dislike penalties
    5. Return top-K with metadata
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        self.models = {}
        self.shard_data = {}
        self.graph_data = None
        self.movie_info = None
        self.user_info = None
        self.shard_assignments = None
        self.all_ratings = None
        self._loaded = False

    def load(self):
        """Load all models, graph data, and metadata."""
        if self._loaded:
            return

        print("Loading recommendation engine...")

        # Load graph data
        self.graph_data = torch.load(DATA_PROCESSED / "graph.pt", weights_only=False)
        self.movie_info = torch.load(DATA_PROCESSED / "movie_info.pt", weights_only=False)
        self.user_info = torch.load(DATA_PROCESSED / "user_info.pt", weights_only=False)
        self.all_ratings = torch.load(DATA_PROCESSED / "all_ratings.pt", weights_only=False)
        self.shard_assignments = torch.load(DATA_PROCESSED / "shard_assignments.pt", weights_only=False)
        self.franchise_data = torch.load(DATA_PROCESSED / "franchise_data.pt", weights_only=False)

        n_shards = self.shard_assignments["n_shards"]

        # Load all shard models
        for shard_id in range(n_shards):
            model_path = MODELS_DIR / f"shard_{shard_id}.pt"
            shard_path = SHARDS_DIR / f"shard_{shard_id}.pt"

            if not model_path.exists() or not shard_path.exists():
                print(f"  Warning: Shard {shard_id} not found, skipping")
                continue

            model_state = torch.load(model_path, weights_only=False)
            shard_data = torch.load(shard_path, weights_only=False)

            model = LightGCN(
                model_state["num_users"],
                model_state["num_movies"],
                model_state["emb_dim"],
                model_state["n_layers"],
            ).to(self.device)
            model.load_state_dict(model_state["model_state_dict"])
            model.eval()

            self.models[shard_id] = model
            self.shard_data[shard_id] = shard_data

        print(f"  Loaded {len(self.models)} shard models")
        self._loaded = True

    def reload_shard(self, shard_id: int):
        """Reload a specific shard model (after retraining)."""
        model_path = MODELS_DIR / f"shard_{shard_id}.pt"
        shard_path = SHARDS_DIR / f"shard_{shard_id}.pt"

        model_state = torch.load(model_path, weights_only=False)
        shard_data = torch.load(shard_path, weights_only=False)

        model = LightGCN(
            model_state["num_users"],
            model_state["num_movies"],
            model_state["emb_dim"],
            model_state["n_layers"],
        ).to(self.device)
        model.load_state_dict(model_state["model_state_dict"])
        model.eval()

        self.models[shard_id] = model
        self.shard_data[shard_id] = shard_data

    def get_user_shard(self, user_id: int) -> int:
        """Get which shard a user belongs to (using original user_id)."""
        user_id_map = self.graph_data["user_id_map"]
        if user_id not in user_id_map:
            raise ValueError(f"User {user_id} not found")

        global_idx = user_id_map[user_id]
        user_to_shard = self.shard_assignments["user_to_shard"]
        return user_to_shard[global_idx]

    def get_watch_history(self, user_id: int) -> list:
        """Get a user's watch history with movie details."""
        if user_id not in self.all_ratings:
            return []

        history = []
        for mid, rating in self.all_ratings[user_id].items():
            info = self.movie_info.get(mid, {})
            history.append({
                "movie_id": mid,
                "title": info.get("title", f"Movie {mid}"),
                "rating": rating,
                "genres": info.get("genres", []),
                "franchise": info.get("franchise"),
            })

        # Sort by rating descending
        history.sort(key=lambda x: x["rating"], reverse=True)
        return history

    def recommend(self, user_id: int, top_k: int = 10) -> dict:
        """
        Get top-K recommendations for a user.

        Returns dict with recommendations lists and metadata.
        """
        self.load()

        shard_id = self.get_user_shard(user_id)
        model = self.models.get(shard_id)
        shard_data = self.shard_data.get(shard_id)

        if model is None:
            return {"error": f"Shard {shard_id} model not loaded"}

        # Get user's local index in the shard
        global_user_idx = self.graph_data["user_id_map"][user_id]
        local_user_map = shard_data["global_to_local_user"]

        if global_user_idx not in local_user_map:
            return {"error": f"User {user_id} not found in shard {shard_id}"}

        local_user_idx = local_user_map[global_user_idx]
        local_edge_index = shard_data["local_edge_index"].to(self.device)

        # Get watched movies to exclude
        watched_original = set(self.all_ratings.get(user_id, {}).keys())
        watched_local = set()
        movie_id_map = self.graph_data["movie_id_map"]
        local_movie_map = shard_data.get("global_to_local_movie", {})
        for mid in watched_original:
            global_mid = movie_id_map.get(mid)
            if global_mid is not None and global_mid in local_movie_map:
                watched_local.add(local_movie_map[global_mid])

        # Get recommendations from model
        with torch.no_grad():
            recs = model.recommend(local_edge_index, local_user_idx,
                                   top_k=top_k + 20, exclude=watched_local)

        # Map local item IDs back to original movie IDs
        local_to_global_movie = shard_data["local_to_global_movie"]
        reverse_movie_map = self.graph_data["reverse_movie_map"]

        recommendations = []
        for local_idx, score in recs:
            global_idx = local_to_global_movie.get(local_idx)
            if global_idx is None:
                continue
            original_mid = reverse_movie_map.get(global_idx)
            if original_mid is None:
                continue

            info = self.movie_info.get(original_mid, {})
            recommendations.append({
                "movie_id": original_mid,
                "title": info.get("title", f"Movie {original_mid}"),
                "genres": info.get("genres", []),
                "franchise": info.get("franchise"),
                "score": round(score, 4),
                "reason": None,  # filled by explain module
            })

        # Apply dislike penalties
        dislikes = load_dislikes()
        user_dislikes = dislikes.get(str(user_id), {})

        if user_dislikes:
            recommendations = penalize_dislikes(
                user_id, recommendations,
                user_dislikes.get("movies", []),
                user_dislikes.get("franchises", []),
                self.franchise_data.get("franchise_to_movies", {}),
            )

        # Final top-K after penalties
        recommendations = recommendations[:top_k]

        # Get disliked franchises for response
        excluded_franchises = user_dislikes.get("franchises", [])

        return {
            "user_id": user_id,
            "shard_id": shard_id,
            "recommendations": recommendations,
            "unlearn_applied": bool(user_dislikes),
            "excluded_franchises": excluded_franchises,
        }


# Module-level singleton
_engine = None

def get_engine(device=None) -> RecommendationEngine:
    """Get or create the global recommendation engine."""
    global _engine
    if _engine is None:
        _engine = RecommendationEngine(device)
    return _engine
