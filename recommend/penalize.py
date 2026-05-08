"""
Dislike Penalty System

Penalizes scores of disliked movies and franchises to ensure
they don't appear in recommendations.
"""

import json
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = PROJECT_ROOT / "backend" / "state"

DISLIKE_PENALTY = 100.0


def load_dislikes() -> dict:
    """
    Load persisted user dislikes from state file.
    
    Format:
    {
        "user_id": {
            "movies": [movie_id, ...],
            "franchises": ["Franchise Name", ...]
        }
    }
    """
    dislikes_path = STATE_DIR / "dislikes.json"
    if dislikes_path.exists():
        with open(dislikes_path, "r") as f:
            return json.load(f)
    return {}


def save_dislikes(dislikes: dict):
    """Save user dislikes to state file."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    dislikes_path = STATE_DIR / "dislikes.json"
    with open(dislikes_path, "w") as f:
        json.dump(dislikes, f, indent=2)


def add_dislike(user_id: int, movie_id: int, franchise: Optional[str] = None,
                scope: str = "movie"):
    """
    Record a user dislike.
    
    Args:
        user_id: the user
        movie_id: the movie to dislike
        franchise: franchise name (if applicable)
        scope: "movie" or "franchise"
    """
    dislikes = load_dislikes()
    uid_str = str(user_id)

    if uid_str not in dislikes:
        dislikes[uid_str] = {"movies": [], "franchises": []}

    if movie_id not in dislikes[uid_str]["movies"]:
        dislikes[uid_str]["movies"].append(movie_id)

    if scope == "franchise" and franchise and franchise not in dislikes[uid_str]["franchises"]:
        dislikes[uid_str]["franchises"].append(franchise)

    save_dislikes(dislikes)
    return dislikes[uid_str]


def remove_dislike(user_id: int, movie_id: Optional[int] = None,
                   franchise: Optional[str] = None):
    """Remove a dislike (for undo functionality)."""
    dislikes = load_dislikes()
    uid_str = str(user_id)

    if uid_str not in dislikes:
        return

    if movie_id and movie_id in dislikes[uid_str]["movies"]:
        dislikes[uid_str]["movies"].remove(movie_id)

    if franchise and franchise in dislikes[uid_str]["franchises"]:
        dislikes[uid_str]["franchises"].remove(franchise)

    save_dislikes(dislikes)


def penalize_dislikes(user_id: int, recommendations: list,
                      disliked_movies: list, disliked_franchises: list,
                      franchise_to_movies: dict) -> list:
    """
    Apply penalties to recommendations based on user dislikes.
    
    Movies that are directly disliked or belong to disliked franchises
    get removed from recommendations entirely.
    
    Args:
        user_id: the user
        recommendations: list of recommendation dicts
        disliked_movies: list of disliked movie IDs
        disliked_franchises: list of disliked franchise names
        franchise_to_movies: mapping from franchise name to movie IDs
    
    Returns:
        Filtered recommendations list
    """
    # Build set of all movies to exclude
    excluded_movie_ids = set(disliked_movies)

    for fname in disliked_franchises:
        franchise_movies = franchise_to_movies.get(fname, [])
        excluded_movie_ids.update(franchise_movies)

    # Filter out excluded movies
    filtered = []
    for rec in recommendations:
        mid = rec["movie_id"]
        franchise = rec.get("franchise")

        if mid in excluded_movie_ids:
            continue
        if franchise and franchise in disliked_franchises:
            continue

        filtered.append(rec)

    return filtered
