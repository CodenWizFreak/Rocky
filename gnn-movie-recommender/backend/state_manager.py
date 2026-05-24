"""
state_manager.py — Read / write user_state.json
"""

import json
import os
from typing import Any

DEFAULT_GENRES = [
    "Action", "Adventure", "Animation", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "History",
    "Horror", "Music", "Mystery", "Romance", "Science Fiction",
    "Sci-Fi", "Thriller", "TV Movie", "War", "Western",
]

LIKE_BOOST   = 0.15
LIKE_CAP     = 1.5
DISLIKE_DECAY = 0.70


def _default_state() -> dict:
    return {
        "liked_movies":        [],
        "disliked_movies":     [],
        "liked_genres":        [],
        "disliked_genres":     [],
        "genre_weights":       {g: 1.0 for g in DEFAULT_GENRES},
        "last_recommendations": [],
        "preference_events":   [],
    }


def initialize_state() -> dict:
    return _default_state()


def load_state(path: str) -> dict:
    """Load state from disk; create defaults if file is missing or empty."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                migrated = False
                # Backfill any missing genre keys
                defaults = _default_state()
                for k, v in defaults.items():
                    if k not in data:
                        data[k] = v
                        migrated = True
                for genre in DEFAULT_GENRES:
                    if genre not in data["genre_weights"]:
                        data["genre_weights"][genre] = 1.0
                        migrated = True
                _dedupe_list(data, "liked_movies")
                _dedupe_list(data, "disliked_movies")
                _dedupe_list(data, "liked_genres")
                _dedupe_list(data, "disliked_genres")
                if migrated:
                    save_state(path, data)
                return data
        except (json.JSONDecodeError, ValueError):
            pass
    state = _default_state()
    save_state(path, state)
    return state


def save_state(path: str, state: dict) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, path)


def _dedupe_list(state: dict, key: str) -> None:
    seen = set()
    deduped = []
    for item in state.get(key, []):
        item_clean = str(item).strip()
        if not item_clean:
            continue
        marker = item_clean.lower()
        if marker not in seen:
            deduped.append(item_clean)
            seen.add(marker)
    state[key] = deduped


def remember_event(state: dict, event: dict) -> None:
    """Persist compact audit trail for state mutations."""
    events = state.setdefault("preference_events", [])
    events.append(event)
    del events[:-50]


def update_liked(state: dict, title: str, genres: list[str]) -> None:
    """Append to liked_movies and boost genre weights."""
    title_clean = title.strip()
    if title_clean not in state["liked_movies"]:
        state["liked_movies"].append(title_clean)

    for genre in genres:
        boost_genre(state, genre)


def update_disliked(state: dict, title: str, genres: list[str]) -> None:
    """Append to disliked_movies and decay genre weights."""
    title_clean = title.strip()
    if title_clean not in state["disliked_movies"]:
        state["disliked_movies"].append(title_clean)

    for genre in genres:
        decay_genre(state, genre)


def boost_genre(state: dict, genre: str) -> None:
    genre = genre.strip()
    if not genre:
        return
    state["genre_weights"].setdefault(genre, 1.0)
    current = state["genre_weights"].get(genre, 1.0)
    state["genre_weights"][genre] = round(min(current + LIKE_BOOST, LIKE_CAP), 4)


def decay_genre(state: dict, genre: str) -> None:
    genre = genre.strip()
    if not genre:
        return
    state["genre_weights"].setdefault(genre, 1.0)
    current = state["genre_weights"].get(genre, 1.0)
    state["genre_weights"][genre] = round(current * DISLIKE_DECAY, 4)


def update_liked_genre(state: dict, genre: str) -> None:
    genre = genre.strip()
    if not genre:
        return
    if genre not in state["liked_genres"]:
        state["liked_genres"].append(genre)
    if genre in state["disliked_genres"]:
        state["disliked_genres"].remove(genre)
    boost_genre(state, genre)


def update_disliked_genre(state: dict, genre: str) -> None:
    genre = genre.strip()
    if not genre:
        return
    if genre not in state["disliked_genres"]:
        state["disliked_genres"].append(genre)
    if genre in state["liked_genres"]:
        state["liked_genres"].remove(genre)
    decay_genre(state, genre)
