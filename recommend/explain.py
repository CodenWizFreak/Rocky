"""
Explainability Module

Generates human-readable explanations for why a movie was recommended
or why it was excluded from recommendations.
"""

from pathlib import Path
from typing import Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def explain_recommendation(user_id: int, movie_id: int,
                           all_ratings: dict, movie_info: dict,
                           dislikes: dict = None) -> str:
    """
    Generate an explanation for why a movie was recommended.

    Considers:
    - Genre overlap with watch history
    - Franchise connections
    - Collaborative signals (similar taste)

    Args:
        user_id: the user
        movie_id: the recommended movie
        all_ratings: all user ratings {uid: {mid: rating}}
        movie_info: movie metadata {mid: {title, genres, franchise, ...}}
        dislikes: user dislike data

    Returns:
        Human-readable explanation string
    """
    user_ratings = all_ratings.get(user_id, {})
    target_info = movie_info.get(movie_id, {})
    target_genres = set(target_info.get("genres", []))
    target_franchise = target_info.get("franchise")
    target_title = target_info.get("title", f"Movie {movie_id}")

    reasons = []

    # 1. Genre overlap
    watched_genres = set()
    highly_rated_genres = set()
    for mid, rating in user_ratings.items():
        info = movie_info.get(mid, {})
        genres = info.get("genres", [])
        watched_genres.update(genres)
        if rating >= 4.0:
            highly_rated_genres.update(genres)

    genre_overlap = target_genres & highly_rated_genres
    if genre_overlap:
        genre_str = ", ".join(sorted(genre_overlap))
        reasons.append(f"Matches your preferred genres: {genre_str}")
    elif target_genres & watched_genres:
        genre_str = ", ".join(sorted(target_genres & watched_genres))
        reasons.append(f"Similar genre to movies you've watched: {genre_str}")

    # 2. Franchise connection
    if target_franchise:
        franchise_movies_watched = [
            movie_info.get(mid, {}).get("title", f"Movie {mid}")
            for mid, rating in user_ratings.items()
            if movie_info.get(mid, {}).get("franchise") == target_franchise
            and rating >= 3.5
        ]
        if franchise_movies_watched:
            reasons.append(
                f"Part of the {target_franchise} series — "
                f"you enjoyed {franchise_movies_watched[0]}"
            )

    # 3. High rated similar movies
    similar_titles = []
    for mid, rating in user_ratings.items():
        if rating >= 4.5:
            info = movie_info.get(mid, {})
            mid_genres = set(info.get("genres", []))
            if len(target_genres & mid_genres) >= 2:
                similar_titles.append(info.get("title", f"Movie {mid}"))
    if similar_titles and not reasons:
        reasons.append(
            f"Similar to highly-rated movies in your history like {similar_titles[0]}"
        )

    # 4. Collaborative filtering signal (generic)
    if not reasons:
        reasons.append("Users with similar taste also enjoyed this movie")

    explanation = f"Recommended because: " + " • ".join(reasons)
    return explanation


def explain_exclusion(user_id: int, movie_id: int,
                      movie_info: dict, dislikes: dict) -> Optional[str]:
    """
    Explain why a movie was excluded from recommendations.
    
    Args:
        user_id: the user
        movie_id: the excluded movie
        movie_info: movie metadata
        dislikes: user dislike data
    
    Returns:
        Explanation string if movie was excluded, None otherwise
    """
    uid_str = str(user_id)
    user_dislikes = dislikes.get(uid_str, {})

    if not user_dislikes:
        return None

    info = movie_info.get(movie_id, {})
    title = info.get("title", f"Movie {movie_id}")
    franchise = info.get("franchise")

    disliked_movies = user_dislikes.get("movies", [])
    disliked_franchises = user_dislikes.get("franchises", [])

    if movie_id in disliked_movies:
        return f'"{title}" was removed from your recommendations because you marked it as disliked.'

    if franchise and franchise in disliked_franchises:
        return (
            f'"{title}" was excluded because you removed the '
            f'{franchise} franchise from your preferences.'
        )

    return None


def batch_explain(user_id: int, recommendations: list,
                  all_ratings: dict, movie_info: dict,
                  dislikes: dict = None) -> list:
    """
    Add explanations to a list of recommendations.
    
    Modifies the 'reason' field of each recommendation dict.
    """
    for rec in recommendations:
        rec["reason"] = explain_recommendation(
            user_id, rec["movie_id"],
            all_ratings, movie_info, dislikes
        )
    return recommendations
