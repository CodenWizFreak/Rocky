"""
Unlearning Router

POST /unlearn → trigger machine unlearning for a user-movie pair
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from recommend.inference import get_engine
from unlearn.embedding_update import fast_unlearn

router = APIRouter()


class UnlearnRequest(BaseModel):
    """Request body for unlearning endpoint."""
    user_id: int
    movie_id: int
    scope: str = "movie"  # "movie" or "franchise"


class UnlearnResponse(BaseModel):
    """Response from unlearning endpoint."""
    success: bool
    level: int
    method: str
    user_id: int
    movie_id: int
    movie_title: Optional[str] = None
    scope: str
    franchise: Optional[str] = None
    shard_id: int
    message: str


@router.post("/unlearn")
def trigger_unlearning(request: UnlearnRequest):
    """
    Trigger machine unlearning for a user-movie interaction.

    Uses Level 1 (fast) unlearning by default:
    - Deletes the user→movie edge from the knowledge graph
    - Records the dislike (movie-level or franchise-level)
    - Effective immediately in recommendations

    Body:
    {
        "user_id": 42,
        "movie_id": 318,
        "scope": "franchise"  // or "movie"
    }
    """
    engine = get_engine()
    engine.load()

    # Validate
    if request.user_id not in engine.user_info:
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
    if request.movie_id not in engine.movie_info:
        raise HTTPException(status_code=404, detail=f"Movie {request.movie_id} not found")
    if request.scope not in ("movie", "franchise"):
        raise HTTPException(status_code=400, detail="Scope must be 'movie' or 'franchise'")

    # Perform Level 1 (fast) unlearning
    result = fast_unlearn(
        user_id=request.user_id,
        movie_id=request.movie_id,
        scope=request.scope,
        graph_data=engine.graph_data,
        shard_assignments=engine.shard_assignments,
        movie_info=engine.movie_info,
        franchise_data=engine.franchise_data,
    )

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Unlearning failed"))

    # Reload the affected shard model to reflect changes
    try:
        shard_id = result.get("shard_id")
        if shard_id is not None:
            engine.reload_shard(shard_id)
    except Exception:
        pass  # Model reload is optional — dislike system handles filtering

    return result


@router.get("/dislikes/{user_id}")
def get_user_dislikes(user_id: int):
    """Get a user's current dislikes."""
    from recommend.penalize import load_dislikes
    engine = get_engine()
    engine.load()

    if user_id not in engine.user_info:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    dislikes = load_dislikes()
    uid_str = str(user_id)

    user_dislikes = dislikes.get(uid_str, {"movies": [], "franchises": []})

    # Enrich with movie titles
    enriched_movies = []
    for mid in user_dislikes.get("movies", []):
        info = engine.movie_info.get(mid, {})
        enriched_movies.append({
            "movie_id": mid,
            "title": info.get("title", f"Movie {mid}"),
            "franchise": info.get("franchise"),
        })

    return {
        "user_id": user_id,
        "disliked_movies": enriched_movies,
        "disliked_franchises": user_dislikes.get("franchises", []),
    }


@router.delete("/dislikes/{user_id}")
def clear_user_dislikes(user_id: int):
    """Clear all dislikes for a user (reset unlearning)."""
    from recommend.penalize import load_dislikes, save_dislikes

    dislikes = load_dislikes()
    uid_str = str(user_id)

    if uid_str in dislikes:
        del dislikes[uid_str]
        save_dislikes(dislikes)

    return {"success": True, "message": f"Dislikes cleared for user {user_id}"}
