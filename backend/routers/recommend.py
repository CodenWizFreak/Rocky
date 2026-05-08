"""
Recommendation Router

GET  /recommend/{user_id}?k=10       → top-K recommendations
GET  /explain/{user_id}/{movie_id}   → why this movie was recommended
GET  /movies/search?q=...            → search movies by title
"""

from fastapi import APIRouter, Query, HTTPException

from recommend.inference import get_engine
from recommend.explain import explain_recommendation, batch_explain
from recommend.penalize import load_dislikes

router = APIRouter()


@router.get("/recommend/{user_id}")
def get_recommendations(user_id: int, k: int = Query(10, ge=1, le=50)):
    """Get top-K movie recommendations for a user."""
    engine = get_engine()
    engine.load()

    # Validate user
    if user_id not in engine.user_info:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    result = engine.recommend(user_id, top_k=k)

    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    # Add explanations
    dislikes = load_dislikes()
    batch_explain(
        user_id, result["recommendations"],
        engine.all_ratings, engine.movie_info, dislikes
    )

    return result


@router.get("/explain/{user_id}/{movie_id}")
def get_explanation(user_id: int, movie_id: int):
    """Get detailed explanation for why a movie was/wasn't recommended."""
    engine = get_engine()
    engine.load()

    if user_id not in engine.user_info:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    if movie_id not in engine.movie_info:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")

    dislikes = load_dislikes()

    explanation = explain_recommendation(
        user_id, movie_id,
        engine.all_ratings, engine.movie_info, dislikes
    )

    # Check if movie was excluded
    from recommend.explain import explain_exclusion
    exclusion = explain_exclusion(user_id, movie_id, engine.movie_info, dislikes)

    return {
        "user_id": user_id,
        "movie_id": movie_id,
        "movie_title": engine.movie_info.get(movie_id, {}).get("title"),
        "explanation": explanation,
        "excluded": exclusion is not None,
        "exclusion_reason": exclusion,
    }


@router.get("/movies/search")
def search_movies(q: str = Query(..., min_length=1)):
    """Search movies by title."""
    engine = get_engine()
    engine.load()

    query_lower = q.lower()
    results = []

    for mid, info in engine.movie_info.items():
        title = info.get("title", "")
        if query_lower in title.lower():
            results.append({
                "movie_id": mid,
                "title": title,
                "genres": info.get("genres", []),
                "franchise": info.get("franchise"),
            })

    # Sort by relevance (title starts with query first, then contains)
    results.sort(key=lambda x: (
        not x["title"].lower().startswith(query_lower),
        x["title"]
    ))

    return {"query": q, "results": results[:20]}
