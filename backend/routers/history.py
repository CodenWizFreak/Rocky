"""
History Router — Users and Watch History

GET  /users              → list of users
GET  /history/{user_id}  → watch history for a user
GET  /graph/{user_id}    → subgraph data for D3 visualisation
"""

from fastapi import APIRouter, HTTPException, Query

from recommend.inference import get_engine

router = APIRouter()


@router.get("/users")
def list_users(limit: int = Query(20, ge=1, le=943)):
    """
    List available users.
    Returns a sample of users with their metadata.
    """
    engine = get_engine()
    engine.load()

    users = []
    for uid, info in list(engine.user_info.items())[:limit]:
        # Count watched movies
        num_watched = len(engine.all_ratings.get(uid, {}))
        users.append({
            "user_id": uid,
            "age": info.get("age"),
            "gender": info.get("gender"),
            "occupation": info.get("occupation"),
            "num_movies_watched": num_watched,
        })

    # Sort by number of movies watched (most active first)
    users.sort(key=lambda x: x["num_movies_watched"], reverse=True)

    return {"users": users, "total": len(engine.user_info)}


@router.get("/history/{user_id}")
def get_watch_history(user_id: int):
    """Get a user's complete watch history with movie details."""
    engine = get_engine()
    engine.load()

    if user_id not in engine.user_info:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    history = engine.get_watch_history(user_id)

    return {
        "user_id": user_id,
        "user_info": engine.user_info.get(user_id),
        "history": history,
        "total_watched": len(history),
    }


@router.get("/graph/{user_id}")
def get_user_subgraph(user_id: int, depth: int = Query(1, ge=1, le=2)):
    """
    Get the knowledge graph subgraph around a user for D3 visualisation.
    
    Returns nodes and links in D3-compatible format.
    """
    engine = get_engine()
    engine.load()

    if user_id not in engine.user_info:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    nodes = []
    links = []
    seen_nodes = set()

    # User node
    user_node_id = f"user_{user_id}"
    user_info = engine.user_info[user_id]
    nodes.append({
        "id": user_node_id,
        "type": "user",
        "label": f"User {user_id}",
        "occupation": user_info.get("occupation"),
    })
    seen_nodes.add(user_node_id)

    # Get user's ratings (use positive only for cleaner graph)
    user_ratings = engine.all_ratings.get(user_id, {})
    positive_movies = {mid: r for mid, r in user_ratings.items() if r >= 3.5}

    # Limit to top 15 movies by rating for readability
    sorted_movies = sorted(positive_movies.items(), key=lambda x: x[1], reverse=True)[:15]

    for mid, rating in sorted_movies:
        movie_info = engine.movie_info.get(mid, {})
        movie_node_id = f"movie_{mid}"

        if movie_node_id not in seen_nodes:
            nodes.append({
                "id": movie_node_id,
                "type": "movie",
                "label": movie_info.get("title", f"Movie {mid}"),
                "genres": movie_info.get("genres", []),
                "franchise": movie_info.get("franchise"),
            })
            seen_nodes.add(movie_node_id)

        links.append({
            "source": user_node_id,
            "target": movie_node_id,
            "type": "watched",
            "rating": rating,
        })

        # Add genre and franchise connections
        if depth >= 1:
            for genre in movie_info.get("genres", []):
                genre_node_id = f"genre_{genre}"
                if genre_node_id not in seen_nodes:
                    nodes.append({
                        "id": genre_node_id,
                        "type": "genre",
                        "label": genre,
                    })
                    seen_nodes.add(genre_node_id)

                links.append({
                    "source": movie_node_id,
                    "target": genre_node_id,
                    "type": "has_genre",
                })

            franchise = movie_info.get("franchise")
            if franchise:
                franchise_node_id = f"franchise_{franchise}"
                if franchise_node_id not in seen_nodes:
                    nodes.append({
                        "id": franchise_node_id,
                        "type": "franchise",
                        "label": franchise,
                    })
                    seen_nodes.add(franchise_node_id)

                links.append({
                    "source": movie_node_id,
                    "target": franchise_node_id,
                    "type": "in_franchise",
                })

    return {
        "user_id": user_id,
        "nodes": nodes,
        "links": links,
    }
