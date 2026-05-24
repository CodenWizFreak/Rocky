"""
api.py — FastAPI server wrapping the movie recommendation engine.

Run with:
    uvicorn api:app --reload --port 8000
"""

import os
import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from groq import Groq
from state_manager import (
    load_state, remember_event, save_state,
    update_disliked, update_disliked_genre,
    update_liked, update_liked_genre,
    decay_genre,
    initialize_state,
)
from intent_parser import parse_intent
from scoring_engine import build_movie_db, score_and_recommend
from embedder import setup_embedder, build_plot_index

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

def _path(env_key, default):
    return os.path.join(ROOT, os.getenv(env_key, default))

METADATA_CSV      = _path("METADATA_CSV",     "data/movies_metadata.csv")
CREDITS_CSV       = _path("CREDITS_CSV",      "data/credits.csv")
RATINGS_CSV       = _path("RATINGS_CSV",      "data/ratings.csv")
STATE_FILE        = _path("STATE_FILE",        "user_state.json")
EMBEDDINGS_CACHE  = _path("EMBEDDINGS_CACHE", "embeddings_cache.npy")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
VOYAGE_API_KEY    = os.getenv("VOYAGE_API_KEY", "")
GROQ_MODEL        = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TEMPERATURE  = float(os.getenv("GROQ_TEMPERATURE", "0.8"))
GROQ_MAX_TOKENS   = int(os.getenv("GROQ_MAX_TOKENS", "1024"))
TOP_N             = int(os.getenv("TOP_N_RESULTS", "5"))

app_state: dict = {}


# ── Lifespan ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Starting up — loading models...")

    _missing = [k for k in ("GROQ_API_KEY", "GROQ_MODEL") if not os.getenv(k)]
    if _missing:
        raise RuntimeError(f"Missing env vars: {_missing}")

    app_state["groq_client"] = Groq()

    setup_embedder(
        cache_path=EMBEDDINGS_CACHE,
        backend=EMBEDDING_BACKEND,
        voyage_key=VOYAGE_API_KEY or None,
    )

    print("[API] Loading movie database...")
    app_state["movie_db"] = build_movie_db(METADATA_CSV, CREDITS_CSV, RATINGS_CSV)

    print("[API] Building plot index...")
    app_state["plot_index"] = build_plot_index(app_state["movie_db"])

    print("[API] Loading user state...")
    app_state["state"] = load_state(STATE_FILE)

    app_state["state"]["last_recommendation_scores"] = {}
    app_state["state"]["last_recommendations"] = []

    app_state["conversation"] = []

    print("[API] Ready ✓")
    yield
    print("[API] Shutting down.")


app = FastAPI(title="GNN Movie Recommender API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

class ResetRequest(BaseModel):
    confirm: bool = True


# ── System prompt ──────────────────────────────────────────────────────────────
def build_system_prompt(state: dict) -> str:
    liked     = state.get("liked_movies", [])
    disliked  = state.get("disliked_movies", [])
    liked_g   = state.get("liked_genres", [])
    blocked_g = state.get("disliked_genres", [])
    genres    = state.get("genre_weights", {})
    top_g     = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3]
    top_g_str = ", ".join(f"{g} ({w:.2f})" for g, w in top_g if w != 1.0)

    return f"""You are a friendly, knowledgeable movie recommendation assistant powered by a GNN (Graph Neural Network) with machine unlearning capability.
You help users discover films they will genuinely love, and can permanently "forget" movies or genres when asked.

CURRENT USER PROFILE:
- Liked movies     : {liked    if liked    else 'none yet'}
- Disliked/blocked : {disliked if disliked else 'none yet'}
- Liked genres     : {liked_g if liked_g else 'none yet'}
- Blocked genres   : {blocked_g if blocked_g else 'none yet'}
- Top genre weights: {top_g_str if top_g_str else 'all genres equal (no preference yet)'}

RULES:
1. Present recommendations as a clean numbered list with a one-line reason for each.
2. NEVER recommend anything in the disliked list: {disliked}.
3. NEVER recommend movies from blocked genres: {blocked_g}.
4. Keep responses concise and warm — 2-3 sentences then the list.
5. Acknowledge likes/dislikes before moving on.
6. Only mention movies from the SYSTEM CONTEXT list — do not invent titles.
7. If a plot/director/actor query was made, briefly explain why each pick matches.
8. If no results were found, say so honestly and ask for different criteria.
9. When unlearning happens, mention that the GNN graph edge has been removed and the shard reweighted.
"""


def format_reco_context(recommendations: list, intent: dict) -> str:
    if not recommendations:
        return "No movies matched the criteria. Tell the user politely and ask them to refine their request."

    directors = intent.get("directors_requested", [])
    actors    = intent.get("actors_requested", [])
    genres    = intent.get("genres_requested", [])
    plot_desc = intent.get("plot_description", "")
    keywords  = intent.get("plot_keywords", [])

    parts = []
    if directors: parts.append("director(s): " + ", ".join(directors))
    if actors:    parts.append("actor(s): "    + ", ".join(actors))
    if genres:    parts.append("genre(s): "    + ", ".join(genres))
    if plot_desc: parts.append("plot: "        + plot_desc)
    if keywords:  parts.append("keywords: "    + ", ".join(keywords))
    criteria = "; ".join(parts) if parts else "general preference"

    lines = [f"Scored results for [{criteria}]:"]
    for i, (title, score) in enumerate(recommendations, 1):
        lines.append(f"  {i}. {title}  (score: {score:.2f})")
    lines.append("\nIMPORTANT: Only recommend movies from this exact list above. Do not substitute or add others.")
    return "\n".join(lines)


# ── Graph builder ──────────────────────────────────────────────────────────────
def get_graph_data(state: dict) -> dict:
    movie_db         = app_state.get("movie_db")
    nodes            = []
    edges            = []
    node_ids         = set()

    liked_movies     = state.get("liked_movies", [])
    disliked_movies  = state.get("disliked_movies", [])
    genre_weights    = state.get("genre_weights", {})
    liked_genres     = state.get("liked_genres", [])
    disliked_genres  = state.get("disliked_genres", [])
    rec_scores: dict = state.get("last_recommendation_scores", {})

    nodes.append({"id": "user", "label": "You", "type": "user", "weight": 1.0})
    node_ids.add("user")

    def safe_node_id(prefix: str, title: str) -> str:
        return f"{prefix}__{title[:30].replace(' ', '_').replace('/', '_')}"

    # Recommended nodes (purple) — exact scores only
    for title, score in rec_scores.items():
        nid = safe_node_id("rec", title)
        if nid not in node_ids:
            nodes.append({"id": nid, "label": title, "type": "recommended", "weight": round(score, 3)})
            node_ids.add(nid)
        edges.append({
            "source": "user", "target": nid,
            "weight": round(score, 3), "type": "recommends",
        })
        if movie_db is not None:
            rows = movie_db[movie_db["title_clean"] == title.lower()]
            if not rows.empty:
                genres_list = rows.iloc[0].get("genres_list", []) or []
                for genre in genres_list[:3]:
                    gid = f"genre__{genre.replace(' ', '_')}"
                    if gid not in node_ids:
                        w = genre_weights.get(genre, 1.0)
                        nodes.append({"id": gid, "label": genre, "type": "genre", "weight": round(w, 3)})
                        node_ids.add(gid)
                    edges.append({
                        "source": nid, "target": gid,
                        "weight": round(genre_weights.get(genre, 1.0), 3),
                        "type": "has_genre",
                    })

    # Liked nodes (blue)
    for title in liked_movies:
        nid     = safe_node_id("liked", title)
        rec_nid = safe_node_id("rec",   title)
        if nid not in node_ids and rec_nid not in node_ids:
            nodes.append({"id": nid, "label": title, "type": "liked", "weight": 1.0})
            node_ids.add(nid)
            edges.append({"source": "user", "target": nid, "weight": 0.8, "type": "liked"})

    # Erased nodes (red)
    for title in disliked_movies:
        nid = safe_node_id("erased", title)
        if nid not in node_ids:
            nodes.append({"id": nid, "label": title, "type": "erased", "weight": 0.0})
            node_ids.add(nid)
            edges.append({"source": "user", "target": nid, "weight": 0.0, "type": "erased"})

    return {
        "nodes": nodes,
        "edges": edges,
        "genre_weights": {k: round(v, 3) for k, v in genre_weights.items() if v != 1.0},
        "stats": {
            "liked_count":     len(liked_movies),
            "disliked_count":  len(disliked_movies),
            "liked_genres":    liked_genres,
            "disliked_genres": disliked_genres,
        },
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": GROQ_MODEL}

@app.get("/state")
async def get_state():
    return app_state.get("state", {})

@app.get("/graph")
async def get_graph():
    return get_graph_data(app_state.get("state", {}))

@app.post("/reset")
async def reset_state(req: ResetRequest):
    if not req.confirm:
        return {"message": "Reset cancelled"}
    new_state = initialize_state()
    new_state["last_recommendation_scores"] = {}
    save_state(STATE_FILE, new_state)
    app_state["state"]        = new_state
    app_state["conversation"] = []
    return {"message": "State reset successfully"}

@app.get("/greet")
async def greet():
    state        = app_state.get("state", {})
    groq_client  = app_state.get("groq_client")
    conversation = app_state.get("conversation", [])

    if conversation:
        return {"greeting": "Welcome back! What would you like to watch today?"}

    sys_prompt = build_system_prompt(state)
    resp = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user",   "content": "Greet me in one friendly sentence and ask what I'm in the mood to watch."},
        ],
        temperature=GROQ_TEMPERATURE,
        max_completion_tokens=150,
        stream=False,
    )
    greeting = resp.choices[0].message.content.strip()
    conversation.append({"role": "user",      "content": "Greet me..."})
    conversation.append({"role": "assistant", "content": greeting})
    app_state["conversation"] = conversation
    return {"greeting": greeting}


@app.post("/chat")
async def chat(req: ChatRequest):
    user_input = req.message.strip()
    if not user_input:
        raise HTTPException(status_code=400, detail="Empty message")

    movie_db     = app_state["movie_db"]
    plot_index   = app_state["plot_index"]
    state        = app_state["state"]
    conversation = app_state["conversation"]
    groq_client  = app_state["groq_client"]

    async def generate():
        intent = parse_intent(
            user_input=user_input,
            movie_db=movie_db,
            groq_client=groq_client,
            groq_model=GROQ_MODEL,
            last_recs=state.get("last_recommendations", []),
            recent_history=conversation[-6:],
            user_state=state,
        )

        state_changed = False

        # ── Liked movies ──────────────────────────────────────────────────────
        for title in intent.get("liked_found", []):
            rows   = movie_db[movie_db["title_clean"] == title.lower()]
            genres = list(rows.iloc[0]["genres_list"]) if not rows.empty else []
            update_liked(state, title, genres)
            remember_event(state, {"type": "like_movie", "title": title, "genres": genres})
            state_changed = True

        # ── Liked genres ──────────────────────────────────────────────────────
        for genre in intent.get("liked_genres", []):
            update_liked_genre(state, genre)
            remember_event(state, {"type": "like_genre", "genre": genre})
            state_changed = True

        # ── Disliked movies ───────────────────────────────────────────────────
        for title in intent.get("disliked_found", []):
            rows   = movie_db[movie_db["title_clean"] == title.lower()]
            genres = list(rows.iloc[0]["genres_list"]) if not rows.empty else []
            update_disliked(state, title, [])
            for genre in genres:
                decay_genre(state, genre)
            remember_event(state, {"type": "dislike_movie", "title": title, "genres": genres})
            state_changed = True

        # ── Hard genre blocks ─────────────────────────────────────────────────
        for genre in intent.get("disliked_genres", []):
            update_disliked_genre(state, genre)
            remember_event(state, {"type": "dislike_genre", "genre": genre})
            state_changed = True

        # ── Soft genre decay only ─────────────────────────────────────────────
        for genre in intent.get("soft_disliked_genres", []):
            decay_genre(state, genre)
            remember_event(state, {"type": "soft_dislike_genre", "genre": genre})
            state_changed = True

        # ── Unblock genre if user explicitly requests it ──────────────────────
        for genre in intent.get("genres_requested", []):
            if genre in state.get("disliked_genres", []):
                state["disliked_genres"].remove(genre)
                state["genre_weights"][genre] = max(state["genre_weights"].get(genre, 1.0), 1.0)
                remember_event(state, {"type": "unblock_genre", "genre": genre})
                state_changed = True

        if state_changed:
            save_state(STATE_FILE, state)

        # ── pure_sentiment MUST be computed before scoring ────────────────────
        has_any_filter = any([
            intent.get("genres_requested"),
            intent.get("actors_requested"),
            intent.get("directors_requested"),
            intent.get("plot_description"),
            intent.get("plot_keywords"),
        ])

        pure_sentiment = (
            state_changed
            and not has_any_filter
            and not intent.get("wants_different")
        )

        # ── Persist genre context when user requests genres ───────────────────
        if intent.get("genres_requested"):
            state["last_requested_genres"] = intent["genres_requested"]
            save_state(STATE_FILE, state)

        # ── Build effective intent for scoring ────────────────────────────────
        effective_intent = dict(intent)

        if intent.get("wants_different") and not has_any_filter:
            effective_intent["genres_requested"]    = []
            effective_intent["actors_requested"]    = []
            effective_intent["directors_requested"] = []
            effective_intent["plot_keywords"]       = []
        elif not has_any_filter and not pure_sentiment:
            # Short message like "No I don't like it" — carry last genre context
            last_genres = state.get("last_requested_genres", [])
            if last_genres:
                effective_intent["genres_requested"] = last_genres

        # ── Score & recommend ─────────────────────────────────────────────────
        recommendations = score_and_recommend(
            movie_db=movie_db,
            intent=effective_intent,
            state=state,
            plot_index=plot_index,
            top_n=TOP_N,
        )

        shown_recommendations = bool(recommendations) and not pure_sentiment

        if shown_recommendations:
            state["last_recommendations"]       = [r[0] for r in recommendations]
            state["last_recommendation_scores"] = {r[0]: r[1] for r in recommendations}
            save_state(STATE_FILE, state)

        # ── Build LLM context ─────────────────────────────────────────────────
        if pure_sentiment:
            reco_context = (
                "The user just expressed a like/dislike. "
                "Acknowledge it warmly, confirm the preference was noted, "
                "and invite them to ask for recommendations."
            )
        elif intent.get("wants_different"):
            reco_context = (
                "User wants different picks from the last batch.\n"
                + format_reco_context(recommendations, effective_intent)
            )
        elif intent.get("sentiment_last_recs") == "negative" and not recommendations:
            reco_context = (
                "The user didn't enjoy the last suggestions and no new results matched. "
                "Apologise briefly, ask them to describe what they'd prefer differently."
            )
        else:
            reco_context = format_reco_context(recommendations, effective_intent)

        # ── Groq call (streaming) ─────────────────────────────────────────────
        sys_prompt = build_system_prompt(state)
        conversation.append({"role": "user", "content": user_input})

        injected = (
            "\n\n[SYSTEM CONTEXT — internal only, do not quote verbatim]\n"
            + reco_context
            + "\nPresent ALL movies from the scored list above as a numbered list. "
            + "CRITICAL: Do not skip, omit, or substitute any title. "
            + "Do not add titles from your own knowledge. "
            + "Present every single movie in the list, no matter what."
        )

        llm_messages = (
            [{"role": "system", "content": sys_prompt}]
            + conversation[:-1]
            + [{"role": "user", "content": user_input + injected}]
        )

        full_reply = []
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=llm_messages,
            temperature=GROQ_TEMPERATURE,
            max_completion_tokens=GROQ_MAX_TOKENS,
            top_p=1,
            stream=True,
            stop=None,
        )
        for chunk in completion:
            tok = chunk.choices[0].delta.content or ""
            if tok:
                full_reply.append(tok)
                yield f"data: {json.dumps({'type': 'token', 'content': tok})}\n\n"
            await asyncio.sleep(0)

        bot_reply = "".join(full_reply)
        conversation.append({"role": "assistant", "content": bot_reply})

        graph_data = get_graph_data(state)
        yield f"data: {json.dumps({'type': 'graph', 'data': graph_data})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"
        
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection":    "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )