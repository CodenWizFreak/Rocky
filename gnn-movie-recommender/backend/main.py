"""
Conversational Movie Recommendation Engine
VS Code Terminal Mode | TMDB Dataset + Groq LLM + Semantic Embeddings
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from groq import Groq
from dotenv import load_dotenv

# ── Load .env before anything else ────────────────────────────────────────────
load_dotenv()

from state_manager import (
    load_state,
    remember_event,
    save_state,
    update_disliked,
    update_disliked_genre,
    update_liked,
    update_liked_genre,
)
from intent_parser  import parse_intent
from scoring_engine import build_movie_db, score_and_recommend
from embedder       import setup_embedder, build_plot_index
from display        import (
    print_banner, print_state_monitor, print_debug, print_system
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — everything comes from .env
# ══════════════════════════════════════════════════════════════════════════════
ROOT = os.path.dirname(os.path.abspath(__file__))

def _path(env_key: str, default: str) -> str:
    return os.path.join(ROOT, os.getenv(env_key, default))

METADATA_CSV      = _path("METADATA_CSV",      "data/movies_metadata.csv")
CREDITS_CSV       = _path("CREDITS_CSV",        "data/credits.csv")
RATINGS_CSV       = _path("RATINGS_CSV",        "data/ratings.csv")
STATE_FILE        = _path("STATE_FILE",          "user_state.json")
EMBEDDINGS_CACHE  = _path("EMBEDDINGS_CACHE",   "embeddings_cache.npy")
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local")
VOYAGE_API_KEY    = os.getenv("VOYAGE_API_KEY",  "")
GROQ_MODEL        = os.getenv("GROQ_MODEL",      "")
GROQ_TEMPERATURE  = float(os.getenv("GROQ_TEMPERATURE", "0.8"))
GROQ_MAX_TOKENS   = int(os.getenv("GROQ_MAX_TOKENS",    "1024"))
TOP_N             = int(os.getenv("TOP_N_RESULTS",       "5"))

# ── Validate required env vars ─────────────────────────────────────────────────
_missing = [k for k in ("GROQ_API_KEY", "GROQ_MODEL") if not os.getenv(k)]
if _missing:
    for k in _missing:
        print(f"[ERROR] {k} is not set in your .env file.")
    sys.exit(1)

# ── Groq client ────────────────────────────────────────────────────────────────
groq_client = Groq()   # auto-reads GROQ_API_KEY from env


# ══════════════════════════════════════════════════════════════════════════════
#  GROQ HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(state: dict) -> str:
    liked    = state.get("liked_movies",    [])
    disliked = state.get("disliked_movies", [])
    liked_g  = state.get("liked_genres",     [])
    blocked_g = state.get("disliked_genres", [])
    genres   = state.get("genre_weights",   {})
    top_g    = sorted(genres.items(), key=lambda x: x[1], reverse=True)[:3]
    top_g_str = ", ".join(f"{g} ({w:.2f})" for g, w in top_g)

    return f"""You are a friendly, knowledgeable movie recommendation assistant.
You help users discover films they will genuinely love.

CURRENT USER PROFILE:
- Liked movies     : {liked    if liked    else 'none yet'}
- Disliked/blocked : {disliked if disliked else 'none yet'}
- Liked genres     : {liked_g if liked_g else 'none yet'}
- Blocked genres   : {blocked_g if blocked_g else 'none yet'}
- Top genre weights: {top_g_str if top_g_str else 'not established yet'}

RULES:
1. Present recommendations as a clean numbered list with a one-line reason for each.
2. NEVER recommend anything in the disliked list: {disliked}.
3. NEVER recommend movies from blocked genres: {blocked_g}.
4. Keep responses concise and warm — 2-3 sentences then the list.
5. Acknowledge likes/dislikes before moving on.
6. Only mention movies from the SYSTEM CONTEXT list — do not invent titles.
7. If a plot/director/actor query was made, briefly explain why each pick matches.
8. If no results were found, say so honestly and ask for different criteria.
"""


def call_groq(messages: list[dict]) -> str:
    """Stream a Groq response; print tokens as they arrive; return full text."""
    completion = groq_client.chat.completions.create(
        model                = GROQ_MODEL,
        messages             = messages,
        temperature          = GROQ_TEMPERATURE,
        max_completion_tokens= GROQ_MAX_TOKENS,
        top_p                = 1,
        stream               = True,
        stop                 = None,
    )
    parts = []
    print("\033[36mBot:\033[0m ", end="", flush=True)
    for chunk in completion:
        tok = chunk.choices[0].delta.content or ""
        print(tok, end="", flush=True)
        parts.append(tok)
    print()
    return "".join(parts)


def format_reco_context(recommendations: list[tuple], intent: dict) -> str:
    """Format scored results + intent metadata for injection into the LLM prompt."""
    if not recommendations:
        return "No movies matched the criteria. Tell the user politely and ask them to refine their request."

    directors = intent.get("directors_requested", [])
    actors    = intent.get("actors_requested",    [])
    genres    = intent.get("genres_requested",    [])
    plot_desc = intent.get("plot_description",    "")
    keywords  = intent.get("plot_keywords",       [])

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
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print_banner()

    # ── 1. Embedding backend ───────────────────────────────────────────────────
    print_system("Initialising embedding backend …")
    setup_embedder(
        cache_path = EMBEDDINGS_CACHE,
        backend    = EMBEDDING_BACKEND,
        voyage_key = VOYAGE_API_KEY or None,
    )

    # ── 2. Load dataset ────────────────────────────────────────────────────────
    print_system("Loading dataset …")
    try:
        movie_db = build_movie_db(METADATA_CSV, CREDITS_CSV, RATINGS_CSV)
    except FileNotFoundError as exc:
        print(f"\n\033[31m[ERROR]\033[0m {exc}")
        print("Check that movies_metadata.csv, credits.csv, and ratings.csv")
        print("are all inside the data/ folder.")
        sys.exit(1)

    # ── 3. Build / load plot index ─────────────────────────────────────────────
    print_system("Building plot embedding index (cached after first run) …")
    plot_index = build_plot_index(movie_db)   # loads cache if exists

    # ── 4. Load user state ─────────────────────────────────────────────────────
    state = load_state(STATE_FILE)
    print_system("State loaded.\n")

    # ── 5. Conversation history ────────────────────────────────────────────────
    conversation: list[dict] = []

    # Greeting
    sys_prompt = build_system_prompt(state)
    conversation.append({
        "role":    "user",
        "content": "Greet me in one friendly sentence and ask what I'm in the mood to watch.",
    })
    greeting = call_groq([{"role": "system", "content": sys_prompt}] + conversation)
    conversation.append({"role": "assistant", "content": greeting})

    # ══════════════════════════════════════════════════════════════════════════
    #  INTERACTIVE LOOP
    # ══════════════════════════════════════════════════════════════════════════
    while True:
        try:
            user_input = input("\n\033[33mUser:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\033[36mBot:\033[0m Catch you later! 🎬")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "bye", "q"):
            print("\033[36mBot:\033[0m Enjoy your movies! 🎬")
            break

        # ── Intent parsing (LLM-first, regex fallback) ─────────────────────────
        intent = parse_intent(
            user_input      = user_input,
            movie_db        = movie_db,
            groq_client     = groq_client,
            groq_model      = GROQ_MODEL,
            last_recs       = state.get("last_recommendations", []),
            recent_history  = conversation[-6:],
            user_state      = state,
        )
        print_debug(intent)

        # ── State updates (likes / dislikes) ───────────────────────────────────
        state_changed = False

        for title in intent.get("liked_found", []):
            rows   = movie_db[movie_db["title_clean"] == title.lower()]
            genres = rows.iloc[0]["genres_list"] if not rows.empty else []
            update_liked(state, title, genres)
            remember_event(state, {"type": "like_movie", "title": title, "genres": genres})
            print(f"\033[32m[Graph]\033[0m Boosting '{title}' genres: {genres}")
            state_changed = True

        for genre in intent.get("liked_genres", []):
            update_liked_genre(state, genre)
            remember_event(state, {"type": "like_genre", "genre": genre})
            print(f"\033[32m[Graph]\033[0m Boosting genre '{genre}'")
            state_changed = True

        for title in intent.get("disliked_found", []):
            rows   = movie_db[movie_db["title_clean"] == title.lower()]
            genres = []
            update_disliked(state, title, genres)
            remember_event(state, {"type": "dislike_movie", "title": title})
            print(f"\033[31m[Graph Eraser]\033[0m Severing '{title}' (weight → 0.0)")
            state_changed = True

        for genre in intent.get("disliked_genres", []):
            update_disliked_genre(state, genre)
            remember_event(state, {"type": "dislike_genre", "genre": genre})
            print(f"\033[31m[Graph Eraser]\033[0m Blocking genre '{genre}', decaying weight")
            state_changed = True

        if state_changed:
            save_state(STATE_FILE, state)

        # ── Widen search if user wants something different ─────────────────────
        effective_intent = dict(intent)
        if intent.get("wants_different") and not any([
            intent.get("genres_requested"),
            intent.get("actors_requested"),
            intent.get("directors_requested"),
            intent.get("plot_description"),
            intent.get("plot_keywords"),
        ]):
            effective_intent["genres_requested"]    = []
            effective_intent["actors_requested"]    = []
            effective_intent["directors_requested"] = []
            effective_intent["plot_keywords"]       = []

        # ── Score & recommend ──────────────────────────────────────────────────
        recommendations = score_and_recommend(
            movie_db   = movie_db,
            intent     = effective_intent,
            state      = state,
            plot_index = plot_index,
            top_n      = TOP_N,
        )

        # ── Build LLM context ──────────────────────────────────────────────────
        pure_sentiment = (
            state_changed
            and not intent.get("genres_requested")
            and not intent.get("actors_requested")
            and not intent.get("directors_requested")
            and not intent.get("plot_description")
            and not intent.get("plot_keywords")
            and not intent.get("wants_different")
        )

        shown_recommendations = bool(recommendations) and not pure_sentiment
        if shown_recommendations:
            state["last_recommendations"] = [r[0] for r in recommendations]
            save_state(STATE_FILE, state)

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

        # ── Call Groq conversational LLM ───────────────────────────────────────
        sys_prompt = build_system_prompt(state)
        conversation.append({"role": "user", "content": user_input})

        injected = (
            "\n\n[SYSTEM CONTEXT — internal only, do not quote verbatim]\n"
            + reco_context
            + "\nPresent results naturally. If directors/actors/plot were requested, "
              "briefly explain each pick's relevance."
        )

        llm_messages = (
            [{"role": "system", "content": sys_prompt}]
            + conversation[:-1]
            + [{"role": "user", "content": user_input + injected}]
        )

        bot_reply = call_groq(llm_messages)
        conversation.append({"role": "assistant", "content": bot_reply})

        # ── Live state monitor ─────────────────────────────────────────────────
        print_state_monitor(state)


if __name__ == "__main__":
    main()
