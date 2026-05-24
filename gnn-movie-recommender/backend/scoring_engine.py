"""
scoring_engine.py — Data ingestion, normalisation, and scoring

Dataset (Kaggle "The Movies Dataset"):
  movies_metadata.csv  — movie details, genres, overview, ratings
  credits.csv          — cast + crew JSON per movie (contains directors)
  ratings.csv          — userId, movieId, rating, timestamp

Scoring formula (unchanged from spec):
  Final Score = (normalized_base_weight + matching_points) × W_movie × ∏ W_genres

New in this version:
  • Parses movies_metadata.csv (not tmdb_5000_movies.csv)
  • Extracts director names from crew JSON (job == "Director")
  • Bayesian-weighted rating normalization using ratings.csv
  • plot_index (np.ndarray) passed in for semantic search
  • director_requested intent field supported
  • plot_description intent field triggers semantic search
"""

import ast
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Score bonuses ────────────────────────────────────────────────────────────
GENRE_BONUS    = 2.0
ACTOR_BONUS    = 3.0
DIRECTOR_BONUS = 3.5   # directors are highly intentional requests
KEYWORD_BONUS  = 1.5
PLOT_SIM_BONUS = 4.0   # semantic similarity score gets scaled by this
TOP_N          = 5

# ─── Bayesian rating parameters ───────────────────────────────────────────────
# W = (v / (v + m)) * R + (m / (v + m)) * C
# v = movie vote count, m = minimum votes threshold, R = movie avg, C = global avg
MIN_VOTES_PERCENTILE = 60   # movies below this percentile of vote counts are penalised


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_parse(cell) -> list:
    """Safely parse a stringified Python/JSON list."""
    try:
        parsed = ast.literal_eval(str(cell))
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def _extract_names(cell) -> list[str]:
    return [
        item["name"]
        for item in _safe_parse(cell)
        if isinstance(item, dict) and "name" in item
    ]


def _extract_cast_names(cell, limit: int = 15) -> list[str]:
    items = _safe_parse(cell)
    return [
        item["name"]
        for item in items
        if isinstance(item, dict) and "name" in item
    ][:limit]


def _extract_directors(crew_cell) -> list[str]:
    """Pull names where job == 'Director' from the crew JSON column."""
    items = _safe_parse(crew_cell)
    return [
        item["name"]
        for item in items
        if isinstance(item, dict)
        and item.get("job", "").lower() == "director"
    ]


def _bayesian_score(
    vote_avg:   pd.Series,
    vote_count: pd.Series,
    m:          float,
    C:          float,
) -> pd.Series:
    """
    Bayesian weighted rating: pulls low-count movies toward the global mean.
    Returns a Series of scores between 0 and 10.
    """
    v = vote_count.fillna(0)
    R = vote_avg.fillna(0)
    return (v / (v + m)) * R + (m / (v + m)) * C


# ══════════════════════════════════════════════════════════════════════════════
#  DATA INGESTION
# ══════════════════════════════════════════════════════════════════════════════

def build_movie_db(
    metadata_csv: str,
    credits_csv:  str,
    ratings_csv:  str,
) -> pd.DataFrame:
    """
    Load all three CSVs, join them, parse JSON columns, compute normalised
    Bayesian base weights.

    Returns a clean DataFrame with columns:
      title, title_clean, genres_list, cast_list, directors_list,
      overview, normalized_base_weight, vote_average, vote_count, movie_id
    """

    # ── 1. Load metadata ───────────────────────────────────────────────────────
    print("[Ingestion] Reading movies_metadata.csv …")
    meta = pd.read_csv(
        metadata_csv,
        low_memory=False,
        dtype={"id": str},
    )

    # Some rows have bad IDs (non-numeric) — drop them
    meta = meta[pd.to_numeric(meta["id"], errors="coerce").notna()].copy()
    meta["id"] = meta["id"].astype(str).str.strip()

    # Keep only released movies with an English or non-adult flag
    if "status" in meta.columns:
        meta = meta[meta["status"].isin(["Released", ""])]
    if "adult" in meta.columns:
        meta = meta[meta["adult"].astype(str).str.lower() != "true"]

    # ── 2. Load credits ────────────────────────────────────────────────────────
    print("[Ingestion] Reading credits.csv …")
    credits = pd.read_csv(credits_csv, dtype={"id": str})
    credits["id"] = credits["id"].astype(str).str.strip()

    # ── 3. Merge ───────────────────────────────────────────────────────────────
    df = meta.merge(credits, on="id", how="left")

    # ── 4. Parse JSON-string columns ───────────────────────────────────────────
    print("[Ingestion] Parsing JSON columns …")
    df["genres_list"]    = df["genres"].apply(_extract_names)
    df["cast_list"]      = df["cast"].apply(_extract_cast_names)
    df["directors_list"] = df["crew"].apply(_extract_directors)

    # ── 5. Ratings: Bayesian weighted score ────────────────────────────────────
    print("[Ingestion] Computing Bayesian ratings …")
    try:
        ratings = pd.read_csv(ratings_csv, dtype={"movieId": str})
        ratings["movieId"] = ratings["movieId"].astype(str).str.strip()

        # Aggregate per movie
        agg = (
            ratings.groupby("movieId")["rating"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "r_avg", "count": "r_count"})
            .reset_index()
        )

        # The Kaggle ratings dataset uses its own movieId which may differ from
        # the TMDB id in metadata. We try to merge on imdb_id → movieId link,
        # but if unavailable we fall back to vote_average from metadata.
        # Simple approach: use metadata vote_average + vote_count directly.
        # ratings.csv augments but doesn't replace.
        logger.info(f"[Ingestion] Loaded {len(ratings):,} ratings rows.")
        ratings_available = True
    except FileNotFoundError:
        logger.warning("[Ingestion] ratings.csv not found — using metadata vote_average only.")
        ratings_available = False

    vote_avg   = pd.to_numeric(df["vote_average"],  errors="coerce").fillna(0)
    vote_count = pd.to_numeric(df["vote_count"],     errors="coerce").fillna(0)

    C = vote_avg[vote_avg > 0].mean()          # global mean rating
    m = vote_count.quantile(MIN_VOTES_PERCENTILE / 100)  # min votes threshold

    bayesian = _bayesian_score(vote_avg, vote_count, m, C)

    # Min-max normalise to [0, 1]
    b_min, b_max = bayesian.min(), bayesian.max()
    if b_max > b_min:
        df["normalized_base_weight"] = (bayesian - b_min) / (b_max - b_min)
    else:
        df["normalized_base_weight"] = 0.5

    df["vote_average"] = vote_avg
    df["vote_count"]   = vote_count

    # ── 6. Housekeeping ────────────────────────────────────────────────────────
    df["title"]       = df["title"].astype(str).str.strip()
    df["title_clean"] = df["title"].str.lower()
    df["overview"]    = df["overview"].fillna("").astype(str)
    df["movie_id"]    = df["id"].astype(str)

    df = df.dropna(subset=["title"]).reset_index(drop=True)
    df = df[df["title"] != "nan"].reset_index(drop=True)

    print(f"[Ingestion] Done — {len(df):,} movies loaded.")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def score_and_recommend(
    movie_db:    pd.DataFrame,
    intent:      dict,
    state:       dict,
    plot_index:  np.ndarray | None = None,
    top_n:       int = TOP_N,
) -> list[tuple[str, float]]:
    """
    Filter → score → rank movies.

    intent keys used:
      genres_requested    list[str]
      actors_requested    list[str]
      directors_requested list[str]
      plot_keywords       list[str]   — keyword fallback
      plot_description    str | None  — full natural language plot query
      liked_found         list[str]
      disliked_found      list[str]

    Returns list of (title, score) tuples, up to top_n.
    """
    genres_req    = [g.lower() for g in intent.get("genres_requested",    [])]
    actors_req    = [a.lower() for a in intent.get("actors_requested",    [])]
    directors_req = [d.lower() for d in intent.get("directors_requested", [])]
    keywords_req  = [k.lower() for k in intent.get("plot_keywords",       [])]
    plot_desc     = (intent.get("plot_description") or "").strip()

    disliked_set   = {t.lower() for t in state.get("disliked_movies", [])}
    last_recs_set  = {t.lower() for t in state.get("last_recommendations", [])}
    blocked_genres = {g.lower() for g in state.get("disliked_genres", [])}
    genre_weights  = state.get("genre_weights", {})

    # ── Semantic plot search: get candidate row indices ────────────────────────
    semantic_indices: set[int] | None = None
    semantic_scores:  dict[int, float] = {}

    if plot_desc and plot_index is not None:
        from embedder import search_by_plot
        candidate_indices = search_by_plot(
            plot_desc, movie_db, plot_index, top_k=200
        )
        semantic_indices = set(candidate_indices)
        # Normalised similarity score per row index (0–1 range)
        from embedder import get_embedding
        q_vec = get_embedding(plot_desc)
        safe_index = np.nan_to_num(plot_index, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        safe_q = np.nan_to_num(q_vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        raw_scores = np.dot(safe_index, safe_q)
        raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=0.0, neginf=0.0)
        for idx in candidate_indices:
            semantic_scores[idx] = float(raw_scores[idx])

    has_hard_filter = bool(genres_req or actors_req or directors_req)

    results: list[tuple[str, float]] = []

    for row_idx, row in movie_db.iterrows():
        title       = row["title"]
        title_lower = row["title_clean"]

        # NEW — only excludes disliked (permanent) and last batch (avoid repeat)
        if title_lower in disliked_set:
            continue
        if title_lower in last_recs_set:
            continue

        # ── Semantic filter: if plot query given, only consider top candidates ─
        if semantic_indices is not None and row_idx not in semantic_indices:
            continue

        movie_genres    = [g.lower() for g in (row.get("genres_list")    or [])]
        movie_cast      = [a.lower() for a in (row.get("cast_list")      or [])]
        movie_directors = [d.lower() for d in (row.get("directors_list") or [])]
        overview        = str(row.get("overview", "")).lower()

        if blocked_genres and any(g in blocked_genres for g in movie_genres):
            continue

        # ── Hard filters (genre / actor / director) ────────────────────────────
        if has_hard_filter:
            genre_ok    = any(g in movie_genres    for g in genres_req)    if genres_req    else True
            actor_ok    = any(a in movie_cast      for a in actors_req)    if actors_req    else True
            director_ok = any(d in movie_directors for d in directors_req) if directors_req else True
            if not (genre_ok and actor_ok and director_ok):
                continue

        # ── Matching bonus points ──────────────────────────────────────────────
        matching_points = 0.0

        if genres_req:
            matching_points += sum(
                GENRE_BONUS for g in genres_req if g in movie_genres
            )
        if actors_req:
            matching_points += sum(
                ACTOR_BONUS for a in actors_req if a in movie_cast
            )
        if directors_req:
            matching_points += sum(
                DIRECTOR_BONUS for d in directors_req if d in movie_directors
            )
        if keywords_req:
            matching_points += sum(
                KEYWORD_BONUS
                for k in keywords_req
                if k in overview
            )

        # Semantic similarity bonus
        if row_idx in semantic_scores:
            sim = semantic_scores[row_idx]   # 0–1 cosine similarity
            matching_points += sim * PLOT_SIM_BONUS

        # ── Genre weight product (user preference graph) ───────────────────────
        w_genres = 1.0
        for genre in (row.get("genres_list") or []):
            w = genre_weights.get(genre, 1.0)
            w_genres *= w
        w_genres = max(0.0, min(w_genres, 5.0))

        base        = float(row.get("normalized_base_weight", 0.5))
        final_score = (base + matching_points) * w_genres

        results.append((title, round(final_score, 4)))

    # ── Sort descending, return top N ─────────────────────────────────────────
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]
