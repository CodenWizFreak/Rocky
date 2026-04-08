# engine.py
# ─────────────────────────────────────────────────────────────
# ML1M-backed recommendation engine.
# Loads MovieLens-1M from disk and exposes get_recommendations(),
# which mirrors the interface your friend's EEMU engine will use.
#
# To wire in the real EEMU backend:
#   Replace ML1MEngine.get_recommendations() with a call to
#   your friend's inference() from dist_eval — the query_payload
#   contract (defined in retrieval.py) stays the same.

import os
import pandas as pd

from config import ML1M_PATH


def load_ml1m() -> pd.DataFrame:
    """Load and join movies + ratings from the ML1M .dat files."""
    movies_df = pd.read_csv(
        os.path.join(ML1M_PATH, "movies.dat"),
        sep="::", engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    movies_df["year"] = (
        movies_df["title"].str.extract(r"\((\d{4})\)")
        .astype(float).fillna(0).astype(int)
    )
    movies_df["title_clean"] = (
        movies_df["title"]
        .str.replace(r"\s*\(\d{4}\)", "", regex=True)
        .str.strip()
    )
    movies_df["genres_list"] = movies_df["genres"].str.split("|")

    ratings_df = pd.read_csv(
        os.path.join(ML1M_PATH, "ratings.dat"),
        sep="::", engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1",
    )
    movie_stats = ratings_df.groupby("movie_id").agg(
        avg_rating=("rating", "mean"),
        num_ratings=("rating", "count"),
    ).reset_index()

    movies_df = movies_df.merge(movie_stats, on="movie_id", how="left")
    movies_df["avg_rating"].fillna(0, inplace=True)
    movies_df["num_ratings"].fillna(0, inplace=True)

    print(f"ML1M loaded: {len(movies_df)} movies, {len(ratings_df)} ratings.")
    return movies_df


class ML1MEngine:
    """
    MovieLens-1M recommendation engine.

    Ranking formula (mocks Two-Tower EEMU scoring):
        score = keyword_overlap_with_NL_query × 10 + avg_rating

    In production, swap get_recommendations() for a call to the
    EEMU inference function — the query dict contract is unchanged.
    """

    def __init__(self, df: pd.DataFrame):
        self.catalogue = [
            {
                "id":          str(row["movie_id"]),
                "title":       row["title_clean"],
                "year":        int(row["year"]),
                "genres":      row["genres_list"],
                # ML1M has no plot text — synthesise from metadata
                "plot":        f"A {'/'.join(row['genres_list'])} film from {row['year']}.",
                "avg_rating":  round(float(row["avg_rating"]), 2),
                "num_ratings": int(row["num_ratings"]),
            }
            for _, row in df.iterrows()
        ]
        print(f"[ML1MEngine] Ready — {len(self.catalogue)} movies.")

    def get_recommendations(self, q: dict) -> list[dict]:
        genres_want  = [g.lower() for g in q.get("genres", [])]
        genres_ban   = [g.lower() for g in q.get("exclude_genres", [])]
        year_range   = q.get("year_range")
        rating_min   = float(q.get("rating_min") or 0.0)
        exclude_ids  = set(q.get("exclude_ids", []))
        nl_query     = q.get("nl_query", "").lower()
        top_k        = int(q.get("top_k", 5))

        # Tokenise NL query; skip very short stop-words
        query_tokens = {t for t in nl_query.split() if len(t) > 3}

        scored = []
        for m in self.catalogue:
            if m["id"] in exclude_ids:
                continue

            glow = [g.lower() for g in m["genres"]]

            # Hard filters
            if genres_want and not any(g in glow for g in genres_want):
                continue
            if any(g in glow for g in genres_ban):
                continue
            if year_range and not (year_range[0] <= m["year"] <= year_range[1]):
                continue
            if m["avg_rating"] < rating_min:
                continue

            # Soft score
            text    = (m["title"] + " " + " ".join(m["genres"])).lower()
            overlap = sum(1 for t in query_tokens if t in text)
            score   = overlap * 10 + m["avg_rating"]
            scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def request_unlearning(self, user_id: str):
        """
        Stub for the EEMU exact-unlearning trigger.
        Replace with your friend's shard-retrain call:
          → identify the SISA shard containing user_id's interactions
          → remove the relevant data points
          → retrain that shard only (EEMU guarantees only 1 shard retrains)
        """
        print(f"[ML1MEngine] Unlearning queued for user '{user_id}'.")
        print("  → In production: locate SISA shard → delete interactions → retrain shard.")