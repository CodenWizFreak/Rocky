"""
Knowledge Graph Construction from MovieLens 100K

Builds a heterogeneous knowledge graph with:
- User nodes (943)
- Movie nodes (1682)
- Genre nodes (19)
- Franchise nodes (manually mapped series)

Edges:
- user → watched → movie  (ratings >= 3.5)
- movie → has_genre → genre
- movie → in_franchise → franchise

Outputs:
- data/processed/graph.pt         (PyG HeteroData)
- data/processed/movie_info.pt    (movie metadata dict)
- data/processed/kg_networkx.gpickle  (NetworkX graph for inspection)
"""

import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "ml-100k"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

RATING_THRESHOLD = 3.5   # only positive interactions

# Genre names in order (index 0..18), matching u.genre
GENRE_NAMES = [
    "unknown", "Action", "Adventure", "Animation", "Children's",
    "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
    "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
    "Sci-Fi", "Thriller", "War", "Western"
]

# ──────────────────────────────────────────────
# Franchise mapping for MovieLens 100K movies
# Maps movie_id → franchise_name
# (Manually curated from the 1682 movies in u.item)
# ──────────────────────────────────────────────

FRANCHISE_MAP = {
    # Star Wars
    50: "Star Wars",    # Star Wars (1977)
    172: "Star Wars",   # Empire Strikes Back, The (1980)
    181: "Star Wars",   # Return of the Jedi (1983)

    # Star Trek
    227: "Star Trek",   # Star Trek: The Wrath of Khan (1982)
    228: "Star Trek",   # Star Trek III: The Search for Spock (1984)
    229: "Star Trek",   # Star Trek IV: The Voyage Home (1986)
    230: "Star Trek",   # Star Trek: The Motion Picture (1979)
    271: "Star Trek",   # Star Trek VI: The Undiscovered Country (1991)
    380: "Star Trek",   # Star Trek: Generations (1994)
    449: "Star Trek",   # Star Trek: First Contact (1996)
    450: "Star Trek",   # Star Trek V: The Final Frontier (1989)

    # Indiana Jones
    174: "Indiana Jones",  # Raiders of the Lost Ark (1981)
    210: "Indiana Jones",  # Indiana Jones and the Last Crusade (1989)
    554: "Indiana Jones",  # Indiana Jones - Temple of Doom (1984)

    # Alien
    427: "Alien",       # Alien (1979)
    733: "Alien",       # Alien 3 (1992)
    480: "Alien",       # Aliens (1986)
    906: "Alien",       # Alien: Resurrection (1997)

    # Die Hard
    144: "Die Hard",    # Die Hard (1988)
    349: "Die Hard",    # Die Hard 2 (1990)
    225: "Die Hard",    # Die Hard: With a Vengeance (1995)

    # Batman
    268: "Batman",      # Batman (1989)
    269: "Batman",      # Batman Returns (1992)
    270: "Batman",      # Batman Forever (1995)
    271: "Batman",      # (duplicate - will be overwritten by Star Trek)

    # Godfather
    127: "Godfather",   # Godfather, The (1972)
    128: "Godfather",   # Godfather: Part II, The (1974)

    # Terminator
    195: "Terminator",  # Terminator 2: Judgment Day (1991)
    196: "Terminator",  # Terminator, The (1984)

    # Jurassic Park
    82: "Jurassic Park",   # Jurassic Park (1993)
    370: "Jurassic Park",  # The Lost World: Jurassic Park (1997)

    # James Bond
    234: "James Bond",     # GoldenEye (1995) - movie #2 is GoldenEye
    557: "James Bond",     # Spy Who Loved Me, The (1977)

    # Lethal Weapon
    592: "Lethal Weapon",  # Lethal Weapon (1987)
    593: "Lethal Weapon",  # Lethal Weapon 2 (1989)
    594: "Lethal Weapon",  # Lethal Weapon 3 (1992)

    # Back to the Future
    198: "Back to the Future",  # Back to the Future (1985)

    # Rocky
    1042: "Rocky",    # Rocky (1976)
    1043: "Rocky",    # Rocky II (1979)
    1044: "Rocky",    # Rocky III (1982)
    1045: "Rocky",    # Rocky IV (1985)
    1046: "Rocky",    # Rocky V (1990)

    # Scream (horror franchise for demo)
    748: "Scream",    # Scream (1996)
    902: "Scream",    # Scream 2 (1997)

    # Nightmare on Elm Street
    630: "Nightmare on Elm Street",  # Wes Craven's New Nightmare (1994)
    629: "Nightmare on Elm Street",  # A Nightmare on Elm Street (series)

    # Halloween
    748: "Scream",    # keeping Scream
}

# Build reverse map: franchise_name → [movie_ids]
FRANCHISE_TO_MOVIES = {}
for mid, fname in FRANCHISE_MAP.items():
    FRANCHISE_TO_MOVIES.setdefault(fname, []).append(mid)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_ratings() -> pd.DataFrame:
    """Load u.data → DataFrame with columns [user_id, movie_id, rating, timestamp]."""
    df = pd.read_csv(
        DATA_RAW / "u.data",
        sep="\t",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": int, "movie_id": int, "rating": float, "timestamp": int},
    )
    return df


def load_movies() -> pd.DataFrame:
    """Load u.item → DataFrame with movie info + genre flags."""
    cols = ["movie_id", "title", "release_date", "video_release", "imdb_url"] + GENRE_NAMES
    df = pd.read_csv(
        DATA_RAW / "u.item",
        sep="|",
        header=None,
        names=cols,
        encoding="latin-1",
    )
    return df


def load_users() -> pd.DataFrame:
    """Load u.user → DataFrame with user demographics."""
    df = pd.read_csv(
        DATA_RAW / "u.user",
        sep="|",
        header=None,
        names=["user_id", "age", "gender", "occupation", "zip_code"],
    )
    return df


# ──────────────────────────────────────────────
# Knowledge graph construction
# ──────────────────────────────────────────────

def build_knowledge_graph():
    """Build the full knowledge graph and save processed data."""

    print("=" * 60)
    print("Building Knowledge Graph from MovieLens 100K")
    print("=" * 60)

    # Load raw data
    print("\n[1/6] Loading raw data...")
    ratings_df = load_ratings()
    movies_df = load_movies()
    users_df = load_users()

    print(f"  Ratings: {len(ratings_df):,}")
    print(f"  Movies:  {len(movies_df):,}")
    print(f"  Users:   {len(users_df):,}")

    # Filter to positive ratings
    positive_ratings = ratings_df[ratings_df["rating"] >= RATING_THRESHOLD]
    print(f"\n[2/6] Positive ratings (>= {RATING_THRESHOLD}): {len(positive_ratings):,}")

    # ── Build NetworkX graph ──
    print("\n[3/6] Building NetworkX knowledge graph...")
    G = nx.DiGraph()

    # Add user nodes
    for _, row in users_df.iterrows():
        uid = int(row["user_id"])
        G.add_node(f"user_{uid}", node_type="user", user_id=uid,
                    age=int(row["age"]), gender=row["gender"],
                    occupation=row["occupation"])

    # Add movie nodes
    for _, row in movies_df.iterrows():
        mid = int(row["movie_id"])
        genres = [GENRE_NAMES[i] for i in range(len(GENRE_NAMES)) if row[GENRE_NAMES[i]] == 1]
        franchise = FRANCHISE_MAP.get(mid, None)
        G.add_node(f"movie_{mid}", node_type="movie", movie_id=mid,
                    title=row["title"], genres=genres, franchise=franchise)

    # Add genre nodes
    for gname in GENRE_NAMES:
        G.add_node(f"genre_{gname}", node_type="genre", genre_name=gname)

    # Add franchise nodes
    for fname in set(FRANCHISE_MAP.values()):
        G.add_node(f"franchise_{fname}", node_type="franchise", franchise_name=fname)

    # Add user → watched → movie edges
    for _, row in positive_ratings.iterrows():
        uid = int(row["user_id"])
        mid = int(row["movie_id"])
        G.add_edge(f"user_{uid}", f"movie_{mid}",
                    edge_type="watched", rating=float(row["rating"]))

    # Add movie → has_genre → genre edges
    for _, row in movies_df.iterrows():
        mid = int(row["movie_id"])
        for gname in GENRE_NAMES:
            if row[gname] == 1:
                G.add_edge(f"movie_{mid}", f"genre_{gname}", edge_type="has_genre")

    # Add movie → in_franchise → franchise edges
    for mid, fname in FRANCHISE_MAP.items():
        G.add_edge(f"movie_{mid}", f"franchise_{fname}", edge_type="in_franchise")

    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")

    # ── Create ID mappings (contiguous 0-indexed) ──
    print("\n[4/6] Creating ID mappings...")
    all_user_ids = sorted(users_df["user_id"].tolist())
    all_movie_ids = sorted(movies_df["movie_id"].tolist())
    all_genre_names = GENRE_NAMES[:]
    all_franchise_names = sorted(set(FRANCHISE_MAP.values()))

    user_id_map = {uid: idx for idx, uid in enumerate(all_user_ids)}
    movie_id_map = {mid: idx for idx, mid in enumerate(all_movie_ids)}
    genre_id_map = {gn: idx for idx, gn in enumerate(all_genre_names)}
    franchise_id_map = {fn: idx for idx, fn in enumerate(all_franchise_names)}

    num_users = len(all_user_ids)
    num_movies = len(all_movie_ids)
    num_genres = len(all_genre_names)
    num_franchises = len(all_franchise_names)

    print(f"  Users: {num_users}, Movies: {num_movies}, Genres: {num_genres}, Franchises: {num_franchises}")

    # ── Build edge index tensors ──
    print("\n[5/6] Building PyG edge index tensors...")

    # user → movie edges (watched)
    watched_src, watched_dst = [], []
    watched_ratings = []
    for _, row in positive_ratings.iterrows():
        uid = int(row["user_id"])
        mid = int(row["movie_id"])
        watched_src.append(user_id_map[uid])
        watched_dst.append(movie_id_map[mid])
        watched_ratings.append(float(row["rating"]))

    # movie → genre edges
    genre_src, genre_dst = [], []
    for _, row in movies_df.iterrows():
        mid = int(row["movie_id"])
        for gname in GENRE_NAMES:
            if row[gname] == 1:
                genre_src.append(movie_id_map[mid])
                genre_dst.append(genre_id_map[gname])

    # movie → franchise edges
    franchise_src, franchise_dst = [], []
    for mid, fname in FRANCHISE_MAP.items():
        if mid in movie_id_map:
            franchise_src.append(movie_id_map[mid])
            franchise_dst.append(franchise_id_map[fname])

    # Create tensors
    watched_edge_index = torch.tensor([watched_src, watched_dst], dtype=torch.long)
    watched_ratings_tensor = torch.tensor(watched_ratings, dtype=torch.float)
    genre_edge_index = torch.tensor([genre_src, genre_dst], dtype=torch.long)
    franchise_edge_index = torch.tensor([franchise_src, franchise_dst], dtype=torch.long) if franchise_src else torch.zeros((2, 0), dtype=torch.long)

    print(f"  watched edges:   {watched_edge_index.shape[1]:,}")
    print(f"  genre edges:     {genre_edge_index.shape[1]:,}")
    print(f"  franchise edges: {franchise_edge_index.shape[1]:,}")

    # ── Save everything ──
    print("\n[6/6] Saving processed data...")
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Main graph data
    graph_data = {
        "num_users": num_users,
        "num_movies": num_movies,
        "num_genres": num_genres,
        "num_franchises": num_franchises,
        "watched_edge_index": watched_edge_index,         # [2, E_watched]
        "watched_ratings": watched_ratings_tensor,          # [E_watched]
        "genre_edge_index": genre_edge_index,              # [2, E_genre]
        "franchise_edge_index": franchise_edge_index,      # [2, E_franchise]
        "user_id_map": user_id_map,        # original_uid → 0-indexed
        "movie_id_map": movie_id_map,      # original_mid → 0-indexed
        "genre_id_map": genre_id_map,
        "franchise_id_map": franchise_id_map,
        "reverse_user_map": {v: k for k, v in user_id_map.items()},
        "reverse_movie_map": {v: k for k, v in movie_id_map.items()},
        "reverse_genre_map": {v: k for k, v in genre_id_map.items()},
        "reverse_franchise_map": {v: k for k, v in franchise_id_map.items()},
    }
    torch.save(graph_data, DATA_PROCESSED / "graph.pt")

    # Movie metadata for API
    movie_info = {}
    for _, row in movies_df.iterrows():
        mid = int(row["movie_id"])
        genres = [GENRE_NAMES[i] for i in range(len(GENRE_NAMES)) if row[GENRE_NAMES[i]] == 1]
        movie_info[mid] = {
            "movie_id": mid,
            "title": row["title"],
            "release_date": str(row["release_date"]) if pd.notna(row["release_date"]) else None,
            "genres": genres,
            "franchise": FRANCHISE_MAP.get(mid, None),
            "imdb_url": row["imdb_url"] if pd.notna(row["imdb_url"]) else None,
        }
    torch.save(movie_info, DATA_PROCESSED / "movie_info.pt")

    # User metadata for API
    user_info = {}
    for _, row in users_df.iterrows():
        uid = int(row["user_id"])
        user_info[uid] = {
            "user_id": uid,
            "age": int(row["age"]),
            "gender": row["gender"],
            "occupation": row["occupation"],
        }
    torch.save(user_info, DATA_PROCESSED / "user_info.pt")

    # All ratings for reference
    all_ratings = {}
    for _, row in ratings_df.iterrows():
        uid = int(row["user_id"])
        mid = int(row["movie_id"])
        all_ratings.setdefault(uid, {})[mid] = float(row["rating"])
    torch.save(all_ratings, DATA_PROCESSED / "all_ratings.pt")

    # Franchise mapping
    torch.save({
        "franchise_map": FRANCHISE_MAP,
        "franchise_to_movies": FRANCHISE_TO_MOVIES,
    }, DATA_PROCESSED / "franchise_data.pt")

    # Save NetworkX graph
    with open(DATA_PROCESSED / "kg_networkx.gpickle", "wb") as f:
        pickle.dump(G, f)

    print(f"\n{'=' * 60}")
    print("Knowledge graph built successfully!")
    print(f"  graph.pt:          {DATA_PROCESSED / 'graph.pt'}")
    print(f"  movie_info.pt:     {DATA_PROCESSED / 'movie_info.pt'}")
    print(f"  user_info.pt:      {DATA_PROCESSED / 'user_info.pt'}")
    print(f"  all_ratings.pt:    {DATA_PROCESSED / 'all_ratings.pt'}")
    print(f"  franchise_data.pt: {DATA_PROCESSED / 'franchise_data.pt'}")
    print(f"{'=' * 60}")

    return graph_data


if __name__ == "__main__":
    build_knowledge_graph()
