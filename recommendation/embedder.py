"""
embedder.py — Semantic embedding engine

Phase 1 (local):  sentence-transformers, all-MiniLM-L6-v2, embeddings cached to .npy
Phase 2 (deploy): set EMBEDDING_BACKEND=voyage in .env — swaps internals only,
                  public API (get_embedding, build_plot_index, search_by_plot) unchanged.

Public API:
    setup_embedder(cache_path, backend, voyage_key)  → call once at startup
    get_embedding(text)                              → list[float]
    build_plot_index(movie_db, force_rebuild)        → np.ndarray  (N × D)
    search_by_plot(query, movie_db, index, top_k)   → list[int]   (row indices)
"""

import os
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Module-level state (set by setup_embedder) ───────────────────────────────
_backend:    str        = "local"
_model                  = None       # sentence-transformers model (phase 1)
_voyage_client          = None       # voyage client (phase 2)
_voyage_model: str      = "voyage-02"
_cache_path:   str      = "embeddings_cache.npy"
_dim:          int      = 384        # MiniLM-L6-v2 output dim


def _sanitize_unit_vectors(arr: np.ndarray) -> np.ndarray:
    """Return finite float32 row-normalised vectors; zero rows stay zero."""
    safe = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if safe.ndim == 1:
        norm = np.linalg.norm(safe)
        return safe / norm if norm > 0 else safe
    norms = np.linalg.norm(safe, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return safe / norms


def setup_embedder(
    cache_path:  str = "embeddings_cache.npy",
    backend:     str = "local",
    voyage_key:  str | None = None,
) -> None:
    """
    Initialise the embedding backend. Call once at startup before any other
    embedder functions.

    backend="local"  → loads sentence-transformers (no API key needed)
    backend="voyage" → initialises Voyage AI client (needs voyage_key)
    """
    global _backend, _model, _voyage_client, _cache_path

    _backend    = backend.lower().strip()
    _cache_path = cache_path

    if _backend == "local":
        _load_local_model()
    elif _backend == "voyage":
        _load_voyage_client(voyage_key)
    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}'. Use 'local' or 'voyage'.")


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — LOCAL (sentence-transformers)
# ══════════════════════════════════════════════════════════════════════════════

def _load_local_model() -> None:
    global _model, _dim
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is not installed.\n"
            "Run: pip install sentence-transformers"
        )
    logger.info("[Embedder] Loading local model all-MiniLM-L6-v2 …")
    print("[Embedder] Loading sentence-transformer model (first run may download ~80 MB) …")
    _model = SentenceTransformer("all-MiniLM-L6-v2")
    _dim   = _model.get_sentence_embedding_dimension()
    logger.info(f"[Embedder] Local model ready. Embedding dim={_dim}")


def _embed_local(texts: list[str]) -> np.ndarray:
    """Batch-encode with sentence-transformers. Returns (N, D) float32 array."""
    vecs = _model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors → cosine = dot product
        convert_to_numpy=True,
    )
    return _sanitize_unit_vectors(vecs)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — VOYAGE AI  (swap-in, same interface)
# ══════════════════════════════════════════════════════════════════════════════

def _load_voyage_client(api_key: str | None) -> None:
    global _voyage_client, _dim
    if not api_key:
        raise ValueError("VOYAGE_API_KEY must be set when EMBEDDING_BACKEND=voyage")
    try:
        import voyageai
    except ImportError:
        raise ImportError(
            "voyageai is not installed.\n"
            "Run: pip install voyageai"
        )
    _voyage_client = voyageai.Client(api_key=api_key)
    _dim = 1024   # voyage-02 output dim
    logger.info("[Embedder] Voyage AI client ready.")


def _embed_voyage(texts: list[str]) -> np.ndarray:
    """Batch-encode with Voyage AI. Returns (N, D) float32 array."""
    BATCH = 128
    all_vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        result = _voyage_client.embed(batch, model=_voyage_model, input_type="document")
        all_vecs.extend(result.embeddings)
    arr = np.array(all_vecs, dtype=np.float32)
    # Normalise so cosine similarity = dot product
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return _sanitize_unit_vectors(arr / norms)


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC INTERFACE
# ══════════════════════════════════════════════════════════════════════════════

def get_embedding(text: str) -> np.ndarray:
    """Return a single normalised embedding vector for text."""
    if _backend == "local":
        vec = _embed_local([text])[0]
    elif _backend == "voyage":
        result = _voyage_client.embed([text], model=_voyage_model, input_type="query")
        vec = np.array(result.embeddings[0], dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
    else:
        raise RuntimeError("Embedder not initialised. Call setup_embedder() first.")
    return _sanitize_unit_vectors(vec)


def build_plot_index(
    movie_db:      pd.DataFrame,
    force_rebuild: bool = False,
) -> np.ndarray:
    """
    Build (or load from cache) a matrix of shape (N, D) where each row is the
    normalised embedding of a movie's overview text.

    Cache is stored at _cache_path as a .npy file.
    If the cache exists and force_rebuild=False, load it — startup takes <2s.
    Otherwise embed all overviews and save the cache — takes ~30–60s first time.

    Returns the index matrix (kept in memory for the session).
    """
    cache = _cache_path

    if not force_rebuild and os.path.exists(cache):
        logger.info(f"[Embedder] Loading plot index from cache: {cache}")
        print(f"[Embedder] Loading plot embeddings from cache …")
        index = np.load(cache)
        if index.shape[0] == len(movie_db):
            index = _sanitize_unit_vectors(index)
            print(f"[Embedder] Plot index ready — {index.shape[0]} movies × {index.shape[1]} dims.")
            return index
        else:
            logger.warning("[Embedder] Cache size mismatch — rebuilding.")

    print(f"[Embedder] Building plot index for {len(movie_db)} movies …")
    print("           (This runs once and caches to disk. Grab a coffee ☕)")

    texts = (
        movie_db["overview"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    if _backend == "local":
        index = _embed_local(texts)
    elif _backend == "voyage":
        index = _embed_voyage(texts)
    else:
        raise RuntimeError("Embedder not initialised.")

    index = _sanitize_unit_vectors(index)
    np.save(cache, index)
    print(f"[Embedder] Plot index saved to {cache}. Shape: {index.shape}")
    return index


def search_by_plot(
    query:    str,
    movie_db: pd.DataFrame,
    index:    np.ndarray,
    top_k:    int = 50,
) -> list[int]:
    """
    Embed query and return the row indices of the top_k most semantically
    similar movies in the index. Uses dot product (cosine, since both are
    normalised unit vectors).

    Returns list of integer row indices into movie_db.
    """
    q_vec = get_embedding(query)           # shape (D,)
    safe_index = _sanitize_unit_vectors(index)
    scores = np.dot(safe_index, q_vec)     # shape (N,) — cosine similarity
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices.tolist()
