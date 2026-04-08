# retrieval.py
# ─────────────────────────────────────────────────────────────
# Module 3 — Retrieval Bridge.
# Converts a DialogueState into the query payload that the
# EEMU / ML1M engine consumes.

from state import DialogueState
from config import DEFAULT_TOP_K


def generate_eemu_query(state: DialogueState) -> dict:
    """
    Converts DialogueState → structured query payload.

    Hard constraints  → passed as discrete filter fields so the engine
                        can do exact metadata filtering before ranking.
    Soft constraints  → flattened into a natural-language string that
                        the Two-Tower item tower would embed in production.
    Long-term taste   → appended to the NL query so past preferences
                        continue to influence ranking across sessions.
    Rejected items    → passed as exclude_ids so the engine never
                        re-surfaces something the user already disliked.
    """
    hc = state.hard_constraints
    sc = state.soft_constraints

    # Flatten soft constraints into NL string
    soft_parts = []
    for k, v in sc.items():
        soft_parts.append(f"{k}: {', '.join(v) if isinstance(v, list) else v}")

    # Inject long-term taste into the NL query
    liked = state.long_term_taste.get("liked", [])
    if liked:
        soft_parts.append(f"previously enjoyed: {', '.join(liked[:5])}")

    nl_query = "; ".join(soft_parts) if soft_parts else "good movie"

    return {
        "user_id":        state.user_id,
        "nl_query":       nl_query,
        "genres":         hc.get("genres", []),
        "exclude_genres": hc.get("exclude_genres", []),
        "year_range":     hc.get("year_range", None),   # [min, max] or None
        "rating_min":     hc.get("rating_min", None),
        "director":       hc.get("director", None),
        "actor":          hc.get("actor", None),
        "exclude_ids":    list(state.rejected_items),
        "top_k":          DEFAULT_TOP_K,
    }