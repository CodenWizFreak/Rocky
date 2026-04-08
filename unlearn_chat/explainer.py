# explainer.py
# ─────────────────────────────────────────────────────────────
# Module 4 — Explainable Response Generator.
# Takes the list of movies returned by the engine and the current
# DialogueState, then asks the LLM to produce a warm, natural
# explanation linking each film to the user's constraints.
# Mirrors the RA-Rec paper's Section 3.4 explanation style.

import json

from llm import _call_llm
from prompts import RESPONSE_GEN_SYSTEM
from state import DialogueState


def generate_explanation(movies: list[dict], state: DialogueState) -> str:
    """
    Generates a conversational recommendation response.

    If no movies matched the filters, returns a friendly fallback
    asking the user to loosen their constraints.
    """
    if not movies:
        return (
            "I couldn't find anything matching those filters. "
            "Try broadening the year range or removing a genre constraint."
        )

    hc_txt = json.dumps(state.hard_constraints) if state.hard_constraints else "none"
    sc_txt = json.dumps(state.soft_constraints) if state.soft_constraints else "none"

    blurbs = "\n".join(
        f"- {m['title']} ({m['year']}) | {', '.join(m['genres'])} | "
        f"avg rating {m['avg_rating']}"
        for m in movies
    )

    prompt = (
        f"User's hard constraints: {hc_txt}\n"
        f"User's mood/vibe (soft constraints): {sc_txt}\n\n"
        f"Engine returned:\n{blurbs}\n\n"
        "Write a warm, conversational recommendation. For each movie use:\n"
        "🎬 [Title] ([Year]) — [one sentence linking it to the user's request].\n"
        "End with a brief follow-up question to refine further."
    )

    return _call_llm(RESPONSE_GEN_SYSTEM, prompt, max_new_tokens=400)