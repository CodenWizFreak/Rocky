# classifier.py
# ─────────────────────────────────────────────────────────────
# Module 2 — Multi-label Intent Classifier & State Updater.
# Calls the LLM with STATE_UPDATER_SYSTEM, extracts a JSON patch,
# and applies it to the DialogueState in-place.

import json, re

from llm import _call_llm
from prompts import STATE_UPDATER_SYSTEM
from state import DialogueState


def _extract_json(raw: str) -> dict:
    """
    Robustly pull the first {...} block from raw LLM output.
    Falls back to a direct json.loads if no braces are found.
    Returns {} on any parse failure so the caller never crashes.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(raw)
    except Exception:
        return {}


def classify_and_update(user_message: str, state: DialogueState) -> dict:
    """
    Main entry point for Module 2.

    1. Builds a prompt from STATE_UPDATER_SYSTEM + current state + user message.
    2. Calls the LLM (greedy, deterministic).
    3. Parses the JSON patch from the response.
    4. Applies the patch to `state` in-place.
    5. Returns the raw patch dict for debugging / UI display.
    """
    context = (
        f"Current state:\n{json.dumps(state.to_dict(), indent=2)}\n\n"
        f"User message: {user_message}"
    )
    raw   = _call_llm(STATE_UPDATER_SYSTEM, context, max_new_tokens=150)
    patch = _extract_json(raw)
    state.apply_patch(patch)
    return patch


def is_off_topic(patch: dict) -> bool:
    """Returns True when the LLM flagged the user message as off-topic."""
    return bool(patch.get("off_topic", False))


def wants_unlearn(user_message: str) -> bool:
    """
    Keyword-based unlearning trigger (no LLM call needed).
    Matches phrases associated with the GDPR 'right to be forgotten'.
    """
    triggers = [
        "forget me", "delete my data", "unlearn", "reset",
        "start over", "clear my history", "right to be forgotten",
    ]
    msg = user_message.lower()
    return any(t in msg for t in triggers)