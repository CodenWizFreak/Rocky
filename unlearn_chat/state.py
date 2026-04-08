# state.py
# ─────────────────────────────────────────────────────────────
# RA-Rec Table 2 "contract": the DialogueState dataclass.
# Persists to JSON across sessions; reset_session() is the
# conversational unlearning trigger.

import os, json, copy
from dataclasses import dataclass, field

from config import PROFILE_PATH


@dataclass
class DialogueState:
    """
    Persistent, JSON-serialisable conversation state.

    hard_constraints : non-negotiable filters passed verbatim to the
                       EEMU engine (genres, year_range, exclude_genres …).
    soft_constraints : nuanced/fuzzy preferences used to build the NL
                       retrieval query (mood, vibe, themes, pacing …).
    history          : what has been surfaced / accepted / rejected so far.
    long_term_taste  : cross-session memory that persists after reset_session().
    """
    user_id: str = "kaggle_user"

    # RA-Rec mandatory keys (Table 2)
    hard_constraints: dict = field(default_factory=dict)
    soft_constraints: dict = field(default_factory=dict)

    # Per-session history
    recommended_items: list = field(default_factory=list)
    accepted_items:    list = field(default_factory=list)
    rejected_items:    list = field(default_factory=list)

    # Cross-session taste — survives reset_session()
    long_term_taste: dict = field(default_factory=dict)

    # ── Serialisation ─────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "user_id":          self.user_id,
            "hard_constraints": copy.deepcopy(self.hard_constraints),
            "soft_constraints": copy.deepcopy(self.soft_constraints),
            "history": {
                "recommended_items": list(self.recommended_items),
                "accepted_items":    list(self.accepted_items),
                "rejected_items":    list(self.rejected_items),
            },
            "long_term_taste": copy.deepcopy(self.long_term_taste),
        }

    # ── Patch application ─────────────────────────────────────

    def apply_patch(self, patch: dict):
        """Merge an LLM-generated JSON patch into the live state."""
        for key, value in patch.items():

            if key == "hard_constraints" and isinstance(value, dict):
                for k, v in value.items():
                    if v is None:
                        self.hard_constraints.pop(k, None)
                    elif isinstance(v, list):
                        existing = self.hard_constraints.get(k, [])
                        self.hard_constraints[k] = list(set(existing + v))
                    else:
                        self.hard_constraints[k] = v

            elif key == "soft_constraints" and isinstance(value, dict):
                for k, v in value.items():
                    if v is None:
                        self.soft_constraints.pop(k, None)
                    elif isinstance(v, list):
                        existing = self.soft_constraints.get(k, [])
                        self.soft_constraints[k] = list(set(existing + v))
                    else:
                        self.soft_constraints[k] = v

            elif key == "accepted_item" and value:
                if value not in self.accepted_items:
                    self.accepted_items.append(value)
                self._update_long_term("liked", value)

            elif key == "rejected_item" and value:
                if value not in self.rejected_items:
                    self.rejected_items.append(value)
                self._update_long_term("disliked", value)

    def _update_long_term(self, sentiment: str, title: str):
        bucket = self.long_term_taste.setdefault(sentiment, [])
        if title not in bucket:
            bucket.append(title)

    # ── Unlearning trigger ────────────────────────────────────

    def reset_session(self):
        """
        Wipe per-session state.
        Long-term taste is intentionally preserved so the user doesn't
        have to re-teach their baseline preferences after a reset.
        For full GDPR-style deletion, call engine.request_unlearning()
        alongside this method.
        """
        self.hard_constraints = {}
        self.soft_constraints = {}
        self.recommended_items.clear()
        self.accepted_items.clear()
        self.rejected_items.clear()
        print("[Session reset — preferences cleared. Long-term taste kept.]")

    # ── Persistence ───────────────────────────────────────────

    def save(self, path: str = PROFILE_PATH):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"[Profile saved → {path}]")

    @classmethod
    def load(cls, path: str = PROFILE_PATH, user_id: str = "kaggle_user"):
        state = cls(user_id=user_id)
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            state.hard_constraints  = data.get("hard_constraints", {})
            state.soft_constraints  = data.get("soft_constraints", {})
            hist                    = data.get("history", {})
            state.recommended_items = hist.get("recommended_items", [])
            state.accepted_items    = hist.get("accepted_items", [])
            state.rejected_items    = hist.get("rejected_items", [])
            state.long_term_taste   = data.get("long_term_taste", {})
            print(f"[Profile loaded from {path}]")
            print(f"  Past likes: {state.long_term_taste.get('liked', [])}")
        else:
            print("[No saved profile found — starting fresh.]")
        return state