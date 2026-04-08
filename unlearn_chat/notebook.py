# ============================================================
# CELL 1 — Installs  (run this cell first, then restart kernel)
# ============================================================
# !pip install -q -U transformers bitsandbytes accelerate
# After it finishes: Kernel → Restart, then run from Cell 2.

# ============================================================
# FIXED CELL 2 — Imports + Model Load
# DROP-IN REPLACEMENT for the existing Cell 2
#
# Root cause of the RuntimeError:
#   device_map="auto" splits the model across cuda:0 AND cuda:1.
#   The input tensors are only sent to cuda:0, so the embedding
#   lookup on cuda:1 throws:
#     "Expected all tensors on the same device, but index is on
#      cuda:0 ... other tensors on cuda:1"
#
# Fix: pin EVERYTHING to a single GPU with device_map={"": 0}
# ============================================================
 
import os, json, re, copy, torch
import pandas as pd
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
 
# ── Explicit single-GPU setup ─────────────────────────────────
# Force all model shards onto cuda:0 only.
# Kaggle T4 has 1 GPU visible as cuda:0; even if Kaggle reports 2,
# pinning to {"": 0} prevents the cross-device index mismatch.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"GPU memory available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
 
# ── 4-bit quantisation (Llama-3.2-1B-Instruct fits in ~1.2 GB) ─
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
 
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
 
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
try:
    secrets = UserSecretsClient()
    HF_TOKEN = secrets.get_secret("HF_TOKEN")
    login(token=HF_TOKEN, add_to_git_credential=False)
    print("✅ Logged in via Kaggle Secret.")
except Exception:
    login()   # fallback: interactive prompt
 
print(f"\nLoading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
 
llm_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    # ↓ THE KEY FIX: "" means "default device" → maps all layers to cuda:0
    device_map={"": 0},
)
llm_model.eval()
print("✅ Model ready.\n")
 
 
# ============================================================
# FIXED CELL 3 — LLM Helper
# DROP-IN REPLACEMENT for the existing Cell 3
#
# Additional fix: input_ids and attention_mask are explicitly
# moved to DEVICE (cuda:0) to match where the model lives.
# ============================================================
 
def _call_llm(system_prompt: str, user_content: str,
              max_new_tokens: int = 256) -> str:
    """
    Calls Llama-3.2-1B-Instruct using the chat template.
    Both the model and inputs are pinned to cuda:0.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]
 
    encoded = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,       # gives us a proper dict, not a bare tensor
    )
 
    # Explicitly move BOTH tensors to cuda:0
    input_ids      = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)
 
    with torch.no_grad():
        output_ids = llm_model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,                        # greedy = deterministic JSON
            pad_token_id=tokenizer.eos_token_id,
        )
 
    # Return only the newly generated tokens (strip the prompt)
    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
 
 
# ── Sanity check ─────────────────────────────────────────────
print("Sanity check ...")
print(_call_llm("Reply in one sentence.", "What is Inception about?"))
# Expected: something like "Inception is a sci-fi thriller about dream heists."


# ============================================================
# CELL 4 — Dialogue State  (RA-Rec Table 2 contract)
# ============================================================

PROFILE_PATH = "/kaggle/working/user_profile.json"

@dataclass
class DialogueState:
    """
    Persistent session state.  Mirrors RA-Rec Table 2:
      hard_constraints — facts the engine filters on
      soft_constraints — mood/vibe used for NL query embedding
      history          — what has been recommended/accepted/rejected
    Also accumulates long_term_taste across sessions via JSON file.
    """
    user_id: str = "kaggle_user"

    # RA-Rec mandatory keys
    hard_constraints: dict = field(default_factory=dict)
    soft_constraints: dict = field(default_factory=dict)

    # Per-session history
    recommended_items: list = field(default_factory=list)
    accepted_items:    list = field(default_factory=list)
    rejected_items:    list = field(default_factory=list)

    # Cross-session taste memory (genres/vibes the user keeps liking)
    long_term_taste: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "hard_constraints": copy.deepcopy(self.hard_constraints),
            "soft_constraints": copy.deepcopy(self.soft_constraints),
            "history": {
                "recommended_items": list(self.recommended_items),
                "accepted_items":    list(self.accepted_items),
                "rejected_items":    list(self.rejected_items),
            },
            "long_term_taste": copy.deepcopy(self.long_term_taste),
        }

    def apply_patch(self, patch: dict):
        """Merge LLM JSON patch into live state."""
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
                # Feed acceptance into long-term taste
                self._update_long_term("liked", value)

            elif key == "rejected_item" and value:
                if value not in self.rejected_items:
                    self.rejected_items.append(value)
                self._update_long_term("disliked", value)

    def _update_long_term(self, sentiment: str, title: str):
        bucket = self.long_term_taste.setdefault(sentiment, [])
        if title not in bucket:
            bucket.append(title)

    def reset_session(self):
        """Wipe session state (conversational unlearning trigger)."""
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
            hist = data.get("history", {})
            state.recommended_items = hist.get("recommended_items", [])
            state.accepted_items    = hist.get("accepted_items", [])
            state.rejected_items    = hist.get("rejected_items", [])
            state.long_term_taste   = data.get("long_term_taste", {})
            print(f"[Profile loaded from {path}]")
            print(f"  Past likes: {state.long_term_taste.get('liked', [])}")
        else:
            print("[No saved profile found — starting fresh.]")
        return state


# ============================================================
# CELL 5 — System Prompts
# ============================================================

# Prompt 1: JSON state patcher
STATE_UPDATER_SYSTEM = """\
You are a strict JSON dialogue-state tracker for a MOVIE recommendation chatbot.
Your ONLY job: read the user message + current state, output a JSON patch.

RULES — follow exactly:
1. Output ONLY valid JSON. No prose, no markdown, no code fences.
2. Only include keys that actually changed.
3. Hard constraints (filterable facts):
   genres         → list of strings  e.g. ["Thriller", "Crime"]
   exclude_genres → list of strings  e.g. ["Comedy"]
   year_range     → list of 2 ints   e.g. [1990, 1999]
   rating_min     → float            e.g. 4.0
   director       → string
   actor          → string
4. Soft constraints (mood / vibe / feel):
   mood, vibe, themes, pacing, tone  → list of strings
5. Acceptance  → {"accepted_item": "<exact movie title>"}
6. Rejection   → {"rejected_item": "<exact movie title>"}
7. No change   → {}
8. Off-topic   → {"off_topic": true}
9. Do NOT recommend movies. Do NOT explain. PATCH ONLY.

EXAMPLES:
User: "I want a dark 90s thriller"
→ {"hard_constraints": {"genres": ["Thriller"], "year_range": [1990, 1999]}, "soft_constraints": {"mood": ["dark"]}}

User: "rainy city melancholic vibe"
→ {"soft_constraints": {"vibe": ["rainy city"], "mood": ["melancholic"]}}

User: "no more comedies please"
→ {"hard_constraints": {"exclude_genres": ["Comedy"]}}

User: "yes the first one is great"
→ {"accepted_item": "Se7en"}

User: "not that one"
→ {"rejected_item": "The Matrix"}

User: "what is the capital of France"
→ {"off_topic": true}
"""

# Prompt 2: Natural language explainer
RESPONSE_GEN_SYSTEM = """\
You are a warm, knowledgeable movie recommendation assistant.
You ONLY discuss movies. For anything else, politely redirect.
Keep responses concise — 3 to 5 sentences max.
Always tie each recommendation back to what the user asked for.
Format each pick as:
  "🎬 [Title] ([Year]) — [one sentence why it matches their request]."
"""


# ============================================================
# CELL 6 — Intent Classifier + State Updater  (Module 2)
# ============================================================

def _extract_json(raw: str) -> dict:
    """Robustly pull the first {...} block out of LLM output."""
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
    Calls the LLM with the STATE_UPDATER_SYSTEM prompt.
    Returns the JSON patch and applies it to state in-place.
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
    return bool(patch.get("off_topic", False))


# ============================================================
# CELL 7 — Retrieval Bridge  (Module 3)
# ============================================================

def generate_eemu_query(state: DialogueState) -> dict:
    """
    Converts DialogueState → query payload for the ML1M engine.
    Soft constraints become a NL query string (would be embedded
    by the Two-Tower item tower in the real EEMU backend).
    Long-term taste is appended to the NL query for preference continuity.
    """
    hc = state.hard_constraints
    sc = state.soft_constraints

    # Flatten soft constraints into NL
    soft_parts = []
    for k, v in sc.items():
        soft_parts.append(f"{k}: {', '.join(v) if isinstance(v, list) else v}")

    # Append long-term liked genres/vibes so past taste influences results
    liked = state.long_term_taste.get("liked", [])
    if liked:
        soft_parts.append(f"previously enjoyed: {', '.join(liked[:5])}")

    nl_query = "; ".join(soft_parts) if soft_parts else "good movie"

    return {
        "user_id":        state.user_id,
        "nl_query":       nl_query,
        "genres":         hc.get("genres", []),
        "exclude_genres": hc.get("exclude_genres", []),
        "year_range":     hc.get("year_range", None),
        "rating_min":     hc.get("rating_min", None),
        "director":       hc.get("director", None),
        "actor":          hc.get("actor", None),
        "exclude_ids":    list(state.rejected_items),
        "top_k":          5,
    }


# ============================================================
# CELL 8 — ML1M Engine  (Module 3 backend)
# ============================================================

ML1M_PATH = "/kaggle/input/datasets/ananyodasgupta/movielens-1m/ml-1m"

# Load movies
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
    movies_df["title"].str.replace(r"\s*\(\d{4}\)", "", regex=True).str.strip()
)
movies_df["genres_list"] = movies_df["genres"].str.split("|")

# Load ratings → average per movie
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


class ML1MEngine:
    """
    MovieLens-1M backed recommendation engine.
    Implements the same interface as MockEEMUEngine so the rest of
    the pipeline is unchanged.

    Ranking mimics EEMU Two-Tower scoring:
      score = keyword_overlap × 10 + avg_rating
    In production, replace get_recommendations() with a call to
    your friend's EEMU inference function — the query_payload
    contract stays the same.
    """

    def __init__(self, df: pd.DataFrame):
        self.catalogue = [
            {
                "id":          str(row["movie_id"]),
                "title":       row["title_clean"],
                "year":        int(row["year"]),
                "genres":      row["genres_list"],
                # ML1M has no plot text — we synthesise one from metadata
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

        # Tokenise NL query — skip stop-words shorter than 4 chars
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
        Stub — replace with your friend's EEMU shard-retrain call.
        EEMU guarantees: only the shard containing this user's
        interactions is retrained, not the whole model.
        """
        print(f"[ML1MEngine] Unlearning queued for '{user_id}'.")
        print("  In production: find user's SISA shard → remove interactions → retrain shard.")


engine = ML1MEngine(movies_df)


# ============================================================
# CELL 9 — Explainable Response Generator  (Module 4)
# ============================================================

def generate_explanation(movies: list[dict], state: DialogueState) -> str:
    """
    Asks the LLM to explain why each returned movie matches the
    user's hard + soft constraints, following RA-Rec's explanation
    style (Section 3.4 of the paper).
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


# ============================================================
# CELL 10 — Widget Chat UI
# ============================================================

import ipywidgets as widgets
from IPython.display import display

# Load profile if it exists, otherwise start fresh
state = DialogueState.load(PROFILE_PATH)

# ── Widgets ──────────────────────────────────────────────────
text_box    = widgets.Text(
    placeholder="Tell me what you're in the mood for...",
    layout=widgets.Layout(width="65%"),
)
send_btn    = widgets.Button(description="Send ▶",  button_style="primary")
reset_btn   = widgets.Button(description="🔄 Reset Session", button_style="warning")
save_btn    = widgets.Button(description="💾 Save Taste",    button_style="success")
chat_output = widgets.Output(
    layout=widgets.Layout(
        border="1px solid #444",
        min_height="350px",
        max_height="550px",
        padding="12px",
        width="95%",
        overflow_y="auto",
    )
)
status_bar  = widgets.HTML(value="<i>Ready.</i>")

def _chat(speaker: str, msg: str):
    icon = "🧑" if speaker == "You" else "🎬"
    with chat_output:
        print(f"\n{icon} {speaker}:\n  {msg}")

def _status(msg: str):
    status_bar.value = f"<i>{msg}</i>"

# ── Send handler ─────────────────────────────────────────────
def on_send(b):
    user_msg = text_box.value.strip()
    if not user_msg:
        return
    text_box.value = ""
    _chat("You", user_msg)
    _status("Updating state ...")

    # Step 1 — classify & update state
    try:
        patch = classify_and_update(user_msg, state)
    except Exception as e:
        with chat_output:
            print(f"  ⚠️  [State update error]: {e}")
        _status("Error in state update.")
        return

    with chat_output:
        print(f"  [state patch]: {patch}")  # debug — remove once stable

    if is_off_topic(patch):
        _chat("Assistant",
              "I'm here for movie recommendations only! "
              "What kind of film are you in the mood for?")
        _status("Ready.")
        return

    # Step 2 — build query & get recommendations
    _status("Fetching recommendations ...")
    try:
        query  = generate_eemu_query(state)
        movies = engine.get_recommendations(query)
    except Exception as e:
        with chat_output:
            print(f"  ⚠️  [Engine error]: {e}")
        _status("Error in engine.")
        return

    # Track what we showed
    for m in movies:
        if m["id"] not in state.recommended_items:
            state.recommended_items.append(m["id"])

    if not movies:
        _chat("Assistant",
              "No movies matched those filters. "
              "Try loosening the year range or genre constraints.")
        _status("Ready.")
        return

    # Step 3 — generate explanation
    _status("Generating response ...")
    try:
        reply = generate_explanation(movies, state)
    except Exception as e:
        with chat_output:
            print(f"  ⚠️  [Explanation error]: {e}")
        _status("Error in explanation.")
        return

    _chat("Assistant", reply)
    _status("Ready.")

# ── Reset handler ─────────────────────────────────────────────
def on_reset(b):
    state.reset_session()
    engine.request_unlearning(state.user_id)
    with chat_output:
        print("\n🔄 Session reset. Preferences cleared (long-term taste kept).\n")
    _status("Session reset.")

# ── Save handler ──────────────────────────────────────────────
def on_save(b):
    state.save(PROFILE_PATH)
    _status(f"Taste saved to {PROFILE_PATH}")

send_btn.on_click(on_send)
reset_btn.on_click(on_reset)
save_btn.on_click(on_save)

# Also send on Enter key in text box
def on_enter(change):
    if change["name"] == "value" and change["new"].endswith("\n"):
        text_box.value = text_box.value.strip()
        on_send(None)

# ── Layout ───────────────────────────────────────────────────
display(widgets.VBox([
    widgets.HTML("<h3>🎬 Movie Recommender — ML1M + RA-Rec + EEMU</h3>"),
    chat_output,
    widgets.HBox([text_box, send_btn]),
    widgets.HBox([reset_btn, save_btn]),
    status_bar,
]))

_chat("Assistant",
      "Hi! I'm your personal movie assistant powered by MovieLens-1M. "
      "Tell me what you're in the mood for — genre, decade, vibe, anything. "
      "For example: 'a dark 90s thriller' or 'something light and funny from the 2000s'.")
