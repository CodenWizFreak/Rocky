# app.py
# ─────────────────────────────────────────────────────────────
# Kaggle notebook entry point.
# Paste each CELL block into a separate notebook cell in order.
#
# Run order:
#   Cell 1  — pip install  (then restart kernel)
#   Cell 2  — model load
#   Cell 3  — data load
#   Cell 4  — launch widget UI
# ─────────────────────────────────────────────────────────────


# ============================================================
# CELL 1 — Installs  (run once, then Kernel → Restart)
# ============================================================
# !pip install -q -U transformers bitsandbytes accelerate


# ============================================================
# CELL 2 — Model Load
# ============================================================

import sys, os
# Make sure Python can find the other modules in this directory.
# Adjust the path if you place the folder somewhere other than /kaggle/working.
MODULE_DIR = "/kaggle/working/movie_rec_brain"
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)

import llm as llm_module

tokenizer, model = llm_module.load_model()
llm_module.init(tokenizer, model)

# Quick sanity check — comment out once confirmed working
print("Sanity check ...")
print(llm_module._call_llm("Reply in one sentence.", "What is Inception about?"))


# ============================================================
# CELL 3 — Data + Engine Load
# ============================================================

from engine import load_ml1m, ML1MEngine
from state import DialogueState
from config import PROFILE_PATH

movies_df = load_ml1m()
engine    = ML1MEngine(movies_df)
state     = DialogueState.load(PROFILE_PATH)


# ============================================================
# CELL 4 — Widget Chat UI
# ============================================================

import ipywidgets as widgets
from IPython.display import display

from classifier import classify_and_update, is_off_topic, wants_unlearn
from retrieval  import generate_eemu_query
from explainer  import generate_explanation

# ── Widgets ──────────────────────────────────────────────────
text_box = widgets.Text(
    placeholder="Tell me what you're in the mood for...",
    layout=widgets.Layout(width="65%"),
)
send_btn  = widgets.Button(description="Send ▶",         button_style="primary")
reset_btn = widgets.Button(description="🔄 Reset Session", button_style="warning")
save_btn  = widgets.Button(description="💾 Save Taste",    button_style="success")

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
status_bar = widgets.HTML(value="<i>Ready.</i>")


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

    # ── Unlearning trigger ────────────────────────────────────
    if wants_unlearn(user_msg):
        state.reset_session()
        engine.request_unlearning(state.user_id)
        _chat("Assistant",
              "Done! I've cleared your session preferences. "
              "Your long-term taste is kept. What kind of movie are you in the mood for?")
        _status("Session reset.")
        return

    # ── Module 2: classify & update state ────────────────────
    _status("Updating state ...")
    try:
        patch = classify_and_update(user_msg, state)
    except Exception as e:
        with chat_output:
            print(f"  ⚠️  [State update error]: {e}")
        _status("Error in state update.")
        return

    with chat_output:
        print(f"  [state patch]: {patch}")   # remove once stable

    if is_off_topic(patch):
        _chat("Assistant",
              "I'm here for movie recommendations only! "
              "What kind of film are you in the mood for?")
        _status("Ready.")
        return

    # ── Module 3: query + retrieve ───────────────────────────
    _status("Fetching recommendations ...")
    try:
        query  = generate_eemu_query(state)
        movies = engine.get_recommendations(query)
    except Exception as e:
        with chat_output:
            print(f"  ⚠️  [Engine error]: {e}")
        _status("Error in engine.")
        return

    for m in movies:
        if m["id"] not in state.recommended_items:
            state.recommended_items.append(m["id"])

    if not movies:
        _chat("Assistant",
              "No movies matched those filters. "
              "Try loosening the year range or genre constraints.")
        _status("Ready.")
        return

    # ── Module 4: explain ────────────────────────────────────
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