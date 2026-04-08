# Movie Recommendation Brain — RA-Rec + EEMU

Conversational movie recommendation chatbot implementing the **RA-Rec** framework
(SIGIR 2024) as the dialogue front-end, backed by the **EEMU** exact-unlearning
engine (PoPETs 2025) and the **MovieLens-1M** dataset.

## Directory Structure

```
movie_rec_brain/
├── config.py       # Paths, device, constants — edit this for your environment
├── llm.py          # Model loading (4-bit Llama-3.2-1B-Instruct) + _call_llm()
├── state.py        # DialogueState dataclass (RA-Rec Table 2 schema)
├── prompts.py      # All LLM system prompts
├── classifier.py   # Module 2: intent classifier + JSON state updater
├── retrieval.py    # Module 3: DialogueState → EEMU query payload
├── engine.py       # ML1M engine (swap get_recommendations() for real EEMU)
├── explainer.py    # Module 4: explainable response generator
├── app.py          # Kaggle notebook entry point (4 cells + widget UI)
└── requirements.txt
```

## How to Use on Kaggle

1. Upload this folder to `/kaggle/working/movie_rec_brain/`.
2. Open a new notebook and paste each `CELL` block from `app.py` into
   a separate notebook cell in order.
3. Run **Cell 1** (`pip install`), then **Kernel → Restart**.
4. Run **Cells 2, 3, 4** in order.
5. The widget UI appears at the bottom of Cell 4.

## Wiring in the Real EEMU Backend

In `engine.py`, replace `ML1MEngine.get_recommendations()` with a call
to your partner's `inference()` function from `dist_eval`. The query
payload contract (defined in `retrieval.py`) stays identical:

```python
{
    "user_id":        str,
    "nl_query":       str,   # NL string from soft constraints
    "genres":         list,
    "exclude_genres": list,
    "year_range":     list | None,
    "rating_min":     float | None,
    "exclude_ids":    list,  # rejected movie IDs
    "top_k":          int,
}
```

`request_unlearning(user_id)` in `engine.py` is the stub to call your
partner's SISA shard-retrain routine.

## References

- **RA-Rec**: Kemper et al., *Retrieval-Augmented Conversational Recommendation
  with Prompt-based Semi-Structured NL State Tracking*, SIGIR 2024.
- **EEMU**: Alshabanah et al., *Meta-Learn to Unlearn: Enhanced Exact Machine
  Unlearning in Recommendation Systems*, PoPETs 2025.