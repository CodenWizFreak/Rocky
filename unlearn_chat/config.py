# config.py
# ─────────────────────────────────────────────────────────────
# Central config — change paths/constants here, nothing else needs editing.

import torch

# ── Model ─────────────────────────────────────────────────────
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"

# ── Device ────────────────────────────────────────────────────
# Pinned to cuda:0 to avoid multi-GPU device-mismatch errors.
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ── Paths ─────────────────────────────────────────────────────
ML1M_PATH    = "/kaggle/input/datasets/ananyodasgupta/movielens-1m/ml-1m"
PROFILE_PATH = "/kaggle/working/user_profile.json"

# ── Engine defaults ───────────────────────────────────────────
DEFAULT_TOP_K = 5