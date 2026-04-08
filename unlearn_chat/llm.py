# llm.py
# ─────────────────────────────────────────────────────────────
# Loads Llama-3.2-1B-Instruct in 4-bit and exposes _call_llm().
# All other modules import _call_llm from here.

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login

from config import MODEL_ID, DEVICE


def load_model():
    """
    Log in to HuggingFace, load tokenizer + model.
    Call this once at notebook startup (Cell 2).

    device_map={"": 0} pins ALL layers to cuda:0, preventing the
    cross-device RuntimeError that occurs with device_map="auto"
    when Kaggle exposes multiple CUDA ordinals.
    """
    # ── HuggingFace login ──────────────────────────────────────
    try:
        from kaggle_secrets import UserSecretsClient
        hf_token = UserSecretsClient().get_secret("HF_TOKEN")
        login(token=hf_token, add_to_git_credential=False)
        print("✅ Logged in via Kaggle Secret.")
    except Exception:
        login()   # fallback: interactive prompt

    # ── Tokenizer ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # ── 4-bit quantisation ────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading {MODEL_ID} on {DEVICE} (4-bit) ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},   # ← pins everything to cuda:0
    )
    model.eval()
    print("✅ Model ready.\n")
    return tokenizer, model


# These are populated by the notebook's Cell 2 via load_model()
_tokenizer = None
_model     = None


def init(tokenizer, model):
    """Call this after load_model() to register the globals used by _call_llm."""
    global _tokenizer, _model
    _tokenizer = tokenizer
    _model     = model


def _call_llm(system_prompt: str, user_content: str,
              max_new_tokens: int = 256) -> str:
    """
    Single LLM call using the Llama-3 chat template.
    Inputs are explicitly moved to DEVICE (cuda:0) to match the model.
    Returns only the newly generated text (prompt stripped).
    """
    if _tokenizer is None or _model is None:
        raise RuntimeError("Call llm.init(tokenizer, model) before using _call_llm.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_content},
    ]

    encoded = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,       # returns dict with input_ids + attention_mask
    )
    input_ids      = encoded["input_ids"].to(DEVICE)
    attention_mask = encoded["attention_mask"].to(DEVICE)

    with torch.no_grad():
        output_ids = _model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,                      # greedy — deterministic JSON patches
            pad_token_id=_tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[-1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True).strip()