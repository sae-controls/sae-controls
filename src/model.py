"""Gemma-2-9B-IT loader.

Eager attention is required so forward hooks see clean residual streams (the
flash-attn / sdpa code paths do not expose intermediate residuals consistently).
"""
from __future__ import annotations

from .config import ATTN_IMPL, DTYPE_STR, MODEL_ID, device


def load_model():
    """Returns (tokenizer, model). Requires accepted access to google/gemma-2-9b-it
    on the Hugging Face account whose token is set as HF_TOKEN."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, DTYPE_STR)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        device_map=device(),
        low_cpu_mem_usage=True,
        attn_implementation=ATTN_IMPL,
    )
    model.eval()
    return tok, model
