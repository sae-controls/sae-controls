"""Forward-hook helpers — the only mechanism for residual capture and ablation
used in the paper.

Five helpers:
  * `capture_last_token_residual` — for self-pair/disambig prompts (paper Sec. 3.1, Sec. 2)
  * `capture_all_position_residuals` — for WikiText paragraphs (Sec. 4 characterization, Sec. 5 polysemy)
  * `forward_with_ablation` — error-preserving SAE feature splice at the target layer's last token (Sec. 2)
  * `baseline_top_logits` / `hit_at_k` — read top-k decoded next-token IDs
  * `greedy_decode` — used for slot detection in Sec. 1

Error-preserving splice formula (paper Sec. 3.3):

    new_h_last = sae.decode(z) + (h_last - sae.decode(sae.encode(h_last)))
               = h_last - sae.decode(z_K)             # since decode is linear
                                                       # and z_K = encoded - z

with feature_ids contributing to z_K (the ablated subset). `feature_ids = []`
yields h_last unchanged (an identity).
"""
from __future__ import annotations

import numpy as np

from .config import TOP_LOGITS_K


def _topk_dict(logits, k):
    import torch
    topk = torch.topk(logits, k=k)
    return {
        "top_ids":    topk.indices.cpu().numpy().astype(np.int32).tolist(),
        "top_logits": topk.values.cpu().numpy().astype(np.float32).tolist(),
    }


# ---------------------------------------------------------------------------
# Residual capture
# ---------------------------------------------------------------------------

def capture_last_token_residual(model, tokenizer, prompt: str, layer: int):
    """Last-prompt-token residual at `model.model.layers[layer]`.
    Returns CPU float32 numpy array of shape (d_model,)."""
    import torch
    dev = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    buf = {}

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        buf["x"] = h[:, -1, :].detach().clone()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    return buf["x"][0].to(dtype=torch.float32, device="cpu").numpy()


def capture_all_position_residuals(model, tokenizer, text: str, layer: int,
                                    max_len: int):
    """All-position residuals at `model.model.layers[layer]`. Used for WikiText.

    Returns:
      input_ids: (n_tokens,) np.int64 — for snippet decoding
      residuals: (n_tokens, d_model) torch tensor still on the model's device

    Caller is responsible for moving the residuals off-device after use.
    """
    import torch
    dev = next(model.parameters()).device
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = enc.input_ids.to(dev)
    captured = {}

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["x"] = h[0, :, :].detach().clone()

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    return input_ids[0].cpu().numpy().astype(np.int64), captured["x"]


# ---------------------------------------------------------------------------
# Forward variants
# ---------------------------------------------------------------------------

def baseline_top_logits(model, tokenizer, prompt: str, k: int = TOP_LOGITS_K) -> dict:
    """No intervention — read top-k logits at the next-token position."""
    import torch
    dev = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    return _topk_dict(out.logits[0, -1, :].float(), k)


def forward_with_ablation(model, tokenizer, prompt: str, layer: int, sae,
                          feature_ids, k: int = TOP_LOGITS_K) -> dict:
    """Error-preserving SAE feature ablation at `layer`, last-token only."""
    import torch
    dev = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    fids = list(feature_ids) if feature_ids else []

    def hook(_mod, _inp, out):
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        last = h[:, -1, :].to(sae.W_enc.dtype)
        z = sae.encode(last)
        if fids:
            z[..., fids] = 0.0
        recon = sae.decode(z)
        full_recon = sae.decode(sae.encode(last))
        err = last - full_recon
        h[:, -1, :] = (recon + err).to(h.dtype)
        return (h,) + out[1:] if is_tuple else h

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    return _topk_dict(out.logits[0, -1, :].float(), k)


def greedy_decode(model, tokenizer, prompt: str, max_new: int = 15) -> str:
    """Generate up to `max_new` tokens greedily (no sampling, no beam) for slot
    detection (paper Sec. 3.1.4)."""
    import torch
    dev = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            num_beams=1,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def forward_with_ablation_then_generate(model, tokenizer, prompt: str, layer: int,
                                         sae, feature_ids, max_new: int = 15) -> str:
    """Greedy-decode with SAE ablation applied ONLY on the prompt forward.
    Used by the generation-flip analysis.

    The hook checks `inp[0].shape[1]` to distinguish the initial prompt forward
    (multi-token) from subsequent KV-cached single-token generation steps
    (length-1). Only the multi-token forward gets ablated; later steps are
    pass-through.

    With `feature_ids = []`, the hook is a true no-op (returns `out` unchanged
    on the multi-token forward) so generation matches `greedy_decode`
    bit-identically; this is the basis for the empty-fid sanity check.
    """
    import torch
    dev = next(model.parameters()).device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
    fids = list(feature_ids) if feature_ids else []

    def hook(_mod, inp, out):
        # Skip the entire hook on KV-cached single-token generation steps
        if inp[0].shape[1] <= 1:
            return out
        # No-op when no features to ablate (used by the sanity check)
        if not fids:
            return out
        is_tuple = isinstance(out, tuple)
        h = out[0] if is_tuple else out
        last = h[:, -1, :].to(sae.W_enc.dtype)
        z = sae.encode(last)
        z[..., fids] = 0.0
        recon = sae.decode(z)
        full_recon = sae.decode(sae.encode(last))
        err = last - full_recon
        h[:, -1, :] = (recon + err).to(h.dtype)
        return (h,) + out[1:] if is_tuple else h

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    finally:
        handle.remove()
    new_ids = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Hit@k scoring
# ---------------------------------------------------------------------------

def hit_at_k(top_ids: list[int], targets: set[int], k: int) -> int:
    """1 iff any of top_ids[:k] is in targets."""
    return int(any(t in targets for t in top_ids[:k]))
