"""Gemma Scope JumpReLU SAE — raw NPZ loader.

Math (verbatim from Lieberum et al. 2024, arXiv:2408.05147):

    z      = relu(x @ W_enc + b_enc) * (preact > threshold)
    x_hat  = z @ W_dec + b_dec

`load_sae(layer)` returns `(sae, meta)` where `meta` includes the resolved repo,
subpath, local cache path, SHA-256, and the dimensions. Always verify
`meta["sha256"] == config.SAE_CHECKPOINT_SHA256` at the start of any pipeline
script that depends on numerical equivalence with the paper.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .config import LAYER, SAE_L0_SEL, SAE_REPOS, SAE_WIDTH, device


def _torch():
    """Local torch import so this module loads without GPU on a laptop."""
    import torch
    return torch


class JumpReLUSAE:
    """JumpReLU SAE. Plain Python (not nn.Module) so import-time has no torch dep."""

    def __init__(self, d_model: int, d_sae: int, device_: str = "cuda", dtype=None):
        torch = _torch()
        self.d_model = d_model
        self.d_sae = d_sae
        self.device = device_
        self.dtype = dtype or torch.bfloat16
        self.W_enc     = torch.zeros(d_model, d_sae, device=device_, dtype=self.dtype)
        self.W_dec     = torch.zeros(d_sae, d_model, device=device_, dtype=self.dtype)
        self.threshold = torch.zeros(d_sae,           device=device_, dtype=self.dtype)
        self.b_enc     = torch.zeros(d_sae,           device=device_, dtype=self.dtype)
        self.b_dec     = torch.zeros(d_model,         device=device_, dtype=self.dtype)

    def encode(self, x):
        torch = _torch()
        with torch.no_grad():
            pre = x @ self.W_enc + self.b_enc
            return torch.relu(pre) * (pre > self.threshold).to(pre.dtype)

    def decode(self, z):
        with _torch().no_grad():
            return z @ self.W_dec + self.b_dec

    def reconstruct(self, x):
        return self.decode(self.encode(x))


def _find_sae_subpath(layer: int, width: str, l0_sel: str, repo: str) -> str | None:
    """Resolve the repo's params.npz for layer/width.
    Preference: exact l0_sel match > 'canonical' folder > any 'average_l0_*'."""
    from huggingface_hub import list_repo_files
    try:
        files = list_repo_files(repo)
    except Exception:
        return None
    prefix = f"layer_{layer}/width_{width}/"
    matches = [f for f in files if f.startswith(prefix) and f.endswith("params.npz")]
    if not matches: return None

    def score(f: str) -> tuple[int, str]:
        if f"/{l0_sel}/" in f: return (0, f)
        if "/canonical/" in f:  return (1, f)
        return (2, f)

    matches.sort(key=score)
    return matches[0]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sae(layer: int = LAYER,
             width: str = SAE_WIDTH,
             l0_sel: str = SAE_L0_SEL,
             device_: str | None = None) -> tuple[JumpReLUSAE, dict]:
    """Download (or HF-cache) the Gemma Scope JumpReLU SAE; return (sae, meta).

    meta = {repo, subpath, local_path, sha256, d_model, d_sae}
    """
    from huggingface_hub import hf_hub_download
    torch = _torch()

    dev = device_ or device()
    last_err = None; tried = []
    for repo in SAE_REPOS:
        sub = _find_sae_subpath(layer, width, l0_sel, repo)
        if sub is None:
            tried.append(f"{repo} (no layer_{layer}/width_{width} files)")
            continue
        try:
            path = Path(hf_hub_download(repo, sub))
            print(f"[load_sae L{layer}] using {repo}/{sub}")
            params = np.load(path)
            d_model, d_sae = params["W_enc"].shape
            sae = JumpReLUSAE(d_model, d_sae, device_=dev)
            sae.W_enc     = torch.from_numpy(params["W_enc"]).to(device=dev, dtype=sae.dtype)
            sae.W_dec     = torch.from_numpy(params["W_dec"]).to(device=dev, dtype=sae.dtype)
            sae.threshold = torch.from_numpy(params["threshold"]).to(device=dev, dtype=sae.dtype)
            sae.b_enc     = torch.from_numpy(params["b_enc"]).to(device=dev, dtype=sae.dtype)
            sae.b_dec     = torch.from_numpy(params["b_dec"]).to(device=dev, dtype=sae.dtype)
            return sae, {
                "repo": repo,
                "subpath": sub,
                "local_path": str(path),
                "sha256": _sha256(path),
                "d_model": int(d_model),
                "d_sae": int(d_sae),
            }
        except Exception as e:
            last_err = e
            tried.append(f"{repo}/{sub} ({type(e).__name__})")
            continue

    raise RuntimeError(
        f"Could not load SAE for layer {layer} (width={width}). Tried: "
        + "; ".join(tried)
        + (f". Last error: {last_err}" if last_err else "")
    )
