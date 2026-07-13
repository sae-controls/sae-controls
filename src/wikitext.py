"""WikiText-2 raw test split — used as the OOD distribution for:
  * the WikiText-shuffle ablation control (paper Sec. 3.2.5)
  * per-feature max-activating contexts (Sec. 4)
  * per-feature position-mode statistics (Sec. 5)

The 800-paragraph subset is selected deterministically: walk the raw test split
in order, drop section headers, keep entries with WIKITEXT_MIN_TOKENS ≤ tokens
≤ WIKITEXT_MAX_TOKENS, take the first WIKITEXT_N_PARAGRAPHS.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

from .config import (
    WIKITEXT_CONFIG, WIKITEXT_DATASET, WIKITEXT_MAX_TOKENS,
    WIKITEXT_MIN_TOKENS, WIKITEXT_N_PARAGRAPHS, WIKITEXT_SPLIT,
)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def load_paragraphs(tokenizer, n: int = WIKITEXT_N_PARAGRAPHS,
                    min_tokens: int = WIKITEXT_MIN_TOKENS,
                    max_tokens: int = WIKITEXT_MAX_TOKENS) -> list[str]:
    """Return the deterministic n-paragraph subset of the WikiText-2 raw test split."""
    from datasets import load_dataset
    ds = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split=WIKITEXT_SPLIT)
    paragraphs: list[str] = []
    for d in ds:
        line = d["text"]
        if not line.strip(): continue
        if line.lstrip().startswith("="): continue          # skip section headers
        ids = tokenizer.encode(line, add_special_tokens=False)
        if min_tokens <= len(ids) <= max_tokens:
            paragraphs.append(line)
        if len(paragraphs) >= n:
            break
    return paragraphs


def paragraph_metadata(paragraphs: Iterable[str], token_lens: Iterable[int]) -> list[dict]:
    """Per-paragraph lightweight metadata — id, length, content hash, snippet."""
    out = []
    for i, (p, n) in enumerate(zip(paragraphs, token_lens)):
        out.append({
            "idx": i,
            "n_tokens_no_special": int(n),
            "sha256_first16": sha256_text(p)[:16],
            "first_60_chars": p.strip()[:60],
        })
    return out
