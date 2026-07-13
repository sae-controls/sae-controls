"""Position-mode classification — which of the 2021 disambiguation-derived
features are dominantly paragraph-initial on WikiText (paper Sec. 3.5)?

A feature is `position_suspect` iff:

    n_nonzero ≥ POSITION_NONZERO_FLOOR
        AND
    pct_at_pos0 ≥ POSITION_PCT_THRESHOLD

where the statistics are computed across all (paragraph, token-position) tuples
on the 800-paragraph WikiText subset:

    n_nonzero    = count of (paragraph, position) with post-JumpReLU activation > 0
    n_at_pos0    = count of paragraphs where position-0 has activation > 0
    pct_at_pos0  = n_at_pos0 / n_nonzero

The threshold (0.80) is calibrated to flag the three known polysemantic
features identified during feature characterization (7187, 9825, 15382) — these have
pct_at_pos0 = 0.82 / 0.82 / 0.86 respectively. A stricter 0.90 threshold
would miss all three.
"""
from __future__ import annotations

from .config import POSITION_NONZERO_FLOOR, POSITION_PCT_THRESHOLD


def is_position_suspect(n_nonzero: int, pct_at_pos0: float,
                         floor: int = POSITION_NONZERO_FLOOR,
                         threshold: float = POSITION_PCT_THRESHOLD) -> bool:
    return (n_nonzero >= floor) and (pct_at_pos0 >= threshold)


def partition_top_k(top_k_features: list[int], suspect_set: set[int]) -> tuple[list[int], list[int]]:
    """Split a top-K list into (position_features, content_features)."""
    pos = [f for f in top_k_features if int(f) in suspect_set]
    cont = [f for f in top_k_features if int(f) not in suspect_set]
    return pos, cont
