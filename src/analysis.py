"""Statistical tests used in paper Sec. 3.4.

All tests are paired at the (P, D_i) self-pair level (n=1103 in the published
data). For multi-draw conditions (Sibling, AmbigQA-Shuffled, WikiText-Shuffled,
Random), the per-(P, D_i) value is the mean over draws (a real number in [0, 1])
and the McNemar binarization rule is `int(mean >= 0.5)`.

Functions:
  per_pair_means          aggregate per-(P, D_i)
  mcnemar_paired           paired McNemar with arbitrary alternative
  wilcoxon_paired          paired Wilcoxon signed-rank with alternative
  bootstrap_diff_ci_pp     bootstrap CI on mean(y - x), in percentage points
  contingency              return the 2x2 cell counts after threshold
"""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Sequence

import numpy as np
from scipy.stats import binomtest, wilcoxon


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def per_pair_means(rows: Sequence[dict],
                   key_fn: Callable[[dict], tuple],
                   value_fn: Callable[[dict], float]) -> dict[tuple, float]:
    """Group rows by key_fn and average value_fn within each group."""
    by = defaultdict(list)
    for r in rows:
        by[key_fn(r)].append(value_fn(r))
    return {k: float(np.mean(v)) for k, v in by.items()}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def contingency(x: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    """Binarize x, y at `threshold` and return the McNemar 2x2 cells."""
    bx = (x >= threshold).astype(int); by = (y >= threshold).astype(int)
    a = int(((bx == 1) & (by == 1)).sum())
    x_kills_only = int(((bx == 0) & (by == 1)).sum())
    y_kills_only = int(((bx == 1) & (by == 0)).sum())
    d = int(((bx == 0) & (by == 0)).sum())
    return {"a_both_hit": a,
            "x_kills_only": x_kills_only,
            "y_kills_only": y_kills_only,
            "d_both_miss": d,
            "n_disc": x_kills_only + y_kills_only}


def mcnemar_paired(x: np.ndarray, y: np.ndarray,
                    alternative: str = "greater",
                    threshold: float = 0.5) -> dict:
    """Exact-binomial McNemar on (x, y) paired arrays.

    Convention: the cell tested is `x_kills_only` = (x=0, y=1) under threshold.
    `alternative="greater"` tests H1: x_kills_only > y_kills_only.
    """
    cells = contingency(x, y, threshold=threshold)
    nd = cells["n_disc"]
    if nd == 0:
        return {**cells, "p": float("nan")}
    bt = binomtest(cells["x_kills_only"], nd, 0.5, alternative=alternative)
    return {**cells, "p": float(bt.pvalue)}


def wilcoxon_paired(x: np.ndarray, y: np.ndarray, alternative: str) -> float:
    """Paired Wilcoxon signed-rank on (x, y). Returns p-value (or NaN if all-zero)."""
    try:
        return float(wilcoxon(x, y, alternative=alternative, zero_method="wilcox").pvalue)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_diff_ci_pp(x: np.ndarray, y: np.ndarray, alpha: float = 0.05,
                          n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap 95% CI on mean(y - x), reported in PERCENTAGE POINTS."""
    rng = np.random.default_rng(seed)
    d = y - x
    ds = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    lo, hi = np.quantile(ds, [alpha / 2, 1 - alpha / 2])
    return float(lo) * 100, float(hi) * 100
