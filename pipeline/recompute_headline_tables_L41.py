"""Recompute the L41 headline tables (paper Tables 1 and 2) directly from the
canonical per-pair artifact.

Source
------
All six conditions derive from ``artifacts/layer_bookends/L41/results_main.json``
(run_meta layer=41), the same artifact every other published L41 number derives
from (``reference_layer_analysis.py`` reads it for T3, Targeted, Sibling, and
the unique/shared decomposition).

This script reaggregates all six conditions per (P, D_i) self-pair using the
project's own ``src.analysis`` helpers (same seed, same bootstrap, same paired
tests as the rest of the paper) and emits Tables 1 and 2.

Key empirical facts established here (assert-checked below):
  * WikiText-shuffled and Random ablations leave hit@1 unchanged on ALL 1103
    pairs at L41 (decoded contribution of inactive features ~ 0 at the last
    residual layer), so their Delta is exactly 0.00 pp with a degenerate CI.
  * T3 (Sibling vs WikiText) = +9.05 pp reproduces the published headline
    exactly, because WikiText == Baseline per pair.

Consistency ties (exact by construction):
  * Random == Baseline per pair  =>  T1 (Targeted vs Random) is the negation of
    the Targeted-vs-Baseline contrast (Table 1 Targeted row).
  * WikiText == Baseline per pair =>  T4 (Shuffled-AmbigQA vs WikiText) is the
    negation of the Shuffled-AmbigQA-vs-Baseline contrast (Table 1 row).

Writes: artifacts/reference_layer/corrected_headline_tables_L41.json
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from pathlib import Path

import numpy as np

from src.analysis import (
    bootstrap_diff_ci_pp, mcnemar_paired, per_pair_means, wilcoxon_paired,
)
from src.config import ARTIFACTS_DIR

RESULTS = ARTIFACTS_DIR / "layer_bookends" / "L41" / "results_main.json"
OUT = ARTIFACTS_DIR / "reference_layer" / "corrected_headline_tables_L41.json"


def main() -> None:
    r = json.load(open(RESULTS))
    kt = lambda x: (x["pair_id"], x["target_idx"])
    ka = lambda x: (x["pair_id"], x["ablate_idx"])  # random rows key on ablate_idx

    base = per_pair_means(r["self_rows"], kt, lambda x: x["base_hit1"])
    targ = per_pair_means(r["self_rows"], kt, lambda x: x["ablate_hit1"])
    sib = per_pair_means(r["cross_rows"], kt, lambda x: x["ablate_hit1"])
    amb = per_pair_means(r["ambigqa_shuffled_rows"], kt, lambda x: x["shuffled_hit1"])
    wt = per_pair_means(r["wikitext_shuffled_rows"], kt, lambda x: x["wt_shuffled_hit1"])
    rnd = per_pair_means(r["random_rows"], ka, lambda x: x["random_hit1"])

    keys = sorted(base)
    B = np.array([base[k] for k in keys])
    arr = lambda d: np.array([d.get(k, base[k]) for k in keys])
    T, Si, Am, Wt, Rn = arr(targ), arr(sib), arr(amb), arr(wt), arr(rnd)

    # --- Hard facts: WikiText and Random are identical to baseline on all pairs ---
    assert int((Wt == B).sum()) == len(B), "WikiText must equal baseline on all pairs"
    assert int((Rn == B).sum()) == len(B), "Random must equal baseline on all pairs"

    def cond_row(A):
        d = (A.mean() - B.mean()) * 100.0
        lo, hi = bootstrap_diff_ci_pp(B, A)  # CI on (A - B)
        return {"hit1": float(A.mean()), "delta_pp": float(d),
                "ci_pp": [float(lo), float(hi)]}

    table1 = {
        "Baseline": {"hit1": float(B.mean()), "delta_pp": 0.0, "ci_pp": [0.0, 0.0]},
        "Targeted": cond_row(T),
        "Sibling": cond_row(Si),
        "Shuffled-AmbigQA": cond_row(Am),
        "WikiText-shuffled": cond_row(Wt),
        "Random": cond_row(Rn),
    }

    def contrast(x, y, alt):
        """delta = mean(y - x); paired tests per project convention."""
        d = (y.mean() - x.mean()) * 100.0
        lo, hi = bootstrap_diff_ci_pp(x, y)  # CI on (y - x)
        wx = wilcoxon_paired(y, x, alternative=alt)
        mc = mcnemar_paired(x, y, alternative=alt)
        return {"delta_pp": float(d), "ci_pp": [float(lo), float(hi)],
                "wilcoxon_p": float(wx), "mcnemar_p": float(mc["p"]),
                "mcnemar_cells": {k: mc[k] for k in
                                  ("a_both_hit", "x_kills_only", "y_kills_only",
                                   "d_both_miss", "n_disc")}}

    table2 = {
        "T1_Targeted_vs_Random": contrast(T, Rn, "greater"),
        "T2_Sibling_vs_ShuffledAmbigQA": contrast(Si, Am, "greater"),
        "T3_Sibling_vs_WikiText": contrast(Si, Wt, "greater"),
        "T4_ShuffledAmbigQA_vs_WikiText": contrast(Am, Wt, "two-sided"),
    }

    out = {
        "source": str(RESULTS.relative_to(Path(__file__).resolve().parents[1])),
        "n_self_pairs": len(keys),
        "note": ("Recomputed from canonical L41 per-pair artifact. WikiText and "
                 "Random are identical to baseline on all 1103 pairs."),
        "table1_corrected": table1,
        "table2_corrected": table2,
        "published_stale_values": {
            "table1": {"Shuffled-AmbigQA": [0.2853, -0.75],
                       "WikiText-shuffled": [0.2956, 0.27],
                       "Random": [0.2937, 0.09]},
            "table2": {"T1": 13.33, "T2": 8.30, "T3": 9.05, "T4": 1.02},
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)

    # ---- human-readable dump ----
    print(f"n = {len(keys)}   source = {RESULTS}")
    print("\n=== CORRECTED TABLE 1 ===")
    for name, v in table1.items():
        ci = v["ci_pp"]
        print(f"  {name:18s} hit1={v['hit1']:.4f}  d={v['delta_pp']:+.2f}pp  "
              f"CI=[{ci[0]:+.2f},{ci[1]:+.2f}]")
    print("\n=== CORRECTED TABLE 2 ===")
    for name, v in table2.items():
        ci = v["ci_pp"]
        print(f"  {name:34s} d={v['delta_pp']:+.2f}  CI=[{ci[0]:+.2f},{ci[1]:+.2f}]  "
              f"Wx={v['wilcoxon_p']:.2e}  McN={v['mcnemar_p']:.2e}  "
              f"disc={v['mcnemar_cells']['x_kills_only']}/{v['mcnemar_cells']['y_kills_only']}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
