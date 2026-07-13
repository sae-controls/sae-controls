"""Paired statistics for the matched/donor controls against the published
L41 per-pair outcomes (offline).

Conventions identical to the published T1-T4 (src/analysis.py): per-pair
means over draws, delta = mean(y) - mean(x) in pp, seeded bootstrap CI,
paired Wilcoxon signed-rank (primary), McNemar exact-binomial
(secondary). All contrasts are computed on the common key set of the two
conditions being compared.

Gates:
  * baseline parity — the GPU rerun's base_hit1 must agree with the
    published base_hit1 (environment/weights identity check). The script
    refuses to report contrasts if agreement < 0.98.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.analysis import (bootstrap_diff_ci_pp, mcnemar_paired,  # noqa: E402
                          wilcoxon_paired)

MC = REPO / "artifacts" / "matched_controls"
L41 = REPO / "artifacts" / "layer_bookends" / "L41"


def per_key_mean(rows, val_key, key=lambda r: (r["pair_id"], r["target_idx"])):
    d = defaultdict(list)
    for r in rows:
        d[key(r)].append(r[val_key])
    return {k: float(np.mean(v)) for k, v in d.items()}


def contrast(x_map, y_map, alt):
    """Published convention: delta = mean(y - x) on common keys."""
    keys = sorted(set(x_map) & set(y_map))
    x = np.array([x_map[k] for k in keys])
    y = np.array([y_map[k] for k in keys])
    d = (y.mean() - x.mean()) * 100.0
    lo, hi = bootstrap_diff_ci_pp(x, y)
    wx = wilcoxon_paired(y, x, alternative=alt)
    mc = mcnemar_paired(x, y, alternative=alt)
    return {"n": len(keys), "delta_pp": float(d),
            "ci_pp": [float(lo), float(hi)],
            "wilcoxon_p": float(wx), "mcnemar_p": float(mc["p"])}


def main() -> None:
    sel = json.load(open(MC / "selection.json"))
    res = json.load(open(MC / "matched_controls_results.json"))
    pub = json.load(open(L41 / "results_main.json"))

    kt = lambda r: (r["pair_id"], r["target_idx"])

    # published per-pair maps
    pub_base = per_key_mean(pub["self_rows"], "base_hit1")
    pub_targ = per_key_mean(pub["self_rows"], "ablate_hit1")
    pub_sib = per_key_mean(pub["cross_rows"], "ablate_hit1")
    pub_amb = per_key_mean(pub["ambigqa_shuffled_rows"], "shuffled_hit1")

    # ---- parity gate ----
    new_base = per_key_mean(res["baseline_rows"], "base_hit1")
    common = sorted(set(new_base) & set(pub_base))
    agree = float(np.mean([new_base[k] == pub_base[k] for k in common]))
    print(f"baseline parity: {agree:.4f} on {len(common)} (pair,target) keys")
    if agree < 0.98:
        raise SystemExit(f"PARITY GATE FAILED ({agree:.4f} < 0.98) — "
                         "do not trust the GPU results; investigate weights/env.")

    # new-condition per-pair maps
    m_act = per_key_mean(res["m_act_rows"], "ablate_hit1")
    m_targ = per_key_mean(res["m_targ_rows"], "ablate_hit1")
    sadq = per_key_mean(res["sadq_rows"], "ablate_hit1")
    w = per_key_mean(res["w_rows"], "ablate_hit1")

    # mass-ratio subgroup for M-act (row-level mean per key)
    mr = per_key_mean(sel["m_act_rows"], "mass_ratio")
    hi_keys = {k for k, v in mr.items() if v >= 0.5}
    m_act_hi = {k: v for k, v in m_act.items() if k in hi_keys}

    out = {
        "parity": {"agreement": agree, "n": len(common)},
        "conditions_delta_vs_baseline_pp": {
            name: contrast(pub_base, cond, "two-sided")
            for name, cond in [("M-act", m_act), ("M-targ", m_targ),
                               ("SA-DQ", sadq), ("W", w),
                               ("M-act (mass>=0.5)", m_act_hi)]
        },
        "paired_contrasts": {
            # T5 (paper Table 2): direct Targeted-vs-Sibling contrast,
            # computed from the published per-pair rows
            "T5_Targeted_vs_Sibling": contrast(pub_targ, pub_sib, "greater"),
            # does the real condition remove more than its matched control?
            "T6_Sibling_vs_Mact": contrast(pub_sib, m_act, "greater"),
            "T6hi_Sibling_vs_Mact_massgte50": contrast(
                {k: v for k, v in pub_sib.items() if k in hi_keys}, m_act_hi,
                "greater"),
            "T7_Targeted_vs_Mtarg": contrast(pub_targ, m_targ, "greater"),
            # donor-constrained shuffle vs plain in-distribution shuffle
            "T8_SADQ_vs_ShuffledAmbigQA": contrast(sadq, pub_amb, "two-sided"),
            "T8b_SADQ_vs_Sibling": contrast(sadq, pub_sib, "two-sided"),
            # wrong-answer features vs sibling / shuffle
            "T9_Sibling_vs_W": contrast(pub_sib, w, "two-sided"),
            "T9b_W_vs_ShuffledAmbigQA": contrast(w, pub_amb, "two-sided"),
        },
        "selection_meta": sel["meta"],
        "run_meta": res["run_meta"],
    }
    json.dump(out, open(MC / "summary.json", "w"), indent=1)

    print("\n=== deltas vs baseline (pp) ===")
    for name, c in out["conditions_delta_vs_baseline_pp"].items():
        print(f"{name:22s} n={c['n']:5d}  Δ={c['delta_pp']:+6.2f}  "
              f"CI[{c['ci_pp'][0]:+.2f},{c['ci_pp'][1]:+.2f}]  Wx p={c['wilcoxon_p']:.2e}")
    print("\n=== paired contrasts (pp) ===")
    for name, c in out["paired_contrasts"].items():
        print(f"{name:34s} n={c['n']:5d}  Δ={c['delta_pp']:+6.2f}  "
              f"CI[{c['ci_pp'][0]:+.2f},{c['ci_pp'][1]:+.2f}]  "
              f"Wx p={c['wilcoxon_p']:.2e}  McN p={c['mcnemar_p']:.2e}")
    print(f"\nwrote {MC/'summary.json'}")


if __name__ == "__main__":
    main()
