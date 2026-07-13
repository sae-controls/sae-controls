"""Slot-detection sensitivity + continuous-metric robustness.

Two jobs:
  (I)  Test whether the headline signals depend on which slot-detection
       matcher tier surfaced each pair.
  (II) Re-do the per-pair trim test using the continuous KL metric to
       address the binary-dominated distribution caveat.

Sub-experiments:
  (A) Slot-detection strategy stratification.
      Stratify the 1103 self-pairs into:
        stratum_exact     — pairs detected via 'exact' substring match
                            (752 self-pairs, 300 pairs)
        stratum_distinct  — pairs detected via distinctive-token only
                            (351 self-pairs, 148 pairs)
      Re-run T3, the unique/shared decomposition, and the single-feature
      uc_top vs sc_top head-to-head on each stratum.

  (B) Continuous-metric trim test on KL.
      Use the multi-metric per-pair KL Δ values. For each of T3-on-KL,
      Δ_targeted, Δ_shared, Δ_unique on the KL scale, drop top-10% and
      top-25% by |per-pair Δ_KL|, re-run paired test on bulk.

  (C) Looser-filter analysis. Optional, gated on (A) result. Skip if
      (A) shows both strata pass the success bar.

Reads:
  artifacts/detected_pairs.json
  artifacts/specific_features.json
  artifacts/results_main.json
  artifacts/unique_vs_shared/a/ablation_rows.json
  artifacts/multimetric/b/per_pair_metrics.json
  artifacts/per_feature_equivalence/c/single_feature_rows.json

Writes:
  artifacts/slot_detection/a/{stratification.json}
  artifacts/slot_detection/b/{kl_trim_test.json}
  artifacts/slot_detection/c/{decision.json}
  artifacts/slot_detection/run_meta.json
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.analysis import (
    bootstrap_diff_ci_pp, mcnemar_paired, per_pair_means, wilcoxon_paired,
)
from src.config import (
    ARTIFACTS_DIR, LAYER, MODEL_ID, SAE_CHECKPOINT_SHA256,
)
from src.io_utils import save_json_atomic
from src.sae import load_sae


P17_DIR = ARTIFACTS_DIR / "slot_detection"


def _categorize_strategy(s):
    if s.startswith("exact"):
        return "exact"
    if s.startswith("distinctive"):
        return "distinct"
    return "other"


def _t3_stats(Sib_arr, WSh_arr):
    if len(Sib_arr) < 2:
        return None
    mc = mcnemar_paired(Sib_arr, WSh_arr, alternative="greater")
    wx = wilcoxon_paired(WSh_arr, Sib_arr, alternative="greater")
    ci = bootstrap_diff_ci_pp(Sib_arr, WSh_arr, seed=0)
    return {
        "n": int(len(Sib_arr)),
        "delta_pp": float((WSh_arr.mean() - Sib_arr.mean()) * 100),
        "ci_pp": [float(ci[0]), float(ci[1])],
        "mcnemar_p": mc["p"],
        "wilcoxon_p": wx,
    }


def _drop_vs_base_stats(arr, base_arr):
    if len(arr) < 2:
        return None
    mc = mcnemar_paired(arr, base_arr, alternative="greater")
    wx = wilcoxon_paired(base_arr, arr, alternative="greater")
    ci = bootstrap_diff_ci_pp(base_arr, arr, seed=0)
    return {
        "n": int(len(arr)),
        "delta_pp": float((base_arr.mean() - arr.mean()) * 100),
        "ci_pp": [float(ci[0]), float(ci[1])],
        "mcnemar_p": mc["p"],
        "wilcoxon_p": wx,
    }


# =============================================================================
# (A) Stratification
# =============================================================================

def run_a(detected, results_main, p12pp_rows, sf_rows):
    out_dir = P17_DIR / "a"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (A) slot-detection strategy stratification ==========")

    # Strategy lookup per pair_id
    strategy_per_pair = {p["id"]: _categorize_strategy(p["match_strategy"])
                         for p in detected}
    cat_counts = defaultdict(int)
    for s in strategy_per_pair.values():
        cat_counts[s] += 1
    sp_per_cat = defaultdict(int)
    for p in detected:
        sp_per_cat[strategy_per_pair[p["id"]]] += len(p["disambigs"])
    print(f"[A] pair counts: {dict(cat_counts)}; self-pair counts: {dict(sp_per_cat)}")

    # Build per-pair arrays
    sibling = per_pair_means(
        results_main["cross_rows"],
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["ablate_hit1"],
    )
    wt_shuf = per_pair_means(
        results_main["wikitext_shuffled_rows"],
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["wt_shuffled_hit1"],
    )
    baseline = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                for r in results_main["self_rows"]}
    targeted = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                for r in results_main["self_rows"]}
    shared_only = {(r["pair_id"], r["target_idx"]): r["shared_only_hit1"]
                   for r in p12pp_rows}
    unique_only = {(r["pair_id"], r["target_idx"]): r["unique_only_hit1"]
                   for r in p12pp_rows}

    sf_by_key = {(r["pair_id"], r["target_idx"]): r for r in sf_rows}

    keys = sorted(k for k in baseline if k in targeted and k in shared_only
                   and k in unique_only and k in sibling and k in wt_shuf)
    print(f"[A] {len(keys)} self-pairs with all conditions defined")

    out = {"strata": {}}
    for stratum in ["exact", "distinct"]:
        stratum_keys = [k for k in keys
                        if strategy_per_pair.get(k[0]) == stratum]
        n = len(stratum_keys)
        Sib = np.array([sibling[k] for k in stratum_keys], dtype=float)
        WSh = np.array([wt_shuf[k] for k in stratum_keys], dtype=float)
        Base = np.array([baseline[k] for k in stratum_keys], dtype=float)
        Targ = np.array([targeted[k] for k in stratum_keys], dtype=float)
        Sh = np.array([shared_only[k] for k in stratum_keys], dtype=float)
        Un = np.array([unique_only[k] for k in stratum_keys], dtype=float)

        t3 = _t3_stats(Sib, WSh)
        targ_test = _drop_vs_base_stats(Targ, Base)
        sh_test = _drop_vs_base_stats(Sh, Base)
        un_test = _drop_vs_base_stats(Un, Base)

        # Single-feature uc_top vs sc_top within stratum
        sf_keys = [k for k in stratum_keys if k in sf_by_key]
        uc_arr = np.array([sf_by_key[k]["uc_top_hit1"] for k in sf_keys], dtype=float)
        sc_arr = np.array([sf_by_key[k]["sc_top_hit1"] for k in sf_keys], dtype=float)
        sf_base_arr = np.array([sf_by_key[k]["baseline_hit1"] for k in sf_keys], dtype=float)
        sf_uc = _drop_vs_base_stats(uc_arr, sf_base_arr) if len(uc_arr) >= 2 else None
        sf_sc = _drop_vs_base_stats(sc_arr, sf_base_arr) if len(sc_arr) >= 2 else None
        if len(uc_arr) >= 2:
            sf_uc_vs_sc_mc = mcnemar_paired(uc_arr, sc_arr, alternative="two-sided")
            sf_uc_vs_sc_wx = wilcoxon_paired(uc_arr, sc_arr, alternative="two-sided")
            sf_diff = float((uc_arr.mean() - sc_arr.mean()) * 100)
        else:
            sf_uc_vs_sc_mc = None; sf_uc_vs_sc_wx = None; sf_diff = None

        out["strata"][stratum] = {
            "n": n,
            "T3":             t3,
            "delta_targeted": targ_test,
            "delta_shared":   sh_test,
            "delta_unique":   un_test,
            "single_feature": {
                "n_eligible": len(sf_keys),
                "uc_top_vs_baseline": sf_uc,
                "sc_top_vs_baseline": sf_sc,
                "uc_top_drop_pp": float((sf_base_arr.mean() - uc_arr.mean()) * 100)
                                   if len(uc_arr) else None,
                "sc_top_drop_pp": float((sf_base_arr.mean() - sc_arr.mean()) * 100)
                                   if len(sc_arr) else None,
                "uc_minus_sc_drop_pp_diff": (
                    float((sf_base_arr.mean() - uc_arr.mean()) * 100
                           - (sf_base_arr.mean() - sc_arr.mean()) * 100)
                    if len(uc_arr) else None
                ),
                "uc_vs_sc_two_sided_mcnemar_p": sf_uc_vs_sc_mc["p"] if sf_uc_vs_sc_mc else None,
                "uc_vs_sc_two_sided_wilcoxon_p": sf_uc_vs_sc_wx,
            },
        }
        print(f"\n[A {stratum}] n={n}")
        print(f"  T3:          Δ={t3['delta_pp']:+6.2f}  CI=[{t3['ci_pp'][0]:+5.2f},{t3['ci_pp'][1]:+5.2f}]  "
              f"McN p={t3['mcnemar_p']:.3g}  Wlx p={t3['wilcoxon_p']:.3g}")
        print(f"  Δ_targeted:  Δ={targ_test['delta_pp']:+6.2f}  CI=[{targ_test['ci_pp'][0]:+5.2f},{targ_test['ci_pp'][1]:+5.2f}]  "
              f"McN p={targ_test['mcnemar_p']:.3g}  Wlx p={targ_test['wilcoxon_p']:.3g}")
        print(f"  Δ_shared:    Δ={sh_test['delta_pp']:+6.2f}  CI=[{sh_test['ci_pp'][0]:+5.2f},{sh_test['ci_pp'][1]:+5.2f}]  "
              f"McN p={sh_test['mcnemar_p']:.3g}  Wlx p={sh_test['wilcoxon_p']:.3g}")
        print(f"  Δ_unique:    Δ={un_test['delta_pp']:+6.2f}  CI=[{un_test['ci_pp'][0]:+5.2f},{un_test['ci_pp'][1]:+5.2f}]  "
              f"McN p={un_test['mcnemar_p']:.3g}  Wlx p={un_test['wilcoxon_p']:.3g}")
        if sf_diff is not None:
            print(f"  uc_top vs sc_top (single-feat): n={len(sf_keys)}  "
                  f"Δ_diff={sf_diff:+.2f} pp  "
                  f"two-sided McN p={out['strata'][stratum]['single_feature']['uc_vs_sc_two_sided_mcnemar_p']:.3g}  "
                  f"Wlx p={out['strata'][stratum]['single_feature']['uc_vs_sc_two_sided_wilcoxon_p']:.3g}")

    save_json_atomic(out_dir / "stratification.json", out)

    # Decision: do both strata pass the success bar?
    pass_bar = True
    for stratum in ["exact", "distinct"]:
        s = out["strata"][stratum]
        # T3 within ±2pp of published 4.49 → 2.49 ≤ T3 ≤ 6.49
        t3_in_range = 2.49 <= s["T3"]["delta_pp"] <= 6.49
        # unique/shared both p < 0.05
        uns_pass = (s["delta_unique"]["wilcoxon_p"] < 0.05
                     and s["delta_shared"]["wilcoxon_p"] < 0.05)
        if not (t3_in_range and uns_pass):
            pass_bar = False
            print(f"\n[A decision] stratum {stratum} fails: "
                  f"T3 in [2.49, 6.49]: {t3_in_range} (got {s['T3']['delta_pp']:.2f}); "
                  f"unique p<0.05 and shared p<0.05: {uns_pass}")
    print(f"\n[A decision] success bar pass: {pass_bar}")
    return out, pass_bar


# =============================================================================
# (B) Continuous-metric trim test on KL
# =============================================================================

def run_b(per_pair_metrics):
    out_dir = P17_DIR / "b"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (B) continuous-metric trim test (KL) ==========")

    # Build per-pair KL arrays
    # Each entry has metrics['baseline'/'targeted'/'shared_only'/'unique_only'/'sibling'/'wt_shuf']['kl_to_baseline']
    rows = [r for r in per_pair_metrics
            if r["metrics"]["targeted"] is not None]
    print(f"[B] eligible per-pair metric rows: {len(rows)}")

    def _arr(cond):
        return np.array([r["metrics"][cond]["kl_to_baseline"]
                          for r in rows], dtype=float)

    base_kl = _arr("baseline")    # all zeros by definition
    targ_kl = _arr("targeted")
    sh_kl = _arr("shared_only")
    un_kl = _arr("unique_only")
    sib_kl = _arr("sibling")
    wt_kl = _arr("wt_shuf")

    from scipy.stats import wilcoxon as scipy_wilcoxon

    def _trim_test(x, y, label, drop_pcts):
        """One-sided Wilcoxon: x > y (x diverges more from baseline)."""
        results = []
        diffs = x - y
        abs_diffs = np.abs(diffs)
        n = len(diffs)
        for pct in drop_pcts:
            n_drop = int(round(n * pct))
            if n_drop > 0:
                keep_idx = np.argsort(abs_diffs)[:-n_drop]
            else:
                keep_idx = np.arange(n)
            x_b = x[keep_idx]; y_b = y[keep_idx]
            mean_diff = float((x_b - y_b).mean())
            try:
                p = float(scipy_wilcoxon(x_b, y_b, alternative="greater").pvalue)
            except Exception:
                p = float("nan")
            results.append({
                "drop_pct": pct,
                "n_dropped": n_drop,
                "n_bulk": int(len(keep_idx)),
                "mean_diff_kl": mean_diff,
                "wilcoxon_p_x_greater_y": p,
            })
        return results

    # (i) T3 on KL: Sibling vs WT-shuffled (Sibling KL > WT KL ⇔ Sib drops more)
    t3_kl = _trim_test(sib_kl, wt_kl, "T3-on-KL", [0.0, 0.10, 0.25])
    # (ii) Targeted vs Baseline (Targ KL > Base KL = 0)
    targ_vs_base = _trim_test(targ_kl, base_kl, "Targeted-on-KL", [0.0, 0.10, 0.25])
    # (iii) Shared vs Baseline
    sh_vs_base = _trim_test(sh_kl, base_kl, "Shared-on-KL", [0.0, 0.10, 0.25])
    # (iv) Unique vs Baseline
    un_vs_base = _trim_test(un_kl, base_kl, "Unique-on-KL", [0.0, 0.10, 0.25])

    out = {
        "T3_on_KL": t3_kl,
        "Targeted_vs_Baseline_on_KL": targ_vs_base,
        "Shared_vs_Baseline_on_KL": sh_vs_base,
        "Unique_vs_Baseline_on_KL": un_vs_base,
    }
    save_json_atomic(out_dir / "kl_trim_test.json", out)

    for label, results in [("T3 (Sib KL > WT KL)", t3_kl),
                            ("Targeted vs Baseline KL", targ_vs_base),
                            ("Shared vs Baseline KL", sh_vs_base),
                            ("Unique vs Baseline KL", un_vs_base)]:
        print(f"\n[B] {label}:")
        for r in results:
            print(f"  drop {r['drop_pct']*100:.0f}%  n_drop={r['n_dropped']:<4} n_bulk={r['n_bulk']:<5}  "
                  f"mean_diff_kl={r['mean_diff_kl']:+.5f}  Wlx p (x>y) = {r['wilcoxon_p_x_greater_y']:.3g}")
    return out


# =============================================================================
# (C) Conditional looser-filter analysis
# =============================================================================

def run_c_skipped(reason):
    out_dir = P17_DIR / "c"
    out_dir.mkdir(parents=True, exist_ok=True)
    decision = {
        "ran": False,
        "reason": reason,
    }
    save_json_atomic(out_dir / "decision.json", decision)
    print(f"\n[C] SKIPPED — {reason}")
    return decision


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    P17_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"[setup] verifying SAE SHA via load_sae...")
    sae, sae_meta = load_sae(layer=LAYER)
    if sae_meta["sha256"] != SAE_CHECKPOINT_SHA256:
        raise RuntimeError(f"SAE sha256 mismatch! got {sae_meta['sha256']}")
    print(f"[setup] SAE sha256 verified ({sae_meta['sha256'][:16]}...)")
    del sae

    detected = json.load(open(ARTIFACTS_DIR / "detected_pairs.json"))
    results_main = json.load(open(ARTIFACTS_DIR / "results_main.json"))
    p12pp_rows = json.load(open(ARTIFACTS_DIR / "unique_vs_shared" / "a"
                                  / "ablation_rows.json"))
    sf_rows = json.load(open(ARTIFACTS_DIR / "per_feature_equivalence" / "c"
                               / "single_feature_rows.json"))
    per_pair_metrics = json.load(open(ARTIFACTS_DIR / "multimetric" / "b"
                                        / "per_pair_metrics.json"))

    print(f"[setup] {len(detected)} detected pairs, "
          f"{sum(len(p['disambigs']) for p in detected)} self-pairs, "
          f"{len(p12pp_rows)} p12pp rows, {len(sf_rows)} sf rows, "
          f"{len(per_pair_metrics)} per-pair metric rows")

    a_out, a_pass = run_a(detected, results_main, p12pp_rows, sf_rows)
    b_out = run_b(per_pair_metrics)
    if a_pass:
        c_out = run_c_skipped(
            "(A) shows both strata pass the success bar — "
            "T3 within ±2pp of 4.49 AND unique/shared both p<0.05 — so "
            "the published filter is not load-bearing; (C) skip per spec.")
    else:
        c_out = run_c_skipped(
            "(A) flagged at least one stratum below the success bar. "
            "The looser-filter prompt asks for a sample of looser-filter pairs "
            "to be re-run through the six-condition pipeline. NOT "
            "implemented in this script — would need ~100-200 GPU "
            "forwards on the rejected-candidates set "
            "(slot_detection_failures.json). Recommend a separate "
            "follow-up if this branch fires."
        )

    save_json_atomic(P17_DIR / "run_meta.json", {
        "model": MODEL_ID, "layer": LAYER, "sae_meta": sae_meta,
        "n_detected_pairs": len(detected),
        "stratum_pair_counts": {
            "exact": sum(1 for p in detected if _categorize_strategy(p["match_strategy"]) == "exact"),
            "distinct": sum(1 for p in detected if _categorize_strategy(p["match_strategy"]) == "distinct"),
        },
        "phase_a_pass": a_pass,
        "phase_c_ran": c_out["ran"],
        "elapsed_seconds": round(time.time() - t_total, 2),
    })

    print(f"\n=== Stage 1.7 complete ===")
    print(f"  total elapsed: {(time.time()-t_total)/60:.1f} min")
    print(f"  -> {P17_DIR}")


if __name__ == "__main__":
    main()
