"""Unique-vs-shared decomposition + overlap-shell anomalies.

T3 could in principle be carried entirely at the cluster level. Whether
ablating only the unique-to-D_i subset of Targeted leaves a residual
D_i-specific signal (i.e., a non-cluster-level mediation mechanism)
determines the interpretation. This script tests that and characterizes
the overlap-shell anomalies.

  (A) Unique-vs-shared decomposition.
      For each (P, D_i) using the published selector (a=0.5, k=10):
        shared_{D_i} = Targeted_{D_i} ∩ (∪_{j≠i} Targeted_{D_j})
        unique_{D_i} = Targeted_{D_i} ∖ shared_{D_i}
      Run shared_only and unique_only ablations. Decompose targeted_drop
      into shared, unique, residual. Per-pair (unique_drop / targeted_drop)
      ratio. Stratified analysis on |unique| ≥ 5.
      For empty sets, use the published baseline_hit (forward_with_ablation
      with empty fids is a no-op, but skipping the call saves time).

  (B) k=2 max-overlap shell diagnostic.
      Mean-over-siblings distribution within k=2 shell, disambig-count
      distribution, position-mode enrichment of the 2 shared features.
      Re-run shell trajectory using mean-over-siblings overlap binning.

  (C) k=10 cell sanity check (≤5 minutes).
      For the 8 max-overlap=10 pairs, check whether Sibling top-10 is
      literally identical to Targeted top-10 and whether the all-zero
      variance comes from always-wrong predictions or some other reason.

Reads:
  artifacts/sae_encodings_L37.npz
  artifacts/detected_pairs.json
  artifacts/specific_features.json
  artifacts/results_main.json
  artifacts/wikitext_position_mode.json

Writes:
  artifacts/unique_vs_shared/a/{decomposition_table.json, per_pair_ratios.json,
                             ratio_histogram.png, stratified_table.json,
                             ablation_rows.json, set_size_dist.json}
  artifacts/unique_vs_shared/b/{k2_diagnostic.json, mean_overlap_shells.json,
                             mean_overlap_trajectory.png}
  artifacts/unique_vs_shared/c/{k10_summary.json}
  artifacts/unique_vs_shared/comparison_table.json
  artifacts/unique_vs_shared/run_meta.json
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
import torch
from tqdm.auto import tqdm

from src.analysis import (
    bootstrap_diff_ci_pp, mcnemar_paired, per_pair_means, wilcoxon_paired,
)
from src.config import (
    ARTIFACTS_DIR, LAYER, MODEL_ID, SAE_CHECKPOINT_SHA256, TOP_LOGITS_K,
)
from src.hooks import forward_with_ablation, hit_at_k
from src.io_utils import save_json_atomic
from src.model import load_model
from src.prompts import build_prompt
from src.sae import load_sae


P12PP_DIR = ARTIFACTS_DIR / "unique_vs_shared"
PUBLISHED_TOP_K = 10


def _t3_or_drop_stats(x_arr, y_arr, alt_mc="greater", alt_wx="greater"):
    """Generic 1-sided paired test on (x, y). Returns dict.
    Convention: tests x_kills_only > y_kills_only (i.e., x drops more).
    For unique_only vs baseline: x = baseline (1=hit), y = unique_only (1=hit).
    Call sites state the test direction explicitly."""
    if len(x_arr) < 2:
        return {"n": int(len(x_arr)), "delta_pp": None,
                "ci_pp": [None, None], "mcnemar": None, "wilcoxon_p": None}
    mc = mcnemar_paired(x_arr, y_arr, alternative=alt_mc)
    wx = wilcoxon_paired(y_arr, x_arr, alternative=alt_wx)
    ci = bootstrap_diff_ci_pp(x_arr, y_arr, seed=0)
    return {
        "n": int(len(x_arr)),
        "x_mean": float(x_arr.mean()),
        "y_mean": float(y_arr.mean()),
        "delta_pp_y_minus_x": float((y_arr.mean() - x_arr.mean()) * 100),
        "ci_pp_y_minus_x": [float(ci[0]), float(ci[1])],
        "mcnemar": mc,
        "wilcoxon_p": wx,
    }


# =============================================================================
# (A) Unique-vs-shared decomposition
# =============================================================================

def run_a(tokenizer, model, sae, detected, a_prompts,
          published_specific, results_main):
    out_dir = P12PP_DIR / "a"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (A) unique-vs-shared decomposition ==========")

    # Index published Targeted
    pub_set = {}
    for entry in published_specific:
        pub_set[(entry["pair_id"], entry["disambig_idx"])] = set(entry["features"])

    # Compute shared/unique sets per (P, D_i)
    shared_unique = {}     # (pair_id, target_idx) -> {'shared': set, 'unique': set, 'targeted': set}
    for p in detected:
        K = len(p["disambigs"])
        for i in range(K):
            ti = pub_set.get((p["id"], i))
            if not ti:
                continue
            sib_union = set()
            for j in range(K):
                if j == i:
                    continue
                sib_union |= pub_set.get((p["id"], j), set())
            shared = ti & sib_union
            unique = ti - shared
            shared_unique[(p["id"], i)] = {
                "targeted": ti, "shared": shared, "unique": unique,
            }

    # Set-size distribution
    shared_sizes = np.array([len(s["shared"]) for s in shared_unique.values()])
    unique_sizes = np.array([len(s["unique"]) for s in shared_unique.values()])
    targeted_sizes = np.array([len(s["targeted"]) for s in shared_unique.values()])
    set_size_dist = {
        "n": len(shared_unique),
        "targeted": {
            "mean": float(targeted_sizes.mean()),
            "median": float(np.median(targeted_sizes)),
            "min": int(targeted_sizes.min()), "max": int(targeted_sizes.max()),
        },
        "shared": {
            "mean": float(shared_sizes.mean()),
            "median": float(np.median(shared_sizes)),
            "min": int(shared_sizes.min()), "max": int(shared_sizes.max()),
            "fraction_eq_0": float((shared_sizes == 0).mean()),
            "histogram": {str(k): int((shared_sizes == k).sum()) for k in range(11)},
        },
        "unique": {
            "mean": float(unique_sizes.mean()),
            "median": float(np.median(unique_sizes)),
            "min": int(unique_sizes.min()), "max": int(unique_sizes.max()),
            "fraction_eq_0": float((unique_sizes == 0).mean()),
            "fraction_ge_5": float((unique_sizes >= 5).mean()),
            "histogram": {str(k): int((unique_sizes == k).sum()) for k in range(11)},
        },
    }
    save_json_atomic(out_dir / "set_size_dist.json", set_size_dist)
    print(f"[A] set sizes — shared: mean {set_size_dist['shared']['mean']:.2f}, "
          f"frac=0 {set_size_dist['shared']['fraction_eq_0']:.3f}; "
          f"unique: mean {set_size_dist['unique']['mean']:.2f}, "
          f"frac=0 {set_size_dist['unique']['fraction_eq_0']:.3f}, "
          f"frac≥5 {set_size_dist['unique']['fraction_ge_5']:.3f}")

    # Pull baseline + targeted hits from published self_rows
    self_rows = results_main["self_rows"]
    baseline_hit = {}; targeted_hit = {}
    for r in self_rows:
        key = (r["pair_id"], r["target_idx"])
        baseline_hit[key] = r["base_hit1"]
        targeted_hit[key] = r["ablate_hit1"]

    # Run shared_only and unique_only forwards (skip empty sets — use baseline)
    ablation_rows = []
    n_skipped_shared = 0
    n_skipped_unique = 0
    for p in tqdm(detected, desc="(A) shared+unique forwards"):
        a_prompt = a_prompts[p["id"]]
        for i in range(len(p["disambigs"])):
            key = (p["id"], i)
            if key not in shared_unique:
                continue
            sets = shared_unique[key]
            target = set(p["disambigs"][i]["first_token_variants"])
            base = baseline_hit.get(key)
            if base is None:
                continue

            # shared_only
            if sets["shared"]:
                ab_s = forward_with_ablation(
                    model, tokenizer, a_prompt, LAYER, sae,
                    feature_ids=list(sets["shared"]), k=TOP_LOGITS_K,
                )
                shared_h = hit_at_k(ab_s["top_ids"], target, 1)
            else:
                shared_h = base
                n_skipped_shared += 1

            # unique_only
            if sets["unique"]:
                ab_u = forward_with_ablation(
                    model, tokenizer, a_prompt, LAYER, sae,
                    feature_ids=list(sets["unique"]), k=TOP_LOGITS_K,
                )
                unique_h = hit_at_k(ab_u["top_ids"], target, 1)
            else:
                unique_h = base
                n_skipped_unique += 1

            ablation_rows.append({
                "pair_id": p["id"], "target_idx": i,
                "n_targeted": len(sets["targeted"]),
                "n_shared": len(sets["shared"]),
                "n_unique": len(sets["unique"]),
                "base_hit1": base,
                "targeted_hit1": targeted_hit.get(key),
                "shared_only_hit1": shared_h,
                "unique_only_hit1": unique_h,
            })
        torch.cuda.empty_cache()

    save_json_atomic(out_dir / "ablation_rows.json", ablation_rows)
    print(f"[A] {len(ablation_rows)} (P, D_i) processed; "
          f"skipped shared forwards (|shared|=0): {n_skipped_shared}; "
          f"skipped unique forwards (|unique|=0): {n_skipped_unique}")

    # Build per-key arrays
    keys = sorted((r["pair_id"], r["target_idx"]) for r in ablation_rows)
    by_key = {(r["pair_id"], r["target_idx"]): r for r in ablation_rows}
    B = np.array([by_key[k]["base_hit1"]          for k in keys], dtype=float)
    T = np.array([by_key[k]["targeted_hit1"]      for k in keys], dtype=float)
    Sh = np.array([by_key[k]["shared_only_hit1"]  for k in keys], dtype=float)
    Un = np.array([by_key[k]["unique_only_hit1"]  for k in keys], dtype=float)
    n = len(keys)

    # Decomposition table
    targeted_drop = (T - B).mean() * 100   # negative pp = drop
    shared_drop   = (Sh - B).mean() * 100
    unique_drop   = (Un - B).mean() * 100
    sum_drops     = shared_drop + unique_drop
    residual      = targeted_drop - sum_drops
    pct_target = lambda x: (x / targeted_drop * 100) if targeted_drop else float("nan")

    # Tests vs baseline (1-sided "drops more"): x=baseline, y=ablation.
    # _t3_or_drop_stats sets up alt_mc="greater" for x_kills_only > y_kills_only,
    # i.e., x=baseline kills more → baseline=0 ablation=1. We want ablation=0
    # baseline=1 (ablation kills more). Flip x and y.
    tests = {}
    tests["shared_vs_baseline"] = _t3_or_drop_stats(Sh, B)   # Sh kills more iff Sh<B
    tests["unique_vs_baseline"] = _t3_or_drop_stats(Un, B)
    tests["targeted_vs_unique"] = _t3_or_drop_stats(T,  Un)  # adding shared adds drop

    decomp = {
        "n": n,
        "table": {
            "Baseline":      {"hit1": float(B.mean()), "delta_pp": 0.0,
                              "pct_of_targeted": 0.0, "ci_pp": [None, None]},
            "Targeted":      {"hit1": float(T.mean()), "delta_pp": float(targeted_drop),
                              "pct_of_targeted": 100.0,
                              "ci_pp": list(bootstrap_diff_ci_pp(B, T, seed=0))},
            "Shared-only":   {"hit1": float(Sh.mean()), "delta_pp": float(shared_drop),
                              "pct_of_targeted": float(pct_target(shared_drop)),
                              "ci_pp": list(bootstrap_diff_ci_pp(B, Sh, seed=0))},
            "Unique-only":   {"hit1": float(Un.mean()), "delta_pp": float(unique_drop),
                              "pct_of_targeted": float(pct_target(unique_drop)),
                              "ci_pp": list(bootstrap_diff_ci_pp(B, Un, seed=0))},
            "Sum (sh + un)": {"hit1": None, "delta_pp": float(sum_drops),
                              "pct_of_targeted": float(pct_target(sum_drops)),
                              "ci_pp": [None, None]},
            "Residual":      {"hit1": None, "delta_pp": float(residual),
                              "pct_of_targeted": float(pct_target(residual)),
                              "ci_pp": [None, None]},
        },
        "tests": tests,
    }
    save_json_atomic(out_dir / "decomposition_table.json", decomp)
    print(f"[A] decomposition (n={n}):")
    for label, row in decomp["table"].items():
        ci = row["ci_pp"]
        ci_s = (f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci[0] is not None else "—")
        print(f"  {label:<14}  hit1={row['hit1']!s:<8}  "
              f"Δ pp={row['delta_pp']:+6.2f}  "
              f"({row['pct_of_targeted']:+6.1f}% of targeted)  CI={ci_s}")
    print(f"[A] tests:")
    for name, t in tests.items():
        print(f"  {name:<24}  Δ y-x={t['delta_pp_y_minus_x']:+6.2f}  "
              f"McN p={t['mcnemar']['p']:.3g}  Wlx p={t['wilcoxon_p']:.3g}")

    # Per-pair (unique_drop / targeted_drop) ratio
    eps = 1e-9
    per_pair_drops = []
    for k in keys:
        td = by_key[k]["targeted_hit1"] - by_key[k]["base_hit1"]
        ud = by_key[k]["unique_only_hit1"] - by_key[k]["base_hit1"]
        per_pair_drops.append({
            "pair_id": k[0], "target_idx": k[1],
            "targeted_drop": td, "unique_drop": ud,
            "shared_drop": by_key[k]["shared_only_hit1"] - by_key[k]["base_hit1"],
            "ratio": (ud / td) if abs(td) > eps else None,
        })
    ratios = [d["ratio"] for d in per_pair_drops if d["ratio"] is not None]
    ratios_arr = np.array(ratios)
    ratio_stats = {
        "n_with_nonzero_targeted_drop": int(len(ratios)),
        "n_skipped_zero_targeted_drop": int(len(per_pair_drops) - len(ratios)),
        "mean": float(ratios_arr.mean()) if len(ratios_arr) else None,
        "median": float(np.median(ratios_arr)) if len(ratios_arr) else None,
        "p25": float(np.percentile(ratios_arr, 25)) if len(ratios_arr) else None,
        "p75": float(np.percentile(ratios_arr, 75)) if len(ratios_arr) else None,
    }
    save_json_atomic(out_dir / "per_pair_ratios.json", {
        "stats": ratio_stats,
        "per_pair_drops": per_pair_drops,
    })
    print(f"[A] (unique_drop / targeted_drop) ratio: "
          f"mean={ratio_stats['mean']:.3f}, median={ratio_stats['median']:.3f} "
          f"(n={len(ratios)}, skipped {len(per_pair_drops)-len(ratios)} "
          f"with zero targeted_drop)")

    # Stratified |unique| ≥ 5 subset
    strat_keys = [k for k in keys if len(shared_unique[k]["unique"]) >= 5]
    strat = {}
    if strat_keys:
        Bs = np.array([by_key[k]["base_hit1"]          for k in strat_keys])
        Ts = np.array([by_key[k]["targeted_hit1"]      for k in strat_keys])
        Shs = np.array([by_key[k]["shared_only_hit1"]  for k in strat_keys])
        Uns = np.array([by_key[k]["unique_only_hit1"]  for k in strat_keys])
        td_s = (Ts - Bs).mean() * 100
        sd_s = (Shs - Bs).mean() * 100
        ud_s = (Uns - Bs).mean() * 100
        strat = {
            "n": len(strat_keys),
            "table": {
                "Baseline":    {"hit1": float(Bs.mean()), "delta_pp": 0.0},
                "Targeted":    {"hit1": float(Ts.mean()), "delta_pp": float(td_s)},
                "Shared-only": {"hit1": float(Shs.mean()), "delta_pp": float(sd_s)},
                "Unique-only": {"hit1": float(Uns.mean()), "delta_pp": float(ud_s)},
            },
            "tests": {
                "shared_vs_baseline": _t3_or_drop_stats(Shs, Bs),
                "unique_vs_baseline": _t3_or_drop_stats(Uns, Bs),
                "targeted_vs_unique": _t3_or_drop_stats(Ts, Uns),
            },
        }
    save_json_atomic(out_dir / "stratified_table.json", strat)
    if strat:
        print(f"[A] stratified (|unique|≥5, n={strat['n']}):")
        for label, row in strat["table"].items():
            print(f"  {label:<14}  hit1={row['hit1']:.4f}  Δ pp={row['delta_pp']:+6.2f}")
        for name, t in strat["tests"].items():
            print(f"  {name:<24}  Δ y-x={t['delta_pp_y_minus_x']:+6.2f}  "
                  f"McN p={t['mcnemar']['p']:.3g}  Wlx p={t['wilcoxon_p']:.3g}")

    # Histogram
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ratios_clip = np.clip(ratios_arr, -1.5, 2.5)
    ax.hist(ratios_clip, bins=40, color="#37a", edgecolor="white")
    ax.axvline(np.median(ratios_arr), color="red", linestyle="--",
               label=f"median = {np.median(ratios_arr):.3f}")
    ax.axvline(np.mean(ratios_arr), color="orange", linestyle=":",
               label=f"mean   = {np.mean(ratios_arr):.3f}")
    ax.axvline(1.0, color="black", linestyle="-",
               label="ratio = 1 (unique fully explains targeted)")
    ax.axvline(0.0, color="gray", linestyle="-",
               label="ratio = 0 (unique has no effect)")
    ax.set_xlabel("per-pair (unique_drop / targeted_drop), clipped to [-1.5, 2.5]")
    ax.set_ylabel("count of (P, D_i)")
    ax.set_title("Unique-only fraction of Targeted ablation drop")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "ratio_histogram.png", dpi=140)
    plt.close()

    return decomp, ratio_stats, strat, set_size_dist


# =============================================================================
# (B) k=2 max-overlap shell diagnostic + mean-overlap trajectory
# =============================================================================

def run_b(detected, published_specific, results_main):
    out_dir = P12PP_DIR / "b"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (B) k=2 shell diagnostic + mean-overlap trajectory ==========")

    pub_set = {}
    for entry in published_specific:
        pub_set[(entry["pair_id"], entry["disambig_idx"])] = set(entry["features"])

    overlaps_max = {}; overlaps_mean = {}; per_pair_K = {}
    pair_id_to_K = {p["id"]: len(p["disambigs"]) for p in detected}
    pair_id_to_pair = {p["id"]: p for p in detected}
    for p in detected:
        K = len(p["disambigs"])
        for i in range(K):
            ti = pub_set.get((p["id"], i))
            if not ti:
                continue
            sib_overlaps = []
            for j in range(K):
                if j == i:
                    continue
                tj = pub_set.get((p["id"], j))
                if not tj:
                    continue
                sib_overlaps.append(len(ti & tj))
            if not sib_overlaps:
                continue
            overlaps_max[(p["id"], i)] = int(max(sib_overlaps))
            overlaps_mean[(p["id"], i)] = float(np.mean(sib_overlaps))
            per_pair_K[(p["id"], i)] = K

    # k=2 shell
    k2_keys = [k for k, v in overlaps_max.items() if v == 2]
    print(f"[B] k=2 shell n={len(k2_keys)}")

    # mean-over-siblings within k=2 shell
    k2_means = [overlaps_mean[k] for k in k2_keys]
    k2_Ks = [per_pair_K[k] for k in k2_keys]
    K_dist = {str(k): int(k2_Ks.count(k)) for k in sorted(set(k2_Ks))}

    # Position-mode enrichment of the 2 shared features per (P, D_i)
    pm = json.load(open(ARTIFACTS_DIR / "wikitext_position_mode.json"))
    pm_per = pm["per_feature"]            # dict: str(fid) -> {position_suspect: bool, ...}
    n_pm_total = pm["n_features_scanned"]
    n_pm_suspect = pm["n_position_suspect"]
    pm_baseline_rate = n_pm_suspect / n_pm_total if n_pm_total else 0.0

    n_shared_features_examined = 0
    n_shared_pos_suspect = 0
    shared_features_in_pm_pool = 0
    for k in k2_keys:
        pid, i = k
        ti = pub_set[(pid, i)]
        # Build sibling union and intersect
        K = pair_id_to_K[pid]
        sib_union = set()
        for j in range(K):
            if j == i:
                continue
            sib_union |= pub_set.get((pid, j), set())
        shared = ti & sib_union
        for fid in shared:
            n_shared_features_examined += 1
            entry = pm_per.get(str(fid))
            if entry is not None:
                shared_features_in_pm_pool += 1
                if entry.get("position_suspect"):
                    n_shared_pos_suspect += 1

    pm_rate_in_shared = (
        n_shared_pos_suspect / shared_features_in_pm_pool
        if shared_features_in_pm_pool else 0.0
    )
    k2_diag = {
        "n_k2_self_pairs": len(k2_keys),
        "mean_overlap_within_k2": {
            "mean": float(np.mean(k2_means)),
            "median": float(np.median(k2_means)),
            "min": float(np.min(k2_means)),
            "max": float(np.max(k2_means)),
        },
        "disambig_count_K_distribution_in_k2": K_dist,
        "shared_feature_position_mode_enrichment": {
            "n_shared_features_examined": n_shared_features_examined,
            "n_in_pm_pool":                shared_features_in_pm_pool,
            "n_position_suspect":          n_shared_pos_suspect,
            "pm_rate_in_shared_features":  pm_rate_in_shared,
            "pm_baseline_rate_full_pool":  pm_baseline_rate,
            "enrichment_ratio":            (
                pm_rate_in_shared / pm_baseline_rate if pm_baseline_rate else None
            ),
        },
    }
    save_json_atomic(out_dir / "k2_diagnostic.json", k2_diag)
    print(f"[B] k=2 mean-overlap: mean={k2_diag['mean_overlap_within_k2']['mean']:.2f}, "
          f"median={k2_diag['mean_overlap_within_k2']['median']:.2f}")
    print(f"[B] K distribution within k=2: {K_dist}")
    print(f"[B] position-mode enrichment in k=2 shared features: "
          f"rate={pm_rate_in_shared:.3f} vs baseline {pm_baseline_rate:.3f}, "
          f"ratio={k2_diag['shared_feature_position_mode_enrichment']['enrichment_ratio']:.3f}")

    # Mean-overlap shell trajectory (binned)
    targeted, sibling, wt_shuf = (
        {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
         for r in results_main["self_rows"]},
        per_pair_means(results_main["cross_rows"],
                       key_fn=lambda r: (r["pair_id"], r["target_idx"]),
                       value_fn=lambda r: r["ablate_hit1"]),
        per_pair_means(results_main["wikitext_shuffled_rows"],
                       key_fn=lambda r: (r["pair_id"], r["target_idx"]),
                       value_fn=lambda r: r["wt_shuffled_hit1"]),
    )
    # Bin mean-overlap into [0,1), [1,2), ..., [9,10]
    bins = []
    for low in range(0, 10):
        high = low + 1
        label = f"[{low},{high})" if low < 9 else f"[{low},{high}]"
        keys_in = [k for k, v in overlaps_mean.items()
                   if (low <= v < high) or (low == 9 and v == 10)]
        keys_in = [k for k in keys_in if k in sibling and k in wt_shuf]
        if len(keys_in) >= 2:
            Sib = np.array([sibling[k] for k in keys_in], dtype=float)
            WSh = np.array([wt_shuf[k] for k in keys_in], dtype=float)
            mc = mcnemar_paired(Sib, WSh, alternative="greater")
            wx = wilcoxon_paired(WSh, Sib, alternative="greater")
            ci = bootstrap_diff_ci_pp(Sib, WSh, seed=0)
            bins.append({
                "bin_label": label, "low": low, "high": high,
                "n": len(keys_in),
                "delta_pp": float((WSh.mean() - Sib.mean()) * 100),
                "ci_pp": [float(ci[0]), float(ci[1])],
                "mcnemar_p": mc["p"],
                "wilcoxon_p": wx,
            })
        else:
            bins.append({
                "bin_label": label, "low": low, "high": high,
                "n": len(keys_in),
                "delta_pp": None, "ci_pp": [None, None],
                "mcnemar_p": None, "wilcoxon_p": None,
            })
    save_json_atomic(out_dir / "mean_overlap_shells.json", {"bins": bins})
    print(f"[B] mean-overlap shells:")
    for b in bins:
        if b["delta_pp"] is None:
            print(f"  {b['bin_label']:<8}  n={b['n']:<5}  (n<2)")
        else:
            print(f"  {b['bin_label']:<8}  n={b['n']:<5}  Δ={b['delta_pp']:+6.2f}  "
                  f"CI=[{b['ci_pp'][0]:+5.2f},{b['ci_pp'][1]:+5.2f}]  "
                  f"McN p={b['mcnemar_p']:.3g}  Wlx p={b['wilcoxon_p']:.3g}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    centers = [(b["low"] + b["high"]) / 2 for b in bins if b["delta_pp"] is not None]
    ds = [b["delta_pp"] for b in bins if b["delta_pp"] is not None]
    los = [b["ci_pp"][0] for b in bins if b["delta_pp"] is not None]
    his = [b["ci_pp"][1] for b in bins if b["delta_pp"] is not None]
    ns = [b["n"] for b in bins if b["delta_pp"] is not None]
    yerr = [[d - lo for d, lo in zip(ds, los)],
            [hi - d for hi, d in zip(his, ds)]]
    ax.errorbar(centers, ds, yerr=yerr, fmt="o", capsize=5,
                color="#37a", ecolor="gray", linewidth=2, markersize=8)
    for c, d, n in zip(centers, ds, ns):
        ax.annotate(f"n={n}", (c, d), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(4.49, color="red", linestyle=":", label="published full-sample T3 = +4.49")
    ax.set_xlabel("mean-over-siblings overlap (top-10), bin midpoint")
    ax.set_ylabel("T3 Δ pp (Sibling − WikiText-shuffled, signed)")
    ax.set_title("T3 by mean-over-siblings overlap")
    ax.set_xticks(range(11))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mean_overlap_trajectory.png", dpi=140)
    plt.close()
    print(f"[B] wrote {out_dir / 'mean_overlap_trajectory.png'}")
    return k2_diag, bins


# =============================================================================
# (C) k=10 cell sanity
# =============================================================================

def run_c(detected, published_specific, results_main):
    out_dir = P12PP_DIR / "c"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (C) k=10 cell sanity ==========")

    pub_set = {}
    for entry in published_specific:
        pub_set[(entry["pair_id"], entry["disambig_idx"])] = list(entry["features"])

    pair_id_to_K = {p["id"]: len(p["disambigs"]) for p in detected}

    # Find k=10 (max-overlap=10) self-pairs
    k10_keys = []
    for p in detected:
        K = len(p["disambigs"])
        for i in range(K):
            ti = set(pub_set.get((p["id"], i), []))
            if not ti:
                continue
            sibs = [set(pub_set.get((p["id"], j), [])) for j in range(K) if j != i]
            sibs = [s for s in sibs if s]
            if not sibs:
                continue
            mx = max(len(ti & s) for s in sibs)
            if mx == 10:
                # Find which sibling(s) achieved overlap 10
                full_overlap_sibs = [j for j in range(K) if j != i
                                     and len(ti & set(pub_set.get((p["id"], j), []))) == 10]
                k10_keys.append({
                    "pair_id": p["id"], "target_idx": i, "K": K,
                    "full_overlap_sibling_idxs": full_overlap_sibs,
                    "targeted_features": list(ti),
                })

    # Cross-reference with results_main rows
    self_by_key = {(r["pair_id"], r["target_idx"]): r for r in results_main["self_rows"]}
    cross_by_key = defaultdict(list)
    for r in results_main["cross_rows"]:
        cross_by_key[(r["pair_id"], r["target_idx"])].append(r)
    wt_by_key = defaultdict(list)
    for r in results_main["wikitext_shuffled_rows"]:
        wt_by_key[(r["pair_id"], r["target_idx"])].append(r)

    summary = []
    for entry in k10_keys:
        key = (entry["pair_id"], entry["target_idx"])
        self_r = self_by_key.get(key)
        cross_rs = cross_by_key.get(key, [])
        wt_rs = wt_by_key.get(key, [])
        sib_hits = [r["ablate_hit1"] for r in cross_rs]
        wt_hits = [r["wt_shuffled_hit1"] for r in wt_rs]
        sib_mean = float(np.mean(sib_hits)) if sib_hits else None
        wt_mean = float(np.mean(wt_hits)) if wt_hits else None
        summary.append({
            "pair_id": entry["pair_id"],
            "target_idx": entry["target_idx"],
            "K": entry["K"],
            "n_full_overlap_siblings": len(entry["full_overlap_sibling_idxs"]),
            "base_hit1": self_r["base_hit1"] if self_r else None,
            "targeted_hit1": self_r["ablate_hit1"] if self_r else None,
            "sibling_per_pair_mean_hit1": sib_mean,
            "wt_shuffled_per_pair_mean_hit1": wt_mean,
            "sibling_eq_wt": (sib_mean == wt_mean) if (sib_mean is not None and wt_mean is not None) else None,
        })

    base_zero_count = sum(1 for r in summary if r["base_hit1"] == 0)
    base_one_count = sum(1 for r in summary if r["base_hit1"] == 1)
    save_json_atomic(out_dir / "k10_summary.json", {
        "n_k10_self_pairs": len(summary),
        "base_zero_count": base_zero_count,
        "base_one_count": base_one_count,
        "all_sibling_eq_wt": all(r["sibling_eq_wt"] for r in summary
                                 if r["sibling_eq_wt"] is not None),
        "details": summary,
    })
    print(f"[C] k=10 cell: n={len(summary)} (P, D_i) self-pairs")
    for r in summary:
        print(f"  pair_id={r['pair_id']}, target_idx={r['target_idx']}, K={r['K']}: "
              f"base={r['base_hit1']}, tgt={r['targeted_hit1']}, "
              f"sib={r['sibling_per_pair_mean_hit1']}, wt={r['wt_shuffled_per_pair_mean_hit1']}, "
              f"sib_eq_wt={r['sibling_eq_wt']}")
    print(f"[C] base_hit1=0 in {base_zero_count}, =1 in {base_one_count}")
    return summary


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    P12PP_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"[setup] loading model + SAE...")
    tokenizer, model = load_model()
    sae, sae_meta = load_sae(layer=LAYER)
    if sae_meta["sha256"] != SAE_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"SAE sha256 mismatch! got {sae_meta['sha256']}"
        )
    print(f"[setup] SAE sha256 verified ({sae_meta['sha256'][:16]}...)")

    detected = json.load(open(ARTIFACTS_DIR / "detected_pairs.json"))
    a_prompts = {p["id"]: build_prompt(tokenizer, p["A_question"])
                 for p in detected}
    published_specific = json.load(open(ARTIFACTS_DIR / "specific_features.json"))
    results_main = json.load(open(ARTIFACTS_DIR / "results_main.json"))

    print(f"[setup] {len(detected)} detected pairs, "
          f"{sum(len(p['disambigs']) for p in detected)} self-pairs")

    decomp, ratios, strat, set_dist = run_a(
        tokenizer, model, sae, detected, a_prompts,
        published_specific, results_main,
    )
    k2_diag, mo_bins = run_b(detected, published_specific, results_main)
    k10_summary = run_c(detected, published_specific, results_main)

    # Comparison table (rolls up the overlap-shell analyses and (A))
    rows = [
        {"row": "published Targeted (full ablation)",
         "n_pairs": decomp["n"],
         "delta_pp": decomp["table"]["Targeted"]["delta_pp"],
         "ci_pp": decomp["table"]["Targeted"]["ci_pp"]},
        {"row": "(A) Shared-only ablation",
         "n_pairs": decomp["n"],
         "delta_pp": decomp["table"]["Shared-only"]["delta_pp"],
         "ci_pp": decomp["table"]["Shared-only"]["ci_pp"],
         "pct_of_targeted": decomp["table"]["Shared-only"]["pct_of_targeted"],
         "mcnemar_p": decomp["tests"]["shared_vs_baseline"]["mcnemar"]["p"],
         "wilcoxon_p": decomp["tests"]["shared_vs_baseline"]["wilcoxon_p"]},
        {"row": "(A) Unique-only ablation",
         "n_pairs": decomp["n"],
         "delta_pp": decomp["table"]["Unique-only"]["delta_pp"],
         "ci_pp": decomp["table"]["Unique-only"]["ci_pp"],
         "pct_of_targeted": decomp["table"]["Unique-only"]["pct_of_targeted"],
         "mcnemar_p": decomp["tests"]["unique_vs_baseline"]["mcnemar"]["p"],
         "wilcoxon_p": decomp["tests"]["unique_vs_baseline"]["wilcoxon_p"]},
    ]
    save_json_atomic(P12PP_DIR / "comparison_table.json", {"rows": rows})

    save_json_atomic(P12PP_DIR / "run_meta.json", {
        "model": MODEL_ID, "layer": LAYER, "sae_meta": sae_meta,
        "n_detected_pairs": len(detected),
        "elapsed_seconds": round(time.time() - t_total, 2),
    })

    print(f"\n=== Unique/shared analysis complete ===")
    print(f"  total elapsed: {(time.time()-t_total)/60:.2f} min")
    print(f"  -> {P12PP_DIR}/comparison_table.json")


if __name__ == "__main__":
    main()
