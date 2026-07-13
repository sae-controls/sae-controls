"""Polysemy decomposition under the two-tier framing.

Two jobs:
  (I)  Robustness-test the published polysemy decomposition (Methodology
       Sec. 5; 77/12/12 at L37) on its own terms (sub-experiments A, B).
  (II) Cross with the new unique-vs-shared partition to test whether the
       answer-unique signal concentrates in semantic-content features
       (sub-experiments C, D).

Sub-experiments:
  (A) Polysemy threshold sweep. For threshold ∈
      {0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95}, rebuild the
      position-suspect partition from artifacts/wikitext_position_mode.json
      (per_feature stats) and re-run content-only and position-only
      ablations. Uses the same partition_top_k logic as polysemy_partition.py.

  (B) Adaptive variance-ratio threshold. Re-scan WikiText through the SAE
      to capture per-feature (sum, sum-of-squares) at position 0 vs other
      positions. Compute var(z at pos 0) / var(z at non-pos-0) per feature.
      Pick a cutoff matching the 0.80-partition count and report Jaccard,
      2x2 confusion, and the 3-feature calibration check.

  (C) Cross-decomposition statistics (CPU). For each disambig-derived
      feature, compute unique-fraction across its appearances in (P, D_i)
      top-10 lists. Joint contingency at the *appearance* level
      (unique × content, etc.). Cramér's V.

  (D) Four-way joint ablation decomposition. Partition each (P, D_i)'s
      Targeted top-10 into (unique_content, unique_position, shared_content,
      shared_position) using the published 0.80 partition × the unique/
      shared assignment. Run four new ablation conditions per (P, D_i),
      reuse baseline for empty subsets. Decomposition + per-feature
      normalization + 4-way residual.

Reads:
  artifacts/sae_encodings_L37.npz (not strictly used here but referenced
                                    by other phases)
  artifacts/detected_pairs.json
  artifacts/specific_features.json
  artifacts/results_main.json
  artifacts/wikitext_position_mode.json    (per_feature stats for (A))
  artifacts/feature_inventory.json

Writes:
  artifacts/four_way/a/{threshold_sweep.json, threshold_sweep.png}
  artifacts/four_way/b/{variance_ratios.json, adaptive_partition.json,
                          confusion_vs_080.json, side_by_side_decomp.json,
                          variance_ratio_histogram.png}
  artifacts/four_way/c/{per_feature_unique_fraction.json,
                          contingency_2x2.json, cramers_v.json,
                          unique_fraction_histogram.png}
  artifacts/four_way/d/{four_way_rows.json, decomposition.json,
                          per_feature_impact.json}
  artifacts/four_way/comparison_table.json
  artifacts/four_way/run_meta.json
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.analysis import (
    bootstrap_diff_ci_pp, mcnemar_paired, per_pair_means, wilcoxon_paired,
)
from src.config import (
    ARTIFACTS_DIR, LAYER, MODEL_ID, POSITION_NONZERO_FLOOR,
    POSITION_PCT_THRESHOLD, SAE_CHECKPOINT_SHA256, TOP_LOGITS_K,
    WIKITEXT_MAX_TOKENS,
)
from src.hooks import (
    capture_all_position_residuals, forward_with_ablation, hit_at_k,
)
from src.io_utils import save_json_atomic
from src.model import load_model
from src.position_mode import is_position_suspect, partition_top_k
from src.prompts import build_prompt
from src.sae import load_sae
from src.wikitext import load_paragraphs as load_wikitext_paragraphs


P14_DIR = ARTIFACTS_DIR / "four_way"
THRESHOLD_SWEEP = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def _suspect_set_at_threshold(per_feature, threshold, floor=POSITION_NONZERO_FLOOR):
    return {
        int(fid) for fid, stats in per_feature.items()
        if stats["n_nonzero"] >= floor and stats["pct_at_pos0"] >= threshold
    }


def _ablation_pair_outcomes(model, tokenizer, sae, detected, a_prompts,
                            partition_per_pair, baseline_existing, targeted_existing,
                            target_sets, desc):
    """Run content-only and position-only forwards per (P, D_i), respecting
    pure-content (empty position → position_only=base, content_only=targ)
    and all-position (empty content → content_only=base, position_only=targ)
    edge cases (matches polysemy_partition.py).

    `partition_per_pair` is dict (pair_id, target_idx) -> {position, content}.
    """
    rows = []
    n_pure_content = n_all_position = n_full = 0
    for key, parts in tqdm(partition_per_pair.items(), desc=desc):
        pid, ti = key
        a_prompt = a_prompts[pid]
        target = target_sets[key]
        base = baseline_existing[key]
        targ = targeted_existing[key]
        n_pos = len(parts["position"])
        n_cont = len(parts["content"])
        if n_pos == 0:
            content_only = targ; position_only = base; n_pure_content += 1
        elif n_cont == 0:
            content_only = base; position_only = targ; n_all_position += 1
        else:
            ab_c = forward_with_ablation(model, tokenizer, a_prompt, LAYER, sae,
                                          feature_ids=parts["content"], k=TOP_LOGITS_K)
            content_only = hit_at_k(ab_c["top_ids"], target, 1)
            ab_p = forward_with_ablation(model, tokenizer, a_prompt, LAYER, sae,
                                          feature_ids=parts["position"], k=TOP_LOGITS_K)
            position_only = hit_at_k(ab_p["top_ids"], target, 1)
            n_full += 1
        rows.append({
            "pair_id": pid, "target_idx": ti,
            "n_position": n_pos, "n_content": n_cont,
            "baseline_hit1": base, "targeted_hit1": targ,
            "content_only_hit1": content_only,
            "position_only_hit1": position_only,
        })
    return rows, {"n_pure_content": n_pure_content,
                  "n_all_position": n_all_position,
                  "n_full_forwards": n_full}


def _decompose(rows):
    n = len(rows)
    base = np.array([r["baseline_hit1"] for r in rows], dtype=float)
    targ = np.array([r["targeted_hit1"] for r in rows], dtype=float)
    cont = np.array([r["content_only_hit1"] for r in rows], dtype=float)
    pos = np.array([r["position_only_hit1"] for r in rows], dtype=float)
    td = (targ - base).mean() * 100
    cd = (cont - base).mean() * 100
    pd_ = (pos - base).mean() * 100
    sum_ = cd + pd_
    res = td - sum_
    pct = lambda x: (x / td * 100) if abs(td) > 1e-12 else float("nan")
    return {
        "n": n,
        "baseline_mean": float(base.mean()),
        "targeted_mean": float(targ.mean()),
        "content_only_mean": float(cont.mean()),
        "position_only_mean": float(pos.mean()),
        "delta_targeted_pp": float(td),
        "delta_content_pp": float(cd),
        "delta_position_pp": float(pd_),
        "delta_sum_pp": float(sum_),
        "residual_pp": float(res),
        "pct_targeted_content": float(pct(cd)),
        "pct_targeted_position": float(pct(pd_)),
        "pct_targeted_sum": float(pct(sum_)),
        "pct_targeted_residual": float(pct(res)),
    }


# =============================================================================
# (A) Threshold sweep
# =============================================================================

def run_a(tokenizer, model, sae, detected, a_prompts, published_specific,
          results_main, per_feature):
    out_dir = P14_DIR / "a"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (A) polysemy threshold sweep ==========")

    pub_top10 = {(e["pair_id"], e["disambig_idx"]): list(e["features"])
                 for e in published_specific}
    self_rows_main = results_main["self_rows"]
    baseline_existing = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                         for r in self_rows_main}
    targeted_existing = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                         for r in self_rows_main}
    target_sets = {(p["id"], di): set(d["first_token_variants"])
                   for p in detected for di, d in enumerate(p["disambigs"])}

    cells = []
    for thr in THRESHOLD_SWEEP:
        suspect = _suspect_set_at_threshold(per_feature, thr)
        partition = {}
        for key, feats in pub_top10.items():
            if key not in target_sets:
                continue
            pos, cont = partition_top_k(feats, suspect)
            partition[key] = {"position": pos, "content": cont}
        rows, counts = _ablation_pair_outcomes(
            model, tokenizer, sae, detected, a_prompts,
            partition, baseline_existing, targeted_existing,
            target_sets, desc=f"(A) thr={thr}",
        )
        decomp = _decompose(rows)
        cell = {"threshold": thr, "n_position_suspect_features": len(suspect),
                **counts, **decomp}
        cells.append(cell)
        print(f"[A thr={thr}] n_suspect={len(suspect):<4}  "
              f"Δ_targ={decomp['delta_targeted_pp']:+6.2f} "
              f"Δ_cont={decomp['delta_content_pp']:+6.2f} "
              f"Δ_pos={decomp['delta_position_pp']:+6.2f}  "
              f"residual={decomp['residual_pp']:+6.2f}  "
              f"(content {decomp['pct_targeted_content']:.0f}% / "
              f"position {decomp['pct_targeted_position']:.0f}% / "
              f"residual {decomp['pct_targeted_residual']:.0f}%)")
        torch.cuda.empty_cache()

    save_json_atomic(out_dir / "threshold_sweep.json", {"cells": cells})

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    thrs = [c["threshold"] for c in cells]
    cont_pct = [c["pct_targeted_content"] for c in cells]
    pos_pct = [c["pct_targeted_position"] for c in cells]
    res_pct = [c["pct_targeted_residual"] for c in cells]
    width = 0.04
    ax.bar([t - width for t in thrs], cont_pct, width, label="content-only", color="#3a7")
    ax.bar(thrs, pos_pct, width, label="position-only", color="#a73")
    ax.bar([t + width for t in thrs], res_pct, width, label="residual", color="#777")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("polysemy threshold (pct_at_pos0)")
    ax.set_ylabel("% of Targeted Δ pp")
    ax.set_title("polysemy decomposition vs threshold")
    ax.legend()
    for c in cells:
        ax.annotate(f"n_sus={c['n_position_suspect_features']}",
                    (c["threshold"], 105), ha="center", fontsize=8, color="#555")
    plt.tight_layout()
    plt.savefig(out_dir / "threshold_sweep.png", dpi=140)
    plt.close()
    return cells


# =============================================================================
# (B) Adaptive variance-ratio threshold
# =============================================================================

def run_b(tokenizer, model, sae, detected, a_prompts, published_specific,
          results_main, per_feature):
    out_dir = P14_DIR / "b"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (B) adaptive variance-ratio threshold ==========")

    # ---- Re-scan WikiText paragraphs for per-feature variance at pos 0 vs other ----
    paragraphs = load_wikitext_paragraphs(tokenizer)
    print(f"[B] reconstructed {len(paragraphs)} WikiText paragraphs")

    inv = json.load(open(ARTIFACTS_DIR / "feature_inventory.json"))
    all_feats = sorted(int(f) for f in inv["freq_full"].keys())
    feat_t = torch.tensor(all_feats, dtype=torch.long, device=sae.W_enc.device)
    F = len(all_feats)
    print(f"[B] computing var ratios for {F} disambig-derived features × {len(paragraphs)} paragraphs")

    n_pos0 = torch.zeros(F, dtype=torch.long, device=sae.W_enc.device)
    s_pos0 = torch.zeros(F, dtype=torch.float32, device=sae.W_enc.device)
    ss_pos0 = torch.zeros(F, dtype=torch.float32, device=sae.W_enc.device)
    n_oth = torch.zeros(F, dtype=torch.long, device=sae.W_enc.device)
    s_oth = torch.zeros(F, dtype=torch.float32, device=sae.W_enc.device)
    ss_oth = torch.zeros(F, dtype=torch.float32, device=sae.W_enc.device)

    for i, p_text in enumerate(tqdm(paragraphs, desc="(B) WT scan")):
        try:
            _, res = capture_all_position_residuals(
                model, tokenizer, p_text, layer=LAYER, max_len=WIKITEXT_MAX_TOKENS)
        except Exception as e:
            print(f"[B] paragraph {i} failed: {type(e).__name__}: {e}")
            continue
        z = sae.encode(res.to(dtype=sae.W_enc.dtype))   # (n_tokens, d_sae)
        z_subset = z.index_select(dim=1, index=feat_t).float()  # (n_tokens, F)
        # Position 0
        z0 = z_subset[0]   # (F,)
        n_pos0 += 1
        s_pos0 += z0
        ss_pos0 += z0 * z0
        # Other positions
        if z_subset.shape[0] > 1:
            zo = z_subset[1:]   # (n_other, F)
            n_oth += zo.shape[0]
            s_oth += zo.sum(dim=0)
            ss_oth += (zo * zo).sum(dim=0)
        del res, z, z_subset, z0
        if i % 50 == 0:
            torch.cuda.empty_cache(); gc.collect()

    n_pos0_np = n_pos0.cpu().numpy().astype(float)
    s_pos0_np = s_pos0.cpu().numpy()
    ss_pos0_np = ss_pos0.cpu().numpy()
    n_oth_np = n_oth.cpu().numpy().astype(float)
    s_oth_np = s_oth.cpu().numpy()
    ss_oth_np = ss_oth.cpu().numpy()

    # Sample variance: var = (sum_sq - sum^2/n) / (n - 1)
    def _var(s, ss, n):
        out = np.zeros_like(s)
        mask = n > 1
        out[mask] = (ss[mask] - s[mask]**2 / n[mask]) / (n[mask] - 1)
        return np.where(out > 0, out, 0.0)

    var0 = _var(s_pos0_np, ss_pos0_np, n_pos0_np)
    var_oth = _var(s_oth_np, ss_oth_np, n_oth_np)
    eps = 1e-9
    ratio = np.where(var_oth > eps, var0 / np.maximum(var_oth, eps), np.inf)
    # If both vars are zero, ratio is undefined; mark with NaN
    ratio = np.where((var0 == 0) & (var_oth == 0), np.nan, ratio)

    save_json_atomic(out_dir / "variance_ratios.json", {
        "n_features": int(F),
        "n_paragraphs_scanned": len(paragraphs),
        "per_feature": {
            int(all_feats[i]): {
                "var_at_pos0": float(var0[i]),
                "var_at_other": float(var_oth[i]),
                "ratio": float(ratio[i]) if np.isfinite(ratio[i]) else None,
                "n_pos0": int(n_pos0_np[i]),
                "n_other": int(n_oth_np[i]),
            }
            for i in range(F)
        },
    })

    # ---- Pick a cutoff matching the 0.80-partition count ----
    suspect_080 = _suspect_set_at_threshold(per_feature, 0.80)
    target_count = len(suspect_080)
    finite_ratios = ratio[np.isfinite(ratio)]
    sorted_desc = np.sort(finite_ratios)[::-1]
    if target_count >= len(sorted_desc):
        cutoff = float(sorted_desc[-1]) if len(sorted_desc) else float("inf")
    else:
        cutoff = float(sorted_desc[target_count - 1])
    print(f"[B] target count from 0.80 partition: {target_count}")
    print(f"[B] cutoff variance-ratio: {cutoff:.4f} (top-{target_count} of finite ratios)")

    suspect_adaptive = {int(all_feats[i]) for i in range(F)
                        if np.isfinite(ratio[i]) and ratio[i] >= cutoff}
    print(f"[B] adaptive suspect count: {len(suspect_adaptive)}")

    # Confusion vs 0.80 partition (universe = 2021 disambig-derived features)
    universe = {int(f) for f in all_feats}
    a_only = suspect_adaptive - suspect_080
    b_only = suspect_080 - suspect_adaptive
    both = suspect_adaptive & suspect_080
    neither = universe - suspect_adaptive - suspect_080
    jaccard = len(both) / max(len(suspect_adaptive | suspect_080), 1)

    confusion = {
        "universe_size": len(universe),
        "n_080_partition": len(suspect_080),
        "n_adaptive_partition": len(suspect_adaptive),
        "both": len(both),
        "adaptive_only": len(a_only),
        "080_only": len(b_only),
        "neither": len(neither),
        "jaccard_index": float(jaccard),
    }
    save_json_atomic(out_dir / "confusion_vs_080.json", confusion)
    print(f"[B] vs 0.80: both={len(both)}, adaptive_only={len(a_only)}, "
          f"080_only={len(b_only)}, jaccard={jaccard:.3f}")

    # 3-feature calibration check
    KNOWN_POLYSEMANTIC = [7187, 9825, 15382]
    calib = {}
    for f in KNOWN_POLYSEMANTIC:
        idx = all_feats.index(f) if f in all_feats else None
        calib[str(f)] = {
            "in_080_partition": (f in suspect_080),
            "in_adaptive_partition": (f in suspect_adaptive),
            "var_at_pos0": float(var0[idx]) if idx is not None else None,
            "var_at_other": float(var_oth[idx]) if idx is not None else None,
            "ratio": (float(ratio[idx]) if idx is not None and np.isfinite(ratio[idx])
                       else None),
        }
    save_json_atomic(out_dir / "adaptive_partition.json", {
        "method": "match-count to 0.80 partition; rank features by variance ratio "
                  "var(z at pos 0) / var(z at non-pos-0) descending",
        "cutoff": cutoff,
        "n_flagged": len(suspect_adaptive),
        "known_polysemantic_features": calib,
        "flagged_feature_ids": sorted(suspect_adaptive)[:200],  # truncate for size
        "flagged_feature_ids_count": len(suspect_adaptive),
    })
    print(f"[B] calibration: {calib}")

    # Histogram of variance ratios (log-scale)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    log_finite = np.log10(np.maximum(finite_ratios, 1e-6))
    ax.hist(log_finite, bins=80, color="#37a", edgecolor="white")
    ax.axvline(np.log10(max(cutoff, 1e-6)), color="red", linestyle="--",
               label=f"adaptive cutoff log10 = {np.log10(max(cutoff,1e-6)):.2f}")
    ax.set_xlabel("log10(var(z at pos 0) / var(z at non-pos-0))")
    ax.set_ylabel("count of features")
    ax.set_title("variance-ratio histogram (n=2021 disambig features)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "variance_ratio_histogram.png", dpi=140)
    plt.close()

    # ---- Side-by-side decomposition under both partitions ----
    pub_top10 = {(e["pair_id"], e["disambig_idx"]): list(e["features"])
                 for e in published_specific}
    self_rows_main = results_main["self_rows"]
    baseline_existing = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                         for r in self_rows_main}
    targeted_existing = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                         for r in self_rows_main}
    target_sets = {(p["id"], di): set(d["first_token_variants"])
                   for p in detected for di, d in enumerate(p["disambigs"])}

    side_by_side = {}
    for label, suspect in [("0.80_published", suspect_080),
                            ("adaptive", suspect_adaptive)]:
        partition = {}
        for key, feats in pub_top10.items():
            if key not in target_sets:
                continue
            pos, cont = partition_top_k(feats, suspect)
            partition[key] = {"position": pos, "content": cont}
        rows, counts = _ablation_pair_outcomes(
            model, tokenizer, sae, detected, a_prompts,
            partition, baseline_existing, targeted_existing,
            target_sets, desc=f"(B) {label}",
        )
        decomp = _decompose(rows)
        side_by_side[label] = {**counts, **decomp,
                                "n_position_suspect_features": len(suspect)}
        print(f"[B {label}] Δ_targ={decomp['delta_targeted_pp']:+6.2f}  "
              f"Δ_cont={decomp['delta_content_pp']:+6.2f}  "
              f"Δ_pos={decomp['delta_position_pp']:+6.2f}  "
              f"residual={decomp['residual_pp']:+6.2f}")
        torch.cuda.empty_cache()

    save_json_atomic(out_dir / "side_by_side_decomp.json", side_by_side)
    return side_by_side, confusion, calib, cutoff


# =============================================================================
# (C) Cross-decomposition statistics — CPU
# =============================================================================

def run_c(detected, published_specific, per_feature):
    out_dir = P14_DIR / "c"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (C) cross-decomposition statistics ==========")

    # 0.80 partition
    suspect_080 = _suspect_set_at_threshold(per_feature, 0.80)
    print(f"[C] 0.80 partition size: {len(suspect_080)}")

    # Build pub_set per (P, D_i)
    pub_set = {(e["pair_id"], e["disambig_idx"]): set(e["features"])
               for e in published_specific}

    # For each (P, D_i): identify unique vs shared per feature
    appearances = []   # (feature_id, is_unique)
    per_feature_appearances = defaultdict(lambda: {"n_total": 0, "n_unique": 0})

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
            for fid in ti:
                is_unique = fid not in sib_union
                appearances.append((int(fid), is_unique))
                per_feature_appearances[int(fid)]["n_total"] += 1
                if is_unique:
                    per_feature_appearances[int(fid)]["n_unique"] += 1

    # Per-feature unique fraction
    per_feature_uf = {
        fid: {
            "n_appearances": v["n_total"],
            "n_unique": v["n_unique"],
            "unique_fraction": v["n_unique"] / v["n_total"] if v["n_total"] else None,
            "position_suspect_080": (fid in suspect_080),
        }
        for fid, v in per_feature_appearances.items()
    }
    save_json_atomic(out_dir / "per_feature_unique_fraction.json", per_feature_uf)
    n_features = len(per_feature_uf)
    n_pos_features = sum(1 for v in per_feature_uf.values() if v["position_suspect_080"])
    n_features_100p_unique = sum(1 for v in per_feature_uf.values() if v["unique_fraction"] == 1.0)
    n_features_100p_shared = sum(1 for v in per_feature_uf.values() if v["unique_fraction"] == 0.0)
    print(f"[C] {n_features} features; "
          f"{n_pos_features} position-suspect (0.80); "
          f"{n_features_100p_unique} are 100%-unique; "
          f"{n_features_100p_shared} are 100%-shared")

    # 2x2 contingency at the APPEARANCE level
    cells = {("unique", "content"): 0, ("unique", "position"): 0,
             ("shared", "content"): 0, ("shared", "position"): 0}
    for fid, is_unique in appearances:
        u_label = "unique" if is_unique else "shared"
        p_label = "position" if fid in suspect_080 else "content"
        cells[(u_label, p_label)] += 1

    # Marginal totals
    n_total = sum(cells.values())
    n_unique = cells[("unique", "content")] + cells[("unique", "position")]
    n_shared = cells[("shared", "content")] + cells[("shared", "position")]
    n_content = cells[("unique", "content")] + cells[("shared", "content")]
    n_position = cells[("unique", "position")] + cells[("shared", "position")]

    # Expected counts under independence
    expected = {
        ("unique", "content"): n_unique * n_content / n_total,
        ("unique", "position"): n_unique * n_position / n_total,
        ("shared", "content"): n_shared * n_content / n_total,
        ("shared", "position"): n_shared * n_position / n_total,
    }
    chi2 = sum(
        ((cells[k] - expected[k]) ** 2) / expected[k] if expected[k] > 0 else 0
        for k in cells
    )
    # Cramér's V for 2x2: V = sqrt(chi2 / (n * min(r-1, c-1))) = sqrt(chi2 / n)
    cramers_v = float((chi2 / n_total) ** 0.5)

    # Phi coefficient (signed for 2x2)
    a = cells[("unique", "content")]
    b = cells[("unique", "position")]
    c = cells[("shared", "content")]
    d = cells[("shared", "position")]
    denom = ((a + b) * (c + d) * (a + c) * (b + d)) ** 0.5
    phi = (a * d - b * c) / denom if denom > 0 else 0.0

    # Fisher's exact test (or chi-squared) for significance
    try:
        from scipy.stats import chi2_contingency
        chi2_stat, p_val, dof, _ = chi2_contingency(
            [[a, b], [c, d]], correction=False
        )
    except Exception:
        chi2_stat, p_val, dof = chi2, None, 1

    contingency = {
        "level": "feature appearances in (P, D_i) top-10 lists",
        "n_total_appearances": n_total,
        "cells_observed": {f"{k[0]}|{k[1]}": v for k, v in cells.items()},
        "cells_expected": {f"{k[0]}|{k[1]}": float(v) for k, v in expected.items()},
        "marginal_totals": {
            "unique": n_unique, "shared": n_shared,
            "content": n_content, "position": n_position,
        },
        "chi2_statistic": float(chi2_stat),
        "chi2_p_value": float(p_val) if p_val is not None else None,
        "dof": int(dof),
    }
    save_json_atomic(out_dir / "contingency_2x2.json", contingency)

    cv_block = {
        "cramers_v": cramers_v,
        "phi_coefficient_signed": float(phi),
        "interpretation_phi_sign": (
            "positive: unique appearances are over-represented in content; "
            "shared appearances over-represented in position. Negative: reverse."
        ),
    }
    save_json_atomic(out_dir / "cramers_v.json", cv_block)
    print(f"[C] contingency cells (unique×content, unique×position, "
          f"shared×content, shared×position): {a}, {b}, {c}, {d}")
    print(f"[C] expected: {expected}")
    print(f"[C] χ² = {chi2:.2f}, p = {p_val if p_val is not None else 'n/a'}")
    print(f"[C] Cramér's V = {cramers_v:.4f}, signed φ = {phi:+.4f}")

    # Histogram of unique fractions
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ufs = [v["unique_fraction"] for v in per_feature_uf.values() if v["unique_fraction"] is not None]
    ax.hist(ufs, bins=20, color="#37a", edgecolor="white")
    ax.axvline(np.mean(ufs), color="red", linestyle="--",
               label=f"mean = {np.mean(ufs):.3f}")
    ax.set_xlabel("per-feature unique fraction (across all (P, D_i) appearances)")
    ax.set_ylabel("count of features")
    ax.set_title(f"unique fraction (n_features={len(ufs)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "unique_fraction_histogram.png", dpi=140)
    plt.close()
    return cells, contingency, cv_block


# =============================================================================
# (D) Four-way joint ablation decomposition
# =============================================================================

def run_d(tokenizer, model, sae, detected, a_prompts, published_specific,
          results_main, per_feature):
    out_dir = P14_DIR / "d"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (D) four-way joint ablation decomposition ==========")

    suspect_080 = _suspect_set_at_threshold(per_feature, 0.80)

    pub_set = {(e["pair_id"], e["disambig_idx"]): set(e["features"])
               for e in published_specific}
    self_rows_main = results_main["self_rows"]
    baseline_existing = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                         for r in self_rows_main}
    targeted_existing = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                         for r in self_rows_main}
    target_sets = {(p["id"], di): set(d["first_token_variants"])
                   for p in detected for di, d in enumerate(p["disambigs"])}

    # Build 4-way partition per (P, D_i)
    parts_per_pair = {}
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
            uc = unique - suspect_080  # unique_content
            up = unique & suspect_080  # unique_position
            sc = shared - suspect_080  # shared_content
            sp = shared & suspect_080  # shared_position
            parts_per_pair[(p["id"], i)] = {
                "unique_content": list(uc), "unique_position": list(up),
                "shared_content": list(sc), "shared_position": list(sp),
            }

    # Run 4 ablations per (P, D_i) — skip empty subsets, use baseline
    rows = []
    n_skipped = {"unique_content": 0, "unique_position": 0,
                 "shared_content": 0, "shared_position": 0}
    n_run = {"unique_content": 0, "unique_position": 0,
             "shared_content": 0, "shared_position": 0}
    for key, parts in tqdm(parts_per_pair.items(), desc="(D) 4-way"):
        pid, ti = key
        a_prompt = a_prompts[pid]
        target = target_sets[key]
        base = baseline_existing[key]
        targ = targeted_existing[key]

        hits = {"baseline": base, "targeted": targ}
        for subset_name in ["unique_content", "unique_position",
                            "shared_content", "shared_position"]:
            feats = parts[subset_name]
            if not feats:
                hits[subset_name] = base
                n_skipped[subset_name] += 1
            else:
                ab = forward_with_ablation(
                    model, tokenizer, a_prompt, LAYER, sae,
                    feature_ids=feats, k=TOP_LOGITS_K,
                )
                hits[subset_name] = hit_at_k(ab["top_ids"], target, 1)
                n_run[subset_name] += 1
        rows.append({
            "pair_id": pid, "target_idx": ti,
            "n_uc": len(parts["unique_content"]),
            "n_up": len(parts["unique_position"]),
            "n_sc": len(parts["shared_content"]),
            "n_sp": len(parts["shared_position"]),
            "baseline_hit1": base, "targeted_hit1": targ,
            "uc_hit1": hits["unique_content"],
            "up_hit1": hits["unique_position"],
            "sc_hit1": hits["shared_content"],
            "sp_hit1": hits["shared_position"],
        })
        if (sum(n_run.values()) + sum(n_skipped.values())) % 200 == 0:
            torch.cuda.empty_cache()
    save_json_atomic(out_dir / "four_way_rows.json", rows)
    print(f"[D] forwards run / skipped per subset:")
    for name in ["unique_content", "unique_position", "shared_content", "shared_position"]:
        print(f"  {name:<16}  run={n_run[name]:<4}  skipped(empty)={n_skipped[name]}")

    # Decomposition table
    n = len(rows)
    base = np.array([r["baseline_hit1"] for r in rows], dtype=float)
    targ = np.array([r["targeted_hit1"] for r in rows], dtype=float)
    uc = np.array([r["uc_hit1"] for r in rows], dtype=float)
    up = np.array([r["up_hit1"] for r in rows], dtype=float)
    sc = np.array([r["sc_hit1"] for r in rows], dtype=float)
    sp = np.array([r["sp_hit1"] for r in rows], dtype=float)

    td = (targ - base).mean() * 100
    d_uc = (uc - base).mean() * 100
    d_up = (up - base).mean() * 100
    d_sc = (sc - base).mean() * 100
    d_sp = (sp - base).mean() * 100
    sum_4way = d_uc + d_up + d_sc + d_sp
    residual = td - sum_4way
    pct = lambda x: (x / td * 100) if abs(td) > 1e-12 else float("nan")

    # CIs and tests vs baseline
    def _vs_base(arr, label):
        mc = mcnemar_paired(arr, base, alternative="greater")  # arr kills more than base
        wx = wilcoxon_paired(base, arr, alternative="greater")
        ci = list(bootstrap_diff_ci_pp(base, arr, seed=0))
        return {"label": label,
                "mean_hit1": float(arr.mean()),
                "delta_pp": float((arr - base).mean() * 100),
                "ci_pp": [float(ci[0]), float(ci[1])],
                "mcnemar": mc, "wilcoxon_p": wx}

    tests = {
        "targeted":  _vs_base(targ, "Targeted"),
        "unique_content":  _vs_base(uc, "unique_content"),
        "unique_position": _vs_base(up, "unique_position"),
        "shared_content":  _vs_base(sc, "shared_content"),
        "shared_position": _vs_base(sp, "shared_position"),
    }

    # Per-feature normalization
    def _mean_size(name, key):
        sizes = [r[key] for r in rows if r[key] > 0]
        return float(np.mean(sizes)) if sizes else 0.0

    mean_sizes = {
        "unique_content":  _mean_size("uc", "n_uc"),
        "unique_position": _mean_size("up", "n_up"),
        "shared_content":  _mean_size("sc", "n_sc"),
        "shared_position": _mean_size("sp", "n_sp"),
    }
    deltas = {
        "unique_content":  d_uc,
        "unique_position": d_up,
        "shared_content":  d_sc,
        "shared_position": d_sp,
    }
    per_feat_impact = {
        name: {
            "delta_pp_aggregate": float(deltas[name]),
            "mean_subset_size_when_nonempty": mean_sizes[name],
            "delta_per_feature_pp":
                float(deltas[name] / mean_sizes[name]) if mean_sizes[name] > 0 else None,
        }
        for name in mean_sizes
    }

    # Cross-decomposition table
    cross_table = {
        "rows": ["unique", "shared"],
        "cols": ["content", "position"],
        "delta_pp": {
            "unique|content": d_uc, "unique|position": d_up,
            "shared|content": d_sc, "shared|position": d_sp,
        },
        "ci_pp": {
            "unique|content": tests["unique_content"]["ci_pp"],
            "unique|position": tests["unique_position"]["ci_pp"],
            "shared|content": tests["shared_content"]["ci_pp"],
            "shared|position": tests["shared_position"]["ci_pp"],
        },
        "p_mcnemar": {
            "unique|content": tests["unique_content"]["mcnemar"]["p"],
            "unique|position": tests["unique_position"]["mcnemar"]["p"],
            "shared|content": tests["shared_content"]["mcnemar"]["p"],
            "shared|position": tests["shared_position"]["mcnemar"]["p"],
        },
        "p_wilcoxon": {
            "unique|content": tests["unique_content"]["wilcoxon_p"],
            "unique|position": tests["unique_position"]["wilcoxon_p"],
            "shared|content": tests["shared_content"]["wilcoxon_p"],
            "shared|position": tests["shared_position"]["wilcoxon_p"],
        },
        "delta_per_feature_pp": {
            "unique|content": per_feat_impact["unique_content"]["delta_per_feature_pp"],
            "unique|position": per_feat_impact["unique_position"]["delta_per_feature_pp"],
            "shared|content": per_feat_impact["shared_content"]["delta_per_feature_pp"],
            "shared|position": per_feat_impact["shared_position"]["delta_per_feature_pp"],
        },
        "mean_subset_sizes": mean_sizes,
        "n_skipped_empty": n_skipped,
        "n_run": n_run,
    }
    decomp = {
        "n": n,
        "delta_targeted_pp": float(td),
        "delta_4way_sum_pp": float(sum_4way),
        "four_way_residual_pp": float(residual),
        "pct_residual_of_targeted": float(pct(residual)),
        "components": {
            "unique_content": float(d_uc),
            "unique_position": float(d_up),
            "shared_content": float(d_sc),
            "shared_position": float(d_sp),
        },
        "pct_of_targeted": {
            "unique_content": float(pct(d_uc)),
            "unique_position": float(pct(d_up)),
            "shared_content": float(pct(d_sc)),
            "shared_position": float(pct(d_sp)),
        },
        "tests_vs_baseline": tests,
        "cross_table": cross_table,
    }
    save_json_atomic(out_dir / "decomposition.json", decomp)
    save_json_atomic(out_dir / "per_feature_impact.json", per_feat_impact)

    print(f"[D] decomposition (n={n}, Δ_targ={td:+6.2f} pp):")
    for name in ["unique_content", "unique_position", "shared_content", "shared_position"]:
        ci = tests[name]["ci_pp"]
        print(f"  {name:<18}  Δ={deltas[name]:+6.2f}  ({pct(deltas[name]):+6.1f}%)  "
              f"CI=[{ci[0]:+5.2f},{ci[1]:+5.2f}]  "
              f"McN p={tests[name]['mcnemar']['p']:.3g}  "
              f"Wlx p={tests[name]['wilcoxon_p']:.3g}  "
              f"per-feat={per_feat_impact[name]['delta_per_feature_pp']}")
    print(f"  Sum (4-way): {sum_4way:+6.2f}  ({pct(sum_4way):+6.1f}%)")
    print(f"  4-way residual: {residual:+6.2f}  ({pct(residual):+6.1f}%)")
    return decomp, per_feat_impact


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    P14_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"[setup] loading model + SAE...")
    tokenizer, model = load_model()
    sae, sae_meta = load_sae(layer=LAYER)
    if sae_meta["sha256"] != SAE_CHECKPOINT_SHA256:
        raise RuntimeError(f"SAE sha256 mismatch! got {sae_meta['sha256']}")
    print(f"[setup] SAE sha256 verified ({sae_meta['sha256'][:16]}...)")

    detected = json.load(open(ARTIFACTS_DIR / "detected_pairs.json"))
    a_prompts = {p["id"]: build_prompt(tokenizer, p["A_question"]) for p in detected}
    published_specific = json.load(open(ARTIFACTS_DIR / "specific_features.json"))
    results_main = json.load(open(ARTIFACTS_DIR / "results_main.json"))
    pos_mode = json.load(open(ARTIFACTS_DIR / "wikitext_position_mode.json"))
    per_feature = pos_mode["per_feature"]

    print(f"[setup] {len(detected)} detected pairs, "
          f"{sum(len(p['disambigs']) for p in detected)} self-pairs")
    print(f"[setup] published 0.80 partition: {pos_mode['n_position_suspect']} "
          f"of {pos_mode['n_features_scanned']}")

    a_cells = run_a(tokenizer, model, sae, detected, a_prompts, published_specific,
                     results_main, per_feature)
    b_side, b_confusion, b_calib, b_cutoff = run_b(
        tokenizer, model, sae, detected, a_prompts, published_specific,
        results_main, per_feature,
    )
    c_cells, c_contingency, c_cv = run_c(detected, published_specific, per_feature)
    d_decomp, d_per_feat = run_d(
        tokenizer, model, sae, detected, a_prompts, published_specific,
        results_main, per_feature,
    )

    # Comparison rollup
    rollup = {
        "phase_a_thresholds": [
            {
                "threshold": c["threshold"],
                "n_position_suspect": c["n_position_suspect_features"],
                "delta_targeted_pp": c["delta_targeted_pp"],
                "pct_content": c["pct_targeted_content"],
                "pct_position": c["pct_targeted_position"],
                "pct_residual": c["pct_targeted_residual"],
            }
            for c in a_cells
        ],
        "phase_b_adaptive": b_side,
        "phase_b_confusion": b_confusion,
        "phase_b_calibration": b_calib,
        "phase_c_cramers_v": c_cv,
        "phase_d_4way": {
            "delta_targeted_pp": d_decomp["delta_targeted_pp"],
            "components": d_decomp["components"],
            "pct_of_targeted": d_decomp["pct_of_targeted"],
            "four_way_residual_pp": d_decomp["four_way_residual_pp"],
        },
    }
    save_json_atomic(P14_DIR / "comparison_table.json", rollup)
    save_json_atomic(P14_DIR / "run_meta.json", {
        "model": MODEL_ID, "layer": LAYER, "sae_meta": sae_meta,
        "n_detected_pairs": len(detected),
        "threshold_sweep": THRESHOLD_SWEEP,
        "elapsed_seconds": round(time.time() - t_total, 2),
    })
    print(f"\n=== Stage 1.4 complete ===")
    print(f"  total elapsed: {(time.time()-t_total)/60:.1f} min")
    print(f"  -> {P14_DIR}")


if __name__ == "__main__":
    main()
