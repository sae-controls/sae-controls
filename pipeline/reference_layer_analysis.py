"""Reference-layer analysis at L41.

Re-runs the layer-specific analysis suite at L41 to produce the data
the paper cites in Sec. 4.2-Sec. 4.5:

  (A) Polysemy partition + threshold sweep at L41
  (B) 4-way joint decomposition at L41
  (C) Cross-decomposition Cramér's V at L41
  (D) Multi-metric: KL + generation-flip at L41 (logit-difference and rank-shift are reported at L37 only; see multimetric.py)

Reuses the saved L41 artifacts where possible:
  - artifacts/layer_bookends/L41/sae_encodings_L41.npz (A/D encodings)
  - artifacts/layer_bookends/L41/specific_features.json (Targeted top-10 sets)
  - artifacts/layer_bookends/L41/results_main.json (six-condition rows;
     hits only, no logits — we re-harvest logits for (D)'s KL).

Reads:
  artifacts/detected_pairs.json
  artifacts/layer_bookends/L41/{sae_encodings_L41.npz, specific_features.json,
                              results_main.json, run_meta.json,
                              unique_shared_decomp.json,
                              per_feature_equivalence.json}

Writes everything to artifacts/reference_layer/.
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
    ARTIFACTS_DIR, MODEL_ID, POSITION_NONZERO_FLOOR,
    POSITION_PCT_THRESHOLD, TOP_LOGITS_K, WIKITEXT_MAX_TOKENS,
)
from src.hooks import (
    baseline_top_logits, capture_all_position_residuals,
    forward_with_ablation, forward_with_ablation_then_generate,
    greedy_decode, hit_at_k,
)
from src.io_utils import save_json_atomic, save_npz_atomic
from src.model import load_model
from src.position_mode import is_position_suspect, partition_top_k
from src.prompts import build_prompt
from src.sae import load_sae
from src.slot_detection import find_best_match
from src.wikitext import load_paragraphs as load_wikitext_paragraphs


R0_DIR = ARTIFACTS_DIR / "reference_layer"
LAYER = 41
SAVE_TOP_K = 50
KL_SUPPORT_K = 10
GEN_MAX_NEW = 15
THRESHOLD_SWEEP = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def _drop_vs_base_stats(arr, base_arr, alt="greater"):
    if len(arr) < 2:
        return None
    mc = mcnemar_paired(arr, base_arr, alternative=alt)
    wx = wilcoxon_paired(base_arr, arr, alternative=alt)
    ci = bootstrap_diff_ci_pp(base_arr, arr, seed=0)
    return {
        "n": int(len(arr)),
        "delta_pp": float((base_arr.mean() - arr.mean()) * 100),
        "ci_pp": [float(ci[0]), float(ci[1])],
        "mcnemar_p": mc["p"],
        "wilcoxon_p": wx,
    }


def _kl_topk(p_logits, p_ids, q_logits, q_ids, k=KL_SUPPORT_K):
    """KL(p || q) on p's top-k support; out-of-q-top-50 ids censored."""
    pl = np.array(p_logits[:k], dtype=np.float64)
    p_top_ids = list(p_ids[:k])
    q_lookup = dict(zip(q_ids, q_logits))
    q_floor = float(min(q_logits)) - 5.0
    q_at_p = np.array([q_lookup.get(i, q_floor) for i in p_top_ids], dtype=np.float64)
    p_norm = pl - pl.max()
    q_norm = q_at_p - q_at_p.max()
    p_dist = np.exp(p_norm) / np.exp(p_norm).sum()
    q_dist = np.exp(q_norm) / np.exp(q_norm).sum()
    eps = 1e-12
    return float(np.sum(p_dist * (np.log(p_dist + eps) - np.log(q_dist + eps))))


# =============================================================================
# (A) Polysemy partition at L41 + threshold sweep
# =============================================================================

def run_a_polysemy_at_L41(model, tokenizer, sae, paragraphs, detected,
                           a_prompts, specific_features):
    out_dir = R0_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (A) polysemy at L41 + threshold sweep ==========")

    # ---- Capture all-position WT residuals + per-feature pos-0 stats ----
    print(f"[A] capturing all-position WT residuals at L{LAYER} (800 paragraphs)...")
    n_features = sae.d_sae
    n_at_pos0 = np.zeros(n_features, dtype=np.int64)
    n_nonzero = np.zeros(n_features, dtype=np.int64)
    sum_act_when_nonzero = np.zeros(n_features, dtype=np.float64)
    sum_sq_act_when_nonzero = np.zeros(n_features, dtype=np.float64)
    n_paragraphs_processed = 0

    for i, p_text in enumerate(tqdm(paragraphs, desc="A WT all-pos")):
        try:
            _, res = capture_all_position_residuals(
                model, tokenizer, p_text, layer=LAYER, max_len=WIKITEXT_MAX_TOKENS,
            )
            # Encode all positions
            z = sae.encode(res.to(dtype=sae.W_enc.dtype))  # (n_pos, d_sae)
            z_np = z.float().cpu().numpy()
            nonzero_mask = z_np > 0
            n_nonzero += nonzero_mask.sum(axis=0)
            sum_act_when_nonzero += np.where(nonzero_mask, z_np, 0).sum(axis=0)
            sum_sq_act_when_nonzero += np.where(nonzero_mask, z_np * z_np, 0).sum(axis=0)
            # Position 0 nonzeros
            if nonzero_mask.shape[0] > 0:
                n_at_pos0 += nonzero_mask[0].astype(np.int64)
            n_paragraphs_processed += 1
            del res, z, z_np
        except Exception as e:
            print(f"[A] paragraph {i} failed: {type(e).__name__}: {e}")
        if i % 50 == 0:
            torch.cuda.empty_cache(); gc.collect()

    pct_at_pos0 = np.where(n_nonzero > 0,
                            n_at_pos0.astype(np.float64) / np.maximum(n_nonzero, 1),
                            0.0)
    mean_act = np.where(n_nonzero > 0,
                         sum_act_when_nonzero / np.maximum(n_nonzero, 1),
                         0.0)
    var_act = np.where(n_nonzero > 1,
                        (sum_sq_act_when_nonzero - n_nonzero * mean_act ** 2)
                        / np.maximum(n_nonzero - 1, 1),
                        0.0)

    # Restrict per_feature output to disambig-derived feature pool (saves space)
    derived_feats = set()
    for entry in specific_features:
        for f in entry["features"]:
            derived_feats.add(int(f))
    print(f"[A] disambig-derived feature pool size: {len(derived_feats)}")

    per_feature = {}
    for f in derived_feats:
        per_feature[str(f)] = {
            "n_nonzero": int(n_nonzero[f]),
            "n_at_pos0": int(n_at_pos0[f]),
            "pct_at_pos0": float(pct_at_pos0[f]),
            "mean_act_when_nonzero": float(mean_act[f]),
            "variance_act_when_nonzero": float(var_act[f]),
            "position_suspect": bool(is_position_suspect(
                int(n_nonzero[f]), float(pct_at_pos0[f]),
                floor=POSITION_NONZERO_FLOOR,
                threshold=POSITION_PCT_THRESHOLD,
            )),
        }
    n_position_suspect = sum(1 for v in per_feature.values() if v["position_suspect"])
    print(f"[A] position-suspect (0.80 partition): {n_position_suspect} / {len(derived_feats)}")

    save_json_atomic(out_dir / "wikitext_position_mode_L41.json", {
        "layer": LAYER,
        "threshold_n_nonzero_floor": POSITION_NONZERO_FLOOR,
        "threshold_pct_at_pos0": POSITION_PCT_THRESHOLD,
        "n_paragraphs_processed": n_paragraphs_processed,
        "n_features_scanned": len(derived_feats),
        "n_position_suspect": n_position_suspect,
        "per_feature": per_feature,
    })

    # ---- Threshold sweep ----
    print(f"\n[A] threshold sweep on n_self_pairs...")
    pub_specific = {(e["pair_id"], e["disambig_idx"]): list(e["features"])
                    for e in specific_features}

    # Pull L41 baseline + targeted hits from the layer-bookends results_main
    p21b_results = json.load(open(ARTIFACTS_DIR / "layer_bookends" / "L41"
                                    / "results_main.json"))
    base_lookup = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                    for r in p21b_results["self_rows"]}
    targ_lookup = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                    for r in p21b_results["self_rows"]}

    sweep_results = []
    for thr in THRESHOLD_SWEEP:
        suspect = {int(f) for f, v in per_feature.items()
                   if v["n_nonzero"] >= POSITION_NONZERO_FLOOR
                   and v["pct_at_pos0"] >= thr}
        # Run content-only and position-only ablations per (P, D_i)
        rows = []
        for p in tqdm(detected, desc=f"A thr={thr:.2f}"):
            a_prompt = a_prompts[p["id"]]
            for i in range(len(p["disambigs"])):
                key = (p["id"], i)
                ti = pub_specific.get(key, [])
                if not ti or key not in base_lookup:
                    continue
                position_feats, content_feats = partition_top_k(ti, suspect)
                target = set(p["disambigs"][i]["first_token_variants"])
                base = base_lookup[key]
                if content_feats:
                    ab = forward_with_ablation(
                        model, tokenizer, a_prompt, LAYER, sae,
                        feature_ids=content_feats, k=TOP_LOGITS_K,
                    )
                    cont_hit = hit_at_k(ab["top_ids"], target, 1)
                else:
                    cont_hit = base
                if position_feats:
                    ab = forward_with_ablation(
                        model, tokenizer, a_prompt, LAYER, sae,
                        feature_ids=position_feats, k=TOP_LOGITS_K,
                    )
                    pos_hit = hit_at_k(ab["top_ids"], target, 1)
                else:
                    pos_hit = base
                rows.append({
                    "pair_id": p["id"], "target_idx": i,
                    "n_content": len(content_feats),
                    "n_position": len(position_feats),
                    "base_hit1": base,
                    "targeted_hit1": targ_lookup[key],
                    "content_only_hit1": cont_hit,
                    "position_only_hit1": pos_hit,
                })
            torch.cuda.empty_cache()

        Bsh = np.array([r["base_hit1"] for r in rows])
        Tsh = np.array([r["targeted_hit1"] for r in rows])
        Csh = np.array([r["content_only_hit1"] for r in rows])
        Psh = np.array([r["position_only_hit1"] for r in rows])
        targ_drop = (Tsh - Bsh).mean() * 100
        cont_drop = (Csh - Bsh).mean() * 100
        pos_drop = (Psh - Bsh).mean() * 100
        sum_ = cont_drop + pos_drop
        residual = targ_drop - sum_
        pct = lambda x: (x / targ_drop * 100) if abs(targ_drop) > 1e-9 else float("nan")
        cell = {
            "threshold": thr,
            "n_position_suspect_features": len(suspect),
            "n": len(rows),
            "delta_targeted_pp": float(targ_drop),
            "delta_content_pp": float(cont_drop),
            "delta_position_pp": float(pos_drop),
            "residual_pp": float(residual),
            "pct_targeted_content": float(pct(cont_drop)),
            "pct_targeted_position": float(pct(pos_drop)),
            "pct_targeted_residual": float(pct(residual)),
        }
        sweep_results.append(cell)
        print(f"[A thr={thr:.2f}] n_suspect={len(suspect):<4}  Δ_targ={targ_drop:+5.2f}  "
              f"Δ_cont={cont_drop:+5.2f} ({pct(cont_drop):.0f}%)  "
              f"Δ_pos={pos_drop:+5.2f} ({pct(pos_drop):.0f}%)  "
              f"residual={residual:+5.2f} ({pct(residual):.0f}%)")

    poly_dir = out_dir / "polysemy"
    poly_dir.mkdir(parents=True, exist_ok=True)
    save_json_atomic(poly_dir / "threshold_sweep_L41.json", {
        "layer": LAYER,
        "thresholds": THRESHOLD_SWEEP,
        "cells": sweep_results,
    })

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    thr_x = [c["threshold"] for c in sweep_results]
    cont = [c["pct_targeted_content"] for c in sweep_results]
    pos = [c["pct_targeted_position"] for c in sweep_results]
    res = [c["pct_targeted_residual"] for c in sweep_results]
    width = 0.018
    ax.bar(thr_x, cont, width=width, color="#2c7fb8", label="Content")
    ax.bar(thr_x, pos, width=width, bottom=cont, color="#7fcdbb",
            label="Position")
    bot2 = [a + b for a, b in zip(cont, pos)]
    ax.bar(thr_x, res, width=width, bottom=bot2, color="#cccccc",
            label="Residual")
    ax.axvline(0.80, color="#c0392b", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlabel("$\\text{pct}_{\\text{pos}0}$ threshold")
    ax.set_ylabel("% of targeted total")
    ax.set_xticks(THRESHOLD_SWEEP)
    ax.set_xticklabels([f"{t:.2f}" for t in THRESHOLD_SWEEP])
    ax.set_title(f"Polysemy decomposition at L{LAYER}\n"
                 f"varying threshold ($n = {sweep_results[0]['n']}$)")
    ax.legend(loc="lower left", frameon=False)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(poly_dir / "threshold_sweep_figure.png", dpi=140)
    plt.close()
    print(f"[A] wrote {poly_dir / 'threshold_sweep_figure.png'}")

    # Return per_feature for later use, plus 0.80 suspect set
    suspect_080 = {int(f) for f, v in per_feature.items() if v["position_suspect"]}
    return per_feature, suspect_080, sweep_results


# =============================================================================
# (B) 4-way joint decomposition at L41
# =============================================================================

def run_b_fourway_at_L41(model, tokenizer, sae, detected, a_prompts,
                          specific_features, suspect_080):
    out_dir = R0_DIR
    print(f"\n========== (B) 4-way joint decomposition at L41 ==========")

    # Build unique/shared partition at L41
    pub_set = {(e["pair_id"], e["disambig_idx"]): set(e["features"])
               for e in specific_features}

    sh_un = {}
    for p in detected:
        K = len(p["disambigs"])
        for i in range(K):
            ti = pub_set.get((p["id"], i))
            if not ti:
                continue
            sib_union = set()
            for j in range(K):
                if j != i:
                    sib_union |= pub_set.get((p["id"], j), set())
            shared = ti & sib_union
            unique = ti - shared
            uc = unique - suspect_080
            up = unique & suspect_080
            sc = shared - suspect_080
            sp = shared & suspect_080
            sh_un[(p["id"], i)] = {
                "targeted": ti, "shared": shared, "unique": unique,
                "unique_content": uc, "unique_position": up,
                "shared_content": sc, "shared_position": sp,
            }

    # Pull L41 base/targ from layer-bookends
    p21b_results = json.load(open(ARTIFACTS_DIR / "layer_bookends" / "L41"
                                    / "results_main.json"))
    base_lookup = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                    for r in p21b_results["self_rows"]}
    targ_lookup = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                    for r in p21b_results["self_rows"]}

    # Run four ablations per (P, D_i)
    rows = []
    n_skipped_empty = defaultdict(int)
    for p in tqdm(detected, desc="B 4-way forwards"):
        a_prompt = a_prompts[p["id"]]
        for i in range(len(p["disambigs"])):
            key = (p["id"], i)
            if key not in sh_un or key not in base_lookup:
                continue
            sets = sh_un[key]
            target = set(p["disambigs"][i]["first_token_variants"])
            base = base_lookup[key]
            row = {
                "pair_id": p["id"], "target_idx": i,
                "n_uc": len(sets["unique_content"]),
                "n_up": len(sets["unique_position"]),
                "n_sc": len(sets["shared_content"]),
                "n_sp": len(sets["shared_position"]),
                "base_hit1": base,
                "targeted_hit1": targ_lookup[key],
            }
            for label, fid_set in [("uc", sets["unique_content"]),
                                    ("up", sets["unique_position"]),
                                    ("sc", sets["shared_content"]),
                                    ("sp", sets["shared_position"])]:
                if fid_set:
                    ab = forward_with_ablation(
                        model, tokenizer, a_prompt, LAYER, sae,
                        feature_ids=list(fid_set), k=TOP_LOGITS_K,
                    )
                    h = hit_at_k(ab["top_ids"], target, 1)
                else:
                    h = base
                    n_skipped_empty[label] += 1
                row[f"{label}_hit1"] = h
            rows.append(row)
        torch.cuda.empty_cache()

    keys = [(r["pair_id"], r["target_idx"]) for r in rows]
    by_key = {(r["pair_id"], r["target_idx"]): r for r in rows}
    Bsh = np.array([by_key[k]["base_hit1"] for k in keys])
    Tsh = np.array([by_key[k]["targeted_hit1"] for k in keys])
    UCsh = np.array([by_key[k]["uc_hit1"] for k in keys])
    UPsh = np.array([by_key[k]["up_hit1"] for k in keys])
    SCsh = np.array([by_key[k]["sc_hit1"] for k in keys])
    SPsh = np.array([by_key[k]["sp_hit1"] for k in keys])

    targ_drop = (Tsh - Bsh).mean() * 100
    uc_drop = (UCsh - Bsh).mean() * 100
    up_drop = (UPsh - Bsh).mean() * 100
    sc_drop = (SCsh - Bsh).mean() * 100
    sp_drop = (SPsh - Bsh).mean() * 100
    sum_4way = uc_drop + up_drop + sc_drop + sp_drop
    residual = targ_drop - sum_4way

    pct = lambda x: (x / targ_drop * 100) if abs(targ_drop) > 1e-9 else float("nan")

    # Per-feature normalized impact
    def _per_feat(arr_drop, label):
        sizes = [by_key[k][f"n_{label}"] for k in keys
                 if by_key[k][f"n_{label}"] > 0]
        return arr_drop / np.mean(sizes) if sizes else None

    out = {
        "layer": LAYER,
        "n": len(keys),
        "delta_targeted_pp": float(targ_drop),
        "delta_4way_sum_pp": float(sum_4way),
        "four_way_residual_pp": float(residual),
        "pct_residual_of_targeted": float(pct(residual)),
        "cross_table": {
            "delta_pp": {
                "unique|content": float(uc_drop),
                "unique|position": float(up_drop),
                "shared|content": float(sc_drop),
                "shared|position": float(sp_drop),
            },
            "delta_per_feature_pp": {
                "unique|content": _per_feat(uc_drop, "uc"),
                "unique|position": _per_feat(up_drop, "up"),
                "shared|content": _per_feat(sc_drop, "sc"),
                "shared|position": _per_feat(sp_drop, "sp"),
            },
            "mean_subset_sizes": {
                "unique_content": float(np.mean([r["n_uc"] for r in rows if r["n_uc"] > 0])) if any(r["n_uc"] > 0 for r in rows) else 0.0,
                "unique_position": float(np.mean([r["n_up"] for r in rows if r["n_up"] > 0])) if any(r["n_up"] > 0 for r in rows) else 0.0,
                "shared_content": float(np.mean([r["n_sc"] for r in rows if r["n_sc"] > 0])) if any(r["n_sc"] > 0 for r in rows) else 0.0,
                "shared_position": float(np.mean([r["n_sp"] for r in rows if r["n_sp"] > 0])) if any(r["n_sp"] > 0 for r in rows) else 0.0,
            },
            "n_skipped_empty": dict(n_skipped_empty),
        },
        "tests_vs_baseline": {
            "uc": _drop_vs_base_stats(UCsh, Bsh),
            "up": _drop_vs_base_stats(UPsh, Bsh),
            "sc": _drop_vs_base_stats(SCsh, Bsh),
            "sp": _drop_vs_base_stats(SPsh, Bsh),
            "targeted": _drop_vs_base_stats(Tsh, Bsh),
        },
    }
    save_json_atomic(out_dir / "four_way_joint_L41.json", out)
    ct = out["cross_table"]
    print(f"[B] 4-way decomposition at L{LAYER} (n={len(keys)}, "
          f"Δ_targ={targ_drop:+.2f}):")
    print(f"  uc Δ={uc_drop:+5.2f} per-feat={ct['delta_per_feature_pp']['unique|content']:+.4f}")
    print(f"  up Δ={up_drop:+5.2f} per-feat={ct['delta_per_feature_pp']['unique|position']:+.4f}")
    print(f"  sc Δ={sc_drop:+5.2f} per-feat={ct['delta_per_feature_pp']['shared|content']:+.4f}")
    print(f"  sp Δ={sp_drop:+5.2f} per-feat={ct['delta_per_feature_pp']['shared|position']:+.4f}")
    print(f"  4-way sum: {sum_4way:+.2f} ({pct(sum_4way):.1f}%)")
    print(f"  4-way residual: {residual:+.2f} ({pct(residual):.1f}%)")
    return out


# =============================================================================
# (C) Cramér's V at L41
# =============================================================================

def run_c_cramers_v_at_L41(detected, specific_features, suspect_080):
    out_dir = R0_DIR
    print(f"\n========== (C) Cramér's V at L41 ==========")

    pub_set = {(e["pair_id"], e["disambig_idx"]): set(e["features"])
               for e in specific_features}

    # Per-appearance contingency
    cells = {("unique", "content"): 0, ("unique", "position"): 0,
              ("shared", "content"): 0, ("shared", "position"): 0}
    for p in detected:
        K = len(p["disambigs"])
        for i in range(K):
            ti = pub_set.get((p["id"], i))
            if not ti:
                continue
            sib_union = set()
            for j in range(K):
                if j != i:
                    sib_union |= pub_set.get((p["id"], j), set())
            for fid in ti:
                us = "unique" if fid not in sib_union else "shared"
                cp = "position" if int(fid) in suspect_080 else "content"
                cells[(us, cp)] += 1

    a = cells[("unique", "content")]
    b = cells[("unique", "position")]
    c = cells[("shared", "content")]
    d = cells[("shared", "position")]
    n = a + b + c + d
    row_u = a + b; row_s = c + d
    col_c = a + c; col_p = b + d
    expected = {
        ("unique", "content"):  row_u * col_c / n if n else 0,
        ("unique", "position"): row_u * col_p / n if n else 0,
        ("shared", "content"):  row_s * col_c / n if n else 0,
        ("shared", "position"): row_s * col_p / n if n else 0,
    }
    chi2 = sum(
        ((cells[k] - expected[k]) ** 2 / expected[k]) if expected[k] > 0 else 0.0
        for k in cells
    )
    from scipy.stats import chi2 as chi2_dist
    chi2_p = float(chi2_dist.sf(chi2, df=1))
    cramers_v = float(np.sqrt(chi2 / n)) if n else 0.0
    phi_signed = float((a * d - b * c) / np.sqrt(row_u * row_s * col_c * col_p)) \
        if all(x > 0 for x in [row_u, row_s, col_c, col_p]) else 0.0

    out = {
        "layer": LAYER,
        "cells_observed": {f"{k[0]}|{k[1]}": v for k, v in cells.items()},
        "cells_expected": {f"{k[0]}|{k[1]}": float(v) for k, v in expected.items()},
        "n": n,
        "chi2_statistic": float(chi2),
        "chi2_p_value": chi2_p,
        "cramers_v": cramers_v,
        "phi_coefficient_signed": phi_signed,
    }
    save_json_atomic(out_dir / "cross_decomp_L41.json", out)
    print(f"[C] L{LAYER} cells: {out['cells_observed']}")
    print(f"[C] χ² = {chi2:.2f}, p = {chi2_p:.3g}, Cramér's V = {cramers_v:.4f}, "
          f"signed φ = {phi_signed:+.4f}")
    return out


# =============================================================================
# (D) Multi-metric: KL + generation-flip at L41
# =============================================================================

def run_d_multimetric_at_L41(model, tokenizer, sae, detected, a_prompts,
                              specific_features, suspect_080):
    """Re-harvest forwards (capturing top-50 logits) for KL + run generations."""
    out_dir = R0_DIR / "multimetric"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (D) multi-metric at L41 (KL + gen-flip) ==========")

    pub_set = {(e["pair_id"], e["disambig_idx"]): set(e["features"])
               for e in specific_features}

    # Build sh/un per (P, D_i)
    sh_un = {}
    for p in detected:
        K = len(p["disambigs"])
        for i in range(K):
            ti = pub_set.get((p["id"], i))
            if not ti:
                continue
            sib_union = set()
            for j in range(K):
                if j != i:
                    sib_union |= pub_set.get((p["id"], j), set())
            shared = ti & sib_union
            unique = ti - shared
            sh_un[(p["id"], i)] = {"targeted": list(ti), "shared": list(shared),
                                    "unique": list(unique)}

    # ---- Re-harvest forwards ----
    print(f"[D] re-harvesting forwards (top-{SAVE_TOP_K} logits) for KL...")
    baseline_logits = {}
    for p in tqdm(detected, desc="D baseline"):
        d = baseline_top_logits(model, tokenizer, a_prompts[p["id"]], k=SAVE_TOP_K)
        baseline_logits[p["id"]] = d

    targeted_logits = {}; shared_logits = {}; unique_logits = {}
    for p in tqdm(detected, desc="D ablations"):
        a_prompt = a_prompts[p["id"]]
        for i in range(len(p["disambigs"])):
            key = (p["id"], i)
            if key not in sh_un:
                continue
            for label, store, fids in [
                ("targeted", targeted_logits, sh_un[key]["targeted"]),
                ("shared",   shared_logits,   sh_un[key]["shared"]),
                ("unique",   unique_logits,   sh_un[key]["unique"]),
            ]:
                if not fids:
                    continue
                d = forward_with_ablation(
                    model, tokenizer, a_prompt, LAYER, sae,
                    feature_ids=fids, k=SAVE_TOP_K,
                )
                store[f"{p['id']}|{i}"] = d
        torch.cuda.empty_cache()

    # ---- Compute per-(P, D_i) KL ----
    print(f"[D] computing KL on baseline's top-{KL_SUPPORT_K}...")
    per_pair = []
    for p in detected:
        for i in range(len(p["disambigs"])):
            key = (p["id"], i)
            if key not in sh_un:
                continue
            base = baseline_logits.get(p["id"])
            if base is None:
                continue
            entry = {"pair_id": p["id"], "target_idx": i, "metrics": {}}
            entry["metrics"]["baseline"] = {"kl_to_baseline": 0.0}
            for label, store in [("targeted", targeted_logits),
                                  ("shared_only", shared_logits),
                                  ("unique_only", unique_logits)]:
                fids = (sh_un[key]["targeted"] if label == "targeted"
                         else sh_un[key]["shared"] if label == "shared_only"
                         else sh_un[key]["unique"])
                if not fids:
                    # Empty subset → equivalent to baseline
                    entry["metrics"][label] = {"kl_to_baseline": 0.0}
                    continue
                cond = store.get(f"{p['id']}|{i}")
                if cond is None:
                    entry["metrics"][label] = None
                    continue
                kl = _kl_topk(base["top_logits"], base["top_ids"],
                               cond["top_logits"], cond["top_ids"])
                entry["metrics"][label] = {"kl_to_baseline": kl}
            per_pair.append(entry)

    # KL decomposition
    def _arr(cond):
        return np.array([r["metrics"][cond]["kl_to_baseline"]
                          for r in per_pair if r["metrics"][cond] is not None],
                         dtype=float)
    base_kl = _arr("baseline")
    targ_kl = _arr("targeted")
    sh_kl = _arr("shared_only")
    un_kl = _arr("unique_only")

    from scipy.stats import wilcoxon as _wlx
    def _wlx_p(b, c):
        try:
            return float(_wlx(c, b, alternative="greater").pvalue)
        except Exception:
            return float("nan")

    targ_d = (targ_kl - base_kl).mean()
    sh_d   = (sh_kl - base_kl).mean()
    un_d   = (un_kl - base_kl).mean()
    sum_   = sh_d + un_d
    res    = targ_d - sum_
    pct    = lambda x: (x / targ_d * 100) if abs(targ_d) > 1e-9 else float("nan")

    decomp_kl = {
        "metric": "kl_to_baseline",
        "n": len(per_pair),
        "table": {
            "Baseline":    {"value": float(base_kl.mean()), "delta": 0.0,
                            "pct_of_targeted": 0.0},
            "Targeted":    {"value": float(targ_kl.mean()), "delta": float(targ_d),
                            "pct_of_targeted": 100.0,
                            "wilcoxon_p_vs_baseline": _wlx_p(base_kl, targ_kl)},
            "Shared-only": {"value": float(sh_kl.mean()), "delta": float(sh_d),
                            "pct_of_targeted": float(pct(sh_d)),
                            "wilcoxon_p_vs_baseline": _wlx_p(base_kl, sh_kl)},
            "Unique-only": {"value": float(un_kl.mean()), "delta": float(un_d),
                            "pct_of_targeted": float(pct(un_d)),
                            "wilcoxon_p_vs_baseline": _wlx_p(base_kl, un_kl)},
            "Sum":         {"delta": float(sum_), "pct_of_targeted": float(pct(sum_))},
            "Residual":    {"delta": float(res), "pct_of_targeted": float(pct(res))},
        },
    }
    save_json_atomic(out_dir / "kl_decomposition_L41.json", decomp_kl)
    print(f"[D KL] Targeted Δ={targ_d:.4f}; Shared Δ={sh_d:.4f} ({pct(sh_d):+.1f}%); "
          f"Unique Δ={un_d:.4f} ({pct(un_d):+.1f}%); Residual {res:.4f} ({pct(res):+.1f}%)")

    # ---- Generation phase ----
    print(f"\n[D] generation phase: 4 conditions × ~{len(per_pair)} (P, D_i)")
    base_gen_cache = {}
    generations = []
    for r in tqdm(per_pair, desc="D generate"):
        pid = r["pair_id"]; i = r["target_idx"]
        a_prompt = a_prompts[pid]
        if pid not in base_gen_cache:
            base_gen_cache[pid] = greedy_decode(model, tokenizer, a_prompt,
                                                  max_new=GEN_MAX_NEW)
        base_g = base_gen_cache[pid]

        targ_g = forward_with_ablation_then_generate(
            model, tokenizer, a_prompt, LAYER, sae,
            feature_ids=sh_un[(pid, i)]["targeted"], max_new=GEN_MAX_NEW,
        ) if sh_un[(pid, i)]["targeted"] else base_g
        sh_g = forward_with_ablation_then_generate(
            model, tokenizer, a_prompt, LAYER, sae,
            feature_ids=sh_un[(pid, i)]["shared"], max_new=GEN_MAX_NEW,
        ) if sh_un[(pid, i)]["shared"] else base_g
        un_g = forward_with_ablation_then_generate(
            model, tokenizer, a_prompt, LAYER, sae,
            feature_ids=sh_un[(pid, i)]["unique"], max_new=GEN_MAX_NEW,
        ) if sh_un[(pid, i)]["unique"] else base_g

        generations.append({
            "pair_id": pid, "target_idx": i,
            "baseline": base_g, "targeted": targ_g,
            "shared_only": sh_g, "unique_only": un_g,
        })
    save_json_atomic(out_dir / "generations_L41.json", generations)

    # ---- Classify generations ----
    pid_to_pair = {p["id"]: p for p in detected}
    classifications = []
    for r in generations:
        pid = r["pair_id"]; i = r["target_idx"]
        pair = pid_to_pair[pid]
        cand_strs = [d["answer"] for d in pair["disambigs"]]
        cls = {}
        for cond in ["baseline", "targeted", "shared_only", "unique_only"]:
            text = r[cond]
            match = find_best_match(text, cand_strs)
            if match.cand_idx < 0:
                slot = "no-match"
            elif match.cand_idx == i:
                slot = "D_i"
            else:
                slot = "D_j"
            cls[cond] = slot
        classifications.append({
            "pair_id": pid, "target_idx": i, "slots": cls,
        })

    # Confusion matrices + flip rates
    confusion = {}
    flip_rates = {}
    slots_list = ["D_i", "D_j", "no-match"]
    for cond in ["targeted", "shared_only", "unique_only"]:
        cm = {a: {b: 0 for b in slots_list} for a in slots_list}
        flips = 0
        for r in classifications:
            base_slot = r["slots"]["baseline"]
            cond_slot = r["slots"][cond]
            cm[base_slot][cond_slot] += 1
            if base_slot != cond_slot:
                flips += 1
        confusion[cond] = cm
        flip_rates[cond] = {"n": len(classifications), "n_flips": flips,
                             "flip_rate": flips / len(classifications)}
    save_json_atomic(out_dir / "gen_confusion_matrices_L41.json", confusion)

    # Generation-flip decomposition
    keys_order = [(r["pair_id"], r["target_idx"]) for r in classifications]
    cls_lookup = {(r["pair_id"], r["target_idx"]): r["slots"] for r in classifications}
    def _hit_arr(cond):
        return np.array([1 if cls_lookup[k][cond] == "D_i" else 0
                          for k in keys_order], dtype=float)
    Bg = _hit_arr("baseline")
    Tg = _hit_arr("targeted")
    Sg = _hit_arr("shared_only")
    Ug = _hit_arr("unique_only")
    td = (Tg - Bg).mean() * 100
    sd = (Sg - Bg).mean() * 100
    ud = (Ug - Bg).mean() * 100
    pct_g = lambda x: (x / td * 100) if abs(td) > 1e-9 else float("nan")

    gen_decomp = {
        "n": len(keys_order),
        "table": {
            "Baseline":    {"hit_D_i": float(Bg.mean()), "delta_pp": 0.0,
                            "pct_of_targeted": 0.0},
            "Targeted":    {"hit_D_i": float(Tg.mean()), "delta_pp": float(td),
                            "pct_of_targeted": 100.0,
                            "ci_pp": list(bootstrap_diff_ci_pp(Bg, Tg, seed=0)),
                            "mcnemar":  mcnemar_paired(Tg, Bg, alternative="greater"),
                            "wilcoxon_p": wilcoxon_paired(Bg, Tg, alternative="greater")},
            "Shared-only": {"hit_D_i": float(Sg.mean()), "delta_pp": float(sd),
                            "pct_of_targeted": float(pct_g(sd)),
                            "ci_pp": list(bootstrap_diff_ci_pp(Bg, Sg, seed=0)),
                            "mcnemar":  mcnemar_paired(Sg, Bg, alternative="greater"),
                            "wilcoxon_p": wilcoxon_paired(Bg, Sg, alternative="greater")},
            "Unique-only": {"hit_D_i": float(Ug.mean()), "delta_pp": float(ud),
                            "pct_of_targeted": float(pct_g(ud)),
                            "ci_pp": list(bootstrap_diff_ci_pp(Bg, Ug, seed=0)),
                            "mcnemar":  mcnemar_paired(Ug, Bg, alternative="greater"),
                            "wilcoxon_p": wilcoxon_paired(Bg, Ug, alternative="greater")},
            "Sum":         {"delta_pp": float(sd + ud),
                            "pct_of_targeted": float(pct_g(sd + ud))},
            "Residual":    {"delta_pp": float(td - (sd + ud)),
                            "pct_of_targeted": float(pct_g(td - (sd + ud)))},
        },
        "flip_rates_from_baseline": flip_rates,
    }
    save_json_atomic(out_dir / "gen_flip_decomposition_L41.json", gen_decomp)
    print(f"[D gen] Targeted Δ={td:+.2f}pp; Shared Δ={sd:+.2f} ({pct_g(sd):+.1f}%); "
          f"Unique Δ={ud:+.2f} ({pct_g(ud):+.1f}%)")
    print(f"[D gen] flip rates: targ={flip_rates['targeted']['flip_rate']:.4f}, "
          f"shared={flip_rates['shared_only']['flip_rate']:.4f}, "
          f"unique={flip_rates['unique_only']['flip_rate']:.4f}")

    return decomp_kl, gen_decomp, confusion


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    R0_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"[setup] loading {MODEL_ID} + L{LAYER} SAE...")
    tokenizer, model = load_model()
    sae, sae_meta = load_sae(layer=LAYER)
    print(f"[setup] L{LAYER} SAE sha256: {sae_meta['sha256'][:16]}...")

    detected = json.load(open(ARTIFACTS_DIR / "detected_pairs.json"))
    a_prompts = {p["id"]: build_prompt(tokenizer, p["A_question"])
                 for p in detected}
    paragraphs = load_wikitext_paragraphs(tokenizer)
    print(f"[setup] {len(detected)} detected pairs, "
          f"{sum(len(p['disambigs']) for p in detected)} self-pairs, "
          f"{len(paragraphs)} WikiText paragraphs")

    # Load the L41 specific_features (Targeted top-10 sets)
    specific_features = json.load(open(ARTIFACTS_DIR / "layer_bookends" / "L41"
                                         / "specific_features.json"))
    print(f"[setup] {len(specific_features)} Targeted top-10 sets at L{LAYER}")

    # ---- (A) Polysemy at L41 ----
    per_feature, suspect_080, sweep_results = run_a_polysemy_at_L41(
        model, tokenizer, sae, paragraphs, detected, a_prompts,
        specific_features,
    )

    # ---- (B) 4-way joint at L41 ----
    fourway = run_b_fourway_at_L41(
        model, tokenizer, sae, detected, a_prompts,
        specific_features, suspect_080,
    )

    # ---- (C) Cramér's V at L41 ----
    cramers = run_c_cramers_v_at_L41(detected, specific_features, suspect_080)

    # ---- (D) Multi-metric at L41 ----
    decomp_kl, gen_decomp, gen_confusion = run_d_multimetric_at_L41(
        model, tokenizer, sae, detected, a_prompts,
        specific_features, suspect_080,
    )

    # ---- L41 summary table (headline numbers cited by the paper) ----
    p21b_decomp = json.load(open(ARTIFACTS_DIR / "layer_bookends" / "L41"
                                   / "unique_shared_decomp.json"))
    p21b_sf = json.load(open(ARTIFACTS_DIR / "layer_bookends" / "L41"
                               / "per_feature_equivalence.json"))
    p21b_results = json.load(open(ARTIFACTS_DIR / "layer_bookends" / "L41"
                                    / "results_main.json"))

    sib = per_pair_means(
        p21b_results["cross_rows"],
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["ablate_hit1"],
    )
    wt = per_pair_means(
        p21b_results["wikitext_shuffled_rows"],
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["wt_shuffled_hit1"],
    )
    keys_t3 = sorted(k for k in sib if k in wt)
    Sib_a = np.array([sib[k] for k in keys_t3])
    WSh_a = np.array([wt[k] for k in keys_t3])
    t3_mc = mcnemar_paired(Sib_a, WSh_a, alternative="greater")
    t3_wx = wilcoxon_paired(WSh_a, Sib_a, alternative="greater")
    t3_ci = bootstrap_diff_ci_pp(Sib_a, WSh_a, seed=0)

    summary_table = {
        "layer": LAYER,
        "sae_meta": sae_meta,
        "n_self_pairs": p21b_results["summary"]["n_self"],
        "headline_hit1": {
            "Baseline": p21b_results["summary"]["self_base_hit1"],
            "Targeted": p21b_results["summary"]["self_ablate_hit1"],
            "Sibling":  float(Sib_a.mean()),
            "WikiTextShuffled": float(WSh_a.mean()),
        },
        "T3": {
            "delta_pp": float((WSh_a.mean() - Sib_a.mean()) * 100),
            "ci_pp": [float(t3_ci[0]), float(t3_ci[1])],
            "mcnemar_p": t3_mc["p"],
            "wilcoxon_p": t3_wx,
        },
        "decomposition_hit1_from_p21b": p21b_decomp["decomposition"]["table"],
        "decomposition_kl": decomp_kl["table"],
        "decomposition_gen_flip": gen_decomp["table"],
        "polysemy_at_080": [c for c in sweep_results if abs(c["threshold"] - 0.80) < 1e-9][0],
        "fourway_joint": {
            "residual_pp": fourway["four_way_residual_pp"],
            "pct_residual_of_targeted": fourway["pct_residual_of_targeted"],
            "cross_table": fourway["cross_table"],
        },
        "cramers_v": cramers["cramers_v"],
        "phi_signed": cramers["phi_coefficient_signed"],
        "single_feature_equivalence": p21b_sf["summary"],
        "gen_flip_rates": gen_decomp["flip_rates_from_baseline"],
        "gen_confusion_matrices": gen_confusion,
    }
    save_json_atomic(R0_DIR / "L41_summary_table.json", summary_table)

    # Cleanup SAE blob
    try:
        local_path = sae_meta.get("local_path")
        if local_path:
            real = Path(local_path).resolve()
            if real.exists():
                real.unlink()
                print(f"  deleted L{LAYER} SAE blob from HF cache to free space")
    except Exception:
        pass

    print(f"\n=== Stage R0 complete ===")
    print(f"  total elapsed: {(time.time()-t_total)/60:.1f} min")
    print(f"  -> {R0_DIR}/L41_summary_table.json")


if __name__ == "__main__":
    main()
