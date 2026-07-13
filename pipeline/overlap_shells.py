"""Overlap-shell analyses of the T3 contrast.

Two analyses, both probing how sibling feature overlap drives T3:
  - T3 falls to +0.67 pp (p=0.39) on the n=75 overlap=0 subsample,
    motivating the per-shell trajectory in (d).
  - A z_A-matched Sibling control is only meaningful if Targeted_{D_i} is
    excluded from the candidate pool; without the exclusion, matches are
    trivially the same features (median |z_A diff| = 0). (c_clean) applies
    the exclusion.

  (c_clean) Clean z_A-matched Sibling. For each (P, D_i, D_j), pick matched-
       Sibling features from D_j_positives ∖ Targeted_{D_i} (instead of
       D_j_positives), apply the same greedy nearest-neighbor on z_A.
       Drop (i, j) cells where the filtered pool has < 10 candidates;
       drop (P, D_i) where all (i, j) drop. Report counts. Compute T3
       vs published WikiText-shuffled.

  (d) Per-overlap-shell T3 trajectory. Pure re-analysis. For each
      k ∈ {0, …, 10}, compute T3 on the shell of (P, D_i) whose
      max-over-siblings overlap equals exactly k. Also cumulative T3 at
      overlap ≤ k. Plot per-shell T3 + CI bars.

Reads:
  artifacts/sae_encodings_L37.npz
  artifacts/detected_pairs.json
  artifacts/specific_features.json
  artifacts/results_main.json

Writes:
  artifacts/overlap_shells/c_prime/{matched_sibling_rows.json, cell_result.json,
                                      match_quality.json, dropped_cells.json}
  artifacts/overlap_shells/d/{per_shell_results.json, cumulative_results.json,
                                shell_trajectory.png}
  artifacts/overlap_shells/comparison_table.json
  artifacts/overlap_shells/run_meta.json
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import time
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


P12P_DIR = ARTIFACTS_DIR / "overlap_shells"
PUBLISHED_TOP_K = 10


# =============================================================================
# Shared helpers (local copies to avoid a cross-module dependency)
# =============================================================================

def _build_pub_per_pair_outcomes(results_main):
    targeted = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                for r in results_main["self_rows"]}
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
    return targeted, sibling, wt_shuf


def _t3_stats(Sib_arr, WSh_arr):
    if len(Sib_arr) < 2:
        return {
            "n": int(len(Sib_arr)),
            "sibling_mean": float(Sib_arr.mean()) if len(Sib_arr) else None,
            "wt_shuf_mean": float(WSh_arr.mean()) if len(WSh_arr) else None,
            "delta_pp": None, "ci_pp": [None, None],
            "mcnemar": None, "wilcoxon_p": None,
        }
    mc = mcnemar_paired(Sib_arr, WSh_arr, alternative="greater")
    wx = wilcoxon_paired(WSh_arr, Sib_arr, alternative="greater")
    ci = bootstrap_diff_ci_pp(Sib_arr, WSh_arr, seed=0)
    return {
        "n": int(len(Sib_arr)),
        "sibling_mean": float(Sib_arr.mean()),
        "wt_shuf_mean": float(WSh_arr.mean()),
        "delta_pp": float((WSh_arr.mean() - Sib_arr.mean()) * 100),
        "ci_pp": [float(ci[0]), float(ci[1])],
        "mcnemar": mc,
        "wilcoxon_p": wx,
    }


def _greedy_zA_match(target_zA_sorted_desc, cand_fids, cand_zA, k):
    used = [False] * len(cand_fids)
    matched = []
    for tz in target_zA_sorted_desc:
        best = -1; best_d = float("inf")
        for j in range(len(cand_fids)):
            if used[j]:
                continue
            d = abs(cand_zA[j] - tz)
            if d < best_d:
                best_d = d; best = j
        if best < 0:
            break
        matched.append(int(cand_fids[best]))
        used[best] = True
        if len(matched) == k:
            break
    return matched


# =============================================================================
# (c_clean) Clean z_A-matched Sibling
# =============================================================================

def run_c_prime(tokenizer, model, sae, enc_dict, detected, a_prompts,
                published_specific, results_main):
    out_dir = P12P_DIR / "c_prime"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (c_clean) clean z_A-matched Sibling ==========")

    # Index published Targeted by (pair_id, di)
    pub = {}
    for entry in published_specific:
        pub[(entry["pair_id"], entry["disambig_idx"])] = list(entry["features"])

    matched_rows = []
    z_A_diffs = []
    z_Dj_diffs = []
    dropped_ij = []   # (pair_id, target_idx, sibling_idx, reason, n_candidates)
    dropped_pi = []   # (pair_id, target_idx) where no surviving (i, j) cell

    for p in tqdm(detected, desc="(c_clean) matched-sibling clean"):
        K = len(p["disambigs"])
        a_prompt = a_prompts[p["id"]]
        z_A = enc_dict[f"A__{p['id']}"]

        for i in range(K):
            target_self = set(p["disambigs"][i]["first_token_variants"])
            tgt_feats = pub.get((p["id"], i), [])
            if not tgt_feats:
                continue
            tgt_set = set(tgt_feats)

            # Sort Targeted by z_A desc; carry feature ids in same order
            tgt_zA_unsorted = [float(z_A[fid].item()) for fid in tgt_feats]
            order = sorted(range(len(tgt_feats)), key=lambda x: -tgt_zA_unsorted[x])
            tgt_feats_sorted = [tgt_feats[k] for k in order]
            tgt_zA_sorted = [tgt_zA_unsorted[k] for k in order]

            survived_any = False
            for j in range(K):
                if j == i:
                    continue
                z_Dj = enc_dict[f"D__{p['id']}__{j}"]
                pos_mask = (z_Dj > 0)
                cand_all = pos_mask.nonzero(as_tuple=False).squeeze(-1).cpu().tolist()
                # Filter: D_j_positives ∖ Targeted_{D_i}
                cand_fids = [f for f in cand_all if f not in tgt_set]
                if len(cand_fids) < PUBLISHED_TOP_K:
                    dropped_ij.append({
                        "pair_id": p["id"], "target_idx": i, "sibling_idx": j,
                        "n_dj_positives": len(cand_all),
                        "n_after_filter": len(cand_fids),
                    })
                    continue

                cand_zA = [float(z_A[f].item()) for f in cand_fids]
                matched = _greedy_zA_match(tgt_zA_sorted, cand_fids, cand_zA,
                                            k=PUBLISHED_TOP_K)
                if len(matched) < PUBLISHED_TOP_K:
                    dropped_ij.append({
                        "pair_id": p["id"], "target_idx": i, "sibling_idx": j,
                        "n_dj_positives": len(cand_all),
                        "n_after_filter": len(cand_fids),
                        "reason": "matched < 10",
                    })
                    continue

                # Match-quality diagnostics, slot by slot
                for slot, mfid in enumerate(matched):
                    z_A_diffs.append(abs(tgt_zA_sorted[slot] - float(z_A[mfid].item())))
                    tgt_fid = tgt_feats_sorted[slot]
                    z_Dj_diffs.append(abs(float(z_Dj[tgt_fid].item())
                                           - float(z_Dj[mfid].item())))

                ab = forward_with_ablation(
                    model, tokenizer, a_prompt, LAYER, sae,
                    feature_ids=matched, k=TOP_LOGITS_K,
                )
                matched_rows.append({
                    "pair_id": p["id"], "target_idx": i, "sibling_idx": j,
                    "n_matched": len(matched),
                    "matched_features": matched,
                    "matched_hit1": hit_at_k(ab["top_ids"], target_self, 1),
                })
                survived_any = True

            if not survived_any:
                dropped_pi.append({"pair_id": p["id"], "target_idx": i})
        torch.cuda.empty_cache()

    save_json_atomic(out_dir / "matched_sibling_rows.json", matched_rows)
    save_json_atomic(out_dir / "dropped_cells.json", {
        "dropped_ij_count": len(dropped_ij),
        "dropped_pi_count": len(dropped_pi),
        "dropped_ij_examples": dropped_ij[:50],
        "dropped_pi_pairs": dropped_pi,
    })

    matched_per_pair = per_pair_means(
        matched_rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["matched_hit1"],
    )

    _, _, wt_shuf = _build_pub_per_pair_outcomes(results_main)

    keys = sorted(k for k in matched_per_pair if k in wt_shuf)
    Sib_m = np.array([matched_per_pair[k] for k in keys], dtype=float)
    WSh = np.array([wt_shuf[k] for k in keys], dtype=float)
    t3 = _t3_stats(Sib_m, WSh)

    zA_arr = np.array(z_A_diffs)
    zDj_arr = np.array(z_Dj_diffs)
    match_diag = {
        "n_feature_pairs_matched": int(len(zA_arr)),
        "z_A_abs_diff": {
            "mean": float(zA_arr.mean()) if len(zA_arr) else None,
            "median": float(np.median(zA_arr)) if len(zA_arr) else None,
            "p95": float(np.percentile(zA_arr, 95)) if len(zA_arr) else None,
        },
        "z_Dj_abs_diff": {
            "mean": float(zDj_arr.mean()) if len(zDj_arr) else None,
            "median": float(np.median(zDj_arr)) if len(zDj_arr) else None,
            "p95": float(np.percentile(zDj_arr, 95)) if len(zDj_arr) else None,
        },
    }
    save_json_atomic(out_dir / "match_quality.json", match_diag)

    cell = {
        "selector": "z_A_matched_sibling_CLEAN (D_j_positives ∖ Targeted_{D_i}, "
                    "greedy NN on z_A)",
        "n_matched_forwards": len(matched_rows),
        "n_self_pairs": len(keys),
        "n_dropped_ij_cells": len(dropped_ij),
        "n_dropped_pi_pairs": len(dropped_pi),
        "T3": t3,
    }
    save_json_atomic(out_dir / "cell_result.json", cell)
    print(f"[c_clean] dropped (i,j) cells: {len(dropped_ij)}, dropped (P, D_i): {len(dropped_pi)}")
    print(f"[c_clean] T3 Δ pp = {t3['delta_pp']:+.2f}  "
          f"CI95 = [{t3['ci_pp'][0]:+.2f}, {t3['ci_pp'][1]:+.2f}]  "
          f"McNemar p={t3['mcnemar']['p']:.3g}  "
          f"Wilcoxon p={t3['wilcoxon_p']:.3g}")
    print(f"[c_clean] |z_A diff|: mean={match_diag['z_A_abs_diff']['mean']:.4f}, "
          f"median={match_diag['z_A_abs_diff']['median']:.4f}")
    print(f"[c_clean] |z_Dj diff|: mean={match_diag['z_Dj_abs_diff']['mean']:.4f}, "
          f"median={match_diag['z_Dj_abs_diff']['median']:.4f}")
    return cell, match_diag


# =============================================================================
# (d) Per-overlap-shell T3 trajectory
# =============================================================================

def run_d(detected, published_specific, results_main):
    out_dir = P12P_DIR / "d"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (d) per-overlap-shell T3 trajectory ==========")

    pub_set = {}
    for entry in published_specific:
        pub_set[(entry["pair_id"], entry["disambig_idx"])] = set(entry["features"])

    overlaps = {}
    for p in detected:
        K = len(p["disambigs"])
        for i in range(K):
            ti = pub_set.get((p["id"], i), set())
            if not ti:
                continue
            sibs = [pub_set.get((p["id"], j), set()) for j in range(K) if j != i]
            sibs = [s for s in sibs if s]
            if not sibs:
                continue
            overlaps[(p["id"], i)] = int(max(len(ti & s) for s in sibs))

    targeted, sibling, wt_shuf = _build_pub_per_pair_outcomes(results_main)

    shells = []
    cums = []
    for k in range(0, PUBLISHED_TOP_K + 1):
        shell_keys = sorted(
            key for key, ov in overlaps.items()
            if ov == k and key in sibling and key in wt_shuf
        )
        cum_keys = sorted(
            key for key, ov in overlaps.items()
            if ov <= k and key in sibling and key in wt_shuf
        )

        Sib_s = np.array([sibling[key] for key in shell_keys], dtype=float)
        WSh_s = np.array([wt_shuf[key] for key in shell_keys], dtype=float)
        Sib_c = np.array([sibling[key] for key in cum_keys], dtype=float)
        WSh_c = np.array([wt_shuf[key] for key in cum_keys], dtype=float)

        shells.append({
            "k": k, "n": len(shell_keys), "T3": _t3_stats(Sib_s, WSh_s),
        })
        cums.append({
            "k_le": k, "n": len(cum_keys), "T3": _t3_stats(Sib_c, WSh_c),
        })

    save_json_atomic(out_dir / "per_shell_results.json", {"shells": shells})
    save_json_atomic(out_dir / "cumulative_results.json", {"cums": cums})

    print(f"[d] per-shell:")
    for s in shells:
        t3 = s["T3"]
        if t3["delta_pp"] is None:
            print(f"  k={s['k']:<2}  n={s['n']:<5}  (skipped, n<2)")
        else:
            print(f"  k={s['k']:<2}  n={s['n']:<5}  Δ={t3['delta_pp']:+6.2f}  "
                  f"CI=[{t3['ci_pp'][0]:+5.2f},{t3['ci_pp'][1]:+5.2f}]  "
                  f"McN p={t3['mcnemar']['p']:.3g}  Wlx p={t3['wilcoxon_p']:.3g}")
    print(f"[d] cumulative (overlap ≤ k):")
    for c in cums:
        t3 = c["T3"]
        print(f"  k≤{c['k_le']:<2}  n={c['n']:<5}  Δ={t3['delta_pp']:+6.2f}  "
              f"CI=[{t3['ci_pp'][0]:+5.2f},{t3['ci_pp'][1]:+5.2f}]  "
              f"McN p={t3['mcnemar']['p']:.3g}  Wlx p={t3['wilcoxon_p']:.3g}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = [s["k"] for s in shells if s["T3"]["delta_pp"] is not None]
    ds = [s["T3"]["delta_pp"] for s in shells if s["T3"]["delta_pp"] is not None]
    los = [s["T3"]["ci_pp"][0] for s in shells if s["T3"]["delta_pp"] is not None]
    his = [s["T3"]["ci_pp"][1] for s in shells if s["T3"]["delta_pp"] is not None]
    ns = [s["n"] for s in shells if s["T3"]["delta_pp"] is not None]
    yerr = [
        [d - lo for d, lo in zip(ds, los)],
        [hi - d for hi, d in zip(his, ds)],
    ]
    ax.errorbar(ks, ds, yerr=yerr, fmt="o", capsize=5, color="#1a7",
                ecolor="gray", linewidth=2, markersize=8)
    for k, d, n in zip(ks, ds, ns):
        ax.annotate(f"n={n}", (k, d), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(4.49, color="red", linestyle=":", label="published full-sample T3 = +4.49")
    ax.set_xlabel("max overlap with any sibling (top-10)")
    ax.set_ylabel("T3 Δ pp (Sibling − WikiText-shuffled, signed)")
    ax.set_title("Per-shell T3 trajectory with 95% bootstrap CIs")
    ax.set_xticks(range(PUBLISHED_TOP_K + 1))
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_png = out_dir / "shell_trajectory.png"
    plt.savefig(out_png, dpi=140)
    plt.close()
    print(f"[d] wrote {out_png}")
    return shells, cums


# =============================================================================
# Comparison + main
# =============================================================================

def write_comparison_table(c_cell, results_main, c_orig_path):
    targeted, sibling, wt_shuf = _build_pub_per_pair_outcomes(results_main)
    keys = sorted(k for k in targeted if k in sibling and k in wt_shuf)
    Sib = np.array([sibling[k] for k in keys], dtype=float)
    WSh = np.array([wt_shuf[k] for k in keys], dtype=float)
    pub_t3 = _t3_stats(Sib, WSh)

    # original matched-sibling cell
    c_orig = json.load(open(c_orig_path))

    rows = [
        {
            "row": "published (a_weight=0.5, top_k=10)",
            "n_pairs": len(keys),
            "delta_pp": pub_t3["delta_pp"],
            "ci_pp": pub_t3["ci_pp"],
            "mcnemar_p": pub_t3["mcnemar"]["p"],
            "wilcoxon_p": pub_t3["wilcoxon_p"],
            "dropped_ij_cells": 0,
            "dropped_pi_pairs": 0,
        },
        {
            "row": "1.2(c) original z_A-matched",
            "n_pairs": c_orig["n_self_pairs"],
            "delta_pp": c_orig["T3"]["delta_pp"],
            "ci_pp": c_orig["T3"]["ci_pp"],
            "mcnemar_p": c_orig["T3"]["mcnemar"]["p"],
            "wilcoxon_p": c_orig["T3"]["wilcoxon_p"],
            "dropped_ij_cells": 0,
            "dropped_pi_pairs": 0,
        },
        {
            "row": "z_A-matched Sibling (clean)",
            "n_pairs": c_cell["n_self_pairs"],
            "delta_pp": c_cell["T3"]["delta_pp"],
            "ci_pp": c_cell["T3"]["ci_pp"],
            "mcnemar_p": c_cell["T3"]["mcnemar"]["p"],
            "wilcoxon_p": c_cell["T3"]["wilcoxon_p"],
            "dropped_ij_cells": c_cell["n_dropped_ij_cells"],
            "dropped_pi_pairs": c_cell["n_dropped_pi_pairs"],
        },
    ]
    save_json_atomic(P12P_DIR / "comparison_table.json", {"rows": rows})
    return rows


def main() -> None:
    P12P_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"[setup] loading model + SAE (model={MODEL_ID}, layer=L{LAYER})...")
    tokenizer, model = load_model()
    sae, sae_meta = load_sae(layer=LAYER)
    if sae_meta["sha256"] != SAE_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"SAE sha256 mismatch! got {sae_meta['sha256']}, "
            f"expected {SAE_CHECKPOINT_SHA256}"
        )
    print(f"[setup] SAE sha256 verified ({sae_meta['sha256'][:16]}...)")

    npz = np.load(ARTIFACTS_DIR / f"sae_encodings_L{LAYER}.npz")
    enc_dict = {
        k: torch.from_numpy(npz[k]).to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        for k in npz.files
    }
    detected = json.load(open(ARTIFACTS_DIR / "detected_pairs.json"))
    a_prompts = {p["id"]: build_prompt(tokenizer, p["A_question"])
                 for p in detected}
    published_specific = json.load(open(ARTIFACTS_DIR / "specific_features.json"))
    results_main = json.load(open(ARTIFACTS_DIR / "results_main.json"))

    print(f"[setup] {len(detected)} detected pairs, "
          f"{sum(len(p['disambigs']) for p in detected)} self-pairs")

    c_cell, c_diag = run_c_prime(
        tokenizer, model, sae, enc_dict, detected, a_prompts,
        published_specific, results_main,
    )
    shells, cums = run_d(detected, published_specific, results_main)

    c_orig_path = ARTIFACTS_DIR / "overlap_shells" / "c_original" / "cell_result.json"
    rows = write_comparison_table(c_cell, results_main, c_orig_path)

    save_json_atomic(P12P_DIR / "run_meta.json", {
        "model": MODEL_ID, "layer": LAYER, "sae_meta": sae_meta,
        "n_detected_pairs": len(detected),
        "elapsed_seconds": round(time.time() - t_total, 2),
    })

    print(f"\n=== Overlap-shell analysis complete ===")
    print(f"  total elapsed: {(time.time()-t_total)/60:.1f} min")
    print(f"  -> {P12P_DIR}/comparison_table.json")
    print(f"\n  Comparison:")
    for r in rows:
        print(f"    {r['row']:<40}  n={r['n_pairs']:<5}  Δ={r['delta_pp']:+6.2f} pp  "
              f"CI=[{r['ci_pp'][0]:+5.2f},{r['ci_pp'][1]:+5.2f}]  "
              f"McN p={r['mcnemar_p']:.2e}  Wlx p={r['wilcoxon_p']:.2e}  "
              f"drop_ij={r['dropped_ij_cells']}, drop_pi={r['dropped_pi_pairs']}")


if __name__ == "__main__":
    main()
