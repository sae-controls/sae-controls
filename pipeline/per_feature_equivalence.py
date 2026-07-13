"""Confounder controls for the orthogonal decompositions.

Two jobs:
  (I)  Test the WT-shuffled baseline against magnitude (A) and selection-
       regime (B) confounders.
  (II) Stress-test the per-feature equivalence between unique_content and
       shared_content from four_way_decomposition.py (D) (C).

Sub-experiments:
  (A) Magnitude-matched WT-shuffled control.
      For each (P, D_i, draw): match the Targeted z_{D_i} profile to
      WikiText paragraph positive features (greedy NN on z magnitudes).
      Re-run WT ablation. Compute T3 vs published Sibling.

  (B) Score-on-WT control.
      Apply the published score function across 800 WikiText paragraphs
      treated as a "disambig pool":
        score(f, p) = 1.5 · z_p(f) − mean_{p'≠p} z_{p'}(f).
      Take top-10 per paragraph (mask z_p(f) > 0). Re-run WT ablation.

  (C) Per-feature equivalence magnitude diagnostic.
      (i) CPU re-analysis: per-pair mean z_{D_i} of unique_content vs
          shared_content; Wilcoxon signed-rank; per-magnitude-bin
          per-feature impact ratio from four_way_decomposition.py (D)'s ablation rows.
      (ii) Stratified GPU: single-feature ablation of the highest-magnitude
           unique_content vs highest-magnitude shared_content per pair.
           ~1,900 forwards.

Reads:
  artifacts/sae_encodings_L37.npz
  artifacts/detected_pairs.json
  artifacts/specific_features.json
  artifacts/results_main.json
  artifacts/wikitext_position_mode.json
  artifacts/shuffle_draws_wikitext.json
  artifacts/unique_vs_shared/a/ablation_rows.json   (for (C)(i))
  artifacts/four_way/d/four_way_rows.json     (for partition reuse)

Writes:
  artifacts/per_feature_equivalence/a/{matched_wt_rows.json, t3.json}
  artifacts/per_feature_equivalence/b/{score_on_wt_rows.json, scored_wt_top10_per_para.json,
                          t3.json}
  artifacts/per_feature_equivalence/c/{per_pair_magnitudes.json, magnitude_test.json,
                          stratified_impact.json, single_feature_rows.json,
                          single_feature_summary.json,
                          unique_vs_shared_magnitude_histogram.png}
  artifacts/per_feature_equivalence/comparison_table.json
  artifacts/per_feature_equivalence/run_meta.json
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
    POSITION_PCT_THRESHOLD, SAE_CHECKPOINT_SHA256,
    SCORE_A_WEIGHT, SHUFFLE_SEED, TOP_LOGITS_K,
    TOP_K_FEATURES, WIKITEXT_MAX_TOKENS,
)
from src.hooks import (
    capture_all_position_residuals, forward_with_ablation, hit_at_k,
)
from src.io_utils import save_json_atomic
from src.model import load_model
from src.position_mode import is_position_suspect
from src.prompts import build_prompt
from src.sae import load_sae
from src.wikitext import load_paragraphs as load_wikitext_paragraphs


P15_DIR = ARTIFACTS_DIR / "per_feature_equivalence"


# =============================================================================
# Helpers
# =============================================================================

def _greedy_match(target_sorted_desc, cand_fids, cand_vals, k):
    """Greedy nearest-neighbor on magnitude. Returns list of cand fids of length
    min(k, len(cand_fids))."""
    used = [False] * len(cand_fids)
    matched = []
    for tz in target_sorted_desc:
        best = -1; best_d = float("inf")
        for j in range(len(cand_fids)):
            if used[j]:
                continue
            d = abs(cand_vals[j] - tz)
            if d < best_d:
                best_d = d; best = j
        if best < 0:
            break
        matched.append(int(cand_fids[best]))
        used[best] = True
        if len(matched) == k:
            break
    return matched


def _t3_stats(Sib_arr, WSh_arr):
    if len(Sib_arr) < 2:
        return None
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


def _capture_wt_lasttok_z(model, tokenizer, sae, paragraphs):
    """Forward each paragraph; capture last-token z (post-JumpReLU) for all
    SAE features. Returns numpy array of shape (n_paragraphs, d_sae)."""
    print(f"[wt-z] capturing last-token z for {len(paragraphs)} paragraphs × "
          f"d_sae={sae.d_sae}")
    out = np.zeros((len(paragraphs), sae.d_sae), dtype=np.float32)
    for i, p_text in enumerate(tqdm(paragraphs, desc="wt-lasttok-z")):
        try:
            _, res = capture_all_position_residuals(
                model, tokenizer, p_text, layer=LAYER, max_len=WIKITEXT_MAX_TOKENS,
            )
            z_last = sae.encode(res[-1:].to(dtype=sae.W_enc.dtype)).squeeze(0)
            out[i] = z_last.float().cpu().numpy()
            del res, z_last
        except Exception as e:
            print(f"[wt-z] paragraph {i} failed: {type(e).__name__}: {e}")
        if i % 50 == 0:
            torch.cuda.empty_cache(); gc.collect()
    return out


# =============================================================================
# (A) Magnitude-matched WT-shuffled control
# =============================================================================

def run_a(tokenizer, model, sae, enc_dict, detected, pid_to_pair, a_prompts,
          published_specific, wt_draws, wt_lasttok_z, results_main):
    out_dir = P15_DIR / "a"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (A) magnitude-matched WT-shuffled ==========")

    pub = {(e["pair_id"], e["disambig_idx"]): list(e["features"])
           for e in published_specific}
    target_sets = {(p["id"], di): set(d["first_token_variants"])
                   for p in detected for di, d in enumerate(p["disambigs"])}

    rows = []
    for draw in tqdm(wt_draws, desc="(A) matched-wt"):
        pid = draw["pair_id"]; ab_i = draw["target_idx"]; di = draw["draw_idx"]
        para_idx = draw["wikitext_paragraph_idx"]
        key = (pid, ab_i)
        feats_pub = pub.get(key, [])
        if not feats_pub:
            continue
        z_Di = enc_dict[f"D__{pid}__{ab_i}"]
        # Profile: z_{D_i} values of the published Targeted top-10
        profile = sorted(
            [float(z_Di[fid].item()) for fid in feats_pub], reverse=True,
        )
        # Candidate pool: WT paragraph's positive-activation features
        z_para = wt_lasttok_z[para_idx]
        cand_idx = np.where(z_para > 0)[0]
        if len(cand_idx) == 0:
            matched = []
        else:
            cand_vals = z_para[cand_idx].tolist()
            matched = _greedy_match(profile, cand_idx.tolist(), cand_vals, k=10)
        target_self = target_sets[key]
        wt = forward_with_ablation(
            model, tokenizer, a_prompts[pid], LAYER, sae,
            feature_ids=matched, k=TOP_LOGITS_K,
        )
        rows.append({
            "pair_id": pid, "target_idx": ab_i, "draw_idx": di,
            "wikitext_paragraph_idx": para_idx,
            "n_matched": len(matched),
            "matched_features": matched,
            "matched_hit1": hit_at_k(wt["top_ids"], target_self, 1),
        })
    save_json_atomic(out_dir / "matched_wt_rows.json", rows)

    # Per-pair WT-shuffled mean
    wt_per_pair = per_pair_means(
        rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["matched_hit1"],
    )

    # Sibling per-pair from published cross_rows
    sibling = per_pair_means(
        results_main["cross_rows"],
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["ablate_hit1"],
    )

    keys = sorted(k for k in wt_per_pair if k in sibling)
    Sib = np.array([sibling[k] for k in keys], dtype=float)
    WSh = np.array([wt_per_pair[k] for k in keys], dtype=float)
    t3 = _t3_stats(Sib, WSh)
    save_json_atomic(out_dir / "t3.json", t3)
    print(f"[A] T3 (Sibling vs magnitude-matched-WT) "
          f"Δ pp = {t3['delta_pp']:+.2f}  "
          f"CI95 = [{t3['ci_pp'][0]:+.2f}, {t3['ci_pp'][1]:+.2f}]  "
          f"McN p={t3['mcnemar']['p']:.3g}  "
          f"Wlx p={t3['wilcoxon_p']:.3g}")
    print(f"[A] Sib mean = {t3['sibling_mean']:.4f}, "
          f"matched-WT mean = {t3['wt_shuf_mean']:.4f}")
    return t3


# =============================================================================
# (B) Score-on-WT control
# =============================================================================

def run_b(tokenizer, model, sae, detected, pid_to_pair, a_prompts,
          wt_draws, wt_lasttok_z, results_main):
    out_dir = P15_DIR / "b"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (B) score-on-WT control ==========")

    target_sets = {(p["id"], di): set(d["first_token_variants"])
                   for p in detected for di, d in enumerate(p["disambigs"])}

    # Compute score(f, p) = 1.5 * z_p(f) - mean_{p'!=p} z_{p'}(f)
    # Vectorized: score = 1.5 * z - (sum_z_total - z) / (P - 1)
    P = wt_lasttok_z.shape[0]
    sum_total = wt_lasttok_z.sum(axis=0)   # shape (d_sae,)
    score = (1.0 + SCORE_A_WEIGHT) * wt_lasttok_z - (
        sum_total[None, :] - wt_lasttok_z) / max(P - 1, 1)
    # Mask: only features positive on this paragraph
    mask = wt_lasttok_z > 0
    score_masked = np.where(mask, score, -1e9)
    # Top-10 per paragraph
    top10_per_para = np.argsort(-score_masked, axis=1)[:, :TOP_K_FEATURES].tolist()
    # Filter out features below threshold (in case mask removed everything)
    scored_top10 = []
    for p_idx, fids in enumerate(top10_per_para):
        kept = [int(f) for f in fids
                if mask[p_idx, f] and score_masked[p_idx, f] > -1e8]
        scored_top10.append({"paragraph_idx": p_idx, "top10_feature_ids": kept})
    save_json_atomic(out_dir / "scored_wt_top10_per_para.json", scored_top10)
    scored_top10_by_idx = {item["paragraph_idx"]: item["top10_feature_ids"]
                            for item in scored_top10}
    n_features_per_para = [len(s["top10_feature_ids"]) for s in scored_top10]
    print(f"[B] scored top-10 per paragraph: mean={np.mean(n_features_per_para):.2f}, "
          f"median={np.median(n_features_per_para)}, "
          f"min={np.min(n_features_per_para)}, max={np.max(n_features_per_para)}")

    rows = []
    for draw in tqdm(wt_draws, desc="(B) score-wt"):
        pid = draw["pair_id"]; ab_i = draw["target_idx"]; di = draw["draw_idx"]
        para_idx = draw["wikitext_paragraph_idx"]
        feats = scored_top10_by_idx.get(para_idx, [])
        target_self = target_sets[(pid, ab_i)]
        wt = forward_with_ablation(
            model, tokenizer, a_prompts[pid], LAYER, sae,
            feature_ids=feats, k=TOP_LOGITS_K,
        )
        rows.append({
            "pair_id": pid, "target_idx": ab_i, "draw_idx": di,
            "wikitext_paragraph_idx": para_idx,
            "n_features": len(feats),
            "scored_wt_hit1": hit_at_k(wt["top_ids"], target_self, 1),
        })
    save_json_atomic(out_dir / "score_on_wt_rows.json", rows)

    wt_per_pair = per_pair_means(
        rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["scored_wt_hit1"],
    )
    sibling = per_pair_means(
        results_main["cross_rows"],
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["ablate_hit1"],
    )

    keys = sorted(k for k in wt_per_pair if k in sibling)
    Sib = np.array([sibling[k] for k in keys], dtype=float)
    WSh = np.array([wt_per_pair[k] for k in keys], dtype=float)
    t3 = _t3_stats(Sib, WSh)
    save_json_atomic(out_dir / "t3.json", t3)
    print(f"[B] T3 (Sibling vs score-on-WT) "
          f"Δ pp = {t3['delta_pp']:+.2f}  "
          f"CI95 = [{t3['ci_pp'][0]:+.2f}, {t3['ci_pp'][1]:+.2f}]  "
          f"McN p={t3['mcnemar']['p']:.3g}  "
          f"Wlx p={t3['wilcoxon_p']:.3g}")
    print(f"[B] Sib mean = {t3['sibling_mean']:.4f}, "
          f"scored-WT mean = {t3['wt_shuf_mean']:.4f}")
    return t3


# =============================================================================
# (C) Per-feature equivalence diagnostic
# =============================================================================

def run_c(tokenizer, model, sae, enc_dict, detected, pid_to_pair, a_prompts,
          published_specific, per_feature_pos):
    out_dir = P15_DIR / "c"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (C) per-feature magnitude diagnostic ==========")

    pub = {(e["pair_id"], e["disambig_idx"]): list(e["features"])
           for e in published_specific}
    suspect_080 = {int(fid) for fid, st in per_feature_pos.items()
                   if st["n_nonzero"] >= POSITION_NONZERO_FLOOR
                   and st["pct_at_pos0"] >= POSITION_PCT_THRESHOLD}

    # Per-pair partition + magnitudes
    per_pair = []
    for p in detected:
        K = len(p["disambigs"])
        for i in range(K):
            ti = pub.get((p["id"], i))
            if not ti:
                continue
            sib_union = set()
            for j in range(K):
                if j == i:
                    continue
                sib_union |= set(pub.get((p["id"], j), []))
            shared = set(ti) & sib_union
            unique = set(ti) - shared
            uc = unique - suspect_080
            sc = shared - suspect_080
            # z_{D_i} magnitudes
            z_Di = enc_dict[f"D__{p['id']}__{i}"]
            mag = lambda f: float(z_Di[int(f)].item())
            uc_mags = [mag(f) for f in uc]
            sc_mags = [mag(f) for f in sc]
            per_pair.append({
                "pair_id": p["id"], "target_idx": i,
                "uc_features": list(uc), "uc_mags": uc_mags,
                "sc_features": list(sc), "sc_mags": sc_mags,
                "uc_mean": float(np.mean(uc_mags)) if uc_mags else None,
                "sc_mean": float(np.mean(sc_mags)) if sc_mags else None,
                "uc_max": float(np.max(uc_mags)) if uc_mags else None,
                "uc_argmax": (int(list(uc)[int(np.argmax(uc_mags))])
                               if uc_mags else None),
                "sc_max": float(np.max(sc_mags)) if sc_mags else None,
                "sc_argmax": (int(list(sc)[int(np.argmax(sc_mags))])
                               if sc_mags else None),
            })
    save_json_atomic(out_dir / "per_pair_magnitudes.json", per_pair)

    # ---- (C)(i) Wilcoxon signed-rank on per-pair (mean uc, mean sc) ----
    paired = [(r["uc_mean"], r["sc_mean"]) for r in per_pair
              if r["uc_mean"] is not None and r["sc_mean"] is not None]
    n_paired = len(paired)
    print(f"[C(i)] paired (uc_mean, sc_mean) per (P, D_i): n={n_paired}")
    if n_paired >= 2:
        uc_arr = np.array([t[0] for t in paired])
        sc_arr = np.array([t[1] for t in paired])
        from scipy.stats import wilcoxon
        try:
            stat = wilcoxon(uc_arr, sc_arr, alternative="two-sided")
            p_two = float(stat.pvalue)
        except Exception:
            p_two = float("nan")
        try:
            stat_g = wilcoxon(uc_arr, sc_arr, alternative="greater")
            p_g = float(stat_g.pvalue)
        except Exception:
            p_g = float("nan")
    else:
        uc_arr = np.array([]); sc_arr = np.array([])
        p_two = float("nan"); p_g = float("nan")

    mag_test = {
        "n_paired": n_paired,
        "mean_uc_z": float(uc_arr.mean()) if len(uc_arr) else None,
        "mean_sc_z": float(sc_arr.mean()) if len(sc_arr) else None,
        "median_uc_z": float(np.median(uc_arr)) if len(uc_arr) else None,
        "median_sc_z": float(np.median(sc_arr)) if len(sc_arr) else None,
        "mean_uc_minus_sc": float((uc_arr - sc_arr).mean()) if len(uc_arr) else None,
        "wilcoxon_two_sided_p": p_two,
        "wilcoxon_uc_greater_p": p_g,
    }
    save_json_atomic(out_dir / "magnitude_test.json", mag_test)
    print(f"[C(i)] uc mean z = {mag_test['mean_uc_z']:.3f}, "
          f"sc mean z = {mag_test['mean_sc_z']:.3f}, "
          f"diff = {mag_test['mean_uc_minus_sc']:+.3f}")
    print(f"[C(i)] Wilcoxon two-sided p = {p_two:.3g}, "
          f"one-sided 'uc > sc' p = {p_g:.3g}")

    # ---- (C)(i) Per-magnitude-bin per-feature impact ratio (CPU, from four_way_decomposition.py (D)) ----
    # Use four_way/d/four_way_rows.json: rows have n_uc, n_sc, uc_hit1, sc_hit1, baseline_hit1
    p14_rows_path = ARTIFACTS_DIR / "four_way" / "d" / "four_way_rows.json"
    stratified = {}
    if p14_rows_path.exists():
        p14_rows = json.load(open(p14_rows_path))
        # Map (pid, ti) -> p14_row
        p14_by_key = {(r["pair_id"], r["target_idx"]): r for r in p14_rows}
        # Bin by magnitude (uc_mean and sc_mean separately are per-pair)
        # Use 3 bins: low, mid, high (thirds of the combined distribution)
        all_means = []
        for r in per_pair:
            if r["uc_mean"] is not None: all_means.append(r["uc_mean"])
            if r["sc_mean"] is not None: all_means.append(r["sc_mean"])
        all_means_arr = np.array(all_means)
        if len(all_means_arr) > 6:
            q1, q2 = np.quantile(all_means_arr, [1/3, 2/3])
        else:
            q1, q2 = 0.0, 0.0

        def _bin(m):
            if m is None: return None
            if m < q1: return "low"
            if m < q2: return "mid"
            return "high"

        # Per-bin aggregate impact for unique_content and shared_content
        bins = ["low", "mid", "high"]
        bin_data = {b: {"uc_drops": [], "sc_drops": [], "uc_sizes": [], "sc_sizes": []}
                    for b in bins}
        for r in per_pair:
            key = (r["pair_id"], r["target_idx"])
            p14r = p14_by_key.get(key)
            if p14r is None:
                continue
            base = p14r["baseline_hit1"]
            uc_drop = (p14r["uc_hit1"] - base) if r["uc_mean"] is not None else None
            sc_drop = (p14r["sc_hit1"] - base) if r["sc_mean"] is not None else None
            uc_bin = _bin(r["uc_mean"])
            sc_bin = _bin(r["sc_mean"])
            if uc_drop is not None and uc_bin is not None and r["uc_mags"]:
                bin_data[uc_bin]["uc_drops"].append(uc_drop)
                bin_data[uc_bin]["uc_sizes"].append(len(r["uc_mags"]))
            if sc_drop is not None and sc_bin is not None and r["sc_mags"]:
                bin_data[sc_bin]["sc_drops"].append(sc_drop)
                bin_data[sc_bin]["sc_sizes"].append(len(r["sc_mags"]))
        stratified["bin_thresholds_q1_q2"] = [float(q1), float(q2)]
        stratified["bins"] = {}
        for b in bins:
            d = bin_data[b]
            if not d["uc_drops"] or not d["sc_drops"]:
                stratified["bins"][b] = {"uc_n": len(d["uc_drops"]),
                                          "sc_n": len(d["sc_drops"]),
                                          "note": "insufficient data"}
                continue
            uc_agg = float(np.mean(d["uc_drops"])) * 100   # in pp
            sc_agg = float(np.mean(d["sc_drops"])) * 100
            uc_size = float(np.mean(d["uc_sizes"]))
            sc_size = float(np.mean(d["sc_sizes"]))
            uc_per_feat = uc_agg / uc_size if uc_size else None
            sc_per_feat = sc_agg / sc_size if sc_size else None
            stratified["bins"][b] = {
                "uc_n": len(d["uc_drops"]), "sc_n": len(d["sc_drops"]),
                "uc_aggregate_pp": uc_agg,
                "sc_aggregate_pp": sc_agg,
                "uc_mean_size": uc_size,
                "sc_mean_size": sc_size,
                "uc_per_feature_pp": uc_per_feat,
                "sc_per_feature_pp": sc_per_feat,
                "ratio_uc_over_sc": (uc_per_feat / sc_per_feat
                                      if sc_per_feat and sc_per_feat != 0 else None),
            }
            print(f"[C(i) bin={b}] uc n={len(d['uc_drops'])}: "
                  f"agg={uc_agg:+.2f}, mean_size={uc_size:.2f}, "
                  f"per-feat={uc_per_feat:+.4f}")
            print(f"[C(i) bin={b}] sc n={len(d['sc_drops'])}: "
                  f"agg={sc_agg:+.2f}, mean_size={sc_size:.2f}, "
                  f"per-feat={sc_per_feat:+.4f}")
    save_json_atomic(out_dir / "stratified_impact.json", stratified)

    # ---- Histogram of uc vs sc mean z ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(uc_arr) > 0:
        ax.hist(uc_arr, bins=30, alpha=0.6, color="#37a", edgecolor="white",
                label=f"unique_content mean z (n={len(uc_arr)})")
    if len(sc_arr) > 0:
        ax.hist(sc_arr, bins=30, alpha=0.6, color="#a73", edgecolor="white",
                label=f"shared_content mean z (n={len(sc_arr)})")
    ax.set_xlabel("per-pair mean z_{D_i} of subset features")
    ax.set_ylabel("count of (P, D_i)")
    ax.set_title("magnitude distribution: unique_content vs shared_content")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "unique_vs_shared_magnitude_histogram.png", dpi=140)
    plt.close()

    # ---- (C)(ii) Stratified single-feature ablation ----
    print(f"\n[C(ii)] stratified single-feature ablations")
    eligible = [r for r in per_pair
                if r["uc_argmax"] is not None and r["sc_argmax"] is not None]
    print(f"[C(ii)] {len(eligible)} (P, D_i) have both uc and sc nonempty "
          f"(eligible for single-feature ablation)")

    # Pull baseline hits from results_main
    self_rows_main = json.load(open(ARTIFACTS_DIR / "results_main.json"))["self_rows"]
    base_hit = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                for r in self_rows_main}
    target_sets = {(p["id"], di): set(d["first_token_variants"])
                   for p in detected for di, d in enumerate(p["disambigs"])}

    sf_rows = []
    for r in tqdm(eligible, desc="(C)(ii) single-feat"):
        pid = r["pair_id"]; ti = r["target_idx"]
        a_prompt = a_prompts[pid]
        target = target_sets[(pid, ti)]
        base = base_hit.get((pid, ti))
        if base is None:
            continue
        uc_fid = r["uc_argmax"]
        sc_fid = r["sc_argmax"]
        ab_uc = forward_with_ablation(
            model, tokenizer, a_prompt, LAYER, sae,
            feature_ids=[uc_fid], k=TOP_LOGITS_K,
        )
        uc_hit = hit_at_k(ab_uc["top_ids"], target, 1)
        ab_sc = forward_with_ablation(
            model, tokenizer, a_prompt, LAYER, sae,
            feature_ids=[sc_fid], k=TOP_LOGITS_K,
        )
        sc_hit = hit_at_k(ab_sc["top_ids"], target, 1)
        sf_rows.append({
            "pair_id": pid, "target_idx": ti,
            "baseline_hit1": base,
            "uc_top_feature_id": uc_fid, "uc_top_z": r["uc_max"],
            "uc_top_hit1": uc_hit,
            "sc_top_feature_id": sc_fid, "sc_top_z": r["sc_max"],
            "sc_top_hit1": sc_hit,
        })
    save_json_atomic(out_dir / "single_feature_rows.json", sf_rows)

    n = len(sf_rows)
    base_arr = np.array([r["baseline_hit1"] for r in sf_rows], dtype=float)
    uc_arr = np.array([r["uc_top_hit1"] for r in sf_rows], dtype=float)
    sc_arr = np.array([r["sc_top_hit1"] for r in sf_rows], dtype=float)
    uc_drop = (uc_arr - base_arr).mean() * 100
    sc_drop = (sc_arr - base_arr).mean() * 100

    # Tests vs baseline
    mc_uc = mcnemar_paired(uc_arr, base_arr, alternative="greater")
    wx_uc = wilcoxon_paired(base_arr, uc_arr, alternative="greater")
    mc_sc = mcnemar_paired(sc_arr, base_arr, alternative="greater")
    wx_sc = wilcoxon_paired(base_arr, sc_arr, alternative="greater")
    # uc vs sc head-to-head
    mc_uvs = mcnemar_paired(uc_arr, sc_arr, alternative="two-sided")
    wx_uvs = wilcoxon_paired(uc_arr, sc_arr, alternative="two-sided")

    sf_summary = {
        "n_eligible": n,
        "uc_top_mean_hit1": float(uc_arr.mean()),
        "sc_top_mean_hit1": float(sc_arr.mean()),
        "baseline_mean_hit1": float(base_arr.mean()),
        "uc_top_drop_pp": float(uc_drop),
        "sc_top_drop_pp": float(sc_drop),
        "uc_top_drop_minus_sc_top_drop_pp": float(uc_drop - sc_drop),
        "uc_vs_baseline":  {"mcnemar": mc_uc, "wilcoxon_p": wx_uc},
        "sc_vs_baseline":  {"mcnemar": mc_sc, "wilcoxon_p": wx_sc},
        "uc_vs_sc_two_sided": {"mcnemar": mc_uvs, "wilcoxon_p": wx_uvs},
    }
    save_json_atomic(out_dir / "single_feature_summary.json", sf_summary)
    print(f"[C(ii)] uc_top:  Δ = {uc_drop:+.2f} pp,  McN p={mc_uc['p']:.3g},  "
          f"Wlx p={wx_uc:.3g}")
    print(f"[C(ii)] sc_top:  Δ = {sc_drop:+.2f} pp,  McN p={mc_sc['p']:.3g},  "
          f"Wlx p={wx_sc:.3g}")
    print(f"[C(ii)] uc_top vs sc_top:  Δ_diff = {uc_drop - sc_drop:+.2f} pp,  "
          f"McN p={mc_uvs['p']:.3g},  Wlx p={wx_uvs:.3g}")
    return mag_test, stratified, sf_summary


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    P15_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"[setup] loading model + SAE...")
    tokenizer, model = load_model()
    sae, sae_meta = load_sae(layer=LAYER)
    if sae_meta["sha256"] != SAE_CHECKPOINT_SHA256:
        raise RuntimeError(f"SAE sha256 mismatch! got {sae_meta['sha256']}")
    print(f"[setup] SAE sha256 verified ({sae_meta['sha256'][:16]}...)")

    npz = np.load(ARTIFACTS_DIR / f"sae_encodings_L{LAYER}.npz")
    enc_dict = {
        k: torch.from_numpy(npz[k]).to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        for k in npz.files
    }
    detected = json.load(open(ARTIFACTS_DIR / "detected_pairs.json"))
    pid_to_pair = {p["id"]: p for p in detected}
    a_prompts = {p["id"]: build_prompt(tokenizer, p["A_question"]) for p in detected}
    published_specific = json.load(open(ARTIFACTS_DIR / "specific_features.json"))
    results_main = json.load(open(ARTIFACTS_DIR / "results_main.json"))
    pos_mode = json.load(open(ARTIFACTS_DIR / "wikitext_position_mode.json"))
    per_feature_pos = pos_mode["per_feature"]
    wt_draws = json.load(open(ARTIFACTS_DIR / "shuffle_draws_wikitext.json"))

    print(f"[setup] {len(detected)} detected pairs, "
          f"{sum(len(p['disambigs']) for p in detected)} self-pairs, "
          f"{len(wt_draws)} WT draws")

    # Capture WT last-token z for (A) and (B)
    paragraphs = load_wikitext_paragraphs(tokenizer)
    print(f"[setup] reconstructed {len(paragraphs)} WikiText paragraphs")
    wt_lasttok_z = _capture_wt_lasttok_z(model, tokenizer, sae, paragraphs)

    a_t3 = run_a(tokenizer, model, sae, enc_dict, detected, pid_to_pair, a_prompts,
                  published_specific, wt_draws, wt_lasttok_z, results_main)
    b_t3 = run_b(tokenizer, model, sae, detected, pid_to_pair, a_prompts,
                  wt_draws, wt_lasttok_z, results_main)
    c_mag, c_strat, c_sf = run_c(tokenizer, model, sae, enc_dict, detected,
                                   pid_to_pair, a_prompts, published_specific,
                                   per_feature_pos)

    # Comparison table
    # Reconstruct published T3 from results_main
    sibling = per_pair_means(
        results_main["cross_rows"],
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["ablate_hit1"],
    )
    wt_pub = per_pair_means(
        results_main["wikitext_shuffled_rows"],
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["wt_shuffled_hit1"],
    )
    keys_pub = sorted(k for k in sibling if k in wt_pub)
    Sib_pub = np.array([sibling[k] for k in keys_pub], dtype=float)
    WSh_pub = np.array([wt_pub[k] for k in keys_pub], dtype=float)
    pub_t3 = _t3_stats(Sib_pub, WSh_pub)

    rollup = {
        "comparison_t3_rows": [
            {"row": "published WT-shuffled (raw activation top-10)",
             "n": pub_t3["n"], "delta_pp": pub_t3["delta_pp"],
             "ci_pp": pub_t3["ci_pp"],
             "mcnemar_p": pub_t3["mcnemar"]["p"],
             "wilcoxon_p": pub_t3["wilcoxon_p"],
             "wt_shuf_mean_hit1": pub_t3["wt_shuf_mean"]},
            {"row": "(A) magnitude-matched WT",
             "n": a_t3["n"], "delta_pp": a_t3["delta_pp"],
             "ci_pp": a_t3["ci_pp"],
             "mcnemar_p": a_t3["mcnemar"]["p"],
             "wilcoxon_p": a_t3["wilcoxon_p"],
             "wt_shuf_mean_hit1": a_t3["wt_shuf_mean"]},
            {"row": "(B) score-on-WT",
             "n": b_t3["n"], "delta_pp": b_t3["delta_pp"],
             "ci_pp": b_t3["ci_pp"],
             "mcnemar_p": b_t3["mcnemar"]["p"],
             "wilcoxon_p": b_t3["wilcoxon_p"],
             "wt_shuf_mean_hit1": b_t3["wt_shuf_mean"]},
        ],
        "magnitude_test": c_mag,
        "stratified_impact": c_strat,
        "single_feature_summary": c_sf,
    }
    save_json_atomic(P15_DIR / "comparison_table.json", rollup)
    save_json_atomic(P15_DIR / "run_meta.json", {
        "model": MODEL_ID, "layer": LAYER, "sae_meta": sae_meta,
        "n_detected_pairs": len(detected),
        "elapsed_seconds": round(time.time() - t_total, 2),
    })

    print(f"\n=== Stage 1.5 complete ===")
    print(f"  total elapsed: {(time.time()-t_total)/60:.1f} min")
    for r in rollup["comparison_t3_rows"]:
        print(f"  {r['row']:<48}  n={r['n']:<5}  Δ_T3={r['delta_pp']:+6.2f} pp  "
              f"CI=[{r['ci_pp'][0]:+5.2f},{r['ci_pp'][1]:+5.2f}]  "
              f"McN p={r['mcnemar_p']:.2e}  Wlx p={r['wilcoxon_p']:.2e}")


if __name__ == "__main__":
    main()
