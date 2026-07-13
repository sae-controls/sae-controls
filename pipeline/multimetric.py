"""Multi-metric decomposition + k=2 dip diagnostic.

The unique/shared analysis established two-tier specificity (Shared 51% + Unique 40% + 9%
residual) on hit@1. This script:

  (A) k=2 mean-overlap dip diagnostic. Characterizes the [2,3) mean-overlap
      shell (n=151): K-stratification, Neuronpedia lookups for sampled
      shared features, and the per-pair Δ_T3 distribution.

  (B) Multi-metric decomposition. Re-run forwards for Baseline, Targeted,
      Shared-only, Unique-only (top_logits k=50 saved here; the main-run
      artifacts store hit@1 only) and WikiText-shuffled (for T3
      continuity). Compute per-(P, D_i) metrics:
        - KL(p_baseline || p_condition) on baseline's top-10 support
        - Logit-diff: logit(D_i first-token) − max_{j≠i} logit(D_j first)
        - Rank shift: rank of D_i first-token (censored at 11)
      Plus generation-level: greedy-decode 15 tokens under each condition,
      classify with src/slot_detection.find_best_match. Per-condition
      flip rate from baseline's committed slot.

Reads:
  artifacts/sae_encodings_L37.npz
  artifacts/detected_pairs.json
  artifacts/specific_features.json
  artifacts/results_main.json
  artifacts/wikitext_position_mode.json     (for context only; pm checked in the unique/shared stage)
  artifacts/wikitext_max_activating.json    (for (A) feature-context fallback)
  artifacts/feature_inventory.json          (for (A) feature frequency sample)
  artifacts/feature_origins_and_ambigqa.json (for (A) feature origin context)
  artifacts/shuffle_draws_wikitext.json
  artifacts/multimetric/wikitext_top30.json (for WT pool)
  artifacts/unique_vs_shared/b/mean_overlap_shells.json (for (A) [2,3) keys)

Writes:
  artifacts/multimetric/a/{k2_stratification.json, k2_feature_lookups.json,
                          k2_per_pair_T3.json, k2_per_pair_histogram.png}
  artifacts/multimetric/b/forwards/{baseline.json, targeted.json,
                                    shared_only.json, unique_only.json,
                                    wt_shuffled.json}
  artifacts/multimetric/b/{per_pair_metrics.json,
                          decomposition_kl.json,
                          decomposition_logit_diff.json,
                          decomposition_rank_shift.json,
                          t3_per_metric.json,
                          generations.json,
                          generation_classifications.json,
                          confusion_matrices.json,
                          generation_decomposition.json,
                          generation_sanity_check.json,
                          censoring_rates.json}
  artifacts/multimetric/run_meta.json
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import math
import os
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
    ARTIFACTS_DIR, LAYER, MODEL_ID, SAE_CHECKPOINT_SHA256, TOP_LOGITS_K,
    WIKITEXT_MAX_TOKENS,
)
from src.hooks import (
    baseline_top_logits, forward_with_ablation,
    forward_with_ablation_then_generate, greedy_decode, hit_at_k,
)
from src.io_utils import save_json_atomic
from src.model import load_model
from src.prompts import build_prompt
from src.sae import load_sae
from src.slot_detection import find_best_match


P13_DIR = ARTIFACTS_DIR / "multimetric"
P12PP_DIR = ARTIFACTS_DIR / "unique_vs_shared"
P11_DIR = ARTIFACTS_DIR / "multimetric"  # bundled WT-pool input (wikitext_top30.json)

# Save logits at k=50 to reduce censoring on logit-diff/rank-shift; KL is
# computed on baseline's top-10 support.
SAVE_TOP_K = 50
KL_SUPPORT_K = 10
RANK_CENSOR = 11
SAMPLE_FEATURES_FOR_LOOKUP = 10
GEN_MAX_NEW = 15
SANITY_N = 20


# =============================================================================
# Forward harvesting (saves top-50 logits)
# =============================================================================

def _harvest_baseline(model, tokenizer, detected, a_prompts, out_path):
    print(f"[B1] baseline forwards: {len(detected)}")
    res = {}
    for p in tqdm(detected, desc="B1 baseline"):
        d = baseline_top_logits(model, tokenizer, a_prompts[p["id"]], k=SAVE_TOP_K)
        res[p["id"]] = d
    save_json_atomic(out_path, res)
    return res


def _harvest_ablations(model, tokenizer, sae, detected, a_prompts,
                       feats_per_key, label, out_path,
                       skip_if_empty=True):
    """`feats_per_key` is dict (pair_id, ab_i) -> list[int] feature ids."""
    print(f"[B1] {label} forwards: {len(feats_per_key)} (skip_empty={skip_if_empty})")
    res = {}
    for (pid, ab_i), feats in tqdm(list(feats_per_key.items()), desc=f"B1 {label}"):
        if skip_if_empty and not feats:
            continue
        a_prompt = a_prompts[pid]
        d = forward_with_ablation(model, tokenizer, a_prompt, LAYER, sae,
                                   feature_ids=feats, k=SAVE_TOP_K)
        res[f"{pid}|{ab_i}"] = d
        if len(res) % 200 == 0:
            torch.cuda.empty_cache()
    save_json_atomic(out_path, res)
    return res


def _harvest_wt_shuf(model, tokenizer, sae, a_prompts, wt_draws, wt_top_by_idx,
                     k_features, out_path):
    print(f"[B1] wt-shuffled forwards: {len(wt_draws)} (top_k={k_features})")
    res = {}
    for draw in tqdm(wt_draws, desc="B1 wt-shuf"):
        pid = draw["pair_id"]; ab_i = draw["target_idx"]; di = draw["draw_idx"]
        para_idx = draw["wikitext_paragraph_idx"]
        feats = wt_top_by_idx.get(para_idx, [])[:k_features]
        d = forward_with_ablation(model, tokenizer, a_prompts[pid], LAYER, sae,
                                   feature_ids=feats, k=SAVE_TOP_K)
        res[f"{pid}|{ab_i}|{di}"] = d
        if len(res) % 200 == 0:
            torch.cuda.empty_cache()
    save_json_atomic(out_path, res)
    return res


# =============================================================================
# Metric helpers
# =============================================================================

def _kl_topk(p_logits, p_ids, q_logits, q_ids, k=KL_SUPPORT_K):
    """KL(p || q) on p's top-k support; out-of-q-top-50 ids censored to
    q's min top-50 logit minus 5.0."""
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


def _logit_diff(top_ids, top_logits, target_first_id, sibling_first_ids):
    lookup = dict(zip(top_ids, top_logits))
    floor = float(min(top_logits)) - 5.0
    tlog = lookup.get(target_first_id, floor)
    if not sibling_first_ids:
        return None, False, False
    slogs = [lookup.get(s, floor) for s in sibling_first_ids]
    target_in = target_first_id in lookup
    any_sib_in = any(s in lookup for s in sibling_first_ids)
    return float(tlog - max(slogs)), bool(target_in), bool(any_sib_in)


def _rank_in_top(top_ids, target_id, k=KL_SUPPORT_K):
    if target_id in top_ids[:k]:
        return top_ids[:k].index(target_id) + 1, False
    return RANK_CENSOR, True


# =============================================================================
# (A) k=2 mean-overlap dip diagnostic
# =============================================================================

def _try_neuronpedia_lookup(feature_id):
    """Best-effort fetch of an auto-interpretation from Neuronpedia. Returns
    a short string description or None on failure / network unavailability.
    Mirrors the public REST URL pattern; tries a couple of source IDs since
    Neuronpedia's source slug for Gemma Scope can vary across mirrors."""
    try:
        import requests
    except ImportError:
        return {"available": False, "reason": "requests not installed"}
    candidate_sources = [
        "37-gemmascope-res-16k",
        "37-res-jb",
        "37-gemma-scope-9b-pt-res-canonical",
    ]
    for src in candidate_sources:
        url = f"https://www.neuronpedia.org/api/feature/gemma-2-9b/{src}/{feature_id}"
        try:
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                continue
            data = r.json()
            expls = data.get("explanations", [])
            if not expls:
                continue
            # Take the first explanation's description
            d = expls[0].get("description", "")
            return {"available": True, "source_used": src, "description": d}
        except Exception as e:
            continue
    return {"available": False, "reason": "no source returned a 200 with explanations"}


def run_a(detected, published_specific, results_main):
    out_dir = P13_DIR / "a"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (A) k=2 mean-overlap dip diagnostic ==========")

    pub_set = {}
    for entry in published_specific:
        pub_set[(entry["pair_id"], entry["disambig_idx"])] = set(entry["features"])

    # Build mean-overlap per (P, D_i) and identify [2, 3) shell
    overlaps_mean = {}
    K_per_pair = {p["id"]: len(p["disambigs"]) for p in detected}
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
            overlaps_mean[(p["id"], i)] = float(np.mean(sib_overlaps))

    shell_keys = [k for k, v in overlaps_mean.items() if 2.0 <= v < 3.0]
    print(f"[A] [2, 3) shell n = {len(shell_keys)}")

    # K stratification
    K2_keys = [k for k in shell_keys if K_per_pair[k[0]] == 2]
    K3plus_keys = [k for k in shell_keys if K_per_pair[k[0]] >= 3]

    # Build per-(P, D_i) Sibling and WT-shuf hits from published
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

    def _t3(keys):
        ks = [k for k in keys if k in sibling and k in wt_shuf]
        if len(ks) < 2:
            return None
        Sib = np.array([sibling[k] for k in ks], dtype=float)
        WSh = np.array([wt_shuf[k] for k in ks], dtype=float)
        mc = mcnemar_paired(Sib, WSh, alternative="greater")
        wx = wilcoxon_paired(WSh, Sib, alternative="greater")
        ci = bootstrap_diff_ci_pp(Sib, WSh, seed=0)
        return {"n": len(ks),
                "delta_pp": float((WSh.mean() - Sib.mean()) * 100),
                "ci_pp": [float(ci[0]), float(ci[1])],
                "mcnemar_p": mc["p"],
                "wilcoxon_p": wx,
                "per_pair_diff_pp": [(wt_shuf[k] - sibling[k]) * 100 for k in ks],
                "keys_order": [{"pair_id": k[0], "target_idx": k[1]} for k in ks]}

    strat = {
        "shell_n": len(shell_keys),
        "K_eq_2": _t3(K2_keys),
        "K_ge_3": _t3(K3plus_keys),
        "shell_all": _t3(shell_keys),
    }
    save_json_atomic(out_dir / "k2_stratification.json", strat)
    print(f"[A] K=2 sub-stratum:   n={strat['K_eq_2']['n'] if strat['K_eq_2'] else 0}  "
          f"Δ={strat['K_eq_2']['delta_pp'] if strat['K_eq_2'] else None:+.2f}  "
          f"Wilcoxon p={strat['K_eq_2']['wilcoxon_p'] if strat['K_eq_2'] else None}")
    print(f"[A] K≥3 sub-stratum:   n={strat['K_ge_3']['n'] if strat['K_ge_3'] else 0}  "
          f"Δ={strat['K_ge_3']['delta_pp'] if strat['K_ge_3'] else None:+.2f}  "
          f"Wilcoxon p={strat['K_ge_3']['wilcoxon_p'] if strat['K_ge_3'] else None}")
    print(f"[A] Full shell:        n={strat['shell_all']['n']}  "
          f"Δ={strat['shell_all']['delta_pp']:+.2f}  "
          f"Wilcoxon p={strat['shell_all']['wilcoxon_p']}")

    # Sample 10 shared-feature IDs and look them up
    rng = np.random.default_rng(0)
    shared_features_in_shell = []   # collect (pair_id, di, fid) tuples
    for (pid, i) in shell_keys:
        ti = pub_set[(pid, i)]
        K = K_per_pair[pid]
        sib_union = set()
        for j in range(K):
            if j == i:
                continue
            sib_union |= pub_set.get((pid, j), set())
        shared = ti & sib_union
        for fid in shared:
            shared_features_in_shell.append((pid, i, int(fid)))
    print(f"[A] {len(shared_features_in_shell)} shared-feature occurrences in shell "
          f"({len(set(t[2] for t in shared_features_in_shell))} unique)")
    unique_fids = sorted(set(t[2] for t in shared_features_in_shell))
    sample_fids = (rng.choice(unique_fids, size=min(SAMPLE_FEATURES_FOR_LOOKUP, len(unique_fids)),
                              replace=False).tolist()
                   if unique_fids else [])

    # Cross-reference with feature_inventory and wikitext_max_activating + origins
    inv = json.load(open(ARTIFACTS_DIR / "feature_inventory.json"))
    inv_freq = {int(k): v for k, v in inv["freq_full"].items()}
    wma = json.load(open(ARTIFACTS_DIR / "wikitext_max_activating.json"))
    origins = json.load(open(ARTIFACTS_DIR / "feature_origins_and_ambigqa.json"))
    origin_lookup = {entry["feature_id"]: entry for entry in origins} \
        if isinstance(origins, list) else {}

    feature_lookups = []
    for fid in sample_fids:
        fid_int = int(fid)
        np_lookup = _try_neuronpedia_lookup(fid_int)
        wma_entry = wma.get(str(fid_int))
        wma_top3 = (wma_entry["top20"][:3] if wma_entry else None)
        origin_entry = origin_lookup.get(fid_int) or origin_lookup.get(str(fid_int))
        feature_lookups.append({
            "feature_id": fid_int,
            "n_lists_appearing_in": inv_freq.get(fid_int, None),
            "neuronpedia": np_lookup,
            "wikitext_max_activating_top3": wma_top3,
            "ambigqa_origin": origin_entry,
        })
    save_json_atomic(out_dir / "k2_feature_lookups.json", {
        "n_unique_shared_fids_in_shell": len(unique_fids),
        "sampled_fids": sample_fids,
        "lookups": feature_lookups,
    })

    # Per-pair Δ_T3 distribution: histogram + outlier flagging
    per_pair_diff = strat["shell_all"]["per_pair_diff_pp"]
    pp_arr = np.array(per_pair_diff)
    mn, sd = float(pp_arr.mean()), float(pp_arr.std())
    outliers = []
    for k, d in zip(strat["shell_all"]["keys_order"], per_pair_diff):
        if abs(d - mn) > 3 * sd:
            outliers.append({"pair_id": k["pair_id"], "target_idx": k["target_idx"],
                             "diff_pp": d})
    save_json_atomic(out_dir / "k2_per_pair_T3.json", {
        "per_pair_diff_pp": per_pair_diff,
        "mean": mn, "std": sd,
        "n_outliers_3sigma": len(outliers),
        "outliers_3sigma": outliers,
    })
    print(f"[A] per-pair Δ_T3 mean={mn:+.2f}, std={sd:.2f}, "
          f"3σ outliers: {len(outliers)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pp_arr, bins=30, color="#37a", edgecolor="white")
    ax.axvline(mn, color="red", linestyle="--", label=f"mean = {mn:+.2f}")
    ax.axvline(0, color="black", linestyle="-")
    ax.set_xlabel("per-pair (WT-shuf − Sibling) hit@1, pp")
    ax.set_ylabel("count")
    ax.set_title(f"k=[2,3) shell per-pair Δ_T3 (n={len(per_pair_diff)})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "k2_per_pair_histogram.png", dpi=140)
    plt.close()
    return strat


# =============================================================================
# (B) Multi-metric decomposition
# =============================================================================

def run_b(tokenizer, model, sae, enc_dict, detected, a_prompts,
          published_specific, results_main):
    out_dir = P13_DIR / "b"
    fwd_dir = out_dir / "forwards"
    fwd_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (B) multi-metric decomposition ==========")

    # ---- Build feature sets per (P, D_i) ----
    pub_set = {}
    for entry in published_specific:
        pub_set[(entry["pair_id"], entry["disambig_idx"])] = list(entry["features"])

    targeted_feats = {}
    shared_feats = {}
    unique_feats = {}
    for p in detected:
        K = len(p["disambigs"])
        for i in range(K):
            ti = set(pub_set.get((p["id"], i), []))
            if not ti:
                continue
            sib_union = set()
            for j in range(K):
                if j == i:
                    continue
                sib_union |= set(pub_set.get((p["id"], j), []))
            sh = ti & sib_union
            un = ti - sh
            targeted_feats[(p["id"], i)] = list(ti)
            shared_feats[(p["id"], i)] = list(sh)
            unique_feats[(p["id"], i)] = list(un)

    # ---- B1. Harvest forwards ----
    baseline_path = fwd_dir / "baseline.json"
    targ_path = fwd_dir / "targeted.json"
    sh_path = fwd_dir / "shared_only.json"
    un_path = fwd_dir / "unique_only.json"
    wt_path = fwd_dir / "wt_shuffled.json"

    baseline_logits = _harvest_baseline(model, tokenizer, detected, a_prompts, baseline_path)
    targeted_logits = _harvest_ablations(model, tokenizer, sae, detected, a_prompts,
                                          targeted_feats, "targeted", targ_path)
    shared_logits = _harvest_ablations(model, tokenizer, sae, detected, a_prompts,
                                        shared_feats, "shared_only", sh_path)
    unique_logits = _harvest_ablations(model, tokenizer, sae, detected, a_prompts,
                                        unique_feats, "unique_only", un_path)

    wt_top = json.load(open(P11_DIR / "wikitext_top30.json"))
    wt_top_by_idx = {item["paragraph_idx"]: item["top_feature_ids"] for item in wt_top}
    wt_draws = json.load(open(ARTIFACTS_DIR / "shuffle_draws_wikitext.json"))
    wt_logits = _harvest_wt_shuf(model, tokenizer, sae, a_prompts, wt_draws,
                                  wt_top_by_idx, k_features=10, out_path=wt_path)

    # ---- B2. Compute per-(P, D_i) metrics ----
    def _entry(d, key):
        return d.get(key) if isinstance(key, str) else d.get(f"{key[0]}|{key[1]}")

    per_pair = []
    censoring = defaultdict(lambda: {"target_censored": 0, "any_sib_censored": 0,
                                      "rank_censored": 0, "n": 0})
    for p in detected:
        K = len(p["disambigs"])
        a_first_per_di = [d["first_token_variants"][0] for d in p["disambigs"]]
        for i in range(K):
            key = (p["id"], i)
            base = _entry(baseline_logits, p["id"])
            if base is None or key not in targeted_feats:
                continue
            target_first = a_first_per_di[i]
            sibling_first = [a_first_per_di[j] for j in range(K) if j != i]

            entry = {"pair_id": p["id"], "target_idx": i,
                     "K": K, "target_first": int(target_first),
                     "sibling_firsts": [int(x) for x in sibling_first],
                     "n_targeted": len(targeted_feats[key]),
                     "n_shared": len(shared_feats[key]),
                     "n_unique": len(unique_feats[key]),
                     "metrics": {}}

            base_kl = 0.0
            base_ld, t_in_b, anysib_in_b = _logit_diff(base["top_ids"], base["top_logits"],
                                                        target_first, sibling_first)
            base_rk, rk_cens_b = _rank_in_top(base["top_ids"], target_first)
            entry["metrics"]["baseline"] = {
                "kl_to_baseline": base_kl,
                "logit_diff": base_ld,
                "rank": base_rk,
                "target_in_topK": t_in_b,
                "rank_censored": rk_cens_b,
            }
            censoring["baseline"]["n"] += 1
            if not t_in_b: censoring["baseline"]["target_censored"] += 1
            if not anysib_in_b: censoring["baseline"]["any_sib_censored"] += 1
            if rk_cens_b: censoring["baseline"]["rank_censored"] += 1

            # Targeted (uses own forward)
            tg = _entry(targeted_logits, key)
            if tg is not None:
                kl = _kl_topk(base["top_logits"], base["top_ids"],
                              tg["top_logits"], tg["top_ids"])
                ld, t_in, sib_in = _logit_diff(tg["top_ids"], tg["top_logits"],
                                                target_first, sibling_first)
                rk, rk_cens = _rank_in_top(tg["top_ids"], target_first)
                entry["metrics"]["targeted"] = {"kl_to_baseline": kl, "logit_diff": ld,
                                                 "rank": rk}
                censoring["targeted"]["n"] += 1
                if not t_in: censoring["targeted"]["target_censored"] += 1
                if not sib_in: censoring["targeted"]["any_sib_censored"] += 1
                if rk_cens: censoring["targeted"]["rank_censored"] += 1
            else:
                entry["metrics"]["targeted"] = None

            # Shared-only
            sh_e = _entry(shared_logits, key)
            if shared_feats[key] and sh_e is not None:
                kl = _kl_topk(base["top_logits"], base["top_ids"],
                              sh_e["top_logits"], sh_e["top_ids"])
                ld, t_in, sib_in = _logit_diff(sh_e["top_ids"], sh_e["top_logits"],
                                                target_first, sibling_first)
                rk, rk_cens = _rank_in_top(sh_e["top_ids"], target_first)
                entry["metrics"]["shared_only"] = {"kl_to_baseline": kl, "logit_diff": ld,
                                                    "rank": rk}
                censoring["shared_only"]["n"] += 1
                if not t_in: censoring["shared_only"]["target_censored"] += 1
                if not sib_in: censoring["shared_only"]["any_sib_censored"] += 1
                if rk_cens: censoring["shared_only"]["rank_censored"] += 1
            else:
                # Empty shared → equivalent to baseline
                entry["metrics"]["shared_only"] = entry["metrics"]["baseline"]

            # Unique-only
            un_e = _entry(unique_logits, key)
            if unique_feats[key] and un_e is not None:
                kl = _kl_topk(base["top_logits"], base["top_ids"],
                              un_e["top_logits"], un_e["top_ids"])
                ld, t_in, sib_in = _logit_diff(un_e["top_ids"], un_e["top_logits"],
                                                target_first, sibling_first)
                rk, rk_cens = _rank_in_top(un_e["top_ids"], target_first)
                entry["metrics"]["unique_only"] = {"kl_to_baseline": kl, "logit_diff": ld,
                                                    "rank": rk}
                censoring["unique_only"]["n"] += 1
                if not t_in: censoring["unique_only"]["target_censored"] += 1
                if not sib_in: censoring["unique_only"]["any_sib_censored"] += 1
                if rk_cens: censoring["unique_only"]["rank_censored"] += 1
            else:
                entry["metrics"]["unique_only"] = entry["metrics"]["baseline"]

            # Sibling: mean over j != i of metrics from forward(ablate=Targeted_{D_j})
            sib_kls = []; sib_lds = []; sib_rks = []
            for j in range(K):
                if j == i:
                    continue
                k_other = (p["id"], j)
                tg_other = _entry(targeted_logits, k_other)
                if tg_other is None:
                    continue
                sib_kls.append(_kl_topk(base["top_logits"], base["top_ids"],
                                          tg_other["top_logits"], tg_other["top_ids"]))
                ld, _, _ = _logit_diff(tg_other["top_ids"], tg_other["top_logits"],
                                        target_first, sibling_first)
                sib_lds.append(ld)
                rk, _ = _rank_in_top(tg_other["top_ids"], target_first)
                sib_rks.append(rk)
            entry["metrics"]["sibling"] = {
                "kl_to_baseline": float(np.mean(sib_kls)) if sib_kls else None,
                "logit_diff": float(np.mean(sib_lds)) if sib_lds else None,
                "rank": float(np.mean(sib_rks)) if sib_rks else None,
            }

            # WikiText-shuffled: mean over 3 draws
            wt_kls = []; wt_lds = []; wt_rks = []
            for d_idx in range(3):
                w_e = _entry(wt_logits, f"{p['id']}|{i}|{d_idx}")
                if w_e is None:
                    continue
                wt_kls.append(_kl_topk(base["top_logits"], base["top_ids"],
                                        w_e["top_logits"], w_e["top_ids"]))
                ld, _, _ = _logit_diff(w_e["top_ids"], w_e["top_logits"],
                                        target_first, sibling_first)
                wt_lds.append(ld)
                rk, _ = _rank_in_top(w_e["top_ids"], target_first)
                wt_rks.append(rk)
            entry["metrics"]["wt_shuf"] = {
                "kl_to_baseline": float(np.mean(wt_kls)) if wt_kls else None,
                "logit_diff": float(np.mean(wt_lds)) if wt_lds else None,
                "rank": float(np.mean(wt_rks)) if wt_rks else None,
            }
            per_pair.append(entry)

    save_json_atomic(out_dir / "per_pair_metrics.json", per_pair)
    save_json_atomic(out_dir / "censoring_rates.json", dict(censoring))
    print(f"[B2] per-pair metrics computed for {len(per_pair)} (P, D_i)")
    print(f"[B2] censoring (target out-of-top-50): " +
          ", ".join(f"{k}={v['target_censored']}/{v['n']}"
                    for k, v in censoring.items()))

    # ---- B3. Decomposition tables per metric ----
    def _drop_decomp(metric_name, sign):
        """sign = +1 if 'higher = more drop' (KL, rank), -1 if 'lower = more drop'
        (hit, logit-diff). For decomposition, Δ vs baseline = condition - baseline,
        always."""
        key_list = [(r["pair_id"], r["target_idx"]) for r in per_pair]
        get = lambda r, c: r["metrics"][c][metric_name] if r["metrics"][c] else None
        B = np.array([get(r, "baseline")    for r in per_pair], dtype=float)
        T = np.array([get(r, "targeted")    for r in per_pair], dtype=float)
        Sh = np.array([get(r, "shared_only") for r in per_pair], dtype=float)
        Un = np.array([get(r, "unique_only") for r in per_pair], dtype=float)
        # Drop magnitude relative to baseline (in metric's natural direction)
        td = (T - B).mean()
        sd = (Sh - B).mean()
        ud = (Un - B).mean()
        sum_ = sd + ud
        residual = td - sum_
        pct = lambda x: (x / td * 100) if abs(td) > 1e-12 else float("nan")

        # 1-sided test "y drops more" — depends on sign convention
        alt_mc = "less" if sign == -1 else "greater"
        alt_wx = "greater" if sign == -1 else "greater"  # see note below
        # Note: mcnemar uses the x_kills_only > y_kills_only convention of src.analysis.mcnemar_paired.
        # For continuous metrics we do Wilcoxon only; mcnemar is N/A on continuous.
        def _wilcoxon_dir(b, c):
            try:
                from scipy.stats import wilcoxon
                # Test whether c is "worse" than b in metric direction
                if sign == 1:
                    p = float(wilcoxon(c, b, alternative="greater").pvalue)
                else:
                    p = float(wilcoxon(c, b, alternative="less").pvalue)
                return p
            except Exception:
                return float("nan")

        ci_T = list(bootstrap_diff_ci_pp(B, T, seed=0))
        ci_Sh = list(bootstrap_diff_ci_pp(B, Sh, seed=0))
        ci_Un = list(bootstrap_diff_ci_pp(B, Un, seed=0))

        return {
            "metric": metric_name,
            "sign_convention": "higher_means_more_drop" if sign == 1 else "lower_means_more_drop",
            "n": len(per_pair),
            "table": {
                "Baseline":    {"value": float(B.mean()), "delta": 0.0,
                                "ci_pp_x100": [None, None], "pct_of_targeted": 0.0},
                "Targeted":    {"value": float(T.mean()), "delta": float(td),
                                "ci_pp_x100": ci_T, "pct_of_targeted": 100.0,
                                "wilcoxon_p_vs_baseline": _wilcoxon_dir(B, T)},
                "Shared-only": {"value": float(Sh.mean()), "delta": float(sd),
                                "ci_pp_x100": ci_Sh, "pct_of_targeted": float(pct(sd)),
                                "wilcoxon_p_vs_baseline": _wilcoxon_dir(B, Sh)},
                "Unique-only": {"value": float(Un.mean()), "delta": float(ud),
                                "ci_pp_x100": ci_Un, "pct_of_targeted": float(pct(ud)),
                                "wilcoxon_p_vs_baseline": _wilcoxon_dir(B, Un)},
                "Sum (Sh+Un)": {"value": None, "delta": float(sum_),
                                "pct_of_targeted": float(pct(sum_))},
                "Residual":    {"value": None, "delta": float(residual),
                                "pct_of_targeted": float(pct(residual))},
            },
        }

    decomp_kl = _drop_decomp("kl_to_baseline", sign=+1)
    decomp_ld = _drop_decomp("logit_diff", sign=-1)
    decomp_rk = _drop_decomp("rank", sign=+1)
    save_json_atomic(out_dir / "decomposition_kl.json", decomp_kl)
    save_json_atomic(out_dir / "decomposition_logit_diff.json", decomp_ld)
    save_json_atomic(out_dir / "decomposition_rank_shift.json", decomp_rk)

    # ---- B4. T3 per metric (Sibling vs WT-shuf) ----
    def _t3_metric(metric_name, sign):
        Sib = np.array([r["metrics"]["sibling"].get(metric_name)
                         for r in per_pair if r["metrics"]["sibling"]
                         and r["metrics"]["sibling"].get(metric_name) is not None],
                        dtype=float)
        WSh = np.array([r["metrics"]["wt_shuf"].get(metric_name)
                         for r in per_pair if r["metrics"]["wt_shuf"]
                         and r["metrics"]["wt_shuf"].get(metric_name) is not None],
                        dtype=float)
        # Need same-length paired arrays
        rows_with_both = [r for r in per_pair
                          if r["metrics"]["sibling"] and r["metrics"]["wt_shuf"]
                          and r["metrics"]["sibling"].get(metric_name) is not None
                          and r["metrics"]["wt_shuf"].get(metric_name) is not None]
        Sib = np.array([r["metrics"]["sibling"][metric_name]
                         for r in rows_with_both], dtype=float)
        WSh = np.array([r["metrics"]["wt_shuf"][metric_name]
                         for r in rows_with_both], dtype=float)
        if len(Sib) < 2:
            return {"metric": metric_name, "n": len(Sib), "delta": None}
        from scipy.stats import wilcoxon
        try:
            if sign == 1:  # higher = more drop
                p = float(wilcoxon(Sib, WSh, alternative="greater").pvalue)
            else:
                p = float(wilcoxon(Sib, WSh, alternative="less").pvalue)
        except Exception:
            p = float("nan")
        ci = list(bootstrap_diff_ci_pp(Sib, WSh, seed=0))
        return {"metric": metric_name, "n": len(Sib),
                "sibling_mean": float(Sib.mean()), "wt_shuf_mean": float(WSh.mean()),
                "delta_sib_minus_wt": float(Sib.mean() - WSh.mean()),
                "ci_pp_x100": ci,
                "wilcoxon_p_sib_drops_more": p}

    t3 = {
        "kl":         _t3_metric("kl_to_baseline", sign=+1),
        "logit_diff": _t3_metric("logit_diff", sign=-1),
        "rank":       _t3_metric("rank", sign=+1),
    }
    save_json_atomic(out_dir / "t3_per_metric.json", t3)

    # ---- B5. Generation ----
    print(f"\n[B5] generation phase (Baseline + 3 ablations)")
    gen_t = time.time()

    # Sanity check: 20 prompts via forward_with_ablation_then_generate with empty fids vs greedy_decode
    print(f"[B5] sanity check: {SANITY_N} prompts, empty-fid vs greedy_decode")
    rng = np.random.default_rng(0)
    sanity_pids = list(rng.choice([p["id"] for p in detected],
                                    size=min(SANITY_N, len(detected)), replace=False))
    sanity_results = []
    for pid in sanity_pids:
        a_prompt = a_prompts[pid]
        g1 = greedy_decode(model, tokenizer, a_prompt, max_new=GEN_MAX_NEW)
        g2 = forward_with_ablation_then_generate(
            model, tokenizer, a_prompt, LAYER, sae,
            feature_ids=[], max_new=GEN_MAX_NEW,
        )
        sanity_results.append({
            "pair_id": pid, "match": g1 == g2,
            "greedy_decode": g1, "ablation_no_op": g2,
        })
    n_match = sum(1 for r in sanity_results if r["match"])
    print(f"[B5] sanity match: {n_match}/{len(sanity_results)}")
    save_json_atomic(out_dir / "generation_sanity_check.json", {
        "n": len(sanity_results), "n_match": n_match,
        "results": sanity_results,
    })

    # Generate per (P, D_i):
    # - Baseline gen: cached per A prompt
    # - Targeted gen: per (P, D_i)
    # - Shared-only gen: per (P, D_i), reuse baseline if shared empty
    # - Unique-only gen: per (P, D_i), reuse baseline if unique empty
    base_gen_cache = {}
    print(f"[B5] generating: 4 conditions × {len(per_pair)} (P, D_i) target slots")

    generations = []
    for r in tqdm(per_pair, desc="B5 generate"):
        pid = r["pair_id"]; i = r["target_idx"]
        a_prompt = a_prompts[pid]
        # Baseline (cache per A prompt)
        if pid not in base_gen_cache:
            base_gen_cache[pid] = greedy_decode(model, tokenizer, a_prompt, max_new=GEN_MAX_NEW)
        base_g = base_gen_cache[pid]

        # Targeted
        targ_g = forward_with_ablation_then_generate(
            model, tokenizer, a_prompt, LAYER, sae,
            feature_ids=targeted_feats[(pid, i)], max_new=GEN_MAX_NEW,
        ) if targeted_feats[(pid, i)] else base_g

        # Shared-only
        sh_g = forward_with_ablation_then_generate(
            model, tokenizer, a_prompt, LAYER, sae,
            feature_ids=shared_feats[(pid, i)], max_new=GEN_MAX_NEW,
        ) if shared_feats[(pid, i)] else base_g

        # Unique-only
        un_g = forward_with_ablation_then_generate(
            model, tokenizer, a_prompt, LAYER, sae,
            feature_ids=unique_feats[(pid, i)], max_new=GEN_MAX_NEW,
        ) if unique_feats[(pid, i)] else base_g

        generations.append({
            "pair_id": pid, "target_idx": i,
            "baseline": base_g, "targeted": targ_g,
            "shared_only": sh_g, "unique_only": un_g,
        })
    print(f"[B5] generation elapsed: {(time.time()-gen_t)/60:.1f} min")
    save_json_atomic(out_dir / "generations.json", generations)

    # ---- Classify each generation against (D_i, D_j, no-match) ----
    # Build classification per (P, D_i, condition):
    # 'D_i' if gen matches D_i's answer, 'D_j' if matches some D_j (j != i), else 'no-match'
    pid_to_pair = {p["id"]: p for p in detected}
    classifications = []
    for r in generations:
        pid = r["pair_id"]; i = r["target_idx"]
        pair = pid_to_pair[pid]
        cand_strs = [d["answer"] for d in pair["disambigs"]]
        cls_per_condition = {}
        for cond in ["baseline", "targeted", "shared_only", "unique_only"]:
            text = r[cond]
            match = find_best_match(text, cand_strs)
            if match.cand_idx < 0:
                slot = "no-match"
            elif match.cand_idx == i:
                slot = "D_i"
            else:
                slot = "D_j"
            cls_per_condition[cond] = slot
        classifications.append({
            "pair_id": pid, "target_idx": i,
            "slots": cls_per_condition,
        })
    save_json_atomic(out_dir / "generation_classifications.json", classifications)

    # Confusion matrices and flip rates
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
    save_json_atomic(out_dir / "confusion_matrices.json", confusion)

    # ---- Generation-flip decomposition ----
    # Treat "stays-on-D_i" as the "hit" analogue.
    keys_order = [(r["pair_id"], r["target_idx"]) for r in classifications]
    cls_lookup = {(r["pair_id"], r["target_idx"]): r["slots"] for r in classifications}
    # Hit = 1 iff slot == "D_i" for that condition
    def _hit_arr(cond):
        return np.array([1 if cls_lookup[k][cond] == "D_i" else 0 for k in keys_order],
                        dtype=float)
    Bg = _hit_arr("baseline"); Tg = _hit_arr("targeted")
    Sg = _hit_arr("shared_only"); Ug = _hit_arr("unique_only")
    td = (Tg - Bg).mean() * 100
    sd = (Sg - Bg).mean() * 100
    ud = (Ug - Bg).mean() * 100
    pct = lambda x: (x / td * 100) if abs(td) > 1e-12 else float("nan")

    def _mc_drop(b, c):
        return mcnemar_paired(c, b, alternative="greater")  # c kills more

    gen_decomp = {
        "n": len(keys_order),
        "table": {
            "Baseline":    {"hit_D_i": float(Bg.mean()), "delta_pp": 0.0, "pct_of_targeted": 0.0},
            "Targeted":    {"hit_D_i": float(Tg.mean()), "delta_pp": float(td),
                            "pct_of_targeted": 100.0,
                            "ci_pp": list(bootstrap_diff_ci_pp(Bg, Tg, seed=0)),
                            "mcnemar": _mc_drop(Bg, Tg),
                            "wilcoxon_p": wilcoxon_paired(Bg, Tg, alternative="greater")},
            "Shared-only": {"hit_D_i": float(Sg.mean()), "delta_pp": float(sd),
                            "pct_of_targeted": float(pct(sd)),
                            "ci_pp": list(bootstrap_diff_ci_pp(Bg, Sg, seed=0)),
                            "mcnemar": _mc_drop(Bg, Sg),
                            "wilcoxon_p": wilcoxon_paired(Bg, Sg, alternative="greater")},
            "Unique-only": {"hit_D_i": float(Ug.mean()), "delta_pp": float(ud),
                            "pct_of_targeted": float(pct(ud)),
                            "ci_pp": list(bootstrap_diff_ci_pp(Bg, Ug, seed=0)),
                            "mcnemar": _mc_drop(Bg, Ug),
                            "wilcoxon_p": wilcoxon_paired(Bg, Ug, alternative="greater")},
            "Sum (Sh+Un)": {"delta_pp": float(sd + ud),
                            "pct_of_targeted": float(pct(sd + ud))},
            "Residual":    {"delta_pp": float(td - (sd + ud)),
                            "pct_of_targeted": float(pct(td - (sd + ud)))},
        },
        "flip_rates_from_baseline": flip_rates,
    }
    save_json_atomic(out_dir / "generation_decomposition.json", gen_decomp)

    # Pretty-print summary
    print(f"\n=== Multi-metric decomposition (n={len(per_pair)}) ===")
    for label, d in [("KL → baseline", decomp_kl), ("Logit-diff", decomp_ld),
                      ("Rank", decomp_rk)]:
        t = d["table"]
        print(f"\n{label}  ({d['sign_convention']}):")
        for k in ["Baseline", "Targeted", "Shared-only", "Unique-only", "Sum (Sh+Un)", "Residual"]:
            row = t[k]
            v = row.get("value")
            v_s = f"{v:.4f}" if v is not None else "—"
            print(f"  {k:<14}  value={v_s:<10}  Δ={row['delta']:+8.4f}  "
                  f"({row['pct_of_targeted']:+6.1f}% of targeted)")
    print(f"\nGeneration-flip (n={gen_decomp['n']}):")
    for k in ["Baseline", "Targeted", "Shared-only", "Unique-only", "Sum (Sh+Un)", "Residual"]:
        row = gen_decomp["table"][k]
        h = row.get("hit_D_i", None)
        h_s = f"{h:.4f}" if h is not None else "—"
        print(f"  {k:<14}  hit_D_i={h_s:<8}  Δ={row['delta_pp']:+6.2f} pp  "
              f"({row['pct_of_targeted']:+6.1f}% of targeted)")
    print(f"\nT3 per metric (Sibling vs WT-shuf):")
    for name, t in t3.items():
        if t.get("delta") is None and t.get("delta_sib_minus_wt") is None:
            print(f"  {name:<12}  insufficient data")
        else:
            print(f"  {name:<12}  Sib={t['sibling_mean']:.4f}  WT={t['wt_shuf_mean']:.4f}  "
                  f"Δ_sib-wt={t['delta_sib_minus_wt']:+.4f}  Wlx p={t['wilcoxon_p_sib_drops_more']:.3g}")
    return decomp_kl, decomp_ld, decomp_rk, gen_decomp, t3


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    P13_DIR.mkdir(parents=True, exist_ok=True)
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
    a_prompts = {p["id"]: build_prompt(tokenizer, p["A_question"]) for p in detected}
    published_specific = json.load(open(ARTIFACTS_DIR / "specific_features.json"))
    results_main = json.load(open(ARTIFACTS_DIR / "results_main.json"))

    print(f"[setup] {len(detected)} detected pairs, "
          f"{sum(len(p['disambigs']) for p in detected)} self-pairs")

    strat_a = run_a(detected, published_specific, results_main)
    decomp_kl, decomp_ld, decomp_rk, gen_decomp, t3 = run_b(
        tokenizer, model, sae, enc_dict, detected, a_prompts,
        published_specific, results_main,
    )

    save_json_atomic(P13_DIR / "run_meta.json", {
        "model": MODEL_ID, "layer": LAYER, "sae_meta": sae_meta,
        "n_detected_pairs": len(detected),
        "save_top_k": SAVE_TOP_K, "kl_support_k": KL_SUPPORT_K,
        "rank_censor": RANK_CENSOR,
        "elapsed_seconds": round(time.time() - t_total, 2),
    })
    print(f"\n=== Stage 1.3 complete ===")
    print(f"  total elapsed: {(time.time()-t_total)/60:.1f} min")
    print(f"  -> {P13_DIR}")


if __name__ == "__main__":
    main()
