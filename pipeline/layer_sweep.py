"""Layer sweep on Gemma-2-9B-IT.

Replicates the headline tests at L26, L30, L34, L40 to test robustness
to the choice of reference layer, and characterize how the cluster-shared
and answer-unique components vary with depth.

Per layer L ∈ {26, 30, 34, 40}:
  1. Recapture last-prompt-token residuals at L for the 448 A + 1103 D
     prompts and the 800 WikiText paragraphs.
  2. Load Gemma Scope width-16k SAE for L; record SHA-256.
  3. SAE-encode all residuals; save to artifacts/layer_sweep/L{L}/.
  4. Re-pick Targeted/Sibling features via published score formula
     (a_weight=0.5, top_k=10) on layer-L encodings.
  5. Re-pick WikiText-shuffled top-10 by raw activation per paragraph.
  6. Run six-condition pipeline on n=1103 self-pairs (shuffle draws and
     random feature IDs reused from the published L37 artifacts so that
     cross-layer comparisons control for sampling).
  7. Run unique/shared decomposition.
  8. Run per-feature equivalence test using the
     published L37 0.80 polysemy partition transferred AS-IS.

Aggregate output:
  artifacts/layer_sweep/layer_sweep_summary.json
  artifacts/layer_sweep/figure_t3_vs_layer.png
  artifacts/layer_sweep/figure_decomp_vs_layer.png

Reuses src/ unmodified; abort on SHA drift only at L37 (other layers
have no pinned SHA).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc
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
    ARTIFACTS_DIR, DTYPE_STR, MODEL_ID, N_SHUFFLES_PER_PAIR,
    POSITION_NONZERO_FLOOR, POSITION_PCT_THRESHOLD, RANDOM_CONTROLS,
    SAE_CHECKPOINT_SHA256, SCORE_A_WEIGHT, SEED, SHUFFLE_SEED,
    TOP_K_FEATURES, TOP_LOGITS_K, WIKITEXT_MAX_TOKENS,
)
from src.features import score_specific_features
from src.hooks import (
    baseline_top_logits, capture_all_position_residuals,
    capture_last_token_residual, forward_with_ablation, hit_at_k,
)
from src.io_utils import save_json_atomic, save_npz_atomic
from src.model import load_model
from src.prompts import build_prompt
from src.sae import load_sae
from src.wikitext import load_paragraphs as load_wikitext_paragraphs


P21_DIR = ARTIFACTS_DIR / "layer_sweep"
LAYERS_TO_SWEEP = [26, 30, 34, 40]
PUBLISHED_LAYER = 37
PUBLISHED_TOP_K = 10


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


def run_layer(layer, model, tokenizer, detected, a_prompts, paragraphs,
              shuffle_draws_ambig, shuffle_draws_wt,
              random_feature_ids_per_pair, suspect_080):
    """Run one layer's full sweep. Returns the layer's summary dict."""
    out_dir = P21_DIR / f"L{layer}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== Layer L={layer} ==========")
    t_layer = time.time()

    # ---- Load SAE ----
    print(f"[L{layer}] loading SAE...")
    sae, sae_meta = load_sae(layer=layer)
    print(f"[L{layer}] SAE sha256: {sae_meta['sha256'][:16]}...")
    if layer == PUBLISHED_LAYER:
        if sae_meta["sha256"] != SAE_CHECKPOINT_SHA256:
            raise RuntimeError(f"L37 SAE sha mismatch: {sae_meta['sha256']}")
        print(f"[L{layer}] SAE sha matches paper-pinned checkpoint")

    # ---- Capture A + D residuals at this layer ----
    print(f"[L{layer}] capturing A + D residuals "
          f"({sum(len(p['disambigs']) for p in detected) + len(detected)} prompts)...")
    residuals = {}
    for p in tqdm(detected, desc=f"L{layer} A/D residuals"):
        residuals[f"A__{p['id']}"] = capture_last_token_residual(
            model, tokenizer, a_prompts[p["id"]], layer=layer,
        )
        for di, d in enumerate(p["disambigs"]):
            d_prompt = build_prompt(tokenizer, d["question"])
            residuals[f"D__{p['id']}__{di}"] = capture_last_token_residual(
                model, tokenizer, d_prompt, layer=layer,
            )
    save_npz_atomic(out_dir / f"residuals_L{layer}.npz", **residuals)

    # ---- Capture WikiText residuals at this layer ----
    print(f"[L{layer}] capturing WikiText last-token residuals (800 paragraphs)...")
    wt_lasttok_z = np.zeros((len(paragraphs), sae.d_sae), dtype=np.float32)
    wt_residuals = {}
    for i, p_text in enumerate(tqdm(paragraphs, desc=f"L{layer} WT residuals")):
        try:
            _, res = capture_all_position_residuals(
                model, tokenizer, p_text, layer=layer, max_len=WIKITEXT_MAX_TOKENS,
            )
            last_res = res[-1].to(dtype=torch.float32, device="cpu").numpy()
            wt_residuals[f"WT__{i}"] = last_res
            z_last = sae.encode(res[-1:].to(dtype=sae.W_enc.dtype)).squeeze(0)
            wt_lasttok_z[i] = z_last.float().cpu().numpy()
            del res, z_last
        except Exception as e:
            print(f"[L{layer}] paragraph {i} failed: {type(e).__name__}: {e}")
        if i % 50 == 0:
            torch.cuda.empty_cache(); gc.collect()
    save_npz_atomic(out_dir / f"wikitext_residuals_L{layer}.npz", **wt_residuals)

    # ---- SAE-encode A/D residuals ----
    print(f"[L{layer}] SAE-encoding A/D residuals...")
    encodings = {}
    for k, r in residuals.items():
        x = torch.from_numpy(r).to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        z = sae.encode(x.unsqueeze(0)).squeeze(0)
        encodings[k] = z.float().cpu().numpy()
    save_npz_atomic(out_dir / f"sae_encodings_L{layer}.npz", **encodings)

    # ---- Pick Targeted features per (P, D_i) ----
    print(f"[L{layer}] picking Targeted top-10 features...")
    enc_torch = {k: torch.from_numpy(v).to(device=sae.W_enc.device,
                                            dtype=sae.W_enc.dtype)
                 for k, v in encodings.items()}
    specific_features = {}
    for p in detected:
        z_A = enc_torch[f"A__{p['id']}"]
        z_D_list = [enc_torch[f"D__{p['id']}__{di}"]
                    for di in range(len(p["disambigs"]))]
        topk_per_d = score_specific_features(
            z_A, z_D_list, top_k=PUBLISHED_TOP_K, a_weight=SCORE_A_WEIGHT,
        )
        for di, feats in enumerate(topk_per_d):
            specific_features[(p["id"], di)] = feats
    save_json_atomic(
        out_dir / "specific_features.json",
        [{"pair_id": pid, "disambig_idx": di, "features": feats}
         for (pid, di), feats in specific_features.items()],
    )

    # ---- Pick WikiText top-10 per paragraph (raw activation) ----
    wt_top10 = []
    for i in range(len(paragraphs)):
        z_p = wt_lasttok_z[i]
        positives = np.where(z_p > 0)[0]
        if len(positives) == 0:
            kept = []
        else:
            sorted_idx = positives[np.argsort(-z_p[positives])]
            kept = [int(f) for f in sorted_idx[:PUBLISHED_TOP_K]]
        wt_top10.append({"paragraph_idx": i, "top10_feature_ids": kept})

    # ---- Run six-condition pipeline ----
    print(f"[L{layer}] running six-condition pipeline...")
    self_rows = []; cross_rows = []; random_rows = []
    ambig_shuf_rows = []; wt_shuf_rows = []
    pid_to_pair = {p["id"]: p for p in detected}
    a_prompts_local = a_prompts

    # Index pre-computed shuffle draws by (pair_id, ab_i, draw_idx)
    ambig_by_draw = {(d["pair_id"], d["target_idx"], d["draw_idx"]):
                      (d["shuffle_pair_id"], d["shuffle_disambig_idx"])
                      for d in shuffle_draws_ambig}
    wt_by_draw = {(d["pair_id"], d["target_idx"], d["draw_idx"]):
                   d["wikitext_paragraph_idx"]
                   for d in shuffle_draws_wt}
    rand_by_pair = {(r["pair_id"], r["ablate_idx"], r["control_idx"]):
                     r["feature_ids"]
                     for r in random_feature_ids_per_pair}

    for p in tqdm(detected, desc=f"L{layer} 6-cond"):
        a_prompt = a_prompts_local[p["id"]]
        # Baseline (one fwd, scored against every disambig)
        base = baseline_top_logits(model, tokenizer, a_prompt, k=TOP_LOGITS_K)
        base_top = base["top_ids"]
        base_hits = []
        for d in p["disambigs"]:
            base_hits.append(hit_at_k(base_top, set(d["first_token_variants"]), 1))

        for ab_i in range(len(p["disambigs"])):
            feats_self = specific_features.get((p["id"], ab_i), [])
            if not feats_self:
                continue
            ab = forward_with_ablation(
                model, tokenizer, a_prompt, layer, sae,
                feature_ids=feats_self, k=TOP_LOGITS_K,
            )
            for tg_j in range(len(p["disambigs"])):
                target = set(p["disambigs"][tg_j]["first_token_variants"])
                row = {
                    "pair_id": p["id"], "ablate_idx": ab_i, "target_idx": tg_j,
                    "is_self": ab_i == tg_j,
                    "n_features": len(feats_self),
                    "feature_ids": list(feats_self),
                    "base_hit1": base_hits[tg_j],
                    "ablate_hit1": hit_at_k(ab["top_ids"], target, 1),
                }
                if ab_i == tg_j:
                    self_rows.append(row)
                else:
                    cross_rows.append(row)

            target_self = set(p["disambigs"][ab_i]["first_token_variants"])

            # Random (3 draws, reuse feature IDs from L37)
            for r_idx in range(RANDOM_CONTROLS):
                rand_feats = rand_by_pair.get((p["id"], ab_i, r_idx))
                if rand_feats is None:
                    continue
                rc = forward_with_ablation(
                    model, tokenizer, a_prompt, layer, sae,
                    feature_ids=rand_feats, k=TOP_LOGITS_K,
                )
                random_rows.append({
                    "pair_id": p["id"], "ablate_idx": ab_i, "control_idx": r_idx,
                    "n_features": len(rand_feats),
                    "base_hit1": base_hits[ab_i],
                    "random_hit1": hit_at_k(rc["top_ids"], target_self, 1),
                })

            # AmbigQA-shuffle (3 draws, reuse (P', k) tuples from L37)
            for d_idx in range(N_SHUFFLES_PER_PAIR):
                tup = ambig_by_draw.get((p["id"], ab_i, d_idx))
                if tup is None:
                    continue
                p_prime, k_idx = tup
                feats_sh = specific_features.get((p_prime, k_idx), [])
                if not feats_sh:
                    continue
                sh = forward_with_ablation(
                    model, tokenizer, a_prompt, layer, sae,
                    feature_ids=feats_sh, k=TOP_LOGITS_K,
                )
                ambig_shuf_rows.append({
                    "pair_id": p["id"], "target_idx": ab_i, "draw_idx": d_idx,
                    "shuffle_pair_id": p_prime, "shuffle_disambig_idx": int(k_idx),
                    "n_features": len(feats_sh),
                    "base_hit1": base_hits[ab_i],
                    "shuffled_hit1": hit_at_k(sh["top_ids"], target_self, 1),
                })

            # WikiText-shuffle (3 draws, reuse paragraph_idx from L37)
            for d_idx in range(N_SHUFFLES_PER_PAIR):
                para_idx = wt_by_draw.get((p["id"], ab_i, d_idx))
                if para_idx is None:
                    continue
                feats_wt = wt_top10[para_idx]["top10_feature_ids"]
                wt = forward_with_ablation(
                    model, tokenizer, a_prompt, layer, sae,
                    feature_ids=feats_wt, k=TOP_LOGITS_K,
                )
                wt_shuf_rows.append({
                    "pair_id": p["id"], "target_idx": ab_i, "draw_idx": d_idx,
                    "wikitext_paragraph_idx": para_idx,
                    "n_features": len(feats_wt),
                    "base_hit1": base_hits[ab_i],
                    "wt_shuffled_hit1": hit_at_k(wt["top_ids"], target_self, 1),
                })
        torch.cuda.empty_cache()

    # Six-condition summary
    def _mean(rows, k):
        rows = list(rows)
        return float(sum(r[k] for r in rows) / len(rows)) if rows else 0.0

    six_cond_summary = {
        "n_self": len(self_rows),
        "n_cross": len(cross_rows),
        "n_random": len(random_rows),
        "n_ambig_shuf": len(ambig_shuf_rows),
        "n_wt_shuf": len(wt_shuf_rows),
        "self_base_hit1":   _mean(self_rows, "base_hit1"),
        "self_ablate_hit1": _mean(self_rows, "ablate_hit1"),
        "cross_base_hit1":  _mean(cross_rows, "base_hit1"),
        "cross_ablate_hit1": _mean(cross_rows, "ablate_hit1"),
        "random_ablate_hit1": _mean(random_rows, "random_hit1"),
        "ambig_shuf_ablate_hit1": _mean(ambig_shuf_rows, "shuffled_hit1"),
        "wt_shuf_ablate_hit1": _mean(wt_shuf_rows, "wt_shuffled_hit1"),
    }
    save_json_atomic(out_dir / "results_main.json", {
        "summary": six_cond_summary,
        "self_rows": self_rows,
        "cross_rows": cross_rows,
        "random_rows": random_rows,
        "ambigqa_shuffled_rows": ambig_shuf_rows,
        "wikitext_shuffled_rows": wt_shuf_rows,
    })

    # T3 + per-pair tests
    sibling = per_pair_means(
        cross_rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["ablate_hit1"],
    )
    wt_per_pair = per_pair_means(
        wt_shuf_rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["wt_shuffled_hit1"],
    )
    a_shuf_per_pair = per_pair_means(
        ambig_shuf_rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["shuffled_hit1"],
    )
    rand_per_pair = per_pair_means(
        random_rows,
        key_fn=lambda r: (r["pair_id"], r["ablate_idx"]),
        value_fn=lambda r: r["random_hit1"],
    )
    base_per_pair = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                     for r in self_rows}
    targ_per_pair = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                     for r in self_rows}

    keys_all = sorted(k for k in base_per_pair
                       if k in sibling and k in wt_per_pair
                       and k in a_shuf_per_pair and k in rand_per_pair)
    n = len(keys_all)
    B = np.array([base_per_pair[k] for k in keys_all])
    T = np.array([targ_per_pair[k] for k in keys_all])
    Sib = np.array([sibling[k] for k in keys_all])
    ASh = np.array([a_shuf_per_pair[k] for k in keys_all])
    WSh = np.array([wt_per_pair[k] for k in keys_all])
    Rnd = np.array([rand_per_pair[k] for k in keys_all])

    # T1-T4
    tests_t = {
        "T1_targeted_vs_random": _drop_vs_base_stats(T, Rnd),
        "T2_sibling_vs_ashuf":   _t3_stats(Sib, ASh),  # Sib vs ASh, "Sib drops more"
        "T3_sibling_vs_wtshuf":  _t3_stats(Sib, WSh),
        "T4_ashuf_vs_wtshuf":    _t3_stats(ASh, WSh),
    }
    headline_table = {
        "Baseline":          float(B.mean()),
        "Targeted":          float(T.mean()),
        "Sibling":           float(Sib.mean()),
        "ShuffledAmbigQA":   float(ASh.mean()),
        "WikiTextShuffled":  float(WSh.mean()),
        "Random":            float(Rnd.mean()),
    }
    print(f"[L{layer}] headline rates:")
    for k, v in headline_table.items():
        print(f"  {k:<18}  {v:.4f}")
    if tests_t["T3_sibling_vs_wtshuf"]:
        t3 = tests_t["T3_sibling_vs_wtshuf"]
        print(f"  T3: Δ={t3['delta_pp']:+.2f} pp, Wlx p={t3['wilcoxon_p']:.3g}")

    # ---- Unique/shared decomposition ----
    print(f"[L{layer}] unique/shared decomposition...")
    pub_set = {(pid, di): set(feats) for (pid, di), feats in specific_features.items()}
    sh_un = {}
    for p in detected:
        K = len(p["disambigs"])
        for i in range(K):
            ti = pub_set.get((p["id"], i), set())
            if not ti:
                continue
            sib_union = set()
            for j in range(K):
                if j == i:
                    continue
                sib_union |= pub_set.get((p["id"], j), set())
            shared = ti & sib_union
            unique = ti - shared
            sh_un[(p["id"], i)] = {"shared": shared, "unique": unique, "targeted": ti}

    # Run shared_only and unique_only forwards (skip empty)
    base_hits_lookup = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                        for r in self_rows}
    targ_hits_lookup = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                        for r in self_rows}
    sh_un_rows = []
    for p in tqdm(detected, desc=f"L{layer} sh+un"):
        a_prompt = a_prompts_local[p["id"]]
        for i in range(len(p["disambigs"])):
            key = (p["id"], i)
            if key not in sh_un or key not in base_hits_lookup:
                continue
            target = set(p["disambigs"][i]["first_token_variants"])
            base = base_hits_lookup[key]
            if sh_un[key]["shared"]:
                ab = forward_with_ablation(
                    model, tokenizer, a_prompt, layer, sae,
                    feature_ids=list(sh_un[key]["shared"]), k=TOP_LOGITS_K,
                )
                shared_hit = hit_at_k(ab["top_ids"], target, 1)
            else:
                shared_hit = base
            if sh_un[key]["unique"]:
                ab = forward_with_ablation(
                    model, tokenizer, a_prompt, layer, sae,
                    feature_ids=list(sh_un[key]["unique"]), k=TOP_LOGITS_K,
                )
                unique_hit = hit_at_k(ab["top_ids"], target, 1)
            else:
                unique_hit = base
            sh_un_rows.append({
                "pair_id": p["id"], "target_idx": i,
                "n_targeted": len(sh_un[key]["targeted"]),
                "n_shared": len(sh_un[key]["shared"]),
                "n_unique": len(sh_un[key]["unique"]),
                "base_hit1": base,
                "targeted_hit1": targ_hits_lookup[key],
                "shared_only_hit1": shared_hit,
                "unique_only_hit1": unique_hit,
            })
        torch.cuda.empty_cache()

    keys_sh = [(r["pair_id"], r["target_idx"]) for r in sh_un_rows]
    by_key = {(r["pair_id"], r["target_idx"]): r for r in sh_un_rows}
    Bsh = np.array([by_key[k]["base_hit1"] for k in keys_sh], dtype=float)
    Tsh = np.array([by_key[k]["targeted_hit1"] for k in keys_sh], dtype=float)
    Shsh = np.array([by_key[k]["shared_only_hit1"] for k in keys_sh], dtype=float)
    Unsh = np.array([by_key[k]["unique_only_hit1"] for k in keys_sh], dtype=float)
    decomp = {
        "n": len(keys_sh),
        "table": {
            "Baseline":    {"hit1": float(Bsh.mean()), "delta_pp": 0.0},
            "Targeted":    {"hit1": float(Tsh.mean()), "delta_pp": float((Tsh.mean() - Bsh.mean()) * 100)},
            "Shared-only": {"hit1": float(Shsh.mean()), "delta_pp": float((Shsh.mean() - Bsh.mean()) * 100)},
            "Unique-only": {"hit1": float(Unsh.mean()), "delta_pp": float((Unsh.mean() - Bsh.mean()) * 100)},
        },
        "tests": {
            "shared_vs_baseline": _drop_vs_base_stats(Shsh, Bsh),
            "unique_vs_baseline": _drop_vs_base_stats(Unsh, Bsh),
            "targeted_vs_unique": _drop_vs_base_stats(Tsh, Unsh),
        },
    }
    save_json_atomic(out_dir / "unique_shared_decomp.json", {
        "decomposition": decomp, "rows": sh_un_rows,
    })
    print(f"[L{layer}] decomp: shared Δ={decomp['table']['Shared-only']['delta_pp']:+.2f}, "
          f"unique Δ={decomp['table']['Unique-only']['delta_pp']:+.2f}")

    # ---- Per-feature equivalence (uc_top vs sc_top single-feature ablation) ----
    print(f"[L{layer}] per-feature equivalence test...")
    sf_rows = []
    for p in tqdm(detected, desc=f"L{layer} sf-eq"):
        a_prompt = a_prompts_local[p["id"]]
        for i in range(len(p["disambigs"])):
            key = (p["id"], i)
            if key not in sh_un or key not in base_hits_lookup:
                continue
            shared_set = sh_un[key]["shared"]
            unique_set = sh_un[key]["unique"]
            uc = unique_set - suspect_080
            sc = shared_set - suspect_080
            if not uc or not sc:
                continue
            z_Di = enc_torch[f"D__{p['id']}__{i}"]
            uc_list = list(uc); sc_list = list(sc)
            uc_zs = [float(z_Di[fid].item()) for fid in uc_list]
            sc_zs = [float(z_Di[fid].item()) for fid in sc_list]
            uc_top_fid = uc_list[int(np.argmax(uc_zs))]
            sc_top_fid = sc_list[int(np.argmax(sc_zs))]
            target = set(p["disambigs"][i]["first_token_variants"])
            base = base_hits_lookup[key]
            ab_uc = forward_with_ablation(
                model, tokenizer, a_prompt, layer, sae,
                feature_ids=[uc_top_fid], k=TOP_LOGITS_K,
            )
            ab_sc = forward_with_ablation(
                model, tokenizer, a_prompt, layer, sae,
                feature_ids=[sc_top_fid], k=TOP_LOGITS_K,
            )
            sf_rows.append({
                "pair_id": p["id"], "target_idx": i,
                "baseline_hit1": base,
                "uc_top_fid": int(uc_top_fid),
                "uc_top_hit1": hit_at_k(ab_uc["top_ids"], target, 1),
                "sc_top_fid": int(sc_top_fid),
                "sc_top_hit1": hit_at_k(ab_sc["top_ids"], target, 1),
            })
        torch.cuda.empty_cache()

    if sf_rows:
        n_sf = len(sf_rows)
        base_arr = np.array([r["baseline_hit1"] for r in sf_rows], dtype=float)
        uc_arr = np.array([r["uc_top_hit1"] for r in sf_rows], dtype=float)
        sc_arr = np.array([r["sc_top_hit1"] for r in sf_rows], dtype=float)
        uc_drop = float((base_arr.mean() - uc_arr.mean()) * 100)
        sc_drop = float((base_arr.mean() - sc_arr.mean()) * 100)
        mc_uvs = mcnemar_paired(uc_arr, sc_arr, alternative="two-sided")
        wx_uvs = wilcoxon_paired(uc_arr, sc_arr, alternative="two-sided")
        mc_uc = mcnemar_paired(uc_arr, base_arr, alternative="greater")
        wx_uc = wilcoxon_paired(base_arr, uc_arr, alternative="greater")
        mc_sc = mcnemar_paired(sc_arr, base_arr, alternative="greater")
        wx_sc = wilcoxon_paired(base_arr, sc_arr, alternative="greater")
        sf_summary = {
            "n_eligible": n_sf,
            "uc_top_drop_pp": uc_drop,
            "sc_top_drop_pp": sc_drop,
            "uc_minus_sc_drop_pp": uc_drop - sc_drop,
            "uc_vs_baseline_mcnemar_p": mc_uc["p"],
            "sc_vs_baseline_mcnemar_p": mc_sc["p"],
            "uc_vs_baseline_wilcoxon_p": wx_uc,
            "sc_vs_baseline_wilcoxon_p": wx_sc,
            "uc_vs_sc_two_sided_mcnemar_p": mc_uvs["p"],
            "uc_vs_sc_two_sided_wilcoxon_p": wx_uvs,
        }
    else:
        sf_summary = {"n_eligible": 0}
    save_json_atomic(out_dir / "per_feature_equivalence.json",
                     {"summary": sf_summary, "rows": sf_rows})
    if sf_summary["n_eligible"] >= 2:
        print(f"[L{layer}] sf-eq: uc_top Δ={sf_summary['uc_top_drop_pp']:+.2f}, "
              f"sc_top Δ={sf_summary['sc_top_drop_pp']:+.2f}, "
              f"head-to-head Wlx p={sf_summary['uc_vs_sc_two_sided_wilcoxon_p']:.3g}")

    # ---- Save run_meta ----
    elapsed = time.time() - t_layer
    save_json_atomic(out_dir / "run_meta.json", {
        "layer": layer,
        "model": MODEL_ID,
        "sae_meta": sae_meta,
        "n_detected_pairs": len(detected),
        "n_self_pairs": six_cond_summary["n_self"],
        "elapsed_seconds": round(elapsed, 2),
    })
    # Free disk: delete heavy residual NPZs (we have SAE encodings).
    # Residuals are recoverable by re-running capture (~5 min).
    for npz_name in [f"residuals_L{layer}.npz", f"wikitext_residuals_L{layer}.npz"]:
        try:
            (out_dir / npz_name).unlink(missing_ok=True)
        except Exception:
            pass
    # Free disk: delete this layer's SAE blob from HF cache (we have the
    # encodings stored). Disk space is limited.
    try:
        local_path = sae_meta.get("local_path")
        if local_path and layer != PUBLISHED_LAYER:
            real = Path(local_path).resolve()
            if real.exists():
                real.unlink()
                print(f"[L{layer}] deleted SAE blob from HF cache to free space")
    except Exception as e:
        print(f"[L{layer}] sae blob cleanup failed: {e}")
    print(f"[L{layer}] elapsed: {elapsed/60:.1f} min")

    return {
        "layer": layer,
        "sae_sha256": sae_meta["sha256"],
        "n_self_pairs": six_cond_summary["n_self"],
        "headline_table": headline_table,
        "tests": tests_t,
        "decomposition": decomp,
        "single_feature_equivalence": sf_summary,
    }


def main() -> None:
    P21_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # ---- Load model ----
    print(f"[setup] loading {MODEL_ID}...")
    tokenizer, model = load_model()

    # ---- Load published artifacts ----
    detected = json.load(open(ARTIFACTS_DIR / "detected_pairs.json"))
    pid_to_pair = {p["id"]: p for p in detected}
    a_prompts = {p["id"]: build_prompt(tokenizer, p["A_question"]) for p in detected}
    print(f"[setup] {len(detected)} detected pairs, "
          f"{sum(len(p['disambigs']) for p in detected)} self-pairs")

    print(f"[setup] reloading WikiText-2 paragraphs...")
    paragraphs = load_wikitext_paragraphs(tokenizer)
    print(f"[setup] {len(paragraphs)} paragraphs reloaded")

    # ---- Load published shuffle draws + random feature IDs ----
    shuffle_draws_ambig = json.load(open(ARTIFACTS_DIR / "shuffle_draws_ambigqa.json"))
    shuffle_draws_wt = json.load(open(ARTIFACTS_DIR / "shuffle_draws_wikitext.json"))
    pub_results = json.load(open(ARTIFACTS_DIR / "results_main.json"))
    random_rows = pub_results["random_rows"]
    print(f"[setup] reusing {len(shuffle_draws_ambig)} ambig draws, "
          f"{len(shuffle_draws_wt)} WT draws, {len(random_rows)} random feature sets")

    # ---- Load L37 polysemy partition (transferred AS-IS for sf-eq test) ----
    pos_mode = json.load(open(ARTIFACTS_DIR / "wikitext_position_mode.json"))
    suspect_080 = {int(fid) for fid, st in pos_mode["per_feature"].items()
                   if st["n_nonzero"] >= POSITION_NONZERO_FLOOR
                   and st["pct_at_pos0"] >= POSITION_PCT_THRESHOLD}
    print(f"[setup] L37 0.80 polysemy partition: {len(suspect_080)} suspect features "
          f"(transferred as-is across layers for the sf-eq test)")

    # ---- Sweep layers (skip already-completed layers) ----
    layer_summaries = []
    for layer in LAYERS_TO_SWEEP:
        meta_path = P21_DIR / f"L{layer}" / "run_meta.json"
        if meta_path.exists():
            print(f"[L{layer}] already complete — loading run_meta.json")
            try:
                # Reconstruct summary from saved artifacts
                rm = json.load(open(meta_path))
                rmain = json.load(open(P21_DIR / f"L{layer}" / "results_main.json"))
                ushd = json.load(open(P21_DIR / f"L{layer}"
                                        / "unique_shared_decomp.json"))
                pfeq = json.load(open(P21_DIR / f"L{layer}"
                                        / "per_feature_equivalence.json"))
                # Recompute T3 from saved rows
                sib = per_pair_means(
                    rmain["cross_rows"],
                    key_fn=lambda r: (r["pair_id"], r["target_idx"]),
                    value_fn=lambda r: r["ablate_hit1"],
                )
                wt = per_pair_means(
                    rmain["wikitext_shuffled_rows"],
                    key_fn=lambda r: (r["pair_id"], r["target_idx"]),
                    value_fn=lambda r: r["wt_shuffled_hit1"],
                )
                ksx = sorted(k for k in sib if k in wt)
                Sib_a = np.array([sib[k] for k in ksx]); WSh_a = np.array([wt[k] for k in ksx])
                t3 = _t3_stats(Sib_a, WSh_a)
                hl = rmain["summary"]
                summary = {
                    "layer": layer,
                    "sae_sha256": rm["sae_meta"]["sha256"],
                    "n_self_pairs": hl["n_self"],
                    "headline_table": {
                        "Baseline":          hl["self_base_hit1"],
                        "Targeted":          hl["self_ablate_hit1"],
                        "Sibling":           hl["cross_ablate_hit1"],
                        "ShuffledAmbigQA":   hl["ambig_shuf_ablate_hit1"],
                        "WikiTextShuffled":  hl["wt_shuf_ablate_hit1"],
                        "Random":            hl["random_ablate_hit1"],
                    },
                    "tests": {"T3_sibling_vs_wtshuf": t3},
                    "decomposition": ushd["decomposition"],
                    "single_feature_equivalence": pfeq["summary"],
                }
                layer_summaries.append(summary)
                continue
            except Exception as e:
                print(f"[L{layer}] failed to load existing artifacts: {e}; rerunning")
        try:
            summary = run_layer(
                layer, model, tokenizer, detected, a_prompts, paragraphs,
                shuffle_draws_ambig, shuffle_draws_wt, random_rows, suspect_080,
            )
            layer_summaries.append(summary)
        except Exception as e:
            print(f"[L{layer}] FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # ---- Aggregate summary (5 layers including L37 from published) ----
    # Pull L37 numbers from published artifacts
    pub_main = pub_results["summary"]
    pub_self_rows = pub_results["self_rows"]
    pub_cross_rows = pub_results["cross_rows"]
    pub_wt_rows = pub_results["wikitext_shuffled_rows"]
    pub_p12pp_rows = json.load(open(ARTIFACTS_DIR / "unique_vs_shared" / "a"
                                     / "ablation_rows.json"))
    pub_sf = json.load(open(ARTIFACTS_DIR / "per_feature_equivalence" / "c"
                             / "single_feature_summary.json"))

    # Compute L37 numbers for the summary
    pub_sib = per_pair_means(
        pub_cross_rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["ablate_hit1"],
    )
    pub_wt = per_pair_means(
        pub_wt_rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["wt_shuffled_hit1"],
    )
    keys_pub = sorted(k for k in pub_sib if k in pub_wt)
    Sib_pub = np.array([pub_sib[k] for k in keys_pub])
    WSh_pub = np.array([pub_wt[k] for k in keys_pub])
    pub_t3 = _t3_stats(Sib_pub, WSh_pub)

    # Pull unique/shared decomposition (L37)
    pub_decomp = json.load(open(ARTIFACTS_DIR / "unique_vs_shared" / "a"
                                 / "decomposition_table.json"))
    l37_summary = {
        "layer": 37,
        "sae_sha256": SAE_CHECKPOINT_SHA256,
        "n_self_pairs": pub_main["n_self"],
        "headline_table": {
            "Baseline":          pub_main["self_base_hit1"],
            "Targeted":          pub_main["self_ablate_hit1"],
            "Sibling":           pub_main["cross_ablate_hit1"],
            "ShuffledAmbigQA":   pub_main["ambigqa_shuffled_ablate"],
            "WikiTextShuffled":  pub_main["wikitext_shuffled_ablate"],
            "Random":            pub_main["random_ablate_hit1"],
        },
        "tests": {
            "T3_sibling_vs_wtshuf": pub_t3,
        },
        "decomposition": {
            "n": pub_decomp["n"],
            "table": {
                "Shared-only": {"delta_pp": pub_decomp["table"]["Shared-only"]["delta_pp"]},
                "Unique-only": {"delta_pp": pub_decomp["table"]["Unique-only"]["delta_pp"]},
            },
            "tests": {
                "shared_vs_baseline": {
                    "wilcoxon_p": pub_decomp["tests"]["shared_vs_baseline"]["wilcoxon_p"],
                },
                "unique_vs_baseline": {
                    "wilcoxon_p": pub_decomp["tests"]["unique_vs_baseline"]["wilcoxon_p"],
                },
            },
        },
        "single_feature_equivalence": {
            "n_eligible": pub_sf["n_eligible"],
            "uc_top_drop_pp": pub_sf["uc_top_drop_pp"],
            "sc_top_drop_pp": pub_sf["sc_top_drop_pp"],
            "uc_minus_sc_drop_pp": pub_sf["uc_top_drop_minus_sc_top_drop_pp"],
            "uc_vs_sc_two_sided_wilcoxon_p": pub_sf["uc_vs_sc_two_sided"]["wilcoxon_p"],
        },
    }

    # Combine: order by layer
    all_summaries = sorted(layer_summaries + [l37_summary],
                            key=lambda s: s["layer"])
    save_json_atomic(P21_DIR / "layer_sweep_summary.json", {
        "model": MODEL_ID,
        "layers": [s["layer"] for s in all_summaries],
        "summaries": all_summaries,
        "elapsed_seconds": round(time.time() - t_total, 2),
    })

    # ---- Figures ----
    print(f"\n[fig] generating layer-vs-T3 and layer-vs-decomp figures")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers_list = [s["layer"] for s in all_summaries]
    t3_d = [s["tests"]["T3_sibling_vs_wtshuf"]["delta_pp"]
            if s["tests"].get("T3_sibling_vs_wtshuf") else 0.0
            for s in all_summaries]
    t3_lo = [s["tests"]["T3_sibling_vs_wtshuf"]["ci_pp"][0]
             if s["tests"].get("T3_sibling_vs_wtshuf") else 0.0
             for s in all_summaries]
    t3_hi = [s["tests"]["T3_sibling_vs_wtshuf"]["ci_pp"][1]
             if s["tests"].get("T3_sibling_vs_wtshuf") else 0.0
             for s in all_summaries]
    yerr = [[d - lo for d, lo in zip(t3_d, t3_lo)],
            [hi - d for hi, d in zip(t3_hi, t3_d)]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(range(len(layers_list)), t3_d, color="#2c7fb8",
                   edgecolor="white", width=0.6)
    ax.errorbar(range(len(layers_list)), t3_d, yerr=yerr, fmt="none",
                ecolor="#222", capsize=4, elinewidth=0.9)
    for i, (l, d) in enumerate(zip(layers_list, t3_d)):
        marker = " *" if l == 37 else ""
        ax.text(i, d + 0.3, f"{d:+.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(layers_list)))
    ax.set_xticklabels([f"L{l}{'*' if l == 37 else ''}" for l in layers_list])
    ax.set_ylabel("T3 Δ pp (Sibling − WikiText-shuffled)")
    ax.set_title("T3 vs layer (L37* = published reference)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(P21_DIR / "figure_t3_vs_layer.png", dpi=140)
    plt.close()

    # Decomp figure: Shared/Unique Δ pp per layer (grouped bars)
    sh_d = [s["decomposition"]["table"]["Shared-only"]["delta_pp"]
            for s in all_summaries]
    un_d = [s["decomposition"]["table"]["Unique-only"]["delta_pp"]
            for s in all_summaries]
    x = np.arange(len(layers_list))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, sh_d, width, label="Shared-only", color="#2980b9",
            edgecolor="white")
    ax.bar(x + width/2, un_d, width, label="Unique-only", color="#27ae60",
            edgecolor="white")
    for i, (s_d, u_d) in enumerate(zip(sh_d, un_d)):
        ax.text(i - width/2, s_d - 0.5, f"{s_d:+.2f}", ha="center", fontsize=7)
        ax.text(i + width/2, u_d - 0.5, f"{u_d:+.2f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{l}{'*' if l == 37 else ''}" for l in layers_list])
    ax.set_ylabel("Δ pp vs baseline")
    ax.set_title("Shared/Unique decomposition vs layer")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(P21_DIR / "figure_decomp_vs_layer.png", dpi=140)
    plt.close()

    print(f"\n=== Stage 2.1 complete ===")
    print(f"  total elapsed: {(time.time()-t_total)/60:.1f} min")
    print(f"  -> {P21_DIR}/layer_sweep_summary.json")
    print(f"  -> {P21_DIR}/figure_t3_vs_layer.png")
    print(f"  -> {P21_DIR}/figure_decomp_vs_layer.png")
    print(f"\n  Layer-by-layer summary:")
    for s in all_summaries:
        t3 = s["tests"].get("T3_sibling_vs_wtshuf", {})
        d = s["decomposition"]["table"]
        sf = s["single_feature_equivalence"]
        marker = " *" if s["layer"] == 37 else ""
        print(f"    L{s['layer']}{marker}: "
              f"T3 Δ={t3.get('delta_pp', float('nan')):+5.2f} (p={t3.get('wilcoxon_p', float('nan')):.2e})  "
              f"Sh Δ={d['Shared-only']['delta_pp']:+5.2f}  Un Δ={d['Unique-only']['delta_pp']:+5.2f}  "
              f"sf head-to-head p={sf.get('uc_vs_sc_two_sided_wilcoxon_p', float('nan')):.3g}")


if __name__ == "__main__":
    main()
