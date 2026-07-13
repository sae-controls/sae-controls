"""Cross-model replication on Gemma-2-2B-IT.

Replicates the headline mechanism at the last residual-stream layer of
Gemma-2-2B-IT (L25 of 26 transformer layers, indexed 0-25).
Mirrors the paper's L41-on-9B choice — the natural cross-model
translation is "the last residual layer" rather than a fixed absolute
index.

Steps:
  1. Re-run slot detection on the AmbigQA candidate pool (1128 raw
     candidates from expanded_dataset.json) using Gemma-2-2B-IT's
     tokenizer + chat template.
  2. Capture L25 last-prompt-token residuals for surviving A + D_i
     prompts.
  3. SAE-encode via Gemma Scope width-16k canonical SAE for L25 of
     Gemma-2-2B (from google/gemma-scope-2b-pt-res).
  4. Capture L25 last-token residuals for the 800 WikiText paragraphs.
  5. Pick Targeted top-10 via published score formula.
  6. Run six conditions (Baseline, Targeted, Sibling, Shuffled-AmbigQA,
     WikiText-shuffled, Random).
  7. T1-T4 paired tests + bootstrap CIs.
  8. Unique/shared decomposition.

Skips polysemy decomposition, per-feature equivalence, multi-metric —
those are second-order refinements; main breadth question is whether T3
+ unique/shared replicate at 2B-scale.

Reuses src/{hooks, features, analysis, prompts, slot_detection,
wikitext, sae} unmodified. Writes to artifacts/replication_gemma2b/.
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
    ARTIFACTS_DIR, ATTN_IMPL, DTYPE_STR, N_SHUFFLES_PER_PAIR,
    RANDOM_CONTROLS, SAE_WIDTH, SCORE_A_WEIGHT, SEED, SHUFFLE_SEED,
    TOP_K_FEATURES, TOP_LOGITS_K, WIKITEXT_MAX_TOKENS, device,
)
from src.features import score_specific_features
from src.hooks import (
    baseline_top_logits, capture_all_position_residuals,
    capture_last_token_residual, forward_with_ablation, greedy_decode,
    hit_at_k,
)
from src.io_utils import save_json_atomic, save_npz_atomic
from src.prompts import build_prompt
from src.sae import JumpReLUSAE
from src.slot_detection import find_best_match, first_token_variants
from src.wikitext import load_paragraphs as load_wikitext_paragraphs


P23_DIR = ARTIFACTS_DIR / "replication_gemma2b"

# Gemma-2-2B settings
MODEL_ID_2B = "google/gemma-2-2b-it"
SAE_REPO_2B = "google/gemma-scope-2b-pt-res"
LAYER_2B = 25                 # last residual-stream layer (26 layers, 0-25)
GEN_MAX_NEW = 15


def _load_2b_model():
    """Mirror of src.model.load_model but for Gemma-2-2B-IT."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype = getattr(torch, DTYPE_STR)
    tok = AutoTokenizer.from_pretrained(MODEL_ID_2B)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID_2B,
        torch_dtype=dtype,
        device_map=device(),
        low_cpu_mem_usage=True,
        attn_implementation=ATTN_IMPL,
    )
    model.eval()
    return tok, model


def _load_2b_sae(layer):
    """Find canonical L0 for Gemma-2-2B at this layer; load + return (sae, meta)."""
    import hashlib
    from huggingface_hub import hf_hub_download, list_repo_files
    files = list_repo_files(SAE_REPO_2B)
    prefix = f"layer_{layer}/width_{SAE_WIDTH}/"
    matches = sorted([f for f in files
                       if f.startswith(prefix) and f.endswith("params.npz")])
    if not matches:
        raise RuntimeError(f"no SAEs at {prefix} in {SAE_REPO_2B}")
    print(f"[sae 2B L{layer}] available L0 variants: {[m.split('/')[-2] for m in matches]}")
    # Pick the lowest-L0 variant (canonical pattern from 9B: most-sparse SAE)
    def _l0_value(path):
        try:
            return int(path.split("/average_l0_")[-1].split("/")[0])
        except Exception:
            return 999
    canonical = min(matches, key=_l0_value)
    print(f"[sae 2B L{layer}] using {canonical}")
    path = Path(hf_hub_download(SAE_REPO_2B, canonical))
    params = np.load(path)
    d_model, d_sae = params["W_enc"].shape
    sae = JumpReLUSAE(d_model, d_sae, device_=device())
    sae.W_enc     = torch.from_numpy(params["W_enc"]).to(device=sae.device, dtype=sae.dtype)
    sae.W_dec     = torch.from_numpy(params["W_dec"]).to(device=sae.device, dtype=sae.dtype)
    sae.threshold = torch.from_numpy(params["threshold"]).to(device=sae.device, dtype=sae.dtype)
    sae.b_enc     = torch.from_numpy(params["b_enc"]).to(device=sae.device, dtype=sae.dtype)
    sae.b_dec     = torch.from_numpy(params["b_dec"]).to(device=sae.device, dtype=sae.dtype)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    return sae, {
        "repo": SAE_REPO_2B,
        "subpath": canonical,
        "local_path": str(path),
        "sha256": sha,
        "d_model": int(d_model), "d_sae": int(d_sae),
        "layer": layer,
    }


def _stats_t3(Sib_arr, WSh_arr):
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


def main() -> None:
    P23_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"[setup] loading {MODEL_ID_2B}...")
    tokenizer, model = _load_2b_model()
    print(f"[setup] model loaded; n_layers={model.config.num_hidden_layers}")
    if model.config.num_hidden_layers != 26:
        print(f"[setup] WARNING: expected 26 layers, got "
              f"{model.config.num_hidden_layers}")

    print(f"[setup] loading SAE for L{LAYER_2B}...")
    sae, sae_meta = _load_2b_sae(LAYER_2B)
    print(f"[setup] SAE sha256: {sae_meta['sha256'][:16]}... (L0 from "
          f"subpath: {sae_meta['subpath'].split('/average_l0_')[-1].split('/')[0]})")
    print(f"[setup] SAE d_model={sae.d_model}, d_sae={sae.d_sae}")

    expanded = json.load(open(ARTIFACTS_DIR / "expanded_dataset.json"))
    candidates = expanded["pairs"]
    print(f"[setup] {len(candidates)} candidate ambiguous questions from stage 1")

    # ---- Stage 2.A: collision filter (with 2B tokenizer) ----
    print(f"\n[2.A] tokenize first_token_variants and apply within-pair "
          f"collision filter (2B tokenizer)")
    after_collision = []
    for c in tqdm(candidates, desc="2.A 2B"):
        di_list = []
        seen_first_tok = set()
        for d in c["disambigs"]:
            ft = first_token_variants(tokenizer, d["answer"])
            if not ft:
                continue
            if ft[0] in seen_first_tok:
                continue
            seen_first_tok.add(ft[0])
            di_list.append({
                "question": d["question"],
                "answer": d["answer"],
                "answer_variants": d.get("answer_variants", [d["answer"]]),
                "first_token_variants": ft,
            })
        if len(di_list) < 2:
            continue
        if len(di_list) > 4:
            di_list = di_list[:4]
        after_collision.append({
            "id": c["id"],
            "A_question": c["A_question"],
            "source_lever": c.get("source_lever", 0),
            "disambigs": di_list,
        })
    print(f"[2.A 2B] kept {len(after_collision)} of {len(candidates)} after "
          f"collision filter")

    # ---- Stage 2.B: greedy-decode + slot detection (with 2B model) ----
    print(f"\n[2.B] greedy-decode + slot detection on {len(after_collision)} pairs")
    detected = []
    rejections = []
    for p in tqdm(after_collision, desc="2.B 2B"):
        a_prompt = build_prompt(tokenizer, p["A_question"])
        gen_text = greedy_decode(model, tokenizer, a_prompt, max_new=GEN_MAX_NEW)
        cand_strs = [d["answer"] for d in p["disambigs"]]
        match = find_best_match(gen_text, cand_strs)
        if match.cand_idx < 0:
            rejections.append({"id": p["id"], "question": p["A_question"],
                                "gen_text": gen_text.strip(), "cands": cand_strs})
            continue
        detected.append({
            **p,
            "A_gen_text": gen_text.strip(),
            "match_strategy": match.strategy,
            "match_cand_idx": int(match.cand_idx),
            "matched_text": match.matched_text,
        })
    n_self_pairs = sum(len(p["disambigs"]) for p in detected)
    print(f"[2.B 2B] detected: {len(detected)} pairs, {n_self_pairs} self-pairs; "
          f"rejected: {len(rejections)}")

    save_json_atomic(P23_DIR / "detected_pairs_2b.json", detected)
    save_json_atomic(P23_DIR / "slot_detection_failures_2b.json", rejections)

    if len(detected) < 30:
        raise RuntimeError(f"too few detected pairs ({len(detected)}); aborting")

    a_prompts = {p["id"]: build_prompt(tokenizer, p["A_question"]) for p in detected}

    # ---- Stage 2.C: residual capture + SAE encoding ----
    print(f"\n[2.C] capturing L{LAYER_2B} residuals "
          f"({len(detected)} A + {n_self_pairs} D = {len(detected) + n_self_pairs})")
    residuals = {}
    for p in tqdm(detected, desc="2.C residuals"):
        residuals[f"A__{p['id']}"] = capture_last_token_residual(
            model, tokenizer, a_prompts[p["id"]], layer=LAYER_2B,
        )
        for di, d in enumerate(p["disambigs"]):
            d_prompt = build_prompt(tokenizer, d["question"])
            residuals[f"D__{p['id']}__{di}"] = capture_last_token_residual(
                model, tokenizer, d_prompt, layer=LAYER_2B,
            )
    save_npz_atomic(P23_DIR / "residuals_L25_2b.npz", **residuals)

    encodings = {}
    for k, r in residuals.items():
        x = torch.from_numpy(r).to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        z = sae.encode(x.unsqueeze(0)).squeeze(0)
        encodings[k] = z.float().cpu().numpy()
    save_npz_atomic(P23_DIR / "sae_encodings_L25_2b.npz", **encodings)

    # ---- Stage 2.D: WikiText residuals + top-10 ----
    paragraphs = load_wikitext_paragraphs(tokenizer)
    print(f"\n[2.D] WikiText: {len(paragraphs)} paragraphs at L{LAYER_2B}")
    wt_lasttok_z = np.zeros((len(paragraphs), sae.d_sae), dtype=np.float32)
    for i, p_text in enumerate(tqdm(paragraphs, desc="2.D WT")):
        try:
            _, res = capture_all_position_residuals(
                model, tokenizer, p_text, layer=LAYER_2B,
                max_len=WIKITEXT_MAX_TOKENS,
            )
            z_last = sae.encode(res[-1:].to(dtype=sae.W_enc.dtype)).squeeze(0)
            wt_lasttok_z[i] = z_last.float().cpu().numpy()
            del res, z_last
        except Exception as e:
            print(f"[2.D] paragraph {i} failed: {type(e).__name__}: {e}")
        if i % 50 == 0:
            torch.cuda.empty_cache(); gc.collect()
    wt_top10 = []
    for i in range(len(paragraphs)):
        z_p = wt_lasttok_z[i]
        positives = np.where(z_p > 0)[0]
        if len(positives) == 0:
            kept = []
        else:
            sorted_idx = positives[np.argsort(-z_p[positives])]
            kept = [int(f) for f in sorted_idx[:TOP_K_FEATURES]]
        wt_top10.append({"paragraph_idx": i, "top10_feature_ids": kept})

    # ---- Pick Targeted features per (P, D_i) ----
    print(f"\n[features] picking Targeted top-10 features (a_weight={SCORE_A_WEIGHT}, "
          f"top_k={TOP_K_FEATURES})")
    enc_torch = {k: torch.from_numpy(v).to(device=sae.W_enc.device,
                                            dtype=sae.W_enc.dtype)
                 for k, v in encodings.items()}
    specific_features = {}
    for p in detected:
        z_A = enc_torch[f"A__{p['id']}"]
        z_D_list = [enc_torch[f"D__{p['id']}__{di}"]
                    for di in range(len(p["disambigs"]))]
        topk_per_d = score_specific_features(
            z_A, z_D_list, top_k=TOP_K_FEATURES, a_weight=SCORE_A_WEIGHT,
        )
        for di, feats in enumerate(topk_per_d):
            specific_features[(p["id"], di)] = feats
    save_json_atomic(
        P23_DIR / "specific_features_2b.json",
        [{"pair_id": pid, "disambig_idx": di, "features": feats}
         for (pid, di), feats in specific_features.items()],
    )

    # ---- Stage 2.E: six-condition pipeline ----
    print(f"\n[2.E] six-condition pipeline on {len(detected)} pairs ({n_self_pairs} self-pairs)")
    self_rows = []; cross_rows = []; random_rows = []
    ambig_shuf_rows = []; wt_shuf_rows = []
    ambig_draws_log = []; wt_draws_log = []

    import random as pyrandom
    rng_main = pyrandom.Random(SEED)
    rng_wt = pyrandom.Random(SHUFFLE_SEED)
    all_specific_keys = list(specific_features.keys())

    for p in tqdm(detected, desc="2.E"):
        a_prompt = a_prompts[p["id"]]
        base = baseline_top_logits(model, tokenizer, a_prompt, k=TOP_LOGITS_K)
        base_top = base["top_ids"]
        base_hits = []
        for d in p["disambigs"]:
            base_hits.append(hit_at_k(base_top, set(d["first_token_variants"]), 1))

        eligible_ambigqa_keys = [k for k in all_specific_keys if k[0] != p["id"]]

        for ab_i in range(len(p["disambigs"])):
            feats_self = specific_features.get((p["id"], ab_i), [])
            if not feats_self:
                continue
            ab = forward_with_ablation(
                model, tokenizer, a_prompt, LAYER_2B, sae,
                feature_ids=feats_self, k=TOP_LOGITS_K,
            )
            for tg_j in range(len(p["disambigs"])):
                target = set(p["disambigs"][tg_j]["first_token_variants"])
                row = {
                    "pair_id": p["id"], "ablate_idx": ab_i, "target_idx": tg_j,
                    "is_self": ab_i == tg_j,
                    "n_features": len(feats_self),
                    "base_hit1": base_hits[tg_j],
                    "ablate_hit1": hit_at_k(ab["top_ids"], target, 1),
                }
                if ab_i == tg_j:
                    self_rows.append(row)
                else:
                    cross_rows.append(row)

            target_self = set(p["disambigs"][ab_i]["first_token_variants"])

            # Random
            for r_idx in range(RANDOM_CONTROLS):
                rand_feats = rng_main.sample(range(sae.d_sae), len(feats_self))
                rc = forward_with_ablation(
                    model, tokenizer, a_prompt, LAYER_2B, sae,
                    feature_ids=rand_feats, k=TOP_LOGITS_K,
                )
                random_rows.append({
                    "pair_id": p["id"], "ablate_idx": ab_i, "control_idx": r_idx,
                    "n_features": len(rand_feats), "feature_ids": rand_feats,
                    "base_hit1": base_hits[ab_i],
                    "random_hit1": hit_at_k(rc["top_ids"], target_self, 1),
                })

            # AmbigQA-shuffle
            for d_idx in range(N_SHUFFLES_PER_PAIR):
                p_prime, k_idx = rng_main.choice(eligible_ambigqa_keys)
                feats_sh = specific_features.get((p_prime, k_idx), [])
                sh = forward_with_ablation(
                    model, tokenizer, a_prompt, LAYER_2B, sae,
                    feature_ids=feats_sh, k=TOP_LOGITS_K,
                )
                ambig_shuf_rows.append({
                    "pair_id": p["id"], "target_idx": ab_i, "draw_idx": d_idx,
                    "shuffle_pair_id": p_prime, "shuffle_disambig_idx": int(k_idx),
                    "n_features": len(feats_sh),
                    "base_hit1": base_hits[ab_i],
                    "shuffled_hit1": hit_at_k(sh["top_ids"], target_self, 1),
                })
                ambig_draws_log.append({
                    "pair_id": p["id"], "target_idx": ab_i, "draw_idx": d_idx,
                    "shuffle_pair_id": p_prime, "shuffle_disambig_idx": int(k_idx),
                })

            # WikiText-shuffle
            for d_idx in range(N_SHUFFLES_PER_PAIR):
                para_idx = rng_wt.choice(range(len(paragraphs)))
                feats_wt = wt_top10[para_idx]["top10_feature_ids"]
                wt = forward_with_ablation(
                    model, tokenizer, a_prompt, LAYER_2B, sae,
                    feature_ids=feats_wt, k=TOP_LOGITS_K,
                )
                wt_shuf_rows.append({
                    "pair_id": p["id"], "target_idx": ab_i, "draw_idx": d_idx,
                    "wikitext_paragraph_idx": para_idx,
                    "n_features": len(feats_wt),
                    "base_hit1": base_hits[ab_i],
                    "wt_shuffled_hit1": hit_at_k(wt["top_ids"], target_self, 1),
                })
                wt_draws_log.append({
                    "pair_id": p["id"], "target_idx": ab_i, "draw_idx": d_idx,
                    "wikitext_paragraph_idx": para_idx,
                })

        torch.cuda.empty_cache()

    # ---- Six-condition headline + T1-T4 ----
    sib = per_pair_means(
        cross_rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["ablate_hit1"],
    )
    wt = per_pair_means(
        wt_shuf_rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["wt_shuffled_hit1"],
    )
    ash = per_pair_means(
        ambig_shuf_rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["shuffled_hit1"],
    )
    rnd = per_pair_means(
        random_rows,
        key_fn=lambda r: (r["pair_id"], r["ablate_idx"]),
        value_fn=lambda r: r["random_hit1"],
    )
    base_pp = {(r["pair_id"], r["target_idx"]): r["base_hit1"] for r in self_rows}
    targ_pp = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"] for r in self_rows}
    keys_all = sorted(k for k in base_pp if k in sib and k in wt
                       and k in ash and k in rnd)
    n = len(keys_all)
    B = np.array([base_pp[k] for k in keys_all], dtype=float)
    T = np.array([targ_pp[k] for k in keys_all], dtype=float)
    Sib = np.array([sib[k] for k in keys_all], dtype=float)
    ASh = np.array([ash[k] for k in keys_all], dtype=float)
    WSh = np.array([wt[k] for k in keys_all], dtype=float)
    Rnd = np.array([rnd[k] for k in keys_all], dtype=float)

    headline = {
        "Baseline": float(B.mean()),
        "Targeted": float(T.mean()),
        "Sibling":  float(Sib.mean()),
        "ShuffledAmbigQA": float(ASh.mean()),
        "WikiTextShuffled": float(WSh.mean()),
        "Random": float(Rnd.mean()),
    }
    print(f"\n[2E] headline rates (n={n}):")
    for k, v in headline.items():
        delta = (v - B.mean()) * 100
        print(f"  {k:<20}  {v:.4f}  (Δ vs base = {delta:+.2f} pp)")

    tests = {
        "T1_targeted_vs_random": _drop_vs_base_stats(T, Rnd),
        "T2_sibling_vs_ashuf":   _stats_t3(Sib, ASh),
        "T3_sibling_vs_wtshuf":  _stats_t3(Sib, WSh),
        "T4_ashuf_vs_wtshuf":    _stats_t3(ASh, WSh),
    }
    print(f"\n[2E] paired tests:")
    for tn, t in tests.items():
        if t is not None:
            print(f"  {tn:<28}  Δ={t['delta_pp']:+5.2f}  CI=[{t['ci_pp'][0]:+5.2f},"
                  f"{t['ci_pp'][1]:+5.2f}]  Wlx p={t['wilcoxon_p']:.3g}")

    save_json_atomic(P23_DIR / "results_main_2b.json", {
        "model": MODEL_ID_2B,
        "layer": LAYER_2B,
        "headline": headline,
        "tests": tests,
        "self_rows": self_rows,
        "cross_rows": cross_rows,
        "random_rows": random_rows,
        "ambigqa_shuffled_rows": ambig_shuf_rows,
        "wikitext_shuffled_rows": wt_shuf_rows,
        "ambigqa_draws_log": ambig_draws_log,
        "wikitext_draws_log": wt_draws_log,
        "n_self_pairs": n,
    })

    # ---- Unique/shared decomposition ----
    print(f"\n[unique/shared] decomposition")
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
                if j != i:
                    sib_union |= pub_set.get((p["id"], j), set())
            sh_un[(p["id"], i)] = {
                "shared": ti & sib_union,
                "unique": ti - (ti & sib_union),
                "targeted": ti,
            }

    sh_un_rows = []
    for p in tqdm(detected, desc="sh+un"):
        a_prompt = a_prompts[p["id"]]
        for i in range(len(p["disambigs"])):
            key = (p["id"], i)
            if key not in sh_un or key not in base_pp:
                continue
            target = set(p["disambigs"][i]["first_token_variants"])
            base = base_pp[key]
            if sh_un[key]["shared"]:
                ab = forward_with_ablation(
                    model, tokenizer, a_prompt, LAYER_2B, sae,
                    feature_ids=list(sh_un[key]["shared"]), k=TOP_LOGITS_K,
                )
                shared_hit = hit_at_k(ab["top_ids"], target, 1)
            else:
                shared_hit = base
            if sh_un[key]["unique"]:
                ab = forward_with_ablation(
                    model, tokenizer, a_prompt, LAYER_2B, sae,
                    feature_ids=list(sh_un[key]["unique"]), k=TOP_LOGITS_K,
                )
                unique_hit = hit_at_k(ab["top_ids"], target, 1)
            else:
                unique_hit = base
            sh_un_rows.append({
                "pair_id": p["id"], "target_idx": i,
                "n_targeted": len(sh_un[key]["targeted"]),
                "n_shared":   len(sh_un[key]["shared"]),
                "n_unique":   len(sh_un[key]["unique"]),
                "base_hit1":  base,
                "targeted_hit1": targ_pp[key],
                "shared_only_hit1": shared_hit,
                "unique_only_hit1": unique_hit,
            })
        torch.cuda.empty_cache()

    keys_sh = [(r["pair_id"], r["target_idx"]) for r in sh_un_rows]
    by_key = {(r["pair_id"], r["target_idx"]): r for r in sh_un_rows}
    Bsh = np.array([by_key[k]["base_hit1"] for k in keys_sh])
    Tsh = np.array([by_key[k]["targeted_hit1"] for k in keys_sh])
    Shsh = np.array([by_key[k]["shared_only_hit1"] for k in keys_sh])
    Unsh = np.array([by_key[k]["unique_only_hit1"] for k in keys_sh])

    targ_drop = (Tsh.mean() - Bsh.mean()) * 100
    sh_drop = (Shsh.mean() - Bsh.mean()) * 100
    un_drop = (Unsh.mean() - Bsh.mean()) * 100
    sum_drops = sh_drop + un_drop
    residual = targ_drop - sum_drops
    pct = lambda x: (x / targ_drop * 100) if abs(targ_drop) > 1e-9 else float("nan")

    decomp = {
        "n": len(keys_sh),
        "table": {
            "Baseline":    {"hit1": float(Bsh.mean()), "delta_pp": 0.0,
                            "pct_of_targeted": 0.0},
            "Targeted":    {"hit1": float(Tsh.mean()), "delta_pp": float(targ_drop),
                            "pct_of_targeted": 100.0},
            "Shared-only": {"hit1": float(Shsh.mean()), "delta_pp": float(sh_drop),
                            "pct_of_targeted": float(pct(sh_drop))},
            "Unique-only": {"hit1": float(Unsh.mean()), "delta_pp": float(un_drop),
                            "pct_of_targeted": float(pct(un_drop))},
            "Sum (Sh+Un)": {"delta_pp": float(sum_drops),
                             "pct_of_targeted": float(pct(sum_drops))},
            "Residual":    {"delta_pp": float(residual),
                             "pct_of_targeted": float(pct(residual))},
        },
        "tests": {
            "shared_vs_baseline": _drop_vs_base_stats(Shsh, Bsh),
            "unique_vs_baseline": _drop_vs_base_stats(Unsh, Bsh),
            "targeted_vs_unique": _drop_vs_base_stats(Tsh, Unsh),
        },
    }
    save_json_atomic(P23_DIR / "unique_shared_decomp_2b.json", {
        "decomposition": decomp,
        "rows": sh_un_rows,
    })
    print(f"[unique/shared] Targeted Δ={targ_drop:+.2f}; "
          f"Shared Δ={sh_drop:+.2f} ({pct(sh_drop):+.1f}%); "
          f"Unique Δ={un_drop:+.2f} ({pct(un_drop):+.1f}%); "
          f"Residual {residual:+.2f} ({pct(residual):+.1f}%)")

    # ---- Cross-model summary table ----
    # Load 9B L41 reference
    try:
        l41_summary = json.load(open(ARTIFACTS_DIR / "reference_layer" / "L41_summary_table.json"))
    except Exception as e:
        print(f"[summary] could not load L41 reference: {e}")
        l41_summary = None

    cross_model_rows = []
    if l41_summary:
        l41_t3 = l41_summary["T3"]
        l41_decomp = l41_summary["decomposition_hit1_from_p21b"]
        cross_model_rows.append({
            "model": "google/gemma-2-9b-it",
            "layer": 41,
            "n_self_pairs": l41_summary["n_self_pairs"],
            "baseline_hit1": l41_summary["headline_hit1"]["Baseline"],
            "targeted_delta_pp": l41_decomp["Targeted"]["delta_pp"],
            "sibling_hit1": l41_summary["headline_hit1"]["Sibling"],
            "wt_shuf_hit1": l41_summary["headline_hit1"]["WikiTextShuffled"],
            "T3_delta_pp": l41_t3["delta_pp"],
            "T3_wilcoxon_p": l41_t3["wilcoxon_p"],
            "shared_only_delta_pp": l41_decomp["Shared-only"]["delta_pp"],
            "unique_only_delta_pp": l41_decomp["Unique-only"]["delta_pp"],
            "residual_pct_of_targeted": (
                (l41_decomp["Targeted"]["delta_pp"]
                 - l41_decomp["Shared-only"]["delta_pp"]
                 - l41_decomp["Unique-only"]["delta_pp"])
                / l41_decomp["Targeted"]["delta_pp"] * 100
            ),
        })
    cross_model_rows.append({
        "model": MODEL_ID_2B,
        "layer": LAYER_2B,
        "n_self_pairs": n,
        "baseline_hit1": float(B.mean()),
        "targeted_delta_pp": float((T.mean() - B.mean()) * 100),
        "sibling_hit1": float(Sib.mean()),
        "wt_shuf_hit1": float(WSh.mean()),
        "T3_delta_pp": tests["T3_sibling_vs_wtshuf"]["delta_pp"]
            if tests["T3_sibling_vs_wtshuf"] else None,
        "T3_wilcoxon_p": tests["T3_sibling_vs_wtshuf"]["wilcoxon_p"]
            if tests["T3_sibling_vs_wtshuf"] else None,
        "shared_only_delta_pp": float(sh_drop),
        "unique_only_delta_pp": float(un_drop),
        "residual_pct_of_targeted": float(pct(residual)),
    })

    save_json_atomic(P23_DIR / "summary.json", {
        "model": MODEL_ID_2B,
        "layer": LAYER_2B,
        "sae_meta": sae_meta,
        "slot_detection": {
            "n_input_candidates":   len(candidates),
            "n_after_collision":    len(after_collision),
            "n_detected_pairs":     len(detected),
            "n_self_pairs":         n_self_pairs,
            "retention_pct":        100 * len(detected) / len(candidates),
        },
        "headline_hit1":     headline,
        "tests":             tests,
        "decomposition":     decomp,
        "cross_model_table": cross_model_rows,
    })

    # ---- Save run_meta ----
    save_json_atomic(P23_DIR / "run_meta.json", {
        "model": MODEL_ID_2B,
        "layer": LAYER_2B,
        "sae_meta": sae_meta,
        "n_input_candidates": len(candidates),
        "n_after_collision":  len(after_collision),
        "n_detected_pairs":   len(detected),
        "n_self_pairs":       n,
        "elapsed_seconds":    round(time.time() - t_total, 2),
    })

    # ---- Cleanup ----
    print(f"\n[cleanup] removing residuals NPZ + SAE blob (keep encodings)")
    try:
        (P23_DIR / "residuals_L25_2b.npz").unlink(missing_ok=True)
    except Exception:
        pass
    try:
        local_path = sae_meta.get("local_path")
        if local_path:
            real = Path(local_path).resolve()
            if real.exists():
                real.unlink()
                print(f"  deleted SAE blob from HF cache")
    except Exception as e:
        print(f"  sae blob cleanup failed: {e}")

    print(f"\n=== Stage 2.3 complete ===")
    print(f"  total elapsed: {(time.time()-t_total)/60:.1f} min")
    print(f"  retention: {len(detected)}/{len(candidates)} = "
          f"{100*len(detected)/len(candidates):.1f}%; n_self_pairs={n}")
    print(f"  -> {P23_DIR}/summary.json")


if __name__ == "__main__":
    main()
