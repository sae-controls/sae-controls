"""Ablation Conditions (paper Sec. 3.2). Single A40 session.

Runs the full causal pipeline. Reads `artifacts/expanded_dataset.json`; writes
all per-instance and aggregated ablation outcomes.

Sub-stages (one forward-pass loop each):

  2.A  Tokenize first_token_variants per disambig + relaxed within-pair
       collision filter (drop colliding cands, keep pair if ≥2 distinct first
       tokens remain).
  2.B  Greedy-decode baseline for every A question; apply slot detection
       (substring + distinctive_tokens). Drop pairs where no annotated answer
       is detected → `detected_pairs.json`.
  2.C  Capture L37 last-prompt-token residuals for every detected A and D_i.
       SAE-encode each. Compute per-(P, D_i) top-10 D_i-specific features via
       score(f, i) = z_{D_i}(f) - mean_{j≠i} z_{D_j}(f) + 0.5·z_A(f).
  2.D  Capture L37 last-token residuals for the 800 WikiText-2 raw test
       paragraphs (≥50 tokens) and SAE-encode each. Take top-10 features per
       paragraph (used as the WikiText-shuffle ablation source).
  2.E  Run six ablation conditions per (P, D_i) self-pair:
         - Baseline (no intervention; one forward per A prompt, scored against
           every D_i)
         - Targeted (D_i's top-10 features ablated)
         - Sibling cross (D_j's top-10, j≠i in same P)
         - AmbigQA-shuffle  (top-10 of D_k from random P'≠P, 3 draws, seed=SEED)
         - WikiText-shuffle (top-10 of random WikiText paragraph, 3 draws,
                              seed=SHUFFLE_SEED)
         - Random feature  (10 random IDs from 16384, 3 draws, seed=SEED)

All ablations use the error-preserving SAE splice at the L37 last token (src/hooks.py).
SAE checkpoint sha256 is verified at start.

Outputs:
  artifacts/detected_pairs.json
  artifacts/residuals_L37.npz
  artifacts/sae_encodings_L37.npz
  artifacts/specific_features.json
  artifacts/wikitext_paragraphs.json
  artifacts/wikitext_last_token_top10.json
  artifacts/results_main.json            (baseline + 5 ablation conditions)
  artifacts/shuffle_draws_ambigqa.json   (auditable AmbigQA-shuffle draws)
  artifacts/shuffle_draws_wikitext.json  (auditable WikiText-shuffle draws)
  artifacts/run_meta.json
  artifacts/slot_detection_failures.json (candidates rejected at 2.B)
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.config import (
    ARTIFACTS_DIR, DTYPE_STR, LAYER, MODEL_ID, N_SHUFFLES_PER_PAIR,
    RANDOM_CONTROLS, SAE_CHECKPOINT_SHA256, SCORE_A_WEIGHT, SAE_L0_SEL,
    SAE_WIDTH, SEED, SHUFFLE_SEED, TOP_K_FEATURES, TOP_LOGITS_K,
    WIKITEXT_MAX_TOKENS, device,
)
from src.features import score_specific_features
from src.hooks import (
    baseline_top_logits, capture_all_position_residuals,
    capture_last_token_residual, forward_with_ablation, greedy_decode,
    hit_at_k,
)
from src.io_utils import save_json_atomic, save_npz_atomic
from src.model import load_model
from src.prompts import build_prompt
from src.sae import load_sae
from src.slot_detection import find_best_match, first_token_variants
from src.wikitext import load_paragraphs as load_wikitext_paragraphs, paragraph_metadata, sha256_text

GEN_MAX_NEW = 15


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    print(f"[setup] model={MODEL_ID}, layer=L{LAYER}, dtype={DTYPE_STR}, "
          f"main_seed={SEED}, shuffle_seed={SHUFFLE_SEED}")

    # Load candidate dataset (build_dataset.py output)
    expanded_path = ARTIFACTS_DIR / "expanded_dataset.json"
    if not expanded_path.exists():
        raise FileNotFoundError(f"missing {expanded_path}; run build_dataset.py first")
    expanded = json.load(open(expanded_path))
    candidates = expanded["pairs"]
    print(f"[setup] candidates from stage 1: {len(candidates)} A pairs, "
          f"{expanded['n_disambigs_total']} disambigs")

    # ---- Load model + SAE ----
    print("[setup] loading model...")
    t = time.time()
    tokenizer, model = load_model()
    print(f"[setup] model loaded in {time.time()-t:.1f}s")
    print("[setup] loading SAE...")
    t = time.time()
    sae, sae_meta = load_sae(layer=LAYER)
    print(f"[setup] SAE loaded in {time.time()-t:.1f}s "
          f"(sha256={sae_meta['sha256'][:16]}...)")
    if sae_meta["sha256"] != SAE_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"SAE sha256 mismatch! got {sae_meta['sha256']}, expected {SAE_CHECKPOINT_SHA256}"
        )
    print(f"[setup] SAE sha256 verified — matches paper-pinned checkpoint")

    # =========================================================================
    # Stage 2.A — tokenize + collision filter
    # =========================================================================
    print(f"\n[2.A] tokenize first_token_variants and apply within-pair collision filter")
    after_collision = []
    for c in tqdm(candidates, desc="2.A"):
        di_list = []; seen_first_tok = set()
        for d in c["disambigs"]:
            ft = first_token_variants(tokenizer, d["answer"])
            if not ft: continue
            if ft[0] in seen_first_tok: continue
            seen_first_tok.add(ft[0])
            di_list.append({
                "question": d["question"],
                "answer": d["answer"],
                "answer_variants": d.get("answer_variants", [d["answer"]]),
                "first_token_variants": ft,
            })
        if len(di_list) < 2: continue
        if len(di_list) > 4: di_list = di_list[:4]
        after_collision.append({
            "id": c["id"],
            "A_question": c["A_question"],
            "source_lever": c.get("source_lever", 0),
            "disambigs": di_list,
        })
    print(f"[2.A] kept {len(after_collision)} of {len(candidates)} after collision filter")

    # =========================================================================
    # Stage 2.B — greedy-decode + slot detection
    # =========================================================================
    print(f"\n[2.B] greedy-decode + slot detection on {len(after_collision)} pairs")
    detected = []
    rejections = []
    for p in tqdm(after_collision, desc="2.B"):
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
    print(f"[2.B] detected: {len(detected)}; rejected: {len(rejections)}")
    save_json_atomic(ARTIFACTS_DIR / "detected_pairs.json", detected)
    save_json_atomic(ARTIFACTS_DIR / "slot_detection_failures.json", rejections)

    if len(detected) < 30:
        raise RuntimeError(f"too few detected pairs ({len(detected)}); aborting")

    # =========================================================================
    # Stage 2.C — residual capture + SAE encoding + feature scoring
    # =========================================================================
    print(f"\n[2.C] capturing L{LAYER} residuals + SAE encoding + feature scoring")
    prompts_A = {p["id"]: build_prompt(tokenizer, p["A_question"]) for p in detected}
    residuals: dict[str, np.ndarray] = {}
    for p in tqdm(detected, desc="2.C residuals"):
        residuals[f"A__{p['id']}"] = capture_last_token_residual(
            model, tokenizer, prompts_A[p["id"]], layer=LAYER,
        )
        for di, d in enumerate(p["disambigs"]):
            d_prompt = build_prompt(tokenizer, d["question"])
            residuals[f"D__{p['id']}__{di}"] = capture_last_token_residual(
                model, tokenizer, d_prompt, layer=LAYER,
            )
    save_npz_atomic(ARTIFACTS_DIR / f"residuals_L{LAYER}.npz", **residuals)

    encodings: dict[str, np.ndarray] = {}
    specific_features: dict[tuple[str, int], list[int]] = {}
    for p in tqdm(detected, desc="2.C SAE+score"):
        x_A = torch.from_numpy(residuals[f"A__{p['id']}"]).to(
            device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        z_A = sae.encode(x_A.unsqueeze(0)).squeeze(0)
        encodings[f"A__{p['id']}"] = z_A.to(dtype=torch.float32, device="cpu").numpy()
        z_D_list = []
        for di in range(len(p["disambigs"])):
            x_D = torch.from_numpy(residuals[f"D__{p['id']}__{di}"]).to(
                device=sae.W_enc.device, dtype=sae.W_enc.dtype)
            z_D = sae.encode(x_D.unsqueeze(0)).squeeze(0)
            encodings[f"D__{p['id']}__{di}"] = z_D.to(dtype=torch.float32, device="cpu").numpy()
            z_D_list.append(z_D)
        topk_per_d = score_specific_features(z_A, z_D_list, top_k=TOP_K_FEATURES,
                                              a_weight=SCORE_A_WEIGHT)
        for di, feats in enumerate(topk_per_d):
            specific_features[(p["id"], di)] = feats
    save_npz_atomic(ARTIFACTS_DIR / f"sae_encodings_L{LAYER}.npz", **encodings)
    save_json_atomic(
        ARTIFACTS_DIR / "specific_features.json",
        [{"pair_id": pid, "disambig_idx": di, "features": feats}
         for (pid, di), feats in specific_features.items()],
    )

    # =========================================================================
    # Stage 2.D — WikiText residuals + SAE encoding (for OOD shuffle)
    # =========================================================================
    print(f"\n[2.D] WikiText residual capture + SAE encoding")
    paragraphs = load_wikitext_paragraphs(tokenizer)
    para_token_lens = []
    for p_text in paragraphs:
        ids = tokenizer.encode(p_text, add_special_tokens=False)
        para_token_lens.append(len(ids))
    paragraph_hashes = [sha256_text(p) for p in paragraphs]

    save_json_atomic(ARTIFACTS_DIR / "wikitext_paragraphs.json",
                     paragraph_metadata(paragraphs, para_token_lens))

    last_token_top10: list[list[int]] = []
    for p_text in tqdm(paragraphs, desc="2.D WT residuals"):
        try:
            _, res = capture_all_position_residuals(
                model, tokenizer, p_text, layer=LAYER, max_len=WIKITEXT_MAX_TOKENS)
        except Exception as e:
            print(f"[2.D] paragraph failed: {type(e).__name__}: {e}")
            last_token_top10.append([])
            continue
        z_last = sae.encode(res[-1:].to(dtype=sae.W_enc.dtype)).squeeze(0)
        topk = torch.topk(z_last, k=TOP_K_FEATURES)
        ids_l = topk.indices.cpu().numpy().astype(int).tolist()
        vals_l = topk.values.float().cpu().numpy().tolist()
        kept = [int(fid) for fid, v in zip(ids_l, vals_l) if v > 0.0]
        last_token_top10.append(kept)
        del res
    save_json_atomic(
        ARTIFACTS_DIR / "wikitext_last_token_top10.json",
        [{"paragraph_idx": i, "top10_feature_ids": last_token_top10[i]}
         for i in range(len(last_token_top10))],
    )
    print(f"[2.D] saved last-token top-10 for {len(paragraphs)} WikiText paragraphs")

    # =========================================================================
    # Stage 2.E — six-condition ablation pipeline
    # =========================================================================
    print(f"\n[2.E] six-condition ablation pipeline on {len(detected)} pairs")
    self_rows = []; cross_rows = []; random_rows = []
    ambigqa_shuf_rows = []; wikitext_shuf_rows = []
    ambigqa_draws = []; wikitext_draws = []

    rng_main = random.Random(SEED)
    rng_wt   = random.Random(SHUFFLE_SEED)
    all_specific_keys = list(specific_features.keys())   # tuples (pair_id, di)

    for p in tqdm(detected, desc="2.E"):
        a_prompt = prompts_A[p["id"]]
        # baseline (one forward, scored against every disambig)
        base = baseline_top_logits(model, tokenizer, a_prompt, k=TOP_LOGITS_K)
        base_top = base["top_ids"]
        base_hits = []
        for d in p["disambigs"]:
            base_hits.append(hit_at_k(base_top, set(d["first_token_variants"]), 1))

        eligible_ambigqa_keys = [k for k in all_specific_keys if k[0] != p["id"]]

        for ab_i in range(len(p["disambigs"])):
            feats_self = specific_features.get((p["id"], ab_i), [])
            if not feats_self: continue

            # one targeted-ablation forward, scored against every target → self + sibling rows
            ab = forward_with_ablation(model, tokenizer, a_prompt, LAYER, sae,
                                        feature_ids=feats_self, k=TOP_LOGITS_K)
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

            # random-feature controls
            for r_idx in range(RANDOM_CONTROLS):
                rand_feats = rng_main.sample(range(sae.d_sae), len(feats_self))
                rc = forward_with_ablation(model, tokenizer, a_prompt, LAYER, sae,
                                            feature_ids=rand_feats, k=TOP_LOGITS_K)
                random_rows.append({
                    "pair_id": p["id"], "ablate_idx": ab_i, "control_idx": r_idx,
                    "n_features": len(rand_feats), "feature_ids": rand_feats,
                    "base_hit1": base_hits[ab_i],
                    "random_hit1": hit_at_k(rc["top_ids"], target_self, 1),
                })

            # AmbigQA-shuffle (3 draws of (P', k) with P' != p)
            for draw_idx in range(N_SHUFFLES_PER_PAIR):
                p_prime, k_idx = rng_main.choice(eligible_ambigqa_keys)
                feats_sh = specific_features.get((p_prime, k_idx), [])
                sh = forward_with_ablation(model, tokenizer, a_prompt, LAYER, sae,
                                            feature_ids=feats_sh, k=TOP_LOGITS_K)
                ambigqa_shuf_rows.append({
                    "pair_id": p["id"], "target_idx": ab_i, "draw_idx": draw_idx,
                    "shuffle_pair_id": p_prime, "shuffle_disambig_idx": int(k_idx),
                    "shuffle_n_features": len(feats_sh),
                    "shuffle_feature_ids": list(feats_sh),
                    "base_hit1": base_hits[ab_i],
                    "shuffled_hit1": hit_at_k(sh["top_ids"], target_self, 1),
                })
                ambigqa_draws.append({
                    "pair_id": p["id"], "target_idx": ab_i, "draw_idx": draw_idx,
                    "shuffle_pair_id": p_prime, "shuffle_disambig_idx": int(k_idx),
                })

            # WikiText-shuffle (3 draws of paragraph index from rng_wt)
            for draw_idx in range(N_SHUFFLES_PER_PAIR):
                para_idx = rng_wt.choice(range(len(paragraphs)))
                feats_wt = last_token_top10[para_idx]
                wt = forward_with_ablation(model, tokenizer, a_prompt, LAYER, sae,
                                            feature_ids=feats_wt, k=TOP_LOGITS_K)
                wikitext_shuf_rows.append({
                    "pair_id": p["id"], "target_idx": ab_i, "draw_idx": draw_idx,
                    "wikitext_paragraph_idx": para_idx,
                    "wikitext_paragraph_sha256_first16": paragraph_hashes[para_idx][:16],
                    "wikitext_n_features": len(feats_wt),
                    "wikitext_feature_ids": list(feats_wt),
                    "base_hit1": base_hits[ab_i],
                    "wt_shuffled_hit1": hit_at_k(wt["top_ids"], target_self, 1),
                })
                wikitext_draws.append({
                    "pair_id": p["id"], "target_idx": ab_i, "draw_idx": draw_idx,
                    "wikitext_paragraph_idx": para_idx,
                    "wikitext_paragraph_sha256_first16": paragraph_hashes[para_idx][:16],
                })

        torch.cuda.empty_cache(); gc.collect()

    # ---- Aggregate ----
    def mean(xs, key):
        xs = list(xs); return float(sum(r[key] for r in xs) / len(xs)) if xs else 0.0

    summary = {
        "n_self":  len(self_rows),
        "n_cross": len(cross_rows),
        "n_random": len(random_rows),
        "n_ambigqa_shuffled": len(ambigqa_shuf_rows),
        "n_wikitext_shuffled": len(wikitext_shuf_rows),
        "self_base_hit1":           mean(self_rows,           "base_hit1"),
        "self_ablate_hit1":         mean(self_rows,           "ablate_hit1"),
        "cross_base_hit1":          mean(cross_rows,          "base_hit1"),
        "cross_ablate_hit1":        mean(cross_rows,          "ablate_hit1"),
        "random_base_hit1":         mean(random_rows,         "base_hit1"),
        "random_ablate_hit1":       mean(random_rows,         "random_hit1"),
        "ambigqa_shuffled_base":    mean(ambigqa_shuf_rows,   "base_hit1"),
        "ambigqa_shuffled_ablate":  mean(ambigqa_shuf_rows,   "shuffled_hit1"),
        "wikitext_shuffled_base":   mean(wikitext_shuf_rows,  "base_hit1"),
        "wikitext_shuffled_ablate": mean(wikitext_shuf_rows,  "wt_shuffled_hit1"),
    }
    save_json_atomic(ARTIFACTS_DIR / "results_main.json", {
        "summary": summary,
        "self_rows": self_rows,
        "cross_rows": cross_rows,
        "random_rows": random_rows,
        "ambigqa_shuffled_rows": ambigqa_shuf_rows,
        "wikitext_shuffled_rows": wikitext_shuf_rows,
    })
    save_json_atomic(ARTIFACTS_DIR / "shuffle_draws_ambigqa.json",  ambigqa_draws)
    save_json_atomic(ARTIFACTS_DIR / "shuffle_draws_wikitext.json", wikitext_draws)

    # ---- Run metadata ----
    save_json_atomic(ARTIFACTS_DIR / "run_meta.json", {
        "model": MODEL_ID, "dtype": DTYPE_STR, "layer": LAYER,
        "sae_width": SAE_WIDTH, "sae_l0_sel": SAE_L0_SEL, "sae_meta": sae_meta,
        "top_k_features": TOP_K_FEATURES, "score_a_weight": SCORE_A_WEIGHT,
        "random_controls": RANDOM_CONTROLS,
        "n_shuffles_per_pair": N_SHUFFLES_PER_PAIR,
        "main_seed": SEED, "shuffle_seed": SHUFFLE_SEED,
        "n_input_candidates":      len(candidates),
        "n_after_collision_filter": len(after_collision),
        "n_detected":              len(detected),
        "n_self_pairs":            len(self_rows),
        "elapsed_seconds": round(time.time() - t_total, 2),
    })

    print(f"\n=== Stage 2 complete ===")
    print(f"  detected ambiguous pairs:    {len(detected)}")
    print(f"  self-pairs (n_self_pairs):   {summary['n_self']}")
    print(f"  cross-triples:               {summary['n_cross']}")
    print(f"  random / ambigqa-sh / wt-sh: {summary['n_random']} / "
          f"{summary['n_ambigqa_shuffled']} / {summary['n_wikitext_shuffled']}")
    print(f"\n  Headline rates (pooled, from per-instance arrays — see stage 3 for tests):")
    print(f"    self     : base={summary['self_base_hit1']:.4f}  abl={summary['self_ablate_hit1']:.4f}")
    print(f"    cross    : base={summary['cross_base_hit1']:.4f}  abl={summary['cross_ablate_hit1']:.4f}")
    print(f"    a-shuf   : base={summary['ambigqa_shuffled_base']:.4f}  abl={summary['ambigqa_shuffled_ablate']:.4f}")
    print(f"    wt-shuf  : base={summary['wikitext_shuffled_base']:.4f}  abl={summary['wikitext_shuffled_ablate']:.4f}")
    print(f"    random   : base={summary['random_base_hit1']:.4f}  abl={summary['random_ablate_hit1']:.4f}")
    print(f"\n  elapsed: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
