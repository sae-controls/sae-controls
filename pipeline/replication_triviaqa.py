"""Cross-substrate replication on TriviaQA aliases.

Tests whether the cluster-shared / answer-unique two-tier mechanism
replicates on a different paired-output substrate (TriviaQA-Web answer
aliases) at the same model + layer (Gemma-2-9B-IT @ L41) as the main
paper.

If T3 holds, the protocol generalizes from "different answers to the
same ambiguous question" (AmbigQA) to "different surface forms of the
same answer" (TriviaQA aliases). The decomposition becomes a stronger
claim: shared/unique structure at the SAE level reflects form-level
discriminability, not just topic-level discriminability.

Substrate construction:
  - Each TriviaQA-Web question Q has answer aliases (e.g., "USA",
    "United States", "U.S.").
  - A_prompt = build_prompt(Q)
  - D_i_prompt = A_prompt + alias_i (string concat; residual captured
    at last alias token = "model has just emitted alias_i").
  - Slot detection: greedy decode under A_prompt, find_best_match →
    committed alias index. Drop questions with no match.
  - Self-pair (P, D_i*) where i* = committed alias.
  - Sibling rows: ablate features picked for D_j (j ≠ i*) on A_prompt;
    target = first-token-variants of alias_{i*}.

Pipeline mirrors layer_sweep.py for L41:
  baseline / Targeted / Sibling / ShuffledTriviaQA / WikiText-shuffled
  / Random + T1-T4 + unique/shared decomposition.

Not re-run on this substrate (replication scope is headline T3 + the
unique/shared decomposition): polysemy partition, multi-metric
(KL/gen-flip), per-feature equivalence.

Outputs to artifacts/replication_triviaqa/.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc
import hashlib
import json
import random
import re
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
    ARTIFACTS_DIR, MODEL_ID, RANDOM_CONTROLS, SCORE_A_WEIGHT, SEED,
    SHUFFLE_SEED, TOP_K_FEATURES, TOP_LOGITS_K, WIKITEXT_MAX_TOKENS,
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
from src.wikitext import load_paragraphs as load_wikitext_paragraphs


# =============================================================================
# Constants
# =============================================================================
P24_DIR = ARTIFACTS_DIR / "replication_triviaqa"
LAYER = 41
PUBLISHED_TOP_K = 10
N_SHUFFLES_PER_PAIR = 3
N_SELF_PAIRS_CAP = 1103   # match AmbigQA n
GEN_MAX_NEW = 15

# TriviaQA filtering
TRIVIA_DATASET = "mandarjoshi/trivia_qa"
TRIVIA_CONFIG = "rc.web.nocontext"  # TriviaQA-Web; omit evidence docs (we only
                                     # use question+aliases; full rc.web pulls
                                     # ~10GB of evidence files we never read)
ANSWER_MAX_WORDS = 3              # match AmbigQA constraint
QUESTION_MIN_CHARS = 15
QUESTION_MAX_CHARS = 140
MAX_ALIASES_PER_QUESTION = 4      # match MAX_DISAMBIGS_PER_PAIR
MIN_ALIASES_AFTER_FILTER = 2
CANDIDATE_PRESAMPLE = 5000        # slot-detection candidate cap. TriviaQA-Web
                                   # has ~62k post-collision candidates; this
                                   # cap keeps slot detection bounded (~17 min
                                   # at 5 it/s on A40) while leaving ~4.5x
                                   # headroom for the 1103 final self-pair cap.

# L41 SAE pinned SHA (from reference_layer / layer_bookends)
L41_SAE_SHA = "65f7ea2b47901603fc09d5c5e8c8824b56fcfd630265b8832fca8da82a203013"


# =============================================================================
# Stage A: TriviaQA loading + filtering
# =============================================================================

def stage_a_load_filter(tokenizer):
    """Load TriviaQA-Web train+validation, filter by length and alias quality.

    Returns:
        list of dicts: {id, question, aliases (filtered list of strings)}
    """
    print(f"\n========== Stage A: load + filter TriviaQA ==========")
    from datasets import load_dataset

    print(f"[A] loading {TRIVIA_DATASET} / {TRIVIA_CONFIG} (train + validation)...")
    ds_train = load_dataset(TRIVIA_DATASET, TRIVIA_CONFIG, split="train")
    ds_val = load_dataset(TRIVIA_DATASET, TRIVIA_CONFIG, split="validation")
    print(f"[A] train: {len(ds_train)}, val: {len(ds_val)}")

    # Dedup by question_id
    seen_qids = set()
    raw = []
    for split_name, ds in [("train", ds_train), ("validation", ds_val)]:
        for ex in ds:
            qid = ex.get("question_id", "") or ex.get("question", "")
            if qid in seen_qids:
                continue
            seen_qids.add(qid)
            ans = ex.get("answer", {})
            aliases = ans.get("aliases", []) or []
            normalized_aliases = ans.get("normalized_aliases", []) or []
            value = ans.get("value", "")
            normalized_value = ans.get("normalized_value", "")
            # Combine — prefer raw aliases (case preserved). Add value too.
            all_aliases = list(aliases)
            if value and value not in all_aliases:
                all_aliases.insert(0, value)
            raw.append({
                "id": qid,
                "question": ex.get("question", ""),
                "aliases_raw": all_aliases,
                "split": split_name,
            })
    print(f"[A] after dedup: {len(raw)}")

    # Step 1: length + format filter on question
    fmt_ok = []
    for r in raw:
        q = r["question"].strip()
        if not (QUESTION_MIN_CHARS <= len(q) <= QUESTION_MAX_CHARS):
            continue
        r["question"] = q
        fmt_ok.append(r)
    print(f"[A] after question length filter ({QUESTION_MIN_CHARS}-{QUESTION_MAX_CHARS} chars): {len(fmt_ok)}")

    # Step 2: filter aliases — ≤3 words, ≥2 chars, dedupe (case-insensitive)
    short_alias_ok = []
    for r in fmt_ok:
        seen = set()
        kept = []
        for a in r["aliases_raw"]:
            a = a.strip()
            if len(a) < 2:
                continue
            if len(a.split()) > ANSWER_MAX_WORDS:
                continue
            key = a.lower()
            if key in seen:
                continue
            seen.add(key)
            kept.append(a)
            if len(kept) >= MAX_ALIASES_PER_QUESTION:
                break
        if len(kept) < MIN_ALIASES_AFTER_FILTER:
            continue
        r["aliases"] = kept
        short_alias_ok.append(r)
    print(f"[A] after alias short/dedupe filter (≥2 aliases ≤{ANSWER_MAX_WORDS} words): {len(short_alias_ok)}")

    # Step 3: within-question collision filter on first token (Gemma-2-9B-IT)
    # Aliases must have DISTINCT first-token sets (else hit@1 is undecidable).
    print(f"[A] applying within-question first-token collision filter...")
    after_collision = []
    for r in tqdm(short_alias_ok, desc="A collision"):
        ftvs = []
        for a in r["aliases"]:
            ftv_set = set(first_token_variants(tokenizer, a))
            ftvs.append(ftv_set)
        # Aliases that share any first-token-variant id collide.
        # We keep aliases greedily, dropping any whose ftv-set overlaps an
        # already-kept alias's ftv-set.
        kept_idx: list[int] = []
        kept_ftvs: list[set[int]] = []
        for i, ftv in enumerate(ftvs):
            if not ftv:
                continue
            collides = any(ftv & k for k in kept_ftvs)
            if collides:
                continue
            kept_idx.append(i)
            kept_ftvs.append(ftv)
        if len(kept_idx) < MIN_ALIASES_AFTER_FILTER:
            continue
        r["aliases"] = [r["aliases"][i] for i in kept_idx]
        r["alias_first_token_variants"] = [list(kept_ftvs[i])
                                            for i in range(len(kept_idx))]
        after_collision.append(r)
    print(f"[A] after collision filter: {len(after_collision)}")

    # Step 4: pre-sample to bound slot-detection compute. Slot detection on
    # 9B greedy-decode runs at ~5 it/s; capping to CANDIDATE_PRESAMPLE keeps
    # this stage at roughly ~17 min while leaving plenty of headroom for
    # the 1103 self-pair target.
    if len(after_collision) > CANDIDATE_PRESAMPLE:
        rng = random.Random(SEED)
        after_collision = rng.sample(after_collision, CANDIDATE_PRESAMPLE)
        print(f"[A] pre-sampled to {CANDIDATE_PRESAMPLE} candidates "
              f"(seed={SEED}) for tractable slot detection")

    return after_collision


# =============================================================================
# Stage B: slot detection
# =============================================================================

def stage_b_slot_detection(model, tokenizer, candidates):
    """Greedy-decode under A_prompt, identify committed alias.

    Returns:
        list of dicts: {id, question, aliases, alias_first_token_variants,
                         committed_idx, generation, match_strategy}
    """
    print(f"\n========== Stage B: slot detection ==========")
    detected = []
    failures = []
    for r in tqdm(candidates, desc="B slot detect"):
        a_prompt = build_prompt(tokenizer, r["question"])
        gen = greedy_decode(model, tokenizer, a_prompt, max_new=GEN_MAX_NEW)
        match = find_best_match(gen, r["aliases"])
        if match.cand_idx < 0:
            failures.append({
                "id": r["id"], "question": r["question"],
                "aliases": r["aliases"], "generation": gen,
            })
            continue
        detected.append({
            "id": r["id"],
            "question": r["question"],
            "aliases": r["aliases"],
            "alias_first_token_variants": r["alias_first_token_variants"],
            "committed_idx": int(match.cand_idx),
            "match_strategy": match.strategy,
            "match_text": match.matched_text,
            "generation": gen,
        })
    print(f"[B] detected: {len(detected)} / {len(candidates)} "
          f"(failure rate {len(failures)/max(len(candidates),1)*100:.1f}%)")
    return detected, failures


# =============================================================================
# Stage C: cap at N_SELF_PAIRS_CAP
# =============================================================================

def stage_c_cap_self_pairs(detected):
    print(f"\n========== Stage C: cap to {N_SELF_PAIRS_CAP} self-pairs ==========")
    if len(detected) <= N_SELF_PAIRS_CAP:
        print(f"[C] {len(detected)} ≤ {N_SELF_PAIRS_CAP}; using all")
        return detected
    rng = random.Random(SEED)
    sampled = rng.sample(detected, N_SELF_PAIRS_CAP)
    print(f"[C] sampled {len(sampled)} from {len(detected)} (seed={SEED})")
    return sampled


# =============================================================================
# Stage D: residual capture + SAE encoding
# =============================================================================

def stage_d_capture_and_encode(model, tokenizer, sae, detected):
    """Capture L41 residuals for A and each alias appendage, SAE-encode."""
    print(f"\n========== Stage D: residual capture at L{LAYER} ==========")
    residuals = {}
    for r in tqdm(detected, desc="D A+alias residuals"):
        a_prompt = build_prompt(tokenizer, r["question"])
        residuals[f"A__{r['id']}"] = capture_last_token_residual(
            model, tokenizer, a_prompt, layer=LAYER,
        )
        for ai, alias in enumerate(r["aliases"]):
            d_prompt = a_prompt + alias
            residuals[f"D__{r['id']}__{ai}"] = capture_last_token_residual(
                model, tokenizer, d_prompt, layer=LAYER,
            )
    save_npz_atomic(P24_DIR / f"residuals_L{LAYER}_triviaqa.npz", **residuals)
    n_total = len(residuals)
    n_A = sum(1 for k in residuals if k.startswith("A__"))
    print(f"[D] captured {n_total} residuals ({n_A} A + {n_total - n_A} alias)")

    print(f"[D] SAE-encoding...")
    encodings = {}
    for k, r in residuals.items():
        x = torch.from_numpy(r).to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        z = sae.encode(x.unsqueeze(0)).squeeze(0)
        encodings[k] = z.float().cpu().numpy()
    save_npz_atomic(P24_DIR / f"sae_encodings_L{LAYER}_triviaqa.npz", **encodings)
    return encodings


# =============================================================================
# Stage E: feature picking per alias
# =============================================================================

def stage_e_pick_features(encodings, detected, sae):
    print(f"\n========== Stage E: pick Targeted top-{PUBLISHED_TOP_K} per alias ==========")
    enc_torch = {
        k: torch.from_numpy(v).to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        for k, v in encodings.items()
    }
    specific_features = {}   # (qid, alias_idx) -> [feat_ids]
    for r in detected:
        z_A = enc_torch[f"A__{r['id']}"]
        z_D_list = [enc_torch[f"D__{r['id']}__{ai}"]
                    for ai in range(len(r["aliases"]))]
        topk_per_d = score_specific_features(
            z_A, z_D_list, top_k=PUBLISHED_TOP_K, a_weight=SCORE_A_WEIGHT,
        )
        for ai, feats in enumerate(topk_per_d):
            specific_features[(r["id"], ai)] = feats
    save_json_atomic(P24_DIR / "specific_features_triviaqa.json",
                     [{"pair_id": pid, "disambig_idx": ai, "features": f}
                      for (pid, ai), f in specific_features.items()])
    n_full = sum(1 for f in specific_features.values() if len(f) == PUBLISHED_TOP_K)
    print(f"[E] picked features for {len(specific_features)} (qid, alias_idx) "
          f"({n_full} with full top-{PUBLISHED_TOP_K})")
    return specific_features, enc_torch


# =============================================================================
# Stage F: WikiText residual capture + per-paragraph top-10
# =============================================================================

def stage_f_wikitext(model, tokenizer, sae, paragraphs):
    print(f"\n========== Stage F: WikiText last-token at L{LAYER} ==========")
    n_paragraphs = len(paragraphs)
    wt_lasttok_z = np.zeros((n_paragraphs, sae.d_sae), dtype=np.float32)
    for i, p_text in enumerate(tqdm(paragraphs, desc="F WT")):
        try:
            _, res = capture_all_position_residuals(
                model, tokenizer, p_text, layer=LAYER, max_len=WIKITEXT_MAX_TOKENS,
            )
            z_last = sae.encode(res[-1:].to(dtype=sae.W_enc.dtype)).squeeze(0)
            wt_lasttok_z[i] = z_last.float().cpu().numpy()
            del res, z_last
        except Exception as e:
            print(f"[F] paragraph {i} failed: {type(e).__name__}: {e}")
        if i % 50 == 0:
            torch.cuda.empty_cache(); gc.collect()
    wt_top10 = []
    for i in range(n_paragraphs):
        z_p = wt_lasttok_z[i]
        positives = np.where(z_p > 0)[0]
        if len(positives) == 0:
            kept = []
        else:
            sorted_idx = positives[np.argsort(-z_p[positives])]
            kept = [int(f) for f in sorted_idx[:PUBLISHED_TOP_K]]
        wt_top10.append({"paragraph_idx": i, "top10_feature_ids": kept})
    return wt_top10


# =============================================================================
# Stage G: shuffle draws (per-pair, deterministic)
# =============================================================================

def stage_g_shuffle_draws(detected, n_paragraphs):
    """Pre-compute (pair_id, alias_idx, draw_idx) -> shuffle target.

    For ShuffledTriviaQA: pick a random OTHER (qid', alias_idx') with feats != self.
    For WikiText-shuffled: pick a random paragraph index.
    """
    print(f"\n========== Stage G: shuffle draws ==========")
    # Build pool of all valid (qid, alias_idx) tuples
    all_keys = []
    for r in detected:
        for ai in range(len(r["aliases"])):
            all_keys.append((r["id"], ai))
    rng_t = random.Random(SEED)
    rng_w = random.Random(SHUFFLE_SEED)

    trivia_draws = []
    wt_draws = []
    for r in detected:
        for ai in range(len(r["aliases"])):
            for d in range(N_SHUFFLES_PER_PAIR):
                # Trivia shuffle: pick another (qid', ai')
                while True:
                    target = rng_t.choice(all_keys)
                    if target[0] != r["id"]:
                        break
                trivia_draws.append({
                    "pair_id": r["id"], "target_idx": ai, "draw_idx": d,
                    "shuffle_pair_id": target[0],
                    "shuffle_disambig_idx": int(target[1]),
                })
                # WT shuffle
                wt_draws.append({
                    "pair_id": r["id"], "target_idx": ai, "draw_idx": d,
                    "wikitext_paragraph_idx": rng_w.randrange(n_paragraphs),
                })
    print(f"[G] {len(trivia_draws)} trivia draws, {len(wt_draws)} WT draws")
    return trivia_draws, wt_draws


def stage_g_random_features(detected, sae):
    """Pre-compute (pair_id, ablate_idx, control_idx) -> random feature ids.
    Seed = pair_id-derived (matches AmbigQA protocol so cross-substrate
    comparisons are not seed-confounded)."""
    rand_per_pair = []
    for r in detected:
        for ai in range(len(r["aliases"])):
            seed = int(hashlib.sha256(f"{r['id']}|{ai}".encode()).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            for c in range(RANDOM_CONTROLS):
                feats = sorted(rng.sample(range(sae.d_sae), TOP_K_FEATURES))
                rand_per_pair.append({
                    "pair_id": r["id"], "ablate_idx": ai, "control_idx": c,
                    "feature_ids": feats,
                })
    return rand_per_pair


# =============================================================================
# Stage H: six-condition pipeline
# =============================================================================

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


def stage_h_six_conditions(model, tokenizer, sae, detected, specific_features,
                             trivia_draws, wt_draws, rand_features, wt_top10):
    """Run baseline + 5 ablation conditions on each (qid, alias_i*).

    Self-pair = (qid, committed_idx). Ablate features for committed alias.
    Sibling rows: ablate features for non-committed aliases (j != i*),
                  hit@1 target = first-tokens of committed alias.
    """
    print(f"\n========== Stage H: six-condition pipeline ==========")
    self_rows = []      # (qid, i=committed) targeted ablation
    cross_rows = []     # (qid, j != committed) sibling
    random_rows = []    # random feature ablation
    trivia_shuf_rows = []
    wt_shuf_rows = []

    trivia_by_draw = {(d["pair_id"], d["target_idx"], d["draw_idx"]):
                       (d["shuffle_pair_id"], d["shuffle_disambig_idx"])
                       for d in trivia_draws}
    wt_by_draw = {(d["pair_id"], d["target_idx"], d["draw_idx"]):
                   d["wikitext_paragraph_idx"]
                   for d in wt_draws}
    rand_by_pair = {(r["pair_id"], r["ablate_idx"], r["control_idx"]):
                     r["feature_ids"]
                     for r in rand_features}

    for r in tqdm(detected, desc="H 6-cond"):
        a_prompt = build_prompt(tokenizer, r["question"])
        i_star = r["committed_idx"]
        target_self = set(r["alias_first_token_variants"][i_star])

        # Baseline
        base = baseline_top_logits(model, tokenizer, a_prompt, k=TOP_LOGITS_K)
        base_top = base["top_ids"]
        base_hit_self = hit_at_k(base_top, target_self, 1)

        # ---------------------- Targeted: ablate features[i*] ----------------
        feats_self = specific_features.get((r["id"], i_star), [])
        if not feats_self:
            continue
        ab_self = forward_with_ablation(
            model, tokenizer, a_prompt, LAYER, sae,
            feature_ids=feats_self, k=TOP_LOGITS_K,
        )
        self_rows.append({
            "pair_id": r["id"], "ablate_idx": i_star, "target_idx": i_star,
            "is_self": True,
            "n_features": len(feats_self),
            "feature_ids": list(feats_self),
            "base_hit1": base_hit_self,
            "ablate_hit1": hit_at_k(ab_self["top_ids"], target_self, 1),
        })

        # ---------------------- Sibling: ablate features[j] for j != i* ------
        for j in range(len(r["aliases"])):
            if j == i_star:
                continue
            feats_j = specific_features.get((r["id"], j), [])
            if not feats_j:
                continue
            ab_j = forward_with_ablation(
                model, tokenizer, a_prompt, LAYER, sae,
                feature_ids=feats_j, k=TOP_LOGITS_K,
            )
            cross_rows.append({
                "pair_id": r["id"], "ablate_idx": j, "target_idx": i_star,
                "is_self": False,
                "n_features": len(feats_j),
                "feature_ids": list(feats_j),
                "base_hit1": base_hit_self,
                "ablate_hit1": hit_at_k(ab_j["top_ids"], target_self, 1),
            })

        # ---------------------- Random: ablate random feat ids ---------------
        for c in range(RANDOM_CONTROLS):
            rand_feats = rand_by_pair.get((r["id"], i_star, c))
            if rand_feats is None:
                continue
            rc = forward_with_ablation(
                model, tokenizer, a_prompt, LAYER, sae,
                feature_ids=rand_feats, k=TOP_LOGITS_K,
            )
            random_rows.append({
                "pair_id": r["id"], "ablate_idx": i_star, "control_idx": c,
                "n_features": len(rand_feats),
                "base_hit1": base_hit_self,
                "random_hit1": hit_at_k(rc["top_ids"], target_self, 1),
            })

        # ---------------------- Shuffled-Trivia ------------------------------
        for d_idx in range(N_SHUFFLES_PER_PAIR):
            tup = trivia_by_draw.get((r["id"], i_star, d_idx))
            if tup is None:
                continue
            p_prime, k_idx = tup
            feats_sh = specific_features.get((p_prime, k_idx), [])
            if not feats_sh:
                continue
            sh = forward_with_ablation(
                model, tokenizer, a_prompt, LAYER, sae,
                feature_ids=feats_sh, k=TOP_LOGITS_K,
            )
            trivia_shuf_rows.append({
                "pair_id": r["id"], "target_idx": i_star, "draw_idx": d_idx,
                "shuffle_pair_id": p_prime, "shuffle_disambig_idx": int(k_idx),
                "n_features": len(feats_sh),
                "base_hit1": base_hit_self,
                "shuffled_hit1": hit_at_k(sh["top_ids"], target_self, 1),
            })

        # ---------------------- WikiText-shuffled ----------------------------
        for d_idx in range(N_SHUFFLES_PER_PAIR):
            para_idx = wt_by_draw.get((r["id"], i_star, d_idx))
            if para_idx is None:
                continue
            feats_wt = wt_top10[para_idx]["top10_feature_ids"]
            if not feats_wt:
                continue
            wt = forward_with_ablation(
                model, tokenizer, a_prompt, LAYER, sae,
                feature_ids=feats_wt, k=TOP_LOGITS_K,
            )
            wt_shuf_rows.append({
                "pair_id": r["id"], "target_idx": i_star, "draw_idx": d_idx,
                "wikitext_paragraph_idx": para_idx,
                "n_features": len(feats_wt),
                "base_hit1": base_hit_self,
                "wt_shuffled_hit1": hit_at_k(wt["top_ids"], target_self, 1),
            })

        torch.cuda.empty_cache()

    # Six-condition aggregate
    def _mean(rows, k):
        return float(sum(r[k] for r in rows) / len(rows)) if rows else 0.0
    six_cond_summary = {
        "n_self": len(self_rows),
        "n_cross": len(cross_rows),
        "n_random": len(random_rows),
        "n_trivia_shuf": len(trivia_shuf_rows),
        "n_wt_shuf": len(wt_shuf_rows),
        "self_base_hit1":   _mean(self_rows, "base_hit1"),
        "self_ablate_hit1": _mean(self_rows, "ablate_hit1"),
        "cross_base_hit1":  _mean(cross_rows, "base_hit1"),
        "cross_ablate_hit1": _mean(cross_rows, "ablate_hit1"),
        "random_ablate_hit1": _mean(random_rows, "random_hit1"),
        "trivia_shuf_ablate_hit1": _mean(trivia_shuf_rows, "shuffled_hit1"),
        "wt_shuf_ablate_hit1": _mean(wt_shuf_rows, "wt_shuffled_hit1"),
    }
    save_json_atomic(P24_DIR / "results_main_triviaqa.json", {
        "summary": six_cond_summary,
        "self_rows": self_rows,
        "cross_rows": cross_rows,
        "random_rows": random_rows,
        "triviaqa_shuffled_rows": trivia_shuf_rows,
        "wikitext_shuffled_rows": wt_shuf_rows,
    })
    print(f"[H] six-condition counts: {six_cond_summary['n_self']} self, "
          f"{six_cond_summary['n_cross']} cross, "
          f"{six_cond_summary['n_random']} rand, "
          f"{six_cond_summary['n_trivia_shuf']} trivia-shuf, "
          f"{six_cond_summary['n_wt_shuf']} WT-shuf")
    return self_rows, cross_rows, random_rows, trivia_shuf_rows, wt_shuf_rows, six_cond_summary


# =============================================================================
# Stage I: T1-T4 + summary
# =============================================================================

def stage_i_paired_tests(self_rows, cross_rows, random_rows,
                           trivia_shuf_rows, wt_shuf_rows):
    print(f"\n========== Stage I: T1-T4 paired tests ==========")
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
    a_shuf = per_pair_means(
        trivia_shuf_rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["shuffled_hit1"],
    )
    rand = per_pair_means(
        random_rows,
        key_fn=lambda r: (r["pair_id"], r["ablate_idx"]),
        value_fn=lambda r: r["random_hit1"],
    )
    base = {(r["pair_id"], r["target_idx"]): r["base_hit1"] for r in self_rows}
    targ = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"] for r in self_rows}

    keys_all = sorted(k for k in base
                       if k in sib and k in wt and k in a_shuf and k in rand)
    n = len(keys_all)
    if n == 0:
        # Pairs without sibling rows still have valid Targeted vs Random/WT
        keys_all = sorted(k for k in base if k in wt and k in rand)
        n = len(keys_all)
    print(f"[I] paired-test n (all conditions present): {len(sorted(k for k in base if k in sib and k in wt and k in a_shuf and k in rand))}")
    print(f"[I] paired-test n (Targeted/WT/Random present): {len(sorted(k for k in base if k in wt and k in rand))}")

    keys_main = sorted(k for k in base if k in wt and k in rand)
    B = np.array([base[k] for k in keys_main])
    T = np.array([targ[k] for k in keys_main])
    WSh = np.array([wt[k] for k in keys_main])
    Rnd = np.array([rand[k] for k in keys_main])

    # T1, T3, T4 require their corresponding rows
    keys_sib = sorted(k for k in base if k in sib and k in wt and k in a_shuf and k in rand)
    Sib = np.array([sib[k] for k in keys_sib]) if keys_sib else np.array([])
    ASh = np.array([a_shuf[k] for k in keys_sib]) if keys_sib else np.array([])
    WSh_t3 = np.array([wt[k] for k in keys_sib]) if keys_sib else np.array([])

    tests = {
        "T1_targeted_vs_random": _drop_vs_base_stats(T, Rnd),
        "T2_sibling_vs_ashuf":   _t3_stats(Sib, ASh) if len(Sib) > 1 else None,
        "T3_sibling_vs_wtshuf":  _t3_stats(Sib, WSh_t3) if len(Sib) > 1 else None,
        "T4_ashuf_vs_wtshuf":    _t3_stats(ASh, WSh_t3) if len(ASh) > 1 else None,
    }
    headline_table = {
        "Baseline":          float(B.mean()) if n else 0.0,
        "Targeted":          float(T.mean()) if n else 0.0,
        "Sibling":           float(Sib.mean()) if len(Sib) else 0.0,
        "ShuffledTriviaQA":  float(ASh.mean()) if len(ASh) else 0.0,
        "WikiTextShuffled":  float(WSh.mean()) if n else 0.0,
        "Random":            float(Rnd.mean()) if n else 0.0,
    }
    print(f"[I] headline rates:")
    for k, v in headline_table.items():
        print(f"    {k:<20}  {v:.4f}")
    if tests["T3_sibling_vs_wtshuf"]:
        t3 = tests["T3_sibling_vs_wtshuf"]
        print(f"    T3: Δ={t3['delta_pp']:+.2f}pp Wlx p={t3['wilcoxon_p']:.3g}")
    return headline_table, tests


# =============================================================================
# Stage J: unique/shared decomposition (committed-alias only)
# =============================================================================

def stage_j_unique_shared(model, tokenizer, sae, detected, specific_features,
                            self_rows):
    print(f"\n========== Stage J: unique/shared decomposition ==========")
    pub_set = {(qid, ai): set(feats)
               for (qid, ai), feats in specific_features.items()}

    # For each detected question, committed alias = i*; targeted = features[i*];
    # shared = features[i*] ∩ ∪_{j != i*} features[j]; unique = targeted - shared.
    sh_un = {}
    for r in detected:
        i_star = r["committed_idx"]
        ti = pub_set.get((r["id"], i_star))
        if not ti:
            continue
        sib_union = set()
        for j in range(len(r["aliases"])):
            if j == i_star:
                continue
            sib_union |= pub_set.get((r["id"], j), set())
        shared = ti & sib_union
        unique = ti - shared
        sh_un[(r["id"], i_star)] = {
            "shared": shared, "unique": unique, "targeted": ti,
        }

    base_lookup = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                   for r in self_rows}
    targ_lookup = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                   for r in self_rows}

    sh_un_rows = []
    for r in tqdm(detected, desc="J sh+un"):
        a_prompt = build_prompt(tokenizer, r["question"])
        i_star = r["committed_idx"]
        key = (r["id"], i_star)
        if key not in sh_un or key not in base_lookup:
            continue
        target = set(r["alias_first_token_variants"][i_star])
        base = base_lookup[key]
        if sh_un[key]["shared"]:
            ab = forward_with_ablation(
                model, tokenizer, a_prompt, LAYER, sae,
                feature_ids=list(sh_un[key]["shared"]), k=TOP_LOGITS_K,
            )
            shared_hit = hit_at_k(ab["top_ids"], target, 1)
        else:
            shared_hit = base
        if sh_un[key]["unique"]:
            ab = forward_with_ablation(
                model, tokenizer, a_prompt, LAYER, sae,
                feature_ids=list(sh_un[key]["unique"]), k=TOP_LOGITS_K,
            )
            unique_hit = hit_at_k(ab["top_ids"], target, 1)
        else:
            unique_hit = base
        sh_un_rows.append({
            "pair_id": r["id"], "target_idx": i_star,
            "n_targeted": len(sh_un[key]["targeted"]),
            "n_shared": len(sh_un[key]["shared"]),
            "n_unique": len(sh_un[key]["unique"]),
            "base_hit1": base,
            "targeted_hit1": targ_lookup[key],
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
    targ_d = float((Tsh.mean() - Bsh.mean()) * 100)
    sh_d = float((Shsh.mean() - Bsh.mean()) * 100)
    un_d = float((Unsh.mean() - Bsh.mean()) * 100)
    res = targ_d - (sh_d + un_d)
    pct = lambda x: (x / targ_d * 100) if abs(targ_d) > 1e-9 else float("nan")

    decomp = {
        "n": len(keys_sh),
        "table": {
            "Baseline":    {"hit1": float(Bsh.mean()), "delta_pp": 0.0,
                             "pct_of_targeted": 0.0},
            "Targeted":    {"hit1": float(Tsh.mean()), "delta_pp": targ_d,
                             "pct_of_targeted": 100.0},
            "Shared-only": {"hit1": float(Shsh.mean()), "delta_pp": sh_d,
                             "pct_of_targeted": float(pct(sh_d))},
            "Unique-only": {"hit1": float(Unsh.mean()), "delta_pp": un_d,
                             "pct_of_targeted": float(pct(un_d))},
            "Sum (Sh+Un)": {"delta_pp": float(sh_d + un_d),
                             "pct_of_targeted": float(pct(sh_d + un_d))},
            "Residual":    {"delta_pp": float(res),
                             "pct_of_targeted": float(pct(res))},
        },
        "tests": {
            "shared_vs_baseline": _drop_vs_base_stats(Shsh, Bsh),
            "unique_vs_baseline": _drop_vs_base_stats(Unsh, Bsh),
            "targeted_vs_unique": _drop_vs_base_stats(Tsh, Unsh),
        },
    }
    save_json_atomic(P24_DIR / "unique_shared_decomp_triviaqa.json",
                     {"decomposition": decomp, "rows": sh_un_rows})
    print(f"[J] Targeted Δ={targ_d:+.2f}; Shared Δ={sh_d:+.2f} ({pct(sh_d):+.1f}%); "
          f"Unique Δ={un_d:+.2f} ({pct(un_d):+.1f}%); Residual {res:+.2f} ({pct(res):+.1f}%)")
    return decomp


# =============================================================================
# Stage K: cross-substrate summary
# =============================================================================

def stage_k_summary(headline_table, tests, decomp, six_cond_summary, sae_meta,
                     filtering_log):
    print(f"\n========== Stage K: cross-substrate summary ==========")
    # Load AmbigQA L41 reference
    l41_summary = json.load(open(ARTIFACTS_DIR / "reference_layer" / "L41_summary_table.json"))
    a_n = l41_summary["n_self_pairs"]
    a_h = l41_summary["headline_hit1"]
    a_t3 = l41_summary["T3"]
    a_decomp = l41_summary["decomposition_hit1_from_p21b"]

    cross_table = [
        {
            "substrate": "AmbigQA",
            "model": "google/gemma-2-9b-it",
            "layer": 41,
            "n_self_pairs": a_n,
            "baseline_hit1": a_h["Baseline"],
            "targeted_delta_pp": a_decomp["Targeted"]["delta_pp"],
            "sibling_hit1": a_h["Sibling"],
            "wt_shuf_hit1": a_h["WikiTextShuffled"],
            "T3_delta_pp": a_t3["delta_pp"],
            "T3_wilcoxon_p": a_t3["wilcoxon_p"],
            "shared_only_delta_pp": a_decomp["Shared-only"]["delta_pp"],
            "unique_only_delta_pp": a_decomp["Unique-only"]["delta_pp"],
            "shared_pct": (a_decomp["Shared-only"]["delta_pp"]
                            / a_decomp["Targeted"]["delta_pp"] * 100),
            "unique_pct": (a_decomp["Unique-only"]["delta_pp"]
                            / a_decomp["Targeted"]["delta_pp"] * 100),
        },
        {
            "substrate": "TriviaQA aliases",
            "model": "google/gemma-2-9b-it",
            "layer": 41,
            "n_self_pairs": six_cond_summary["n_self"],
            "baseline_hit1": headline_table["Baseline"],
            "targeted_delta_pp": decomp["table"]["Targeted"]["delta_pp"],
            "sibling_hit1": headline_table["Sibling"],
            "wt_shuf_hit1": headline_table["WikiTextShuffled"],
            "T3_delta_pp": tests["T3_sibling_vs_wtshuf"]["delta_pp"]
                            if tests.get("T3_sibling_vs_wtshuf") else None,
            "T3_wilcoxon_p": tests["T3_sibling_vs_wtshuf"]["wilcoxon_p"]
                              if tests.get("T3_sibling_vs_wtshuf") else None,
            "shared_only_delta_pp": decomp["table"]["Shared-only"]["delta_pp"],
            "unique_only_delta_pp": decomp["table"]["Unique-only"]["delta_pp"],
            "shared_pct": decomp["table"]["Shared-only"]["pct_of_targeted"],
            "unique_pct": decomp["table"]["Unique-only"]["pct_of_targeted"],
        },
    ]

    summary = {
        "model": MODEL_ID,
        "layer": LAYER,
        "substrate": "TriviaQA-Web aliases",
        "sae_meta": sae_meta,
        "filtering_log": filtering_log,
        "headline_hit1": headline_table,
        "tests": tests,
        "decomposition": decomp,
        "cross_substrate_table": cross_table,
        "six_condition_counts": {
            "n_self": six_cond_summary["n_self"],
            "n_cross": six_cond_summary["n_cross"],
            "n_random": six_cond_summary["n_random"],
            "n_trivia_shuf": six_cond_summary["n_trivia_shuf"],
            "n_wt_shuf": six_cond_summary["n_wt_shuf"],
        },
    }
    save_json_atomic(P24_DIR / "summary.json", summary)
    return summary


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    P24_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"[setup] loading {MODEL_ID}...")
    tokenizer, model = load_model()
    print(f"[setup] loading L{LAYER} SAE...")
    sae, sae_meta = load_sae(layer=LAYER)
    print(f"[setup] L{LAYER} SAE sha256: {sae_meta['sha256'][:16]}...")
    if sae_meta["sha256"] != L41_SAE_SHA:
        raise RuntimeError(f"L41 SAE SHA mismatch: got {sae_meta['sha256']}, "
                           f"expected {L41_SAE_SHA}")
    print(f"[setup] SAE SHA matches the paper-pinned reference")

    # Stages A-C: filter + slot-detect. Resume from saved artifacts if present.
    detected_path = P24_DIR / "detected_pairs_triviaqa.json"
    log_path = P24_DIR / "triviaqa_filtering_log.json"
    if detected_path.exists() and log_path.exists():
        detected = json.load(open(detected_path))
        filtering_log = json.load(open(log_path))
        n_self_pairs = len(detected)
        print(f"[resume] loaded {n_self_pairs} detected pairs from "
              f"{detected_path.name}; skipping Stages A-C")
        print(f"[filtering] {filtering_log}")
    else:
        candidates = stage_a_load_filter(tokenizer)
        n_after_presample = len(candidates)
        detected, failures = stage_b_slot_detection(model, tokenizer, candidates)
        save_json_atomic(P24_DIR / "slot_detection_failures_triviaqa.json", failures)
        n_after_slot = len(detected)
        detected = stage_c_cap_self_pairs(detected)
        n_self_pairs = len(detected)
        save_json_atomic(detected_path, detected)
        filtering_log = {
            "n_after_presample": n_after_presample,
            "candidate_presample_cap": CANDIDATE_PRESAMPLE,
            "n_after_slot_detection": n_after_slot,
            "n_self_pairs_final": n_self_pairs,
            "n_slot_failures": len(failures),
        }
        save_json_atomic(log_path, filtering_log)
        print(f"[filtering] presampled: {n_after_presample} → "
              f"slot-detected: {n_after_slot} → final: {n_self_pairs}")

    # Stage D: residuals + encodings
    encodings = stage_d_capture_and_encode(model, tokenizer, sae, detected)

    # Stage E: pick features
    specific_features, enc_torch = stage_e_pick_features(encodings, detected, sae)

    # Stage F: WikiText
    paragraphs = load_wikitext_paragraphs(tokenizer)
    wt_top10 = stage_f_wikitext(model, tokenizer, sae, paragraphs)

    # Stage G: shuffle draws + random features
    trivia_draws, wt_draws = stage_g_shuffle_draws(detected, len(paragraphs))
    save_json_atomic(P24_DIR / "shuffle_draws_triviaqa.json", trivia_draws)
    save_json_atomic(P24_DIR / "shuffle_draws_wikitext.json", wt_draws)
    rand_features = stage_g_random_features(detected, sae)

    # Stage H: six-condition pipeline
    self_rows, cross_rows, random_rows, trivia_shuf_rows, wt_shuf_rows, six_cond_summary = \
        stage_h_six_conditions(
            model, tokenizer, sae, detected, specific_features,
            trivia_draws, wt_draws, rand_features, wt_top10,
        )

    # Stage I: T1-T4 + headline table
    headline_table, tests = stage_i_paired_tests(
        self_rows, cross_rows, random_rows, trivia_shuf_rows, wt_shuf_rows,
    )

    # Stage J: unique/shared decomposition
    decomp = stage_j_unique_shared(
        model, tokenizer, sae, detected, specific_features, self_rows,
    )

    # Stage K: cross-substrate summary
    summary = stage_k_summary(
        headline_table, tests, decomp, six_cond_summary, sae_meta, filtering_log,
    )

    # run_meta
    elapsed = time.time() - t_total
    save_json_atomic(P24_DIR / "run_meta.json", {
        "substrate": "TriviaQA-Web aliases",
        "model": MODEL_ID,
        "layer": LAYER,
        "sae_meta": sae_meta,
        "filtering_log": filtering_log,
        "elapsed_seconds": round(elapsed, 2),
    })

    # Cleanup: delete heavy residual NPZ + SAE blob
    try:
        (P24_DIR / f"residuals_L{LAYER}_triviaqa.npz").unlink(missing_ok=True)
    except Exception:
        pass
    try:
        local_path = sae_meta.get("local_path")
        if local_path:
            real = Path(local_path).resolve()
            if real.exists():
                real.unlink()
                print(f"[cleanup] deleted L{LAYER} SAE blob from HF cache")
    except Exception:
        pass

    print(f"\n=== Stage 2.4 complete ===")
    print(f"  total elapsed: {elapsed/60:.1f} min")
    print(f"  -> {P24_DIR}/summary.json")
    if tests.get("T3_sibling_vs_wtshuf"):
        t3 = tests["T3_sibling_vs_wtshuf"]
        print(f"  T3 (Sibling vs WT-shuf): Δ={t3['delta_pp']:+.2f}pp, p={t3['wilcoxon_p']:.3g}")
    print(f"  Targeted Δ={decomp['table']['Targeted']['delta_pp']:+.2f}pp")
    print(f"  Shared-only Δ={decomp['table']['Shared-only']['delta_pp']:+.2f}pp "
          f"({decomp['table']['Shared-only']['pct_of_targeted']:.1f}%)")
    print(f"  Unique-only Δ={decomp['table']['Unique-only']['delta_pp']:+.2f}pp "
          f"({decomp['table']['Unique-only']['pct_of_targeted']:.1f}%)")


if __name__ == "__main__":
    main()
