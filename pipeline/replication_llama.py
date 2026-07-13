"""Cross-architecture replication on Llama-3.1-8B-Instruct.

Tests whether the cluster-shared / answer-unique two-tier mechanism
replicates on a different model architecture. AmbigQA substrate; same
scope as the Gemma-2-2B replication (headline T3 + decomposition only;
no polysemy / multi-metric / per-feature equivalence).

Self-contained: defines its own Llama loader, SAE wrapper, and prompt
builder so that src/ stays Gemma-pinned. Reuses src/features.py,
src/slot_detection.py, src/hooks.py — the hooks abstract over SAE via
duck typing on .encode / .decode / .W_enc.

Cross-architecture comparison:
  - Gemma-2-9B-IT @ L41 (paper reference, AmbigQA, n=1103)
  - Gemma-2-2B-IT @ L25 (AmbigQA, n=655)
  - Llama-3.1-8B-Instruct @ L31 (this script, AmbigQA)

Key technical deltas vs the Gemma-2-2B replication (which mirrored Gemma 9B onto 2B):
  - Different chat template (Llama-3 format, not Gemma).
  - Llama Scope SAE has dataset-wise activation normalization (by 74.5):
      z = jumprelu(W_enc · (x / 74.5) + b_enc)
      x_recon = (W_dec · z + b_dec) · 74.5
    The hooks expect identity-norm SAEs; we wrap encode/decode to apply
    the norm internally so the error-preserving splice in src.hooks
    works unchanged.
  - SAE was trained on Llama-3.1-8B-BASE, not Instruct. Same architecture,
    different fine-tuning. Disclosed in the paper's Limitations section (Sec. 7).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc
import hashlib
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
    ARTIFACTS_DIR, RANDOM_CONTROLS, SCORE_A_WEIGHT, SEED, SHUFFLE_SEED,
    SYSTEM_PREAMBLE, TOP_K_FEATURES, TOP_LOGITS_K, WIKITEXT_MAX_TOKENS,
)
from src.features import score_specific_features
from src.hooks import (
    baseline_top_logits, capture_all_position_residuals,
    capture_last_token_residual, forward_with_ablation, greedy_decode,
    hit_at_k,
)
from src.io_utils import save_json_atomic, save_npz_atomic
from src.slot_detection import find_best_match, first_token_variants
from src.wikitext import load_paragraphs as load_wikitext_paragraphs


# =============================================================================
# Constants
# =============================================================================
P25_DIR = ARTIFACTS_DIR / "replication_llama"
LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_LAYER = 31
LLAMA_SAE_REPO = "fnlp/Llama3_1-8B-Base-LXR-8x"
LLAMA_SAE_SUBPATH = "Llama3_1-8B-Base-L31R-8x/checkpoints/final.safetensors"
LLAMA_SAE_HYPERPARAMS = "Llama3_1-8B-Base-L31R-8x/hyperparams.json"
# Pinned at probe time; verified in _load_llama_sae:
LLAMA_SAE_SHA = "a26dc30d15305d26d582b6e0abfdd20a4553a3b1fa9b507ae34456da56786504"

PUBLISHED_TOP_K = 10
N_SHUFFLES_PER_PAIR = 3
GEN_MAX_NEW = 15


# =============================================================================
# Llama loader
# =============================================================================

def _load_llama_model():
    """Load Llama-3.1-8B-Instruct in bf16 + eager attention.

    Mirrors src.model.load_model but for Llama. Eager attention is
    required for forward-hook-based interventions.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[setup] loading {LLAMA_MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA_MODEL_ID,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.to("cuda")
    model.eval()
    n_layers = len(model.model.layers)
    print(f"[setup] Llama model loaded; n_layers={n_layers}, "
          f"d_model={model.config.hidden_size}")
    if LLAMA_LAYER >= n_layers:
        raise RuntimeError(f"Layer {LLAMA_LAYER} out of range (n_layers={n_layers})")
    return tokenizer, model


# =============================================================================
# Llama Scope SAE wrapper
# =============================================================================

class LlamaScopeSAE:
    """Llama Scope L31R-8x wrapper that exposes the Gemma-Scope-style
    .encode / .decode / .W_enc interface expected by src.hooks and
    src.features.

    Implementation note: the published Llama Scope hyperparams advertise
    a `dataset_average_activation_norm = 74.5` and `norm_activation =
    dataset-wise`. Empirically, applying a /74.5 to inputs at inference
    sets *every* JumpReLU pre-activation below threshold (zero firing).
    Feeding the SAE raw residuals fires ~20-45 features per token with
    explained-variance ~0.55-0.60 — consistent with the published
    sparsity target (top-k=50 during training). We therefore treat the
    saved weights as "raw-input-ready": the /74.5 has been baked into
    the stored encoder/decoder, and inference is plain
        z = jumprelu(W_enc · x + b_enc)
        x_recon = W_dec · z + b_dec
    This also keeps encode/decode linear in x — required by the
    error-preserving splice in src.hooks.forward_with_ablation.

    The Instruct-model activations have norm ~50 vs base-trained ~74.5:
    a known base→Instruct distribution shift. Disclosed in the report.
    """

    def __init__(self, ckpt_path: str, hyperparams: dict, device: str = "cuda"):
        from safetensors.torch import load_file
        params = load_file(ckpt_path, device=device)
        self.W_enc = params["encoder.weight"].t().contiguous()       # (d_model, d_sae)
        self.b_enc = params["encoder.bias"]                           # (d_sae,)
        self.W_dec = params["decoder.weight"].t().contiguous()       # (d_sae, d_model)
        self.b_dec = params["decoder.bias"]                           # (d_model,)
        self.d_model = int(self.W_enc.shape[0])
        self.d_sae = int(self.W_enc.shape[1])
        self.threshold = float(hyperparams["jump_relu_threshold"])
        # Stored only as informational metadata (not applied at inference):
        self.dataset_norm = float(
            hyperparams["dataset_average_activation_norm"]["in"]
        )
        self.dtype = self.W_enc.dtype
        self.device = device

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Linear → JumpReLU. Input dtype must match self.dtype (bf16)."""
        pre = x @ self.W_enc + self.b_enc
        gate = (pre > self.threshold).to(pre.dtype)
        return pre * gate

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Linear; identity scale (the /74.5 baked into stored weights)."""
        return z @ self.W_dec + self.b_dec


def _load_llama_sae(device: str = "cuda"):
    """Download Llama Scope L31R-8x, verify SHA, instantiate wrapper.
    Returns (sae, sae_meta) — meta is dict for run_meta.json."""
    from huggingface_hub import hf_hub_download
    print(f"[setup] loading Llama Scope SAE: {LLAMA_SAE_REPO} / {LLAMA_SAE_SUBPATH}")
    ckpt_path = hf_hub_download(LLAMA_SAE_REPO, LLAMA_SAE_SUBPATH)
    hp_path = hf_hub_download(LLAMA_SAE_REPO, LLAMA_SAE_HYPERPARAMS)
    hp = json.load(open(hp_path))

    sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
    print(f"[setup] Llama Scope SAE sha256: {sha[:16]}...")
    if sha != LLAMA_SAE_SHA:
        raise RuntimeError(f"Llama Scope SAE SHA mismatch: got {sha}, "
                           f"expected {LLAMA_SAE_SHA}")
    print(f"[setup] SHA matches pinned reference")

    sae = LlamaScopeSAE(ckpt_path, hp, device=device)
    print(f"[setup] Llama SAE: d_model={sae.d_model}, d_sae={sae.d_sae}, "
          f"threshold={sae.threshold}, dataset_norm_metadata={sae.dataset_norm}")
    sae_meta = {
        "repo": LLAMA_SAE_REPO,
        "subpath": LLAMA_SAE_SUBPATH,
        "local_path": ckpt_path,
        "sha256": sha,
        "d_model": sae.d_model,
        "d_sae": sae.d_sae,
        "layer": LLAMA_LAYER,
        "act_fn": hp.get("act_fn"),
        "expansion_factor": hp.get("expansion_factor"),
        "jumprelu_threshold": sae.threshold,
        "dataset_norm_metadata": sae.dataset_norm,
        "norm_applied_at_inference": False,   # /74.5 was baked into stored weights
        "trained_on": "Llama-3.1-8B-Base (cross-fine-tune use on -Instruct)",
    }
    return sae, sae_meta


# =============================================================================
# Llama prompt builder
# =============================================================================

def build_llama_prompt(tokenizer, question: str) -> str:
    """Apply Llama-3 chat template: system + user turns, generation prompt suffix.

    The Llama-3.1 default system prompt prepends a 'Cutting Knowledge Date'
    notice; we let it stand (it's the model's default conditioning). We
    pass our SYSTEM_PREAMBLE as part of the user message, matching the
    Gemma path which has no separate system role.
    """
    combined = f"{SYSTEM_PREAMBLE}\n\n{question}"
    messages = [{"role": "user", "content": combined}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )


# =============================================================================
# Stat helpers (duplicated from reference_layer; lightweight)
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


# =============================================================================
# Main pipeline
# =============================================================================

def main() -> None:
    P25_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # ---- Load model + SAE ----
    tokenizer, model = _load_llama_model()
    sae, sae_meta = _load_llama_sae(device="cuda")

    # ---- Quick chat-template verification ----
    print(f"\n[verify] sample chat-template formatting on 5 prompts...")
    sample_questions = [
        "Who was the first president of the United States?",
        "What is the capital of France?",
        "Who painted the Mona Lisa?",
        "What year did World War II end?",
        "Who wrote the play Hamlet?",
    ]
    chat_template_sample = []
    for q in sample_questions:
        p = build_llama_prompt(tokenizer, q)
        ids = tokenizer(p, return_tensors="pt").input_ids[0]
        chat_template_sample.append({
            "question": q,
            "prompt_chars": len(p),
            "n_tokens": int(ids.shape[0]),
            "last_5_token_ids": ids[-5:].cpu().tolist(),
            "last_5_decoded": [tokenizer.decode([i]) for i in ids[-5:].tolist()],
        })
    print(f"[verify] sample 0: '{sample_questions[0]}'")
    print(f"  → {chat_template_sample[0]['n_tokens']} tokens, "
          f"last 5 decoded: {chat_template_sample[0]['last_5_decoded']}")

    # ---- Stage 2.A: collision-filter the AmbigQA candidate set ----
    print(f"\n========== Stage 2.A: collision filter on Llama tokenizer ==========")
    expanded = json.load(open(ARTIFACTS_DIR / "expanded_dataset.json"))
    pairs = expanded["pairs"]
    print(f"[2.A] loaded {len(pairs)} candidate pairs from expanded_dataset.json")

    after_collision = []
    for p in tqdm(pairs, desc="2.A llama collision"):
        cands = p["disambigs"]
        ftvs = []
        for c in cands:
            ftv = first_token_variants(tokenizer, c["answer"])
            ftvs.append(set(ftv))
        # Keep only disambigs whose first-token set is disjoint from siblings'
        keep_idx = []
        for i, ftv_i in enumerate(ftvs):
            if not ftv_i:
                continue
            collides = any(ftv_i & ftv_j
                            for j, ftv_j in enumerate(ftvs) if j != i)
            if collides:
                continue
            keep_idx.append(i)
        if len(keep_idx) < 2:
            continue
        new_disambigs = []
        for i in keep_idx:
            d = dict(cands[i])
            d["first_token_variants"] = list(ftvs[i])
            new_disambigs.append(d)
        after_collision.append({**p, "disambigs": new_disambigs})
    print(f"[2.A] after collision filter: {len(after_collision)} of {len(pairs)}")

    # ---- Stage 2.B: slot detection ----
    print(f"\n========== Stage 2.B: slot detection ==========")
    detected = []
    failures = []
    for p in tqdm(after_collision, desc="2.B llama slot"):
        a_prompt = build_llama_prompt(tokenizer, p["A_question"])
        gen = greedy_decode(model, tokenizer, a_prompt, max_new=GEN_MAX_NEW)
        cand_strs = [d["answer"] for d in p["disambigs"]]
        match = find_best_match(gen, cand_strs)
        if match.cand_idx < 0:
            failures.append({
                "id": p["id"], "question": p["A_question"],
                "candidates": cand_strs, "generation": gen,
            })
            continue
        # Note: like AmbigQA, all disambigs are kept; matched index is
        # informational only (which one the model committed to). The
        # six-condition pipeline runs over all (P, D_i) self-pairs.
        detected.append({**p,
                          "committed_idx": int(match.cand_idx),
                          "match_strategy": match.strategy,
                          "match_text": match.matched_text,
                          "generation": gen})
    save_json_atomic(P25_DIR / "slot_detection_failures_llama.json", failures)
    save_json_atomic(P25_DIR / "detected_pairs_llama.json", detected)
    n_self_pairs_total = sum(len(p["disambigs"]) for p in detected)
    print(f"[2.B] detected: {len(detected)} pairs, "
          f"{n_self_pairs_total} self-pairs "
          f"(retention {len(detected)/max(len(after_collision),1)*100:.1f}%)")

    # ---- Stage 2.C: capture L31 residuals for A and D_i ----
    print(f"\n========== Stage 2.C: residual capture at L{LLAMA_LAYER} ==========")
    residuals = {}
    for p in tqdm(detected, desc="2.C llama A/D residuals"):
        a_prompt = build_llama_prompt(tokenizer, p["A_question"])
        residuals[f"A__{p['id']}"] = capture_last_token_residual(
            model, tokenizer, a_prompt, layer=LLAMA_LAYER,
        )
        for di, d in enumerate(p["disambigs"]):
            d_prompt = build_llama_prompt(tokenizer, d["question"])
            residuals[f"D__{p['id']}__{di}"] = capture_last_token_residual(
                model, tokenizer, d_prompt, layer=LLAMA_LAYER,
            )
    save_npz_atomic(P25_DIR / f"residuals_L{LLAMA_LAYER}_llama.npz", **residuals)
    print(f"[2.C] captured {len(residuals)} residuals")

    # ---- Stage 2.D: SAE-encode ----
    print(f"\n========== Stage 2.D: SAE-encoding ==========")
    encodings = {}
    for k, r in residuals.items():
        x = torch.from_numpy(r).to(device="cuda", dtype=sae.dtype)
        z = sae.encode(x.unsqueeze(0)).squeeze(0)
        encodings[k] = z.float().cpu().numpy()
    save_npz_atomic(P25_DIR / f"sae_encodings_L{LLAMA_LAYER}_llama.npz", **encodings)

    # ---- Stage 2.E: pick Targeted top-10 per (P, D_i) ----
    print(f"\n========== Stage 2.E: pick Targeted top-{PUBLISHED_TOP_K} ==========")
    enc_torch = {
        k: torch.from_numpy(v).to(device="cuda", dtype=sae.dtype)
        for k, v in encodings.items()
    }
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
    save_json_atomic(P25_DIR / "specific_features_llama.json",
                     [{"pair_id": pid, "disambig_idx": di, "features": f}
                      for (pid, di), f in specific_features.items()])
    n_full = sum(1 for f in specific_features.values() if len(f) == PUBLISHED_TOP_K)
    print(f"[2.E] picked features for {len(specific_features)} (P, D_i); "
          f"{n_full} have full top-{PUBLISHED_TOP_K}")

    # ---- Stage 2.F: WikiText residuals + per-paragraph top-10 ----
    print(f"\n========== Stage 2.F: WikiText last-token at L{LLAMA_LAYER} ==========")
    paragraphs = load_wikitext_paragraphs(tokenizer)
    n_paragraphs = len(paragraphs)
    wt_lasttok_z = np.zeros((n_paragraphs, sae.d_sae), dtype=np.float32)
    for i, p_text in enumerate(tqdm(paragraphs, desc="2.F WT")):
        try:
            _, res = capture_all_position_residuals(
                model, tokenizer, p_text, layer=LLAMA_LAYER,
                max_len=WIKITEXT_MAX_TOKENS,
            )
            z_last = sae.encode(res[-1:].to(dtype=sae.dtype)).squeeze(0)
            wt_lasttok_z[i] = z_last.float().cpu().numpy()
            del res, z_last
        except Exception as e:
            print(f"[2.F] paragraph {i} failed: {type(e).__name__}: {e}")
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

    # ---- Stage 2.G: pre-compute shuffle draws + random feature ids ----
    print(f"\n========== Stage 2.G: shuffle draws ==========")
    import random as _rnd
    all_keys = []
    for p in detected:
        for di in range(len(p["disambigs"])):
            all_keys.append((p["id"], di))
    rng_t = _rnd.Random(SEED)
    rng_w = _rnd.Random(SHUFFLE_SEED)
    ambig_draws = []
    wt_draws = []
    for p in detected:
        for ai in range(len(p["disambigs"])):
            for d in range(N_SHUFFLES_PER_PAIR):
                while True:
                    target = rng_t.choice(all_keys)
                    if target[0] != p["id"]:
                        break
                ambig_draws.append({
                    "pair_id": p["id"], "target_idx": ai, "draw_idx": d,
                    "shuffle_pair_id": target[0],
                    "shuffle_disambig_idx": int(target[1]),
                })
                wt_draws.append({
                    "pair_id": p["id"], "target_idx": ai, "draw_idx": d,
                    "wikitext_paragraph_idx": rng_w.randrange(n_paragraphs),
                })
    save_json_atomic(P25_DIR / "shuffle_draws_ambigqa.json", ambig_draws)
    save_json_atomic(P25_DIR / "shuffle_draws_wikitext.json", wt_draws)

    rand_features = []
    for p in detected:
        for ai in range(len(p["disambigs"])):
            seed = int(hashlib.sha256(f"llama|{p['id']}|{ai}".encode())
                        .hexdigest(), 16) % (2**32)
            rng = _rnd.Random(seed)
            for c in range(RANDOM_CONTROLS):
                feats = sorted(rng.sample(range(sae.d_sae), TOP_K_FEATURES))
                rand_features.append({
                    "pair_id": p["id"], "ablate_idx": ai, "control_idx": c,
                    "feature_ids": feats,
                })

    # ---- Stage 2.H: six-condition pipeline ----
    print(f"\n========== Stage 2.H: six-condition pipeline ==========")
    self_rows = []; cross_rows = []; random_rows = []
    ambig_shuf_rows = []; wt_shuf_rows = []
    ambig_by_draw = {(d["pair_id"], d["target_idx"], d["draw_idx"]):
                      (d["shuffle_pair_id"], d["shuffle_disambig_idx"])
                      for d in ambig_draws}
    wt_by_draw = {(d["pair_id"], d["target_idx"], d["draw_idx"]):
                   d["wikitext_paragraph_idx"]
                   for d in wt_draws}
    rand_by_pair = {(r["pair_id"], r["ablate_idx"], r["control_idx"]):
                     r["feature_ids"]
                     for r in rand_features}

    for p in tqdm(detected, desc="2.H 6-cond"):
        a_prompt = build_llama_prompt(tokenizer, p["A_question"])
        # Baseline
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
                model, tokenizer, a_prompt, LLAMA_LAYER, sae,
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

            # Random
            for c in range(RANDOM_CONTROLS):
                rand_feats = rand_by_pair.get((p["id"], ab_i, c))
                if rand_feats is None:
                    continue
                rc = forward_with_ablation(
                    model, tokenizer, a_prompt, LLAMA_LAYER, sae,
                    feature_ids=rand_feats, k=TOP_LOGITS_K,
                )
                random_rows.append({
                    "pair_id": p["id"], "ablate_idx": ab_i, "control_idx": c,
                    "n_features": len(rand_feats),
                    "base_hit1": base_hits[ab_i],
                    "random_hit1": hit_at_k(rc["top_ids"], target_self, 1),
                })

            # AmbigQA-shuffle
            for d_idx in range(N_SHUFFLES_PER_PAIR):
                tup = ambig_by_draw.get((p["id"], ab_i, d_idx))
                if tup is None:
                    continue
                p_prime, k_idx = tup
                feats_sh = specific_features.get((p_prime, k_idx), [])
                if not feats_sh:
                    continue
                sh = forward_with_ablation(
                    model, tokenizer, a_prompt, LLAMA_LAYER, sae,
                    feature_ids=feats_sh, k=TOP_LOGITS_K,
                )
                ambig_shuf_rows.append({
                    "pair_id": p["id"], "target_idx": ab_i, "draw_idx": d_idx,
                    "shuffle_pair_id": p_prime, "shuffle_disambig_idx": int(k_idx),
                    "n_features": len(feats_sh),
                    "base_hit1": base_hits[ab_i],
                    "shuffled_hit1": hit_at_k(sh["top_ids"], target_self, 1),
                })

            # WikiText-shuffle
            for d_idx in range(N_SHUFFLES_PER_PAIR):
                para_idx = wt_by_draw.get((p["id"], ab_i, d_idx))
                if para_idx is None:
                    continue
                feats_wt = wt_top10[para_idx]["top10_feature_ids"]
                if not feats_wt:
                    continue
                wt = forward_with_ablation(
                    model, tokenizer, a_prompt, LLAMA_LAYER, sae,
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

    def _mean(rows, k):
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
    save_json_atomic(P25_DIR / "results_main_llama.json", {
        "summary": six_cond_summary,
        "self_rows": self_rows,
        "cross_rows": cross_rows,
        "random_rows": random_rows,
        "ambigqa_shuffled_rows": ambig_shuf_rows,
        "wikitext_shuffled_rows": wt_shuf_rows,
    })
    print(f"[2.H] six-condition counts: {six_cond_summary['n_self']} self, "
          f"{six_cond_summary['n_cross']} cross, "
          f"{six_cond_summary['n_random']} rand, "
          f"{six_cond_summary['n_ambig_shuf']} ambig-shuf, "
          f"{six_cond_summary['n_wt_shuf']} WT-shuf")

    # ---- Stage 2.I: T1-T4 + headline ----
    print(f"\n========== Stage 2.I: T1-T4 paired tests ==========")
    sib = per_pair_means(
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
    base = {(r["pair_id"], r["target_idx"]): r["base_hit1"] for r in self_rows}
    targ = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"] for r in self_rows}

    keys_main = sorted(k for k in base if k in wt_per_pair and k in rand_per_pair)
    keys_sib = sorted(k for k in base if k in sib and k in wt_per_pair
                       and k in a_shuf_per_pair and k in rand_per_pair)
    print(f"[2.I] main n: {len(keys_main)}, sibling-paired n: {len(keys_sib)}")

    B = np.array([base[k] for k in keys_main])
    T = np.array([targ[k] for k in keys_main])
    WSh = np.array([wt_per_pair[k] for k in keys_main])
    Rnd = np.array([rand_per_pair[k] for k in keys_main])
    Sib = np.array([sib[k] for k in keys_sib]) if keys_sib else np.array([])
    ASh = np.array([a_shuf_per_pair[k] for k in keys_sib]) if keys_sib else np.array([])
    WSh_t3 = np.array([wt_per_pair[k] for k in keys_sib]) if keys_sib else np.array([])

    tests = {
        "T1_targeted_vs_random": _drop_vs_base_stats(T, Rnd),
        "T2_sibling_vs_ashuf":   _t3_stats(Sib, ASh) if len(Sib) > 1 else None,
        "T3_sibling_vs_wtshuf":  _t3_stats(Sib, WSh_t3) if len(Sib) > 1 else None,
        "T4_ashuf_vs_wtshuf":    _t3_stats(ASh, WSh_t3) if len(ASh) > 1 else None,
    }
    headline_table = {
        "Baseline":          float(B.mean()) if len(keys_main) else 0.0,
        "Targeted":          float(T.mean()) if len(keys_main) else 0.0,
        "Sibling":           float(Sib.mean()) if len(Sib) else 0.0,
        "ShuffledAmbigQA":   float(ASh.mean()) if len(ASh) else 0.0,
        "WikiTextShuffled":  float(WSh.mean()) if len(keys_main) else 0.0,
        "Random":            float(Rnd.mean()) if len(keys_main) else 0.0,
    }
    print(f"[2.I] headline rates:")
    for k, v in headline_table.items():
        print(f"    {k:<20}  {v:.4f}")
    if tests["T3_sibling_vs_wtshuf"]:
        t3 = tests["T3_sibling_vs_wtshuf"]
        print(f"    T3: Δ={t3['delta_pp']:+.2f}pp Wlx p={t3['wilcoxon_p']:.3g}")

    # ---- Stage 2.J: unique/shared decomposition ----
    print(f"\n========== Stage 2.J: unique/shared decomposition ==========")
    pub_set = {(pid, di): set(feats)
               for (pid, di), feats in specific_features.items()}
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
            sh_un[(p["id"], i)] = {"shared": shared, "unique": unique,
                                     "targeted": ti}

    base_lookup = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                   for r in self_rows}
    targ_lookup = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                   for r in self_rows}

    sh_un_rows = []
    for p in tqdm(detected, desc="2.J sh+un"):
        a_prompt = build_llama_prompt(tokenizer, p["A_question"])
        for i in range(len(p["disambigs"])):
            key = (p["id"], i)
            if key not in sh_un or key not in base_lookup:
                continue
            target = set(p["disambigs"][i]["first_token_variants"])
            base_v = base_lookup[key]
            if sh_un[key]["shared"]:
                ab = forward_with_ablation(
                    model, tokenizer, a_prompt, LLAMA_LAYER, sae,
                    feature_ids=list(sh_un[key]["shared"]), k=TOP_LOGITS_K,
                )
                shared_hit = hit_at_k(ab["top_ids"], target, 1)
            else:
                shared_hit = base_v
            if sh_un[key]["unique"]:
                ab = forward_with_ablation(
                    model, tokenizer, a_prompt, LLAMA_LAYER, sae,
                    feature_ids=list(sh_un[key]["unique"]), k=TOP_LOGITS_K,
                )
                unique_hit = hit_at_k(ab["top_ids"], target, 1)
            else:
                unique_hit = base_v
            sh_un_rows.append({
                "pair_id": p["id"], "target_idx": i,
                "n_targeted": len(sh_un[key]["targeted"]),
                "n_shared": len(sh_un[key]["shared"]),
                "n_unique": len(sh_un[key]["unique"]),
                "base_hit1": base_v,
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
    save_json_atomic(P25_DIR / "unique_shared_decomp_llama.json",
                     {"decomposition": decomp, "rows": sh_un_rows})
    print(f"[2.J] Targeted Δ={targ_d:+.2f}; Shared Δ={sh_d:+.2f} ({pct(sh_d):+.1f}%); "
          f"Unique Δ={un_d:+.2f} ({pct(un_d):+.1f}%); Residual {res:+.2f} ({pct(res):+.1f}%)")

    # ---- Stage 2.K: cross-architecture summary ----
    print(f"\n========== Stage 2.K: cross-architecture summary ==========")
    # Gemma 9B L41 ref
    gemma9b = json.load(open(ARTIFACTS_DIR / "reference_layer" / "L41_summary_table.json"))
    g9_h = gemma9b["headline_hit1"]
    g9_t3 = gemma9b["T3"]
    g9_d = gemma9b["decomposition_hit1_from_p21b"]
    # Gemma 2B L25 ref
    gemma2b = json.load(open(ARTIFACTS_DIR / "replication_gemma2b" / "summary.json"))
    g2 = gemma2b["cross_model_table"][1]   # Gemma-2-2B row

    cross_arch_table = [
        {
            "model": "google/gemma-2-9b-it",
            "layer": 41,
            "n_self_pairs": gemma9b["n_self_pairs"],
            "baseline_hit1": g9_h["Baseline"],
            "targeted_delta_pp": g9_d["Targeted"]["delta_pp"],
            "sibling_hit1": g9_h["Sibling"],
            "wt_shuf_hit1": g9_h["WikiTextShuffled"],
            "T3_delta_pp": g9_t3["delta_pp"],
            "T3_wilcoxon_p": g9_t3["wilcoxon_p"],
            "shared_only_delta_pp": g9_d["Shared-only"]["delta_pp"],
            "unique_only_delta_pp": g9_d["Unique-only"]["delta_pp"],
            "shared_pct": (g9_d["Shared-only"]["delta_pp"]
                            / g9_d["Targeted"]["delta_pp"] * 100),
            "unique_pct": (g9_d["Unique-only"]["delta_pp"]
                            / g9_d["Targeted"]["delta_pp"] * 100),
        },
        {
            "model": "google/gemma-2-2b-it",
            "layer": 25,
            "n_self_pairs": g2["n_self_pairs"],
            "baseline_hit1": g2["baseline_hit1"],
            "targeted_delta_pp": g2["targeted_delta_pp"],
            "sibling_hit1": g2["sibling_hit1"],
            "wt_shuf_hit1": g2["wt_shuf_hit1"],
            "T3_delta_pp": g2["T3_delta_pp"],
            "T3_wilcoxon_p": g2["T3_wilcoxon_p"],
            "shared_only_delta_pp": g2["shared_only_delta_pp"],
            "unique_only_delta_pp": g2["unique_only_delta_pp"],
            "shared_pct": (g2["shared_only_delta_pp"]
                            / g2["targeted_delta_pp"] * 100),
            "unique_pct": (g2["unique_only_delta_pp"]
                            / g2["targeted_delta_pp"] * 100),
        },
        {
            "model": LLAMA_MODEL_ID,
            "layer": LLAMA_LAYER,
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
        "model": LLAMA_MODEL_ID,
        "layer": LLAMA_LAYER,
        "sae_meta": sae_meta,
        "chat_template_sample": chat_template_sample,
        "filtering": {
            "n_after_collision": len(after_collision),
            "n_detected_pairs": len(detected),
            "n_self_pairs_total": n_self_pairs_total,
            "n_slot_failures": len(failures),
        },
        "headline_hit1": headline_table,
        "tests": tests,
        "decomposition": decomp,
        "cross_arch_table": cross_arch_table,
        "six_condition_counts": {
            "n_self": six_cond_summary["n_self"],
            "n_cross": six_cond_summary["n_cross"],
            "n_random": six_cond_summary["n_random"],
            "n_ambig_shuf": six_cond_summary["n_ambig_shuf"],
            "n_wt_shuf": six_cond_summary["n_wt_shuf"],
        },
    }
    save_json_atomic(P25_DIR / "summary.json", summary)

    # ---- run_meta + cleanup ----
    elapsed = time.time() - t_total
    save_json_atomic(P25_DIR / "run_meta.json", {
        "model": LLAMA_MODEL_ID,
        "layer": LLAMA_LAYER,
        "sae_meta": sae_meta,
        "n_after_collision": len(after_collision),
        "n_detected_pairs": len(detected),
        "n_self_pairs_total": n_self_pairs_total,
        "n_slot_failures": len(failures),
        "elapsed_seconds": round(elapsed, 2),
    })

    # Cleanup heavy NPZ + SAE blob
    try:
        (P25_DIR / f"residuals_L{LLAMA_LAYER}_llama.npz").unlink(missing_ok=True)
        print(f"[cleanup] removed residuals NPZ")
    except Exception:
        pass
    try:
        local_path = sae_meta.get("local_path")
        if local_path:
            real = Path(local_path).resolve()
            if real.exists():
                real.unlink()
                print(f"[cleanup] deleted Llama Scope SAE blob")
    except Exception:
        pass

    print(f"\n=== Stage 2.5 complete ===")
    print(f"  total elapsed: {elapsed/60:.1f} min")
    print(f"  -> {P25_DIR}/summary.json")
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
