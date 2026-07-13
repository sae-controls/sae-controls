"""Polysemy Correction (paper Sec. 3.5).

Decomposes the targeted-ablation effect into content / position / interaction
components by:

  Phase A  Re-encoding the 800 WikiText paragraphs (the same 800 paragraphs
           loaded by main_ablation.py)
           through the SAE and computing per-feature position-mode statistics
           for ALL features that appear in any (P, D_i) top-10 list (≈ 2000).
  Phase B  Flagging features as `position_suspect` if
              n_nonzero ≥ POSITION_NONZERO_FLOOR  AND
              pct_at_pos0 ≥ POSITION_PCT_THRESHOLD
           and verifying the three known polysemantic features (7187, 9825,
           15382) are flagged.
  Phase C  Partitioning each (P, D_i)'s top-10 into `position_features`
           (suspect) and `content_features` (rest).
  Phase D  Running content-only and position-only ablations (skip pairs where
           one of the partitions is empty by construction).

Outputs:
  artifacts/wikitext_position_mode.json     per-feature stats + suspect flag
  artifacts/results_polysemy_decomp.json    per-instance + summary
  artifacts/run_meta_polysemy.json           run metadata (seeds, threshold, SAE hash, elapsed)

CRITICAL: this stage requires (a) the same model + SAE used in main_ablation.py, and
(b) the WikiText paragraphs reconstructed deterministically. SAE checkpoint
sha256 is verified at start. The published run took 4.0 min on an A40 with
hot HF cache.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.config import (
    ARTIFACTS_DIR, DTYPE_STR, LAYER, MODEL_ID, POSITION_NONZERO_FLOOR,
    POSITION_PCT_THRESHOLD, SAE_CHECKPOINT_SHA256, SEED, TOP_LOGITS_K,
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


KNOWN_POLYSEMANTIC = [7187, 9825, 15382]   # validated from feature characterization


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()
    np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"[stage 5] model={MODEL_ID}, layer=L{LAYER}, seed={SEED}")
    print(f"[stage 5] threshold: n_nonzero ≥ {POSITION_NONZERO_FLOOR}, "
          f"pct_at_pos0 ≥ {POSITION_PCT_THRESHOLD}")

    # Inputs
    inv = json.load(open(ARTIFACTS_DIR / "feature_inventory.json"))
    all_disambig_features = sorted(int(f) for f in inv["freq_full"].keys())
    print(f"[stage 5] disambig-derived feature pool: {len(all_disambig_features)}")
    sf = json.load(open(ARTIFACTS_DIR / "specific_features.json"))
    detected = json.load(open(ARTIFACTS_DIR / "detected_pairs.json"))

    # Model + SAE
    print("[stage 5] loading model + SAE...")
    tokenizer, model = load_model()
    sae, sae_meta = load_sae(layer=LAYER)
    if sae_meta["sha256"] != SAE_CHECKPOINT_SHA256:
        raise RuntimeError(f"SAE sha256 mismatch: {sae_meta['sha256']}")
    print(f"[stage 5] SAE sha256 verified ({sae_meta['sha256'][:16]}...)")

    # Reconstruct WikiText paragraphs deterministically
    paragraphs = load_wikitext_paragraphs(tokenizer)
    print(f"[stage 5] reconstructed {len(paragraphs)} WikiText paragraphs")

    # ---- Phase A: per-feature stats ----
    feat_t = torch.tensor(all_disambig_features, dtype=torch.long, device=sae.W_enc.device)
    n_nz = torch.zeros(len(all_disambig_features), dtype=torch.long,    device=sae.W_enc.device)
    n_p0 = torch.zeros(len(all_disambig_features), dtype=torch.long,    device=sae.W_enc.device)
    s_act = torch.zeros(len(all_disambig_features), dtype=torch.float32, device=sae.W_enc.device)
    s_sq  = torch.zeros(len(all_disambig_features), dtype=torch.float32, device=sae.W_enc.device)

    print(f"\n[5.A] scanning {len(all_disambig_features)} features × {len(paragraphs)} paragraphs")
    for i, p_text in enumerate(tqdm(paragraphs, desc="5.A")):
        try:
            _, res = capture_all_position_residuals(
                model, tokenizer, p_text, layer=LAYER, max_len=WIKITEXT_MAX_TOKENS)
        except Exception as e:
            print(f"[5.A] paragraph {i} failed: {type(e).__name__}: {e}")
            continue
        z = sae.encode(res.to(dtype=sae.W_enc.dtype))
        z_subset = z.index_select(dim=1, index=feat_t)
        nonzero = z_subset > 0
        n_nz += nonzero.sum(dim=0).to(torch.long)
        n_p0 += nonzero[0:1, :].to(torch.long).flatten()
        z_f = z_subset.float()
        s_act += z_f.sum(dim=0)
        s_sq  += (z_f * z_f).sum(dim=0)
        del res, z, z_subset, nonzero, z_f
        if i % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    n_nz_np = n_nz.cpu().numpy()
    n_p0_np = n_p0.cpu().numpy()
    s_np    = s_act.cpu().numpy()
    s_sq_np = s_sq.cpu().numpy()

    pct = np.where(n_nz_np > 0, n_p0_np / np.maximum(n_nz_np, 1), 0.0)
    mean_act = np.where(n_nz_np > 0, s_np / np.maximum(n_nz_np, 1), 0.0)
    var_act = np.where(n_nz_np > 1,
                        (s_sq_np - (s_np * s_np) / np.maximum(n_nz_np, 1))
                          / np.maximum(n_nz_np - 1, 1), 0.0)

    suspect = np.array([
        is_position_suspect(int(n_nz_np[i]), float(pct[i]))
        for i in range(len(all_disambig_features))
    ])
    n_suspect = int(suspect.sum())
    print(f"\n[5.B] position-suspect features: {n_suspect} / {len(all_disambig_features)}")

    # Verify
    print(f"[5.B] known-polysemantic features check:")
    for f in KNOWN_POLYSEMANTIC:
        idx = all_disambig_features.index(f)
        print(f"   feature {f}: n_nonzero={n_nz_np[idx]}, pct_at_pos0={pct[idx]:.3f}, flagged={bool(suspect[idx])}")

    # Save per-feature stats
    pos_mode = {
        "threshold_n_nonzero_floor": POSITION_NONZERO_FLOOR,
        "threshold_pct_at_pos0":     POSITION_PCT_THRESHOLD,
        "n_features_scanned":        len(all_disambig_features),
        "n_position_suspect":        n_suspect,
        "n_paragraphs_processed":    len(paragraphs),
        "per_feature": {
            int(all_disambig_features[i]): {
                "n_nonzero":                int(n_nz_np[i]),
                "n_at_pos0":                int(n_p0_np[i]),
                "pct_at_pos0":              float(pct[i]),
                "mean_act_when_nonzero":    float(mean_act[i]),
                "variance_act_when_nonzero": float(var_act[i]),
                "position_suspect":         bool(suspect[i]),
            }
            for i in range(len(all_disambig_features))
        },
    }
    save_json_atomic(ARTIFACTS_DIR / "wikitext_position_mode.json", pos_mode)

    suspect_set = {int(all_disambig_features[i])
                    for i in range(len(all_disambig_features)) if suspect[i]}

    # ---- Phase C: partition top-10 ----
    print(f"\n[5.C] partitioning top-10 lists into content/position")
    pair_partition = []
    for r in sf:
        feats = list(r["features"])
        pos, cont = partition_top_k(feats, suspect_set)
        pair_partition.append({
            "pair_id": r["pair_id"], "target_idx": int(r["disambig_idx"]),
            "n_total": len(feats),
            "n_position_features": len(pos),
            "n_content_features":  len(cont),
            "position_features":   pos,
            "content_features":    cont,
        })

    # ---- Phase D: ablations ----
    print(f"\n[5.D] content-only + position-only ablations")
    R_main = json.load(open(ARTIFACTS_DIR / "results_main.json"))
    targeted_existing = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                          for r in R_main["self_rows"]}
    baseline_existing = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                          for r in R_main["self_rows"]}

    prompts_A = {p["id"]: build_prompt(tokenizer, p["A_question"]) for p in detected}
    target_sets = {(p["id"], di): set(d["first_token_variants"])
                    for p in detected for di, d in enumerate(p["disambigs"])}

    rows = []
    n_pure = n_all_pos = n_run = 0
    for p_info in tqdm(pair_partition, desc="5.D"):
        key = (p_info["pair_id"], p_info["target_idx"])
        target = target_sets[key]
        a_prompt = prompts_A[p_info["pair_id"]]
        n_pos = p_info["n_position_features"]
        n_cont = p_info["n_content_features"]
        base = baseline_existing[key]
        targ = targeted_existing[key]

        if n_pos == 0:
            content_only = targ; position_only = base; n_pure += 1
        elif n_cont == 0:
            content_only = base; position_only = targ; n_all_pos += 1
        else:
            ab_c = forward_with_ablation(model, tokenizer, a_prompt, LAYER, sae,
                                          feature_ids=p_info["content_features"], k=TOP_LOGITS_K)
            content_only = hit_at_k(ab_c["top_ids"], target, 1)
            ab_p = forward_with_ablation(model, tokenizer, a_prompt, LAYER, sae,
                                          feature_ids=p_info["position_features"], k=TOP_LOGITS_K)
            position_only = hit_at_k(ab_p["top_ids"], target, 1)
            n_run += 1

        rows.append({
            **p_info,
            "baseline_hit1":      base,
            "targeted_hit1":      targ,
            "content_only_hit1":  content_only,
            "position_only_hit1": position_only,
        })

    n = len(rows)
    base_mean = sum(r["baseline_hit1"]      for r in rows) / n
    targ_mean = sum(r["targeted_hit1"]      for r in rows) / n
    cont_mean = sum(r["content_only_hit1"]  for r in rows) / n
    pos_mean  = sum(r["position_only_hit1"] for r in rows) / n
    summary = {
        "n_self_pairs":            n,
        "n_pairs_run_both_forwards": n_run,
        "n_pairs_pure_content":     n_pure,
        "n_pairs_all_position":     n_all_pos,
        "baseline_mean_hit1":       base_mean,
        "targeted_mean_hit1":       targ_mean,
        "content_only_mean_hit1":   cont_mean,
        "position_only_mean_hit1":  pos_mean,
        "delta_targeted_pp":        (targ_mean - base_mean) * 100,
        "delta_content_only_pp":    (cont_mean - base_mean) * 100,
        "delta_position_only_pp":   (pos_mean  - base_mean) * 100,
        "decomposition_residual_pp":
            ((targ_mean - base_mean) - ((cont_mean - base_mean) + (pos_mean - base_mean))) * 100,
        "sae_meta": sae_meta,
    }
    save_json_atomic(ARTIFACTS_DIR / "results_polysemy_decomp.json",
                     {"summary": summary, "rows": rows})
    save_json_atomic(ARTIFACTS_DIR / "run_meta_polysemy.json", {
        "model": MODEL_ID, "dtype": DTYPE_STR, "layer": LAYER,
        "sae_meta": sae_meta, "seed": SEED,
        "position_threshold": {
            "n_nonzero_floor":     POSITION_NONZERO_FLOOR,
            "pct_at_pos0_threshold": POSITION_PCT_THRESHOLD,
        },
        "n_position_suspect_features": n_suspect,
        "n_self_pairs": n,
        "elapsed_seconds": round(time.time() - t_total, 2),
    })

    print(f"\n=== Stage 5 complete ===")
    print(f"  baseline:        {base_mean:.4f}")
    print(f"  targeted:        {targ_mean:.4f}    Δ = {summary['delta_targeted_pp']:+.2f} pp")
    print(f"  content-only:    {cont_mean:.4f}    Δ = {summary['delta_content_only_pp']:+.2f} pp")
    print(f"  position-only:   {pos_mean:.4f}    Δ = {summary['delta_position_only_pp']:+.2f} pp")
    print(f"  residual (interaction) = {summary['decomposition_residual_pp']:+.3f} pp")
    print(f"  pairs (run/pure/all-position): {n_run}/{n_pure}/{n_all_pos}")
    print(f"  elapsed: {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
