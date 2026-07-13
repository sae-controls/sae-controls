"""SAE L0 sweep on Gemma-2-9B-IT @ L37 and L41.

Tests whether the headline T3 mechanism and the unique/shared
decomposition depend on the SAE's L0 sparsity choice within Gemma Scope's
published family. Runs at two layer anchors:
  L37 (published reference): non-canonical L0 ∈ {20, 34, 63, 124, 266}
  L41 (depth-trajectory peak): non-canonical L0 ∈ {16, 28, 52, 113, 270}

Per (layer, L0) cell:
  - Load SAE with explicit L0 subpath; pin SHA-256.
  - SAE-encode the layer's A/D residuals (cached residuals_L37.npz for
    L37; recaptured for L41).
  - SAE-encode last-token WikiText residuals (recaptured per layer once).
  - Pick Targeted top-10 via published score formula.
  - Pick WT top-10 by raw activation per paragraph.
  - Run three causal conditions: Targeted (1103), Sibling (from cross
    rows of Targeted forwards, no extra forwards), WikiText-shuffled
    (3309 = 3 draws × 1103).
  - Run unique/shared decomposition (Shared-only ablation, Unique-only
    ablation).
  - Compute T3 + bootstrap CI + Wilcoxon + McNemar.

Aggregates a per-layer L0-table and a T3-vs-L0 figure with both layers
overlaid. Auto-cleanup of SAE blob from HF cache after each cell to limit disk usage.

Reads:
  artifacts/detected_pairs.json
  artifacts/residuals_L37.npz                              (for L37 A/D)
  artifacts/shuffle_draws_wikitext.json                    (paragraph draws)
  artifacts/results_main.json                              (random rows reuse)
  Re-loads WikiText paragraphs from HF for residual recapture.

Writes:
  artifacts/l0_sweep/L{37,41}/L0_{val}/{sae_encodings.npz,
    specific_features.json, results.json, unique_shared_decomp.json,
    run_meta.json}
  artifacts/l0_sweep/l0_sweep_summary.json
  artifacts/l0_sweep/figure_t3_vs_l0.png
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

from src.analysis import (
    bootstrap_diff_ci_pp, mcnemar_paired, per_pair_means, wilcoxon_paired,
)
from src.config import (
    ARTIFACTS_DIR, MODEL_ID, N_SHUFFLES_PER_PAIR, SAE_REPOS, SAE_WIDTH,
    SCORE_A_WEIGHT, TOP_K_FEATURES, TOP_LOGITS_K, WIKITEXT_MAX_TOKENS,
)
from src.features import score_specific_features
from src.hooks import (
    capture_all_position_residuals, capture_last_token_residual,
    forward_with_ablation, hit_at_k,
)
from src.io_utils import save_json_atomic, save_npz_atomic
from src.model import load_model
from src.prompts import build_prompt
from src.sae import JumpReLUSAE
from src.wikitext import load_paragraphs as load_wikitext_paragraphs


P22_DIR = ARTIFACTS_DIR / "l0_sweep"

# Canonical L0s (used in the layer sweep). Skip these.
CANONICAL_L0 = {37: 11, 41: 10}

# Non-canonical L0s to sweep, per layer
SWEEP_L0 = {
    37: [20, 34, 63, 124, 266],
    41: [16, 28, 52, 113, 270],
}


def _load_sae_explicit(layer, l0):
    """Load Gemma Scope SAE with an explicit L0 subpath. Returns (sae, meta)
    or raises RuntimeError if not found."""
    import hashlib
    from huggingface_hub import hf_hub_download
    sub = f"layer_{layer}/width_{SAE_WIDTH}/average_l0_{l0}/params.npz"
    last_err = None
    for repo in SAE_REPOS:
        try:
            path = Path(hf_hub_download(repo, sub))
            params = np.load(path)
            d_model, d_sae = params["W_enc"].shape
            sae = JumpReLUSAE(d_model, d_sae, device_="cuda")
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
                "repo": repo, "subpath": sub, "local_path": str(path),
                "sha256": sha,
                "d_model": int(d_model), "d_sae": int(d_sae),
                "layer": layer, "l0": l0,
            }
        except Exception as e:
            last_err = e
    raise RuntimeError(f"SAE not found at {sub}: {last_err}")


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


def _capture_or_load_residuals(model, tokenizer, layer, detected, a_prompts):
    """Return dict of residual arrays keyed by 'A__pid' or 'D__pid__di'.
    Use published artifacts/residuals_L37.npz for layer 37; recapture for
    other layers."""
    if layer == 37 and (ARTIFACTS_DIR / "residuals_L37.npz").exists():
        npz = np.load(ARTIFACTS_DIR / "residuals_L37.npz")
        out = {k: npz[k] for k in npz.files}
        print(f"[L{layer}] loaded {len(out)} residuals from published artifact")
        return out
    print(f"[L{layer}] capturing A + D residuals "
          f"({sum(len(p['disambigs']) for p in detected) + len(detected)} prompts)...")
    out = {}
    for p in tqdm(detected, desc=f"L{layer} residuals"):
        out[f"A__{p['id']}"] = capture_last_token_residual(
            model, tokenizer, a_prompts[p["id"]], layer=layer,
        )
        for di, d in enumerate(p["disambigs"]):
            d_prompt = build_prompt(tokenizer, d["question"])
            out[f"D__{p['id']}__{di}"] = capture_last_token_residual(
                model, tokenizer, d_prompt, layer=layer,
            )
    return out


def _capture_wt_residuals(model, tokenizer, layer, paragraphs):
    """Return per-paragraph last-token residual array of shape (n_para, d_model)."""
    print(f"[L{layer}] capturing WikiText last-token residuals "
          f"({len(paragraphs)} paragraphs)...")
    n_para = len(paragraphs)
    d_model = None
    for i, p_text in enumerate(tqdm(paragraphs, desc=f"L{layer} WT residuals")):
        try:
            _, res = capture_all_position_residuals(
                model, tokenizer, p_text, layer=layer, max_len=WIKITEXT_MAX_TOKENS,
            )
            last = res[-1].to(dtype=torch.float32, device="cpu").numpy()
            if d_model is None:
                d_model = last.shape[0]
                wt_res = np.zeros((n_para, d_model), dtype=np.float32)
            wt_res[i] = last
            del res
        except Exception as e:
            print(f"[L{layer}] WT paragraph {i} failed: {type(e).__name__}: {e}")
            if d_model is not None and i < n_para:
                pass
        if i % 50 == 0:
            torch.cuda.empty_cache(); gc.collect()
    return wt_res


def run_cell(layer, l0, model, tokenizer, sae, sae_meta, ad_residuals,
             wt_residuals, detected, a_prompts, wt_draws):
    """Run one (layer, L0) cell. Returns summary dict."""
    out_dir = P22_DIR / f"L{layer}" / f"L0_{l0}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n----- L{layer} × L0={l0} -----")
    t_cell = time.time()
    print(f"  SAE sha256: {sae_meta['sha256'][:16]}...")

    # ---- Encode A/D residuals ----
    encodings = {}
    for k, r in ad_residuals.items():
        x = torch.from_numpy(r).to(device=sae.W_enc.device, dtype=sae.W_enc.dtype)
        z = sae.encode(x.unsqueeze(0)).squeeze(0)
        encodings[k] = z.float().cpu().numpy()
    save_npz_atomic(out_dir / "sae_encodings.npz", **encodings)

    # ---- Encode WT residuals (last token per paragraph) ----
    n_para = wt_residuals.shape[0]
    wt_z = np.zeros((n_para, sae.d_sae), dtype=np.float32)
    for i in range(n_para):
        x = torch.from_numpy(wt_residuals[i]).to(device=sae.W_enc.device,
                                                  dtype=sae.W_enc.dtype)
        z = sae.encode(x.unsqueeze(0)).squeeze(0)
        wt_z[i] = z.float().cpu().numpy()

    # ---- Pick Targeted features per (P, D_i) ----
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
        out_dir / "specific_features.json",
        [{"pair_id": pid, "disambig_idx": di, "features": feats}
         for (pid, di), feats in specific_features.items()],
    )

    # ---- WT top-10 per paragraph ----
    wt_top10 = []
    for i in range(n_para):
        z_p = wt_z[i]
        positives = np.where(z_p > 0)[0]
        if len(positives) == 0:
            kept = []
        else:
            sorted_idx = positives[np.argsort(-z_p[positives])]
            kept = [int(f) for f in sorted_idx[:TOP_K_FEATURES]]
        wt_top10.append({"paragraph_idx": i, "top10_feature_ids": kept})

    # ---- Run Targeted (1103 forwards, gives Sibling for free) ----
    wt_by_draw = {(d["pair_id"], d["target_idx"], d["draw_idx"]):
                   d["wikitext_paragraph_idx"]
                   for d in wt_draws}

    self_rows = []; cross_rows = []; wt_rows = []
    for p in tqdm(detected, desc=f"L{layer}.L0={l0} targ+wt"):
        a_prompt = a_prompts[p["id"]]
        # Baseline hits per disambig (one fwd-free per pair)
        from src.hooks import baseline_top_logits
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
                    "base_hit1": base_hits[tg_j],
                    "ablate_hit1": hit_at_k(ab["top_ids"], target, 1),
                }
                if ab_i == tg_j:
                    self_rows.append(row)
                else:
                    cross_rows.append(row)

            target_self = set(p["disambigs"][ab_i]["first_token_variants"])
            for d_idx in range(N_SHUFFLES_PER_PAIR):
                para_idx = wt_by_draw.get((p["id"], ab_i, d_idx))
                if para_idx is None:
                    continue
                feats_wt = wt_top10[para_idx]["top10_feature_ids"]
                wt = forward_with_ablation(
                    model, tokenizer, a_prompt, layer, sae,
                    feature_ids=feats_wt, k=TOP_LOGITS_K,
                )
                wt_rows.append({
                    "pair_id": p["id"], "target_idx": ab_i, "draw_idx": d_idx,
                    "wikitext_paragraph_idx": para_idx,
                    "n_features": len(feats_wt),
                    "base_hit1": base_hits[ab_i],
                    "wt_shuffled_hit1": hit_at_k(wt["top_ids"], target_self, 1),
                })
        torch.cuda.empty_cache()

    # ---- Unique/Shared decomposition ----
    pub_set = {(pid, di): set(feats) for (pid, di), feats in specific_features.items()}
    sh_un_rows = []
    base_lookup = {(r["pair_id"], r["target_idx"]): r["base_hit1"] for r in self_rows}
    targ_lookup = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"] for r in self_rows}
    for p in tqdm(detected, desc=f"L{layer}.L0={l0} sh+un"):
        a_prompt = a_prompts[p["id"]]
        K = len(p["disambigs"])
        for i in range(K):
            key = (p["id"], i)
            ti = pub_set.get(key, set())
            if not ti or key not in base_lookup:
                continue
            sib_union = set()
            for j in range(K):
                if j != i:
                    sib_union |= pub_set.get((p["id"], j), set())
            shared = ti & sib_union
            unique = ti - shared
            target = set(p["disambigs"][i]["first_token_variants"])
            base = base_lookup[key]
            if shared:
                ab = forward_with_ablation(
                    model, tokenizer, a_prompt, layer, sae,
                    feature_ids=list(shared), k=TOP_LOGITS_K,
                )
                shared_hit = hit_at_k(ab["top_ids"], target, 1)
            else:
                shared_hit = base
            if unique:
                ab = forward_with_ablation(
                    model, tokenizer, a_prompt, layer, sae,
                    feature_ids=list(unique), k=TOP_LOGITS_K,
                )
                unique_hit = hit_at_k(ab["top_ids"], target, 1)
            else:
                unique_hit = base
            sh_un_rows.append({
                "pair_id": p["id"], "target_idx": i,
                "n_targeted": len(ti), "n_shared": len(shared),
                "n_unique": len(unique),
                "base_hit1": base,
                "targeted_hit1": targ_lookup[key],
                "shared_only_hit1": shared_hit,
                "unique_only_hit1": unique_hit,
            })
        torch.cuda.empty_cache()

    # ---- Aggregate ----
    sib = per_pair_means(
        cross_rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["ablate_hit1"],
    )
    wt_per_pair = per_pair_means(
        wt_rows,
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["wt_shuffled_hit1"],
    )
    keys_t3 = sorted(k for k in sib if k in wt_per_pair)
    Sib_arr = np.array([sib[k] for k in keys_t3], dtype=float)
    WSh_arr = np.array([wt_per_pair[k] for k in keys_t3], dtype=float)
    t3 = _t3_stats(Sib_arr, WSh_arr)

    sh_keys = [(r["pair_id"], r["target_idx"]) for r in sh_un_rows]
    by_key = {(r["pair_id"], r["target_idx"]): r for r in sh_un_rows}
    Bsh = np.array([by_key[k]["base_hit1"] for k in sh_keys], dtype=float)
    Tsh = np.array([by_key[k]["targeted_hit1"] for k in sh_keys], dtype=float)
    Shsh = np.array([by_key[k]["shared_only_hit1"] for k in sh_keys], dtype=float)
    Unsh = np.array([by_key[k]["unique_only_hit1"] for k in sh_keys], dtype=float)
    decomp = {
        "n": len(sh_keys),
        "table": {
            "Baseline":    {"hit1": float(Bsh.mean()), "delta_pp": 0.0},
            "Targeted":    {"hit1": float(Tsh.mean()), "delta_pp": float((Tsh.mean() - Bsh.mean()) * 100)},
            "Shared-only": {"hit1": float(Shsh.mean()), "delta_pp": float((Shsh.mean() - Bsh.mean()) * 100)},
            "Unique-only": {"hit1": float(Unsh.mean()), "delta_pp": float((Unsh.mean() - Bsh.mean()) * 100)},
        },
        "tests": {
            "shared_vs_baseline": _drop_vs_base_stats(Shsh, Bsh),
            "unique_vs_baseline": _drop_vs_base_stats(Unsh, Bsh),
        },
    }
    save_json_atomic(out_dir / "results.json", {
        "summary": {
            "self_base_hit1":   float(np.mean([r["base_hit1"] for r in self_rows])),
            "self_ablate_hit1": float(np.mean([r["ablate_hit1"] for r in self_rows])),
            "cross_ablate_hit1": float(np.mean([r["ablate_hit1"] for r in cross_rows])),
            "wt_shuf_ablate_hit1": float(np.mean([r["wt_shuffled_hit1"] for r in wt_rows])),
            "n_self": len(self_rows), "n_cross": len(cross_rows), "n_wt": len(wt_rows),
        },
        "self_rows": self_rows,
        "cross_rows": cross_rows,
        "wikitext_shuffled_rows": wt_rows,
        "T3": t3,
    })
    save_json_atomic(out_dir / "unique_shared_decomp.json", {
        "decomposition": decomp,
        "rows": sh_un_rows,
    })

    # ---- Cleanup SAE blob ----
    elapsed = time.time() - t_cell
    save_json_atomic(out_dir / "run_meta.json", {
        "layer": layer, "l0": l0, "model": MODEL_ID,
        "sae_meta": sae_meta,
        "n_self_pairs": len(sh_keys),
        "elapsed_seconds": round(elapsed, 2),
    })
    try:
        local_path = sae_meta.get("local_path")
        if local_path:
            real = Path(local_path).resolve()
            if real.exists():
                real.unlink()
                print(f"  deleted SAE blob from HF cache to free space")
    except Exception as e:
        print(f"  sae blob cleanup failed: {e}")
    print(f"  T3 Δ={t3['delta_pp']:+.2f} (Wlx p={t3['wilcoxon_p']:.3g}); "
          f"Sh Δ={decomp['table']['Shared-only']['delta_pp']:+.2f}; "
          f"Un Δ={decomp['table']['Unique-only']['delta_pp']:+.2f}; "
          f"{elapsed/60:.1f} min")

    return {
        "layer": layer,
        "l0": l0,
        "sae_sha256": sae_meta["sha256"],
        "n_self_pairs": len(sh_keys),
        "T3": t3,
        "decomposition": decomp,
    }


def main() -> None:
    P22_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # ---- Setup ----
    print(f"[setup] loading {MODEL_ID}...")
    tokenizer, model = load_model()
    detected = json.load(open(ARTIFACTS_DIR / "detected_pairs.json"))
    a_prompts = {p["id"]: build_prompt(tokenizer, p["A_question"])
                 for p in detected}
    wt_draws = json.load(open(ARTIFACTS_DIR / "shuffle_draws_wikitext.json"))
    paragraphs = load_wikitext_paragraphs(tokenizer)
    print(f"[setup] {len(detected)} detected pairs, "
          f"{sum(len(p['disambigs']) for p in detected)} self-pairs, "
          f"{len(paragraphs)} WikiText paragraphs, {len(wt_draws)} WT draws")

    layer_summaries = {}
    for layer in [37, 41]:
        print(f"\n========== Layer L{layer} ==========")
        # Capture (or load) layer's residuals once
        ad_residuals = _capture_or_load_residuals(
            model, tokenizer, layer, detected, a_prompts,
        )
        wt_residuals = _capture_wt_residuals(model, tokenizer, layer, paragraphs)

        layer_cells = []
        for l0 in SWEEP_L0[layer]:
            cell_meta_path = P22_DIR / f"L{layer}" / f"L0_{l0}" / "run_meta.json"
            if cell_meta_path.exists():
                print(f"[L{layer}.L0={l0}] already complete — skipping")
                # Reload cell summary
                rm = json.load(open(cell_meta_path))
                ushd = json.load(open(P22_DIR / f"L{layer}" / f"L0_{l0}"
                                       / "unique_shared_decomp.json"))
                results = json.load(open(P22_DIR / f"L{layer}" / f"L0_{l0}"
                                          / "results.json"))
                layer_cells.append({
                    "layer": layer, "l0": l0,
                    "sae_sha256": rm["sae_meta"]["sha256"],
                    "n_self_pairs": rm["n_self_pairs"],
                    "T3": results["T3"],
                    "decomposition": ushd["decomposition"],
                })
                continue
            try:
                sae, sae_meta = _load_sae_explicit(layer, l0)
            except RuntimeError as e:
                print(f"[L{layer}.L0={l0}] SAE load failed: {e}")
                continue
            try:
                summary = run_cell(
                    layer, l0, model, tokenizer, sae, sae_meta,
                    ad_residuals, wt_residuals, detected, a_prompts, wt_draws,
                )
                layer_cells.append(summary)
            except Exception as e:
                print(f"[L{layer}.L0={l0}] FAILED: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            del sae
            torch.cuda.empty_cache(); gc.collect()
        layer_summaries[layer] = layer_cells
        del ad_residuals, wt_residuals
        gc.collect()

    # ---- Aggregate ----
    save_json_atomic(P22_DIR / "l0_sweep_summary.json", {
        "model": MODEL_ID,
        "canonical_l0_per_layer": CANONICAL_L0,
        "layers": list(layer_summaries.keys()),
        "summaries": {str(L): cells for L, cells in layer_summaries.items()},
        "elapsed_seconds": round(time.time() - t_total, 2),
    })

    # ---- Figure: T3 vs L0, two lines ----
    print(f"\n[fig] generating T3-vs-L0 figure")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = {37: "#2c7fb8", 41: "#c0392b"}
    canon_t3 = {37: 4.49, 41: 9.05}     # canonical-L0 T3 from the layer sweep
    for L, cells in layer_summaries.items():
        if not cells:
            continue
        cells_sorted = sorted(cells, key=lambda c: c["l0"])
        l0_x = [c["l0"] for c in cells_sorted]
        t3_y = [c["T3"]["delta_pp"] for c in cells_sorted]
        t3_lo = [c["T3"]["ci_pp"][0] for c in cells_sorted]
        t3_hi = [c["T3"]["ci_pp"][1] for c in cells_sorted]
        yerr = [[d - lo for d, lo in zip(t3_y, t3_lo)],
                [hi - d for hi, d in zip(t3_hi, t3_y)]]
        # Add canonical point
        canon = CANONICAL_L0[L]
        ax.errorbar(l0_x, t3_y, yerr=yerr, fmt="o-",
                    color=colors[L], ecolor="#888", capsize=4,
                    markersize=7, linewidth=1.4,
                    label=f"L{L} (canonical L0={canon}: T3=+{canon_t3[L]:.2f}pp)")
        ax.scatter([canon], [canon_t3[L]], marker="*", s=180,
                    color=colors[L], edgecolor="white", linewidth=1.2, zorder=5)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("SAE average L0 (lower = sparser)")
    ax.set_ylabel("T3 Δ pp (Sibling − WikiText-shuffled)")
    ax.set_title("T3 vs L0 across published Gemma Scope SAEs\n"
                 "(★ = canonical L0)")
    ax.set_xscale("log")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.3, linewidth=0.5)
    plt.tight_layout()
    fig_path = P22_DIR / "figure_t3_vs_l0.png"
    plt.savefig(fig_path, dpi=140)
    plt.close()
    print(f"[fig] wrote {fig_path}")

    # ---- Stdout summary ----
    print(f"\n=== Stage 2.2 complete ===")
    print(f"  total elapsed: {(time.time()-t_total)/60:.1f} min")
    for L, cells in layer_summaries.items():
        print(f"\n  L{L}:")
        canon = CANONICAL_L0[L]
        print(f"    L0={canon} (canonical): T3={canon_t3[L]:+.2f}")
        for c in sorted(cells, key=lambda c: c["l0"]):
            t3 = c["T3"]; d = c["decomposition"]["table"]
            print(f"    L0={c['l0']:<3}: T3 Δ={t3['delta_pp']:+5.2f} "
                  f"(p={t3['wilcoxon_p']:.2e})  Sh Δ={d['Shared-only']['delta_pp']:+5.2f}  "
                  f"Un Δ={d['Unique-only']['delta_pp']:+5.2f}")


if __name__ == "__main__":
    main()
