"""Trajectory bookends (L20, L41).

Wraps the layer-sweep run_layer to add bookend layers L20 and L41 to the
T3 trajectory characterization, then aggregates a 7-point summary
(L20, L26, L30, L34, L37*, L40, L41) using:
  - L20, L41:           run fresh by this script
  - L26, L30, L34, L40: read from artifacts/layer_sweep/L{L}/
  - L37 (published):    read from artifacts/results_main.json + published
                         decomposition + single-feature artifacts

Reuses run_layer + helpers from pipeline.layer_sweep.
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

from src.analysis import (
    bootstrap_diff_ci_pp, mcnemar_paired, per_pair_means, wilcoxon_paired,
)
from src.config import (
    ARTIFACTS_DIR, MODEL_ID, POSITION_NONZERO_FLOOR, POSITION_PCT_THRESHOLD,
    SAE_CHECKPOINT_SHA256,
)
from src.io_utils import save_json_atomic
from src.model import load_model
from src.prompts import build_prompt
from src.wikitext import load_paragraphs as load_wikitext_paragraphs

# Import shared helpers + run_layer from layer_sweep
import pipeline.layer_sweep as p21
from pipeline.layer_sweep import (
    run_layer, _t3_stats, _drop_vs_base_stats,
)


P21_DIR_PARENT = ARTIFACTS_DIR / "layer_sweep"   # for reading L26/30/34/40
P21B_DIR = ARTIFACTS_DIR / "layer_bookends"
LAYERS_NEW = [20, 41]   # L41 is the last layer in Gemma-2-9B (42 layers, 0-indexed)
ALL_LAYERS_ORDER = [20, 26, 30, 34, 37, 40, 41]


def _reload_layer_summary_from_disk(p21_dir, layer):
    """Reconstruct a layer's summary dict from saved JSON artifacts."""
    rm = json.load(open(p21_dir / f"L{layer}" / "run_meta.json"))
    rmain = json.load(open(p21_dir / f"L{layer}" / "results_main.json"))
    ushd = json.load(open(p21_dir / f"L{layer}" / "unique_shared_decomp.json"))
    pfeq = json.load(open(p21_dir / f"L{layer}" / "per_feature_equivalence.json"))
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
    return {
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


def _l37_summary_from_published():
    pub_results = json.load(open(ARTIFACTS_DIR / "results_main.json"))
    pub_main = pub_results["summary"]
    pub_decomp = json.load(open(ARTIFACTS_DIR / "unique_vs_shared" / "a"
                                 / "decomposition_table.json"))
    pub_sf = json.load(open(ARTIFACTS_DIR / "per_feature_equivalence" / "c"
                             / "single_feature_summary.json"))
    pub_sib = per_pair_means(
        pub_results["cross_rows"],
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["ablate_hit1"],
    )
    pub_wt = per_pair_means(
        pub_results["wikitext_shuffled_rows"],
        key_fn=lambda r: (r["pair_id"], r["target_idx"]),
        value_fn=lambda r: r["wt_shuffled_hit1"],
    )
    keys_pub = sorted(k for k in pub_sib if k in pub_wt)
    Sib_pub = np.array([pub_sib[k] for k in keys_pub])
    WSh_pub = np.array([pub_wt[k] for k in keys_pub])
    pub_t3 = _t3_stats(Sib_pub, WSh_pub)
    return {
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
        "tests": {"T3_sibling_vs_wtshuf": pub_t3},
        "decomposition": {
            "n": pub_decomp["n"],
            "table": {
                "Shared-only": {"delta_pp": pub_decomp["table"]["Shared-only"]["delta_pp"]},
                "Unique-only": {"delta_pp": pub_decomp["table"]["Unique-only"]["delta_pp"]},
            },
            "tests": {
                "shared_vs_baseline": {"wilcoxon_p": pub_decomp["tests"]["shared_vs_baseline"]["wilcoxon_p"]},
                "unique_vs_baseline": {"wilcoxon_p": pub_decomp["tests"]["unique_vs_baseline"]["wilcoxon_p"]},
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


def main() -> None:
    P21B_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # Override the parent module's output dir so run_layer writes to layer_bookends
    p21.P21_DIR = P21B_DIR

    # ---- Setup (mirror parent main) ----
    print(f"[setup] loading {MODEL_ID}...")
    tokenizer, model = load_model()
    detected = json.load(open(ARTIFACTS_DIR / "detected_pairs.json"))
    a_prompts = {p["id"]: build_prompt(tokenizer, p["A_question"])
                 for p in detected}
    paragraphs = load_wikitext_paragraphs(tokenizer)
    print(f"[setup] {len(detected)} detected pairs, "
          f"{sum(len(p['disambigs']) for p in detected)} self-pairs, "
          f"{len(paragraphs)} WikiText paragraphs")

    shuffle_draws_ambig = json.load(open(ARTIFACTS_DIR / "shuffle_draws_ambigqa.json"))
    shuffle_draws_wt = json.load(open(ARTIFACTS_DIR / "shuffle_draws_wikitext.json"))
    pub_results = json.load(open(ARTIFACTS_DIR / "results_main.json"))
    random_rows = pub_results["random_rows"]

    pos_mode = json.load(open(ARTIFACTS_DIR / "wikitext_position_mode.json"))
    suspect_080 = {int(fid) for fid, st in pos_mode["per_feature"].items()
                   if st["n_nonzero"] >= POSITION_NONZERO_FLOOR
                   and st["pct_at_pos0"] >= POSITION_PCT_THRESHOLD}

    # ---- Run L20 and L41 (auto-skip if complete) ----
    new_summaries = []
    for layer in LAYERS_NEW:
        meta_path = P21B_DIR / f"L{layer}" / "run_meta.json"
        if meta_path.exists():
            print(f"[L{layer}] already complete — loading run_meta.json")
            try:
                new_summaries.append(_reload_layer_summary_from_disk(P21B_DIR, layer))
                continue
            except Exception as e:
                print(f"[L{layer}] failed to load existing artifacts: {e}; rerunning")
        try:
            summary = run_layer(
                layer, model, tokenizer, detected, a_prompts, paragraphs,
                shuffle_draws_ambig, shuffle_draws_wt, random_rows, suspect_080,
            )
            new_summaries.append(summary)
        except Exception as e:
            print(f"[L{layer}] FAILED: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # ---- Aggregate 7 layers ----
    print(f"\n[aggregate] building 7-point trajectory...")
    all_summaries = list(new_summaries)
    for layer in [26, 30, 34, 40]:
        try:
            all_summaries.append(_reload_layer_summary_from_disk(P21_DIR_PARENT, layer))
        except Exception as e:
            print(f"[aggregate] could not reload L{layer} from parent: {e}")
    try:
        all_summaries.append(_l37_summary_from_published())
    except Exception as e:
        print(f"[aggregate] could not load L37 published: {e}")
    all_summaries.sort(key=lambda s: s["layer"])

    save_json_atomic(P21B_DIR / "trajectory_summary.json", {
        "model": MODEL_ID,
        "layers": [s["layer"] for s in all_summaries],
        "summaries": all_summaries,
        "elapsed_seconds": round(time.time() - t_total, 2),
    })

    # ---- 7-point trajectory figure ----
    print(f"[fig] generating 7-point T3 trajectory figure")
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

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.errorbar(layers_list, t3_d, yerr=yerr, fmt="o-", capsize=4,
                color="#2c7fb8", ecolor="#666", elinewidth=0.9,
                markersize=8, markerfacecolor="#2c7fb8",
                markeredgecolor="white", markeredgewidth=0.7,
                linewidth=1.4)
    for x, d in zip(layers_list, t3_d):
        marker = "*" if x == 37 else ""
        ax.text(x, d + 0.4, f"{d:+.2f}{marker}", ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("residual-stream layer")
    ax.set_ylabel("T3 Δ pp (Sibling − WikiText-shuffled, signed)")
    ax.set_title("T3 trajectory across 7 layers\n"
                 "(L37* = published reference)")
    ax.set_xticks(layers_list)
    ax.set_xticklabels([f"L{l}{'*' if l == 37 else ''}" for l in layers_list])
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig_path = P21B_DIR / "figure_t3_trajectory_7pt.png"
    plt.savefig(fig_path, dpi=140)
    plt.close()
    print(f"[fig] wrote {fig_path}")

    # ---- Stdout summary ----
    print(f"\n=== Stage 2.1b complete ===")
    print(f"  total elapsed: {(time.time()-t_total)/60:.1f} min")
    print(f"  -> {P21B_DIR}/trajectory_summary.json")
    print(f"  -> {fig_path}")
    print(f"\n  7-layer trajectory:")
    for s in all_summaries:
        L = s["layer"]; t3 = s["tests"].get("T3_sibling_vs_wtshuf", {})
        d = s["decomposition"]["table"]
        sf = s["single_feature_equivalence"]
        marker = " *" if L == 37 else "  "
        print(f"    L{L}{marker}: T3 Δ={t3.get('delta_pp', float('nan')):+5.2f} "
              f"(p={t3.get('wilcoxon_p', float('nan')):.2e})  "
              f"Sh Δ={d['Shared-only']['delta_pp']:+5.2f}  "
              f"Un Δ={d['Unique-only']['delta_pp']:+5.2f}  "
              f"sf head-to-head Wlx p={sf.get('uc_vs_sc_two_sided_wilcoxon_p', float('nan')):.3g}")


if __name__ == "__main__":
    main()
