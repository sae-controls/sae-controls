"""Per-pair distribution audit of the orthogonal decompositions.

Mostly CPU re-analysis of existing per-pair data. Three sub-experiments:

  (A) Per-pair distributions of four headline metrics:
        Δ_T3      = (Sibling hit) − (WT-shuffled hit)
        Δ_targeted = (baseline hit) − (Targeted hit)
        Δ_shared   = (baseline hit) − (Shared-only hit)
        Δ_unique   = (baseline hit) − (Unique-only hit)
      Histograms with vertical line at aggregate Δ; mean/median/std/IQR
      and percent-of-pairs in expected direction.

  (B) Trim tests. Drop top-10% and top-25% of pairs by |per-pair Δ|.
      Re-run T3 and Δ_unique vs baseline / Δ_shared vs baseline on the
      bulk. Confirms whether signals are bulk- vs tail-driven.

  (C) Per-pair correlates regression + power curve.
      Predictors: n_disambigs, score margin, baseline hit@1, len(A),
      n_position_features, mean magnitude, |unique|, |shared|, max
      overlap. Outcomes: Δ_T3, Δ_unique, Δ_shared. Linear regression
      (HC3 robust SEs) + standardized β + partial R².
      Power curve for T3 across n ∈ {100, 250, 500, 750, 1000, 1103,
      1500, 2000} using subsample (n < 1103) or bootstrap (n > 1103),
      1000 reps each.

Reads:
  artifacts/detected_pairs.json
  artifacts/specific_features.json
  artifacts/results_main.json
  artifacts/wikitext_position_mode.json
  artifacts/sae_encodings_L37.npz
  artifacts/unique_vs_shared/a/ablation_rows.json   (shared/unique hits per pair)

Writes:
  artifacts/per_pair_audit/a/{distribution_table.json, headline_histograms.png}
  artifacts/per_pair_audit/b/trim_test.json
  artifacts/per_pair_audit/c/{predictors_per_pair.json, regression_results.json,
                          power_curve.json, power_curve.png}
  artifacts/per_pair_audit/run_meta.json
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    ARTIFACTS_DIR, LAYER, MODEL_ID, POSITION_NONZERO_FLOOR,
    POSITION_PCT_THRESHOLD, SAE_CHECKPOINT_SHA256, SCORE_A_WEIGHT,
)
from src.io_utils import save_json_atomic
from src.position_mode import is_position_suspect
from src.sae import load_sae


P16_DIR = ARTIFACTS_DIR / "per_pair_audit"


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
    """Test whether arr drops more than base (ablation kills more)."""
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
# (A) Per-pair distributions
# =============================================================================

def run_a(detected, results_main, p12pp_rows):
    out_dir = P16_DIR / "a"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (A) per-pair distributions ==========")

    # Build per-pair arrays
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
    baseline = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                for r in results_main["self_rows"]}
    targeted = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                for r in results_main["self_rows"]}
    shared_only = {(r["pair_id"], r["target_idx"]): r["shared_only_hit1"]
                   for r in p12pp_rows}
    unique_only = {(r["pair_id"], r["target_idx"]): r["unique_only_hit1"]
                   for r in p12pp_rows}

    keys = sorted(k for k in baseline if k in targeted and k in shared_only
                   and k in unique_only and k in sibling and k in wt_shuf)
    n = len(keys)
    print(f"[A] n_self_pairs with all 4 metrics defined: {n}")

    Sib = np.array([sibling[k] for k in keys], dtype=float)
    WSh = np.array([wt_shuf[k] for k in keys], dtype=float)
    Base = np.array([baseline[k] for k in keys], dtype=float)
    Targ = np.array([targeted[k] for k in keys], dtype=float)
    Sh = np.array([shared_only[k] for k in keys], dtype=float)
    Un = np.array([unique_only[k] for k in keys], dtype=float)

    delta_T3 = Sib - WSh                # negative = Sibling drops more (T3 supports)
    delta_targeted = Base - Targ        # positive = ablation drops baseline
    delta_shared = Base - Sh
    delta_unique = Base - Un

    def _stats(arr, name, support_dir):
        """support_dir: '+' = positive favors ablation-drops-D_i, '-' = negative favors."""
        d = {
            "metric": name,
            "support_direction": support_dir,
            "n": len(arr),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "std": float(arr.std(ddof=1)),
            "iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
            "min": float(arr.min()),
            "max": float(arr.max()),
            "pct_geq_0":   float(100 * np.mean(arr >= 0)),
            "pct_geq_0p5": float(100 * np.mean(arr >= 0.5)),
            "pct_leq_0":   float(100 * np.mean(arr <= 0)),
            "pct_leq_neg0p5": float(100 * np.mean(arr <= -0.5)),
            "aggregate_delta_pp": float(arr.mean() * 100),
        }
        return d

    dist_table = {
        "Δ_T3 (Sibling − WT-shuffled)":
            _stats(delta_T3, "Δ_T3", "-"),
        "Δ_targeted (baseline − Targeted)":
            _stats(delta_targeted, "Δ_targeted", "+"),
        "Δ_shared (baseline − Shared-only)":
            _stats(delta_shared, "Δ_shared", "+"),
        "Δ_unique (baseline − Unique-only)":
            _stats(delta_unique, "Δ_unique", "+"),
    }
    save_json_atomic(out_dir / "distribution_table.json", dist_table)
    for label, s in dist_table.items():
        if s["support_direction"] == "+":
            pct_supp = s["pct_geq_0p5"]
            pct_neutral_or_supp = s["pct_geq_0"]
            supp_word = "≥0.5"
        else:
            pct_supp = s["pct_leq_neg0p5"]
            pct_neutral_or_supp = s["pct_leq_0"]
            supp_word = "≤−0.5"
        print(f"[A] {label:<40}  mean={s['mean']:+.4f}  median={s['median']:+.4f}  "
              f"std={s['std']:.4f}  IQR=[{s['iqr'][0]:+.4f},{s['iqr'][1]:+.4f}]  "
              f"% in support direction (Δ {supp_word}): {pct_supp:.1f}%  "
              f"(% supporting at all: {pct_neutral_or_supp:.1f}%)")

    # ---- Histograms ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, (arr, label, agg, sd) in zip(axes, [
        (delta_T3,       "Δ_T3 (Sib − WT)",       delta_T3.mean()*100,       "-"),
        (delta_targeted, "Δ_targeted (B − T)",    delta_targeted.mean()*100, "+"),
        (delta_shared,   "Δ_shared (B − Sh)",     delta_shared.mean()*100,   "+"),
        (delta_unique,   "Δ_unique (B − Un)",     delta_unique.mean()*100,   "+"),
    ]):
        bins = np.linspace(arr.min(), arr.max(), 41) if arr.max() > arr.min() else 10
        ax.hist(arr, bins=bins, color="#37a", edgecolor="white")
        ax.axvline(arr.mean(), color="red", linestyle="--", linewidth=2,
                   label=f"aggregate = {agg:+.2f} pp")
        ax.axvline(0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("per-pair value")
        ax.set_ylabel("count")
        ax.set_title(f"{label}\nsupport direction: {sd}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "headline_histograms.png", dpi=140)
    plt.close()
    return dist_table, (Sib, WSh, Base, Targ, Sh, Un, keys)


# =============================================================================
# (B) Trim tests
# =============================================================================

def run_b(Sib, WSh, Base, Sh, Un):
    out_dir = P16_DIR / "b"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (B) trim tests ==========")

    def _trim(arr_a, arr_b, drop_pct, t3_mode=False):
        """Drop top-(drop_pct) by |arr_a - arr_b|. Re-run paired test on bulk.
        t3_mode=True uses _t3_stats, otherwise _drop_vs_base_stats."""
        diffs = arr_a - arr_b
        abs_diffs = np.abs(diffs)
        n = len(abs_diffs)
        n_drop = int(round(n * drop_pct))
        if n_drop > 0:
            keep_idx = np.argsort(abs_diffs)[:-n_drop]
        else:
            keep_idx = np.arange(n)
        a_b = arr_a[keep_idx]; b_b = arr_b[keep_idx]
        if t3_mode:
            stats = _t3_stats(a_b, b_b)
        else:
            stats = _drop_vs_base_stats(a_b, b_b)
        return {"drop_pct": drop_pct, "n_dropped": n_drop, **stats}

    out = {}
    for name, kwargs in [
        ("T3 (Sibling vs WT-shuffled)", dict(arr_a=Sib, arr_b=WSh, t3_mode=True)),
        ("Δ_unique vs baseline",         dict(arr_a=Un,  arr_b=Base)),
        ("Δ_shared vs baseline",         dict(arr_a=Sh,  arr_b=Base)),
    ]:
        rows = []
        for pct in [0.0, 0.10, 0.25]:
            rows.append(_trim(drop_pct=pct, **kwargs))
        out[name] = rows
        print(f"[B] {name}:")
        for r in rows:
            print(f"  drop {r['drop_pct']*100:.0f}%  n={r['n']:<5}  Δ={r['delta_pp']:+6.2f}  "
                  f"CI=[{r['ci_pp'][0]:+5.2f},{r['ci_pp'][1]:+5.2f}]  "
                  f"McN p={r['mcnemar_p']:.3g}  Wlx p={r['wilcoxon_p']:.3g}")
    save_json_atomic(out_dir / "trim_test.json", out)
    return out


# =============================================================================
# (C) Per-pair regression + power curve
# =============================================================================

def run_c(detected, results_main, p12pp_rows, encodings, per_feature_pos,
          pid_to_pair):
    out_dir = P16_DIR / "c"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== (C) regression + power curve ==========")

    # Build predictors per (P, D_i)
    suspect_080 = {int(fid) for fid, st in per_feature_pos.items()
                   if st["n_nonzero"] >= POSITION_NONZERO_FLOOR
                   and st["pct_at_pos0"] >= POSITION_PCT_THRESHOLD}

    published_specific = json.load(open(ARTIFACTS_DIR / "specific_features.json"))
    pub = {(e["pair_id"], e["disambig_idx"]): list(e["features"])
           for e in published_specific}

    base_hit = {(r["pair_id"], r["target_idx"]): r["base_hit1"]
                for r in results_main["self_rows"]}
    targ_hit = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"]
                for r in results_main["self_rows"]}
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
    shared_only = {(r["pair_id"], r["target_idx"]): r["shared_only_hit1"]
                   for r in p12pp_rows}
    unique_only = {(r["pair_id"], r["target_idx"]): r["unique_only_hit1"]
                   for r in p12pp_rows}

    # Pair-level predictors that don't depend on D_i
    pair_K = {p["id"]: len(p["disambigs"]) for p in detected}
    pair_A_len = {p["id"]: len(p["A_question"]) for p in detected}

    rows = []
    for p in detected:
        K = len(p["disambigs"])
        for i in range(K):
            key = (p["id"], i)
            ti = pub.get(key)
            if not ti or key not in base_hit or key not in shared_only:
                continue

            # Score margin: max score - second-max score within Targeted top-10
            z_A = encodings[f"A__{p['id']}"]
            z_D_list = [encodings[f"D__{p['id']}__{di}"] for di in range(K)]
            z_D_others = [z_D_list[j] for j in range(K) if j != i]
            mean_other = (np.stack(z_D_others, axis=0).mean(axis=0)
                           if z_D_others else np.zeros_like(z_D_list[i]))
            scores_in_top10 = np.array([
                z_D_list[i][int(f)] - mean_other[int(f)] + SCORE_A_WEIGHT * z_A[int(f)]
                for f in ti
            ])
            sorted_scores = np.sort(scores_in_top10)[::-1]
            score_margin = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.0
            mean_z_Di_top10 = float(np.mean([float(z_D_list[i][int(f)]) for f in ti]))

            # n_position_features in Targeted top-10
            n_pos = sum(1 for f in ti if int(f) in suspect_080)

            # |unique|, |shared|, max-overlap
            sib_union = set()
            for j in range(K):
                if j == i:
                    continue
                sib_union |= set(pub.get((p["id"], j), []))
            shared = set(ti) & sib_union
            unique = set(ti) - shared
            sib_overlaps = [len(set(ti) & set(pub.get((p["id"], j), [])))
                            for j in range(K) if j != i]
            max_overlap = int(max(sib_overlaps)) if sib_overlaps else 0

            row = {
                "pair_id": p["id"], "target_idx": i,
                "n_disambigs": int(K),
                "score_margin": float(score_margin),
                "baseline_hit1": float(base_hit[key]),
                "len_A_chars": int(pair_A_len[p["id"]]),
                "n_position_features": int(n_pos),
                "mean_zDi_top10": float(mean_z_Di_top10),
                "n_unique": int(len(unique)),
                "n_shared": int(len(shared)),
                "max_overlap": int(max_overlap),
                # Outcomes
                "delta_T3": float(sibling[key] - wt_shuf[key]) if (key in sibling and key in wt_shuf) else None,
                "delta_unique": float(base_hit[key] - unique_only[key]),
                "delta_shared": float(base_hit[key] - shared_only[key]),
                "delta_targeted": float(base_hit[key] - targ_hit[key]),
            }
            rows.append(row)
    save_json_atomic(out_dir / "predictors_per_pair.json", rows)
    print(f"[C] built {len(rows)} per-pair predictor rows")

    # ---- Regression ----
    import statsmodels.api as sm

    PREDICTOR_COLS = [
        "n_disambigs", "score_margin", "baseline_hit1", "len_A_chars",
        "n_position_features", "mean_zDi_top10", "n_unique", "n_shared",
        "max_overlap",
    ]
    X = np.array([[r[c] for c in PREDICTOR_COLS] for r in rows], dtype=float)
    # Standardize
    Xz_mean = X.mean(axis=0); Xz_std = X.std(axis=0, ddof=1)
    Xz = (X - Xz_mean) / np.where(Xz_std > 0, Xz_std, 1.0)
    Xz_with_intercept = sm.add_constant(Xz)

    reg_results = {}
    for outcome in ["delta_T3", "delta_unique", "delta_shared"]:
        y = np.array([r[outcome] for r in rows
                      if r[outcome] is not None], dtype=float)
        valid_idx = [i for i, r in enumerate(rows) if r[outcome] is not None]
        Xz_v = Xz_with_intercept[valid_idx]
        if len(y) < 30:
            print(f"[C] {outcome}: insufficient data ({len(y)})")
            continue
        try:
            model = sm.OLS(y, Xz_v).fit(cov_type="HC3")
        except Exception as e:
            print(f"[C] {outcome}: OLS failed: {e}")
            continue
        full_R2 = float(model.rsquared)
        # Partial R^2: 1 - (RSS_full / RSS_reduced) for each predictor
        rss_full = float(np.sum(model.resid ** 2))
        partial_R2 = {}
        for j, name in enumerate(PREDICTOR_COLS):
            cols_drop_j = [k for k in range(Xz_v.shape[1]) if k != (j + 1)]
            X_sub = Xz_v[:, cols_drop_j]
            try:
                m_red = sm.OLS(y, X_sub).fit()
                rss_red = float(np.sum(m_red.resid ** 2))
                partial = 1.0 - (rss_full / rss_red) if rss_red > 0 else None
            except Exception:
                partial = None
            partial_R2[name] = partial

        coefs = model.params
        pvals = model.pvalues
        ses = model.bse
        coef_table = []
        for j, name in enumerate(PREDICTOR_COLS):
            coef_table.append({
                "predictor": name,
                "beta_std": float(coefs[j + 1]),
                "se_HC3": float(ses[j + 1]),
                "p_value": float(pvals[j + 1]),
                "partial_R2": partial_R2[name],
            })
        reg_results[outcome] = {
            "n": len(y),
            "full_R2": full_R2,
            "intercept": float(coefs[0]),
            "coefficients": coef_table,
        }
        print(f"[C] {outcome}: n={len(y)}, R²={full_R2:.4f}")
        for c in coef_table:
            sig = "***" if c["p_value"] < 0.001 else "**" if c["p_value"] < 0.01 else "*" if c["p_value"] < 0.05 else ""
            print(f"  {c['predictor']:<20}  β_std={c['beta_std']:+.4f}  "
                  f"SE={c['se_HC3']:.4f}  p={c['p_value']:.3g}{sig:<3}  "
                  f"partial R²={c['partial_R2']}")

    save_json_atomic(out_dir / "regression_results.json", reg_results)

    # ---- Power curve for T3 ----
    print(f"\n[C] computing T3 power curve")
    from scipy.stats import wilcoxon as scipy_wilcoxon
    keys_T3 = [(r["pair_id"], r["target_idx"]) for r in rows
               if r["delta_T3"] is not None]
    Sib_arr = np.array([sibling[k] for k in keys_T3])
    WSh_arr = np.array([wt_shuf[k] for k in keys_T3])
    n_full = len(Sib_arr)
    print(f"[C] full T3 n = {n_full}")

    POWER_NS = [100, 250, 500, 750, 1000, n_full, 1500, 2000]
    N_REPS = 1000
    rng = np.random.default_rng(0)
    power_curve = []
    for n_target in POWER_NS:
        n_significant = 0
        for rep in range(N_REPS):
            if n_target <= n_full:
                idx = rng.choice(n_full, size=n_target, replace=False)
            else:
                idx = rng.choice(n_full, size=n_target, replace=True)
            try:
                p = float(scipy_wilcoxon(WSh_arr[idx], Sib_arr[idx],
                                          alternative="greater").pvalue)
            except Exception:
                p = 1.0
            if p < 0.001:
                n_significant += 1
        power = n_significant / N_REPS
        power_curve.append({"n_target": n_target, "power_at_p<0.001": float(power)})
        print(f"  n={n_target:<5}  power = {power:.3f}")
    save_json_atomic(out_dir / "power_curve.json", {"power_curve": power_curve})

    # n at 80% power (interpolated)
    n_at_80 = None
    for i in range(len(power_curve) - 1):
        a, b = power_curve[i], power_curve[i + 1]
        if a["power_at_p<0.001"] <= 0.80 <= b["power_at_p<0.001"]:
            # Linear interp
            t = (0.80 - a["power_at_p<0.001"]) / max(
                b["power_at_p<0.001"] - a["power_at_p<0.001"], 1e-9)
            n_at_80 = a["n_target"] + t * (b["n_target"] - a["n_target"])
            break

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = [p["n_target"] for p in power_curve]
    ps = [p["power_at_p<0.001"] for p in power_curve]
    ax.plot(ns, ps, "o-", linewidth=2, color="#37a", markersize=8)
    ax.axhline(0.80, color="red", linestyle="--", label="80% power")
    if n_at_80 is not None:
        ax.axvline(n_at_80, color="orange", linestyle=":",
                   label=f"n at 80% = {n_at_80:.0f}")
    ax.set_xlabel("n self-pairs")
    ax.set_ylabel("power: P(Wilcoxon p < 0.001)")
    ax.set_title("T3 power curve (1-sided Sibling drops more)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "power_curve.png", dpi=140)
    plt.close()
    print(f"[C] n at 80% power: {n_at_80}")
    return reg_results, power_curve, n_at_80


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    P16_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    print(f"[setup] verifying SAE SHA via load_sae...")
    sae, sae_meta = load_sae(layer=LAYER)
    if sae_meta["sha256"] != SAE_CHECKPOINT_SHA256:
        raise RuntimeError(f"SAE sha256 mismatch! got {sae_meta['sha256']}")
    print(f"[setup] SAE sha256 verified ({sae_meta['sha256'][:16]}...)")
    del sae   # not needed — re-analysis only

    detected = json.load(open(ARTIFACTS_DIR / "detected_pairs.json"))
    pid_to_pair = {p["id"]: p for p in detected}
    results_main = json.load(open(ARTIFACTS_DIR / "results_main.json"))
    p12pp_rows = json.load(open(ARTIFACTS_DIR / "unique_vs_shared" / "a"
                                  / "ablation_rows.json"))
    pos_mode = json.load(open(ARTIFACTS_DIR / "wikitext_position_mode.json"))
    per_feature_pos = pos_mode["per_feature"]
    encodings = {k: np.load(ARTIFACTS_DIR / f"sae_encodings_L{LAYER}.npz")[k]
                 for k in np.load(ARTIFACTS_DIR / f"sae_encodings_L{LAYER}.npz").files}

    print(f"[setup] {len(detected)} detected pairs, "
          f"{sum(len(p['disambigs']) for p in detected)} self-pairs, "
          f"{len(p12pp_rows)} p12pp rows")

    dist_table, arrays = run_a(detected, results_main, p12pp_rows)
    Sib, WSh, Base, Targ, Sh, Un, keys = arrays
    trim_table = run_b(Sib, WSh, Base, Sh, Un)
    reg_results, power_curve, n_at_80 = run_c(
        detected, results_main, p12pp_rows, encodings, per_feature_pos,
        pid_to_pair,
    )

    save_json_atomic(P16_DIR / "run_meta.json", {
        "model": MODEL_ID, "layer": LAYER, "sae_meta": sae_meta,
        "n_self_pairs": len(keys),
        "n_at_80_power_for_T3": n_at_80,
        "elapsed_seconds": round(time.time() - t_total, 2),
    })
    print(f"\n=== Stage 1.6 complete ===")
    print(f"  total elapsed: {(time.time()-t_total)/60:.1f} min")
    print(f"  -> {P16_DIR}")


if __name__ == "__main__":
    main()
