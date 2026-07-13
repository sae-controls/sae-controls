"""Sensitivity of the shared/unique PARTITION COMPOSITION to the score
hyperparameter a_weight (Eq. 1's coefficient on z_A), at L41.

Motivation
----------
Eq. 1 selects features by score(f,i) = z_{D_i}(f) - mean_{j!=i} z_{D_j}(f)
+ a*z_A(f), with the published a = 0.5. A natural question
is whether the cluster-shared / answer-unique split is an artifact of this
coefficient. This script measures, with no model forward passes, how the
COMPOSITION of the partition (what fraction of selected features are shared
vs. unique) moves as a_weight is swept --- using the project's own scoring
formula on the saved L41 SAE encodings.

It does NOT recompute the CAUSAL split (the % of the hit@1 drop from ablating
each subset); that requires Gemma-2-9B forward passes (Shared-only/Unique-only
ablations at each a_weight) and should be run on the model with
``src.features.score_specific_features(..., a_weight=w)`` feeding the
unique/shared ablation pipeline. This composition sweep is the model-free
companion that isolates the selection effect.

Faithfulness check: at a_weight = 0.5 the reconstructed top-k sets are compared
to the published ``specific_features.json`` (mean Jaccard reported; should be
~1.0, confirming the numpy reimplementation matches src.features).

Writes: artifacts/reference_layer/score_weight_count_sensitivity_L41.json
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.config import ARTIFACTS_DIR

L41 = ARTIFACTS_DIR / "layer_bookends" / "L41"
OUT = ARTIFACTS_DIR / "reference_layer" / "score_weight_count_sensitivity_L41.json"
TOP_K = 10
A_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0, 2.0]


def topk_features(z_i, z_others_mean, z_A, a_weight, k=TOP_K):
    """Reimplements src.features.score_specific_features for one disambiguation.
    score = (z_i - mean_others) + a*z_A; mask to z_i>0; top-k by score."""
    score = (z_i - z_others_mean) + a_weight * z_A
    mask = z_i > 0
    score = np.where(mask, score, -1e9)
    n_pos = int(mask.sum())
    k = min(k, n_pos)
    if k <= 0:
        return set()
    # top-k indices (descending). np.argpartition then sort for determinism.
    idx = np.argpartition(-score, k - 1)[:k]
    idx = idx[np.argsort(-score[idx], kind="stable")]
    return set(int(i) for i in idx)


def main() -> None:
    enc = np.load(L41 / "sae_encodings_L41.npz")
    sf = json.load(open(L41 / "specific_features.json"))

    # pair_id -> sorted list of disambig indices; and published sets for check
    by_pair = defaultdict(list)
    pub_sets = {}
    for e in sf:
        by_pair[e["pair_id"]].append(e["disambig_idx"])
        pub_sets[(e["pair_id"], e["disambig_idx"])] = set(e["features"])
    for pid in by_pair:
        by_pair[pid] = sorted(by_pair[pid])

    results = []
    for a in A_WEIGHTS:
        sum_F = sum_shared = sum_unique = 0
        n_units = 0
        jacc = []  # faithfulness vs published at a=0.5
        for pid, idxs in by_pair.items():
            zA = enc[f"A__{pid}"]
            zD = {i: enc[f"D__{pid}__{i}"] for i in idxs}
            # build F_i for every disambiguation in this pair
            Fsets = {}
            for i in idxs:
                others = [zD[j] for j in idxs if j != i]
                mean_other = np.mean(others, axis=0) if others else np.zeros_like(zD[i])
                Fsets[i] = topk_features(zD[i], mean_other, zA, a)
            for i in idxs:
                Fi = Fsets[i]
                sib_union = set().union(*[Fsets[j] for j in idxs if j != i]) if len(idxs) > 1 else set()
                shared = Fi & sib_union
                unique = Fi - shared
                sum_F += len(Fi); sum_shared += len(shared); sum_unique += len(unique)
                n_units += 1
                if abs(a - 0.5) < 1e-9:
                    pub = pub_sets[(pid, i)]
                    u = len(Fi | pub)
                    jacc.append(len(Fi & pub) / u if u else 1.0)
        cell = {
            "a_weight": a,
            "n_units": n_units,
            "mean_set_size": sum_F / n_units,
            "pct_shared_by_count": 100.0 * sum_shared / sum_F,
            "pct_unique_by_count": 100.0 * sum_unique / sum_F,
        }
        if jacc:
            cell["faithfulness_mean_jaccard_vs_published"] = float(np.mean(jacc))
        results.append(cell)
        extra = (f"  (Jaccard vs published = {cell['faithfulness_mean_jaccard_vs_published']:.3f})"
                 if "faithfulness_mean_jaccard_vs_published" in cell else "")
        print(f"a={a:>4}:  shared={cell['pct_shared_by_count']:5.1f}%  "
              f"unique={cell['pct_unique_by_count']:5.1f}%  "
              f"mean|F|={cell['mean_set_size']:.2f}{extra}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"layer": 41, "top_k": TOP_K, "note":
               "Count-composition of shared/unique partition vs a_weight "
               "(model-free; selection effect only). Causal split needs forward passes.",
               "cells": results}, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
