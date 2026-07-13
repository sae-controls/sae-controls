"""Build the matched/donor control sets at L41 (offline; no GPU).

Four control conditions are constructed:
  (a) activation-matched   -> M-act  (mirrors each published Sibling row)
                              M-targ (mirrors each published Targeted row)
  (b) frequency-matched    -> folded into the greedy matcher (frequency is
                              the tie-break) + full frequency diagnostics;
                              at L41 the non-cluster active pool is too
                              small (median 2) for an independent
                              frequency-matched condition.
  (c) same-answer/different-question -> SA-DQ (donor-constrained shuffle)
  (d) plausible-wrong-answer         -> W (alias-style capture on the GPU
                                          host, following the alias-capture
                                          design of replication_triviaqa.py)

Design facts this script relies on (verified against the L41 artifacts):
  * The error-preserving splice is exactly the identity for features with
    z_A = 0 (decode is linear; zeroing an already-zero coordinate is a
    no-op). Published WikiText-shuffled and Random conditions are
    identical to baseline on all 1,103 pairs. Therefore only the
    z_A-ACTIVE members of a feature set carry causal effect, and a
    matched control needs to match only the active members.
  * At L41 (canonical L0 ~ 10) the answer-cluster sets nearly exhaust
    the prompt-active features: the non-cluster active pool has
    median 2, mean 3.0 features, and is empty for 13.6 % of pairs.
    Matching is therefore best-effort up to |pool|, with the achieved
    activation-mass ratio recorded per row and used for subgroup
    analysis. The pool-size structure is itself a reported finding.

Output: artifacts/matched_controls/selection.json (consumed by
pipeline/matched_controls_run.py on the GPU host).
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
A = REPO / "artifacts"
L41 = A / "layer_bookends" / "L41"
OUT = A / "matched_controls"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 0  # project convention (src/config.py)


def norm_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[.,!?\"']+$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def main() -> None:
    z = np.load(L41 / "sae_encodings_L41.npz")
    sf = json.load(open(L41 / "specific_features.json"))
    res = json.load(open(L41 / "results_main.json"))
    detected = json.load(open(A / "detected_pairs.json"))

    by_pair = {p["pair_id"]: p for p in [dict(pair_id=d["id"], **d) for d in detected]}
    feats = {(e["pair_id"], e["disambig_idx"]): e["features"] for e in sf}

    # ---- in-domain firing frequency over all 1,551 cached encodings ----
    keys = list(z.keys())
    freq = np.zeros(16384, dtype=np.float64)
    for k in keys:
        freq += (z[k] > 0)
    freq /= len(keys)

    # ---- per-pair pools ----
    cluster = defaultdict(set)
    for (pid, di), fl in feats.items():
        cluster[pid].update(fl)

    zA = {pid: z[f"A__{pid}"] for pid in by_pair}
    pool = {}
    for pid in by_pair:
        act = set(np.nonzero(zA[pid] > 0)[0].tolist())
        pool[pid] = sorted(act - cluster[pid])

    pool_sizes = np.array([len(v) for v in pool.values()])

    # ---- greedy activation-matched selection (frequency tie-break) ----
    def match_row(pid: str, feature_set: list[int]) -> dict | None:
        """Greedy nearest-|z_A| match for each active member of feature_set,
        drawn without replacement from the pair's non-cluster active pool.
        Returns None when the pool is empty (row attrition)."""
        za = zA[pid]
        active = sorted((f for f in feature_set if za[f] > 0),
                        key=lambda f: -za[f])
        avail = list(pool[pid])
        if not avail:
            return None
        matched, gaps, fgaps = [], [], []
        for f in active:
            if not avail:
                break
            best = min(avail, key=lambda g: (abs(float(za[g] - za[f])),
                                             abs(float(freq[g] - freq[f]))))
            avail.remove(best)
            matched.append(int(best))
            gaps.append(abs(float(za[best] - za[f])))
            fgaps.append(abs(float(freq[best] - freq[f])))
        need_mass = float(sum(za[f] for f in active)) or 1e-9
        got_mass = float(sum(za[g] for g in matched))
        return {
            "features": matched,
            "n_active_needed": len(active),
            "n_active_matched": len(matched),
            "mass_ratio": got_mass / need_mass,
            "zA_gap_mean": float(np.mean(gaps)) if gaps else 0.0,
            "freq_gap_mean": float(np.mean(fgaps)) if fgaps else 0.0,
        }

    m_act_rows, m_targ_rows = [], []
    dropped_act = dropped_targ = 0
    for r in res["cross_rows"]:
        m = match_row(r["pair_id"], feats[(r["pair_id"], r["ablate_idx"])])
        if m is None:
            dropped_act += 1
            continue
        m_act_rows.append({"pair_id": r["pair_id"], "ablate_idx": r["ablate_idx"],
                           "target_idx": r["target_idx"], **m})
    for r in res["self_rows"]:
        m = match_row(r["pair_id"], feats[(r["pair_id"], r["ablate_idx"])])
        if m is None:
            dropped_targ += 1
            continue
        m_targ_rows.append({"pair_id": r["pair_id"], "ablate_idx": r["ablate_idx"],
                            "target_idx": r["target_idx"], **m})

    # ---- SA-DQ: same-answer / different-question donors ----
    ans_index = defaultdict(list)  # norm answer -> [(pid, di)]
    for p in detected:
        for di, d in enumerate(p["disambigs"]):
            for v in {norm_answer(d["answer"]), *map(norm_answer, d["answer_variants"])}:
                ans_index[v].append((p["id"], di))

    rng = random.Random(SEED)
    N_DRAWS = 3  # published shuffle-condition convention
    sadq_rows = []
    for p in detected:
        for di, d in enumerate(p["disambigs"]):
            variants = {norm_answer(d["answer"]), *map(norm_answer, d["answer_variants"])}
            donors = sorted({(qid, k) for v in variants for (qid, k) in ans_index[v]
                             if qid != p["id"] and (qid, k) in feats and feats[(qid, k)]})
            if not donors:
                continue
            picks = donors if len(donors) <= N_DRAWS else rng.sample(donors, N_DRAWS)
            for draw_idx, donor in enumerate(picks):
                sadq_rows.append({
                    "pair_id": p["id"], "target_idx": di, "draw_idx": draw_idx,
                    "donor_pair_id": donor[0], "donor_disambig_idx": donor[1],
                    "features": feats[donor],
                })

    # ---- W: plausible-wrong-answer per pair (type-matched, collision-free) ----
    wh_of = {p["id"]: p["A_question"].strip().lower().split()[0] for p in detected}
    w_rows = []
    no_w = 0
    for p in detected:
        own_norms = {norm_answer(d["answer"]) for d in p["disambigs"]}
        own_norms |= {norm_answer(v) for d in p["disambigs"] for v in d["answer_variants"]}
        own_first = {t for d in p["disambigs"] for t in d["first_token_variants"]}
        cands = []
        for q in detected:
            if q["id"] == p["id"] or wh_of[q["id"]] != wh_of[p["id"]]:
                continue
            for d in q["disambigs"]:
                if norm_answer(d["answer"]) in own_norms:
                    continue
                if set(d["first_token_variants"]) & own_first:
                    continue
                cands.append(d["answer"])
        if not cands:
            no_w += 1
            continue
        uniq = sorted(set(cands))
        picks = uniq if len(uniq) <= N_DRAWS else rng.sample(uniq, N_DRAWS)
        for draw_idx, w in enumerate(picks):
            w_rows.append({"pair_id": p["id"], "draw_idx": draw_idx,
                           "wrong_answer": w})

    # ---- pack pair metadata the GPU host needs ----
    pairs_meta = [{
        "pair_id": p["id"],
        "A_question": p["A_question"],
        "disambigs": [{"first_token_variants": d["first_token_variants"]}
                      for d in p["disambigs"]],
    } for p in detected]

    sel = {
        "meta": {
            "layer": 41, "seed": SEED, "n_pairs": len(detected),
            "pool_stats": {"mean": float(pool_sizes.mean()),
                           "median": float(np.median(pool_sizes)),
                           "pct_empty": float((pool_sizes == 0).mean())},
            "m_act": {"rows": len(m_act_rows), "dropped": dropped_act,
                      "mass_ratio_mean": float(np.mean([r["mass_ratio"] for r in m_act_rows])),
                      "mass_ratio_median": float(np.median([r["mass_ratio"] for r in m_act_rows])),
                      "zA_gap_mean": float(np.mean([r["zA_gap_mean"] for r in m_act_rows]))},
            "m_targ": {"rows": len(m_targ_rows), "dropped": dropped_targ,
                       "mass_ratio_mean": float(np.mean([r["mass_ratio"] for r in m_targ_rows])),
                       "mass_ratio_median": float(np.median([r["mass_ratio"] for r in m_targ_rows]))},
            "sadq": {"rows": len(sadq_rows),
                     "unique_targets": len({(r['pair_id'], r['target_idx']) for r in sadq_rows})},
            "w": {"rows": len(w_rows), "pairs_without_candidate": no_w},
        },
        "pairs": pairs_meta,
        "m_act_rows": m_act_rows,
        "m_targ_rows": m_targ_rows,
        "sadq_rows": sadq_rows,
        "w_rows": w_rows,
    }
    out = OUT / "selection.json"
    json.dump(sel, open(out, "w"))
    print(json.dumps(sel["meta"], indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
