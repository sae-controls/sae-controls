"""Run the matched/donor control ablations at L41 (GPU).

Consumes artifacts/matched_controls/selection.json (built offline by
matched_controls_select.py). Produces matched_controls_results.json with
per-row outcomes in the same schema family as results_main.json.

Conditions:
  baseline   one identity-splice forward per pair (environment parity
             gate against the published base_hit1)
  m_act      activation-matched non-cluster sets mirroring Sibling rows
  m_targ     same, mirroring Targeted rows
  sadq       same-answer/different-question donor sets
  w          plausible-wrong-answer sets: capture z at the last token of
             build_prompt(A) + wrong_answer (the alias-capture convention
             of replication_triviaqa.py), score with score_specific_features
             against the pair's real disambiguation encodings, ablate
             the top-10 on A's forward.

Run:  python pipeline/matched_controls_run.py
Env:  HF_HOME should point at a large cache volume.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from src.model import load_model                     # noqa: E402
from src.sae import load_sae                          # noqa: E402
from src.prompts import build_prompt                  # noqa: E402
from src.features import score_specific_features      # noqa: E402
from src.hooks import (capture_last_token_residual,   # noqa: E402
                       forward_with_ablation, hit_at_k)

LAYER = 41
SAE_SHA_PREFIX = "65f7ea2b"
SEL = REPO / "artifacts" / "matched_controls" / "selection.json"
OUT = REPO / "artifacts" / "matched_controls" / "matched_controls_results.json"
PARTIAL = OUT.with_suffix(".partial.json")


def main() -> None:
    import torch

    t0 = time.time()
    limit = int(os.environ.get("MC_LIMIT", "0"))  # smoke test: rows per section
    sel = json.load(open(SEL))
    if limit:
        sel["pairs"] = sel["pairs"][:limit]
        keep = {p["pair_id"] for p in sel["pairs"]}
        for k in ("m_act_rows", "m_targ_rows", "sadq_rows", "w_rows"):
            sel[k] = [r for r in sel[k] if r["pair_id"] in keep][:limit]
    enc = np.load(REPO / "artifacts" / "matched_controls" / "sae_encodings_L41.npz")

    pairs = {p["pair_id"]: p for p in sel["pairs"]}
    variants = {(p["pair_id"], i): set(d["first_token_variants"])
                for p in sel["pairs"] for i, d in enumerate(p["disambigs"])}

    tok, model = load_model()
    sae, meta = load_sae(LAYER)
    assert meta["sha256"].startswith(SAE_SHA_PREFIX), \
        f"SAE checkpoint mismatch: {meta['sha256'][:8]} != {SAE_SHA_PREFIX}"
    dev = next(model.parameters()).device

    prompts = {pid: build_prompt(tok, p["A_question"]) for pid, p in pairs.items()}
    results: dict[str, list] = {"baseline_rows": [], "m_act_rows": [],
                                "m_targ_rows": [], "sadq_rows": [], "w_rows": []}
    from src.config import MODEL_ID
    run_meta = {"layer": LAYER, "sae_sha256": meta["sha256"],
                "sae_repo": meta["repo"], "sae_subpath": meta["subpath"],
                "model_id": MODEL_ID, "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0),
                "started_unix": int(t0)}

    def flush() -> None:
        json.dump({"run_meta": run_meta, **results}, open(PARTIAL, "w"))

    n_done = 0

    def tick(label: str, total: int) -> None:
        nonlocal n_done
        n_done += 1
        if n_done % 200 == 0:
            print(f"[{time.time()-t0:7.0f}s] {label}: {n_done} rows "
                  f"(section total {total})", flush=True)
            flush()

    # ---- 1. baseline parity gate -------------------------------------
    print("baseline forwards...", flush=True)
    for pid, p in pairs.items():
        top = forward_with_ablation(model, tok, prompts[pid], LAYER, sae, [])
        for i in range(len(p["disambigs"])):
            results["baseline_rows"].append({
                "pair_id": pid, "target_idx": i,
                "base_hit1": hit_at_k(top["top_ids"], variants[(pid, i)], 1)})
        tick("baseline", len(pairs))

    # ---- 2-4. precomputed feature-set conditions ---------------------
    for section, key in (("m_act_rows", "m_act_rows"),
                         ("m_targ_rows", "m_targ_rows"),
                         ("sadq_rows", "sadq_rows")):
        rows = sel[key]
        print(f"{section}: {len(rows)} ablations...", flush=True)
        n_done = 0
        for r in rows:
            pid = r["pair_id"]
            top = forward_with_ablation(model, tok, prompts[pid], LAYER, sae,
                                        r["features"])
            out = {k: v for k, v in r.items() if k != "features"}
            out["n_features"] = len(r["features"])
            out["ablate_hit1"] = hit_at_k(top["top_ids"],
                                          variants[(pid, r["target_idx"])], 1)
            results[section].append(out)
            tick(section, len(rows))
        flush()

    # ---- 5. wrong-answer: capture, score, ablate ---------------------
    rows = sel["w_rows"]
    print(f"w_rows: {len(rows)} capture+ablate...", flush=True)
    n_done = 0
    for r in rows:
        pid = r["pair_id"]
        p = pairs[pid]
        d_prompt = prompts[pid] + r["wrong_answer"]
        h = capture_last_token_residual(model, tok, d_prompt, LAYER)
        z_w = sae.encode(torch.from_numpy(h).to(dev, sae.W_enc.dtype))
        z_w = z_w.to(torch.float32).cpu()
        z_a = torch.from_numpy(enc[f"A__{pid}"])
        z_ds = [torch.from_numpy(enc[f"D__{pid}__{i}"])
                for i in range(len(p["disambigs"]))]
        f_w = score_specific_features(z_a, [z_w] + z_ds)[0]
        top = forward_with_ablation(model, tok, prompts[pid], LAYER, sae, f_w)
        for i in range(len(p["disambigs"])):
            results["w_rows"].append({
                "pair_id": pid, "target_idx": i, "draw_idx": r["draw_idx"],
                "wrong_answer": r["wrong_answer"], "n_features": len(f_w),
                "ablate_hit1": hit_at_k(top["top_ids"], variants[(pid, i)], 1)})
        tick("w", len(rows))
    flush()

    run_meta["elapsed_seconds"] = round(time.time() - t0, 1)
    json.dump({"run_meta": run_meta, **results}, open(OUT, "w"))
    PARTIAL.unlink(missing_ok=True)
    print(f"DONE in {run_meta['elapsed_seconds']}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
