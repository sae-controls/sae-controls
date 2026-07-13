"""Data (paper Sec. 3.1).

Builds the candidate dataset for the cross-ablation experiment from AmbigQA.
Produces `artifacts/expanded_dataset.json`. No GPU required.

Two source levers (matches the published expansion):

  Lever 1  data/raw/kau_extended.jsonl — 128 A pairs (relaxed within-pair
           first-token-collision filter; recovers the 28 dropped under the
           strict filter).
  Lever 2  Random sample of 1000 A pairs from the raw AmbigQA train+val
           multipleQAs pool, after applying the basic format/length filter,
           ≥2 short-answer (≤3 words) candidates per pair, dedupe by primary
           answer. Sample with seed=SEED.

Slot detection is NOT applied here — it requires the model and runs in main_ablation.py.
This stage only produces the candidate set.

Inputs:
  data/raw/ambigqa_raw_train.jsonl     (10036 raw items; AmbigQA, HF sewon/ambig_qa)
  data/raw/ambigqa_raw_validation.jsonl (2002 raw items; AmbigQA, HF sewon/ambig_qa)
  data/raw/kau_extended.jsonl   (128 A pairs, lever 1)

Output:
  artifacts/expanded_dataset.json
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib
import json
import random
import re
from pathlib import Path

from src.config import ARTIFACTS_DIR, REPO_ROOT, SEED
from src.io_utils import save_json_atomic

# Lever-2 sample size and pre-tokenization caps (paper-pinned):
LEVER_2_SAMPLE_SIZE = 1000
ANSWER_MIN_CHARS = 2
ANSWER_MAX_WORDS = 3
QUESTION_MIN_CHARS = 15
QUESTION_MAX_CHARS = 140
MAX_DISAMBIGS_PER_PAIR = 4

# Raw inputs live in data/raw/ (NOT bundled; see data/raw/README.md).
V3_ROOT = REPO_ROOT / "data" / "raw"
RAW_TRAIN = V3_ROOT / "ambigqa_raw_train.jsonl"
RAW_VAL   = V3_ROOT / "ambigqa_raw_validation.jsonl"
KAU_EXT   = V3_ROOT / "kau_extended.jsonl"

QUESTION_RE = re.compile(r"^(who|what|when|where|which|how many)\b", re.IGNORECASE)


def is_multipleQA(r: dict) -> bool:
    return "multipleQAs" in r.get("annotations", {}).get("type", [])


def basic_format_filter(r: dict) -> bool:
    q = r.get("question", "").strip()
    return QUESTION_MIN_CHARS <= len(q) <= QUESTION_MAX_CHARS and bool(QUESTION_RE.match(q))


def parse_qa_pairs(r: dict) -> list[dict]:
    """Extract (disambig_question, primary_answer, all_answer_variants) from
    AmbigQA's first annotator. Schema:
        annotations.qaPairs : list of annotator entries
        each entry          : {question: [str, ...], answer: [[str_variants], ...]}
    parallel by index.
    """
    qa_outer = r.get("annotations", {}).get("qaPairs", [])
    if not qa_outer: return []
    entry = qa_outer[0]
    if not isinstance(entry, dict): return []
    qs  = entry.get("question", []) or []
    ans = entry.get("answer", [])   or []
    out = []
    for i in range(min(len(qs), len(ans))):
        if isinstance(ans[i], list) and ans[i]:
            primary = ans[i][0]
            variants = list(ans[i])
            out.append({"question": qs[i], "answer": primary, "answer_variants": variants})
    return out


def short_answer(a: str) -> bool:
    if not isinstance(a, str): return False
    a = a.strip()
    return len(a) >= ANSWER_MIN_CHARS and len(a.split()) <= ANSWER_MAX_WORDS


def filter_disambig_set(r: dict) -> list[dict] | None:
    cands = [c for c in parse_qa_pairs(r) if short_answer(c["answer"])]
    if len(cands) < 2: return None
    if len(cands) > MAX_DISAMBIGS_PER_PAIR:
        cands = cands[:MAX_DISAMBIGS_PER_PAIR]
    seen = set(); uniq = []
    for c in cands:
        key = c["answer"].strip().lower()
        if key in seen: continue
        seen.add(key); uniq.append(c)
    if len(uniq) < 2: return None
    return uniq


def main() -> None:
    print(f"[stage1] building expanded candidate dataset (seed={SEED})")

    # Lever 1: kau_extended (128 A pairs)
    if not KAU_EXT.exists():
        raise FileNotFoundError(f"missing lever 1 source: {KAU_EXT}")
    kau_ext = [json.loads(l) for l in open(KAU_EXT)]
    A_ext = [r for r in kau_ext if r.get("condition") == "A"]
    print(f"[lever 1] kau_extended A pairs: {len(A_ext)}")

    # Build map: id -> raw row (so we can recover the disambig questions for kau_extended)
    raw = []
    for f in (RAW_TRAIN, RAW_VAL):
        if not f.exists():
            raise FileNotFoundError(f"missing raw AmbigQA: {f}")
        raw.extend(json.loads(l) for l in open(f))
    raw_by_id = {str(r["id"]): r for r in raw}
    print(f"[setup]   raw AmbigQA train+val: {len(raw)}")

    expanded = []
    seen_ids: set[str] = set()

    # Lever 1
    for a_row in A_ext:
        pid = str(a_row["id"])
        if pid in seen_ids: continue
        rr = raw_by_id.get(pid)
        if rr is None: continue
        cands = filter_disambig_set(rr)
        if cands is None: continue
        expanded.append({
            "id": pid,
            "A_question": a_row["question"],
            "source_lever": 1,
            "disambigs": cands,
        })
        seen_ids.add(pid)
    n_lever1 = len(expanded)
    print(f"[lever 1] kept {n_lever1} pairs")

    # Lever 2: sample 1000 from raw multi-QA pool minus lever 1
    multi   = [r for r in raw if is_multipleQA(r)]
    fmt_ok  = [r for r in multi if basic_format_filter(r)]
    pool: list[dict] = []
    for r in fmt_ok:
        pid = str(r["id"])
        if pid in seen_ids: continue
        cands = filter_disambig_set(r)
        if cands is None: continue
        pool.append({"id": pid, "A_question": r["question"], "source_lever": 2, "disambigs": cands})
    print(f"[lever 2] eligible raw pool (after format / short-answer / dedupe): {len(pool)}")

    rng = random.Random(SEED)
    sample = rng.sample(pool, min(LEVER_2_SAMPLE_SIZE, len(pool)))
    expanded.extend(sample)
    print(f"[lever 2] sampled {len(sample)} (seed={SEED})")

    # Save
    n_total = len(expanded)
    n_di = sum(len(p["disambigs"]) for p in expanded)
    id_set_hash = hashlib.sha256(
        json.dumps([p["id"] for p in expanded]).encode()
    ).hexdigest()[:16]

    out = {
        "n_total": n_total,
        "n_lever_1_kau_extended": n_lever1,
        "n_lever_2_raw_sample": len(sample),
        "n_disambigs_total": n_di,
        "lever_2_seed": SEED,
        "lever_2_sample_size": LEVER_2_SAMPLE_SIZE,
        "id_set_sha256_first16": id_set_hash,
        "filters": {
            "question_min_chars": QUESTION_MIN_CHARS,
            "question_max_chars": QUESTION_MAX_CHARS,
            "answer_min_chars":   ANSWER_MIN_CHARS,
            "answer_max_words":   ANSWER_MAX_WORDS,
            "max_disambigs_per_pair": MAX_DISAMBIGS_PER_PAIR,
        },
        "pairs": expanded,
    }
    save_json_atomic(ARTIFACTS_DIR / "expanded_dataset.json", out)

    print(f"\n=== Stage 1 complete ===")
    print(f"  total candidate A pairs:              {n_total}")
    print(f"  total candidate (A, D_i) self-pairs:  {n_di}")
    print(f"  id-set sha256 (first 16):             {id_set_hash}")
    print(f"  → artifacts/expanded_dataset.json")


if __name__ == "__main__":
    main()
