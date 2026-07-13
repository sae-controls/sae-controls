"""AmbigQA pair loader.

The dataset (`data/patching_pairs.jsonl`) is the derived 51-pair set which
seeds the slot-detection pipeline. The published 1103 self-pair result uses an
EXPANDED candidate pool — see pipeline/build_dataset.py for how that is built.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import PAIRS_PATH


@dataclass
class Disambig:
    question: str
    answer: str
    first_token_variants: list[int]   # candidate token IDs for the answer's first token


@dataclass
class PatchPair:
    id: str
    A_question: str
    disambigs: list[Disambig]


def load_pairs(path: Path | None = None) -> list[PatchPair]:
    """Load pairs from JSONL (default: data/patching_pairs.jsonl).

    Each line must have: id, A_question, disambigs[].{question, answer, first_token_variants}.
    Other fields (e.g. flippy, A_gen_winner) from the source dataset are ignored.
    """
    p = Path(path) if path is not None else PAIRS_PATH
    if not p.exists():
        raise FileNotFoundError(f"pairs file missing: {p}")
    rows = [json.loads(l) for l in open(p)]
    return [
        PatchPair(
            id=str(r["id"]),
            A_question=r["A_question"],
            disambigs=[
                Disambig(
                    question=d["question"],
                    answer=d["answer"],
                    first_token_variants=[int(x) for x in d["first_token_variants"]],
                )
                for d in r["disambigs"]
            ],
        )
        for r in rows
    ]
