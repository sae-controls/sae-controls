"""Slot detection — does Gemma's greedy-decoded output mention any annotated
answer? Used in paper Sec. 3.1.4 to filter candidate ambiguous questions to
those where the model commits to one of the human-annotated disambiguations.

Two-tier matching:

  1. Exact (case-insensitive, word-boundary aware) substring match on
     normalize_candidate(answer)
  2. Distinctive-token match on tokens extracted from the answer:
       - 4-digit years (19xx / 20xx)
       - All-caps codes (e.g. "USA", "FBI")
       - Last capitalized word (often the surname for person names)
       - Standalone short digit sequences (1-3 digits)

Returns the candidate index (0-based) that the model is judged to have
committed to, or -1 if no match.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
CAPS_CODE_RE = re.compile(r"\b[A-Z]{2,}\b")
TURN_SEP_RE = re.compile(r"<(?:start_of_turn|end_of_turn)>", re.IGNORECASE)


def normalize_candidate(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def distinctive_tokens(cand: str) -> list[str]:
    """Extract distinctive tokens that uniquely identify the candidate
    (years, codes, last capitalized word, short digit sequences)."""
    out = []
    for m in YEAR_RE.findall(cand):
        out.append(m)
    for m in CAPS_CODE_RE.findall(cand):
        out.append(m)
    words = [w for w in re.split(r"\s+", cand.strip()) if w]
    caps = [w for w in words if len(w) >= 3 and w[0].isupper() and not w.isupper()]
    if caps:
        out.append(caps[-1])
    for m in re.findall(r"\b\d{1,4}\b", cand):
        if len(m) != 4 or not m.startswith(("1", "2")):
            out.append(m)
    seen, final = set(), []
    for x in out:
        xn = normalize_candidate(x)
        if xn and xn not in seen:
            seen.add(xn)
            final.append(xn)
    return final


@dataclass
class MatchResult:
    cand_idx: int
    strategy: str
    matched_text: str


def find_best_match(raw_gen: str, candidates: list[str]) -> MatchResult:
    """Return the FIRST candidate matched in raw_gen via exact substring
    or distinctive tokens. cand_idx = -1 if none match."""
    if not candidates:
        return MatchResult(-1, "none", "")
    lower = raw_gen.lower()
    lower_chars = list(lower)
    for m in TURN_SEP_RE.finditer(lower):
        for k in range(m.start(), m.end()):
            lower_chars[k] = " "
    search = "".join(lower_chars)
    hits = []
    for ci, cand in enumerate(candidates):
        nc = normalize_candidate(cand)
        if not nc: continue
        pat = re.escape(nc)
        if nc[0].isalnum():
            pat = r"(?<!\w)" + pat + r"(?!\w)"
        m = re.search(pat, search)
        if m:
            hits.append((m.start(), len(nc), "exact", ci, nc))
            continue
        for frag in distinctive_tokens(cand):
            p = re.escape(frag)
            if frag[0].isalnum():
                p = r"(?<!\w)" + p + r"(?!\w)"
            m = re.search(p, search)
            if m:
                hits.append((m.start(), len(frag), f"distinctive:{frag}", ci, frag))
                break
    if not hits:
        return MatchResult(-1, "none", "")
    hits.sort(key=lambda h: (h[0], -h[1]))   # earliest, longest
    best = hits[0]
    return MatchResult(cand_idx=best[3], strategy=best[2], matched_text=best[4])


def first_token_variants(tokenizer, cand: str) -> list[int]:
    """Token IDs for both " {cand}" and "{cand}" first-token (deduped).
    Used as the hit@1 target set (paper Sec. 3.4)."""
    out = []
    for probe in (" " + cand.strip(), cand.strip()):
        ids = tokenizer.encode(probe, add_special_tokens=False)
        if ids:
            out.append(ids[0])
    return list(dict.fromkeys(out))
