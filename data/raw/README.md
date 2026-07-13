# data/raw/ — regenerating the candidate pool from scratch

The frozen `artifacts/expanded_dataset.json` (SHA-256 pinned in
`repro.txt`) is what every downstream script reads, so the pipeline runs
end-to-end without anything in this directory. This directory exists for
one purpose: regenerating that artifact from raw AmbigQA with
`pipeline/build_dataset.py`.

**Included:**

| File | What it is |
|---|---|
| `kau_extended.reconstructed.jsonl` | The 128 lever-1 candidate rows (`{"id", "question", "condition": "A"}`), reconstructed from the lever tags in the frozen artifact |

**To fetch (public, Hugging Face `sewon/ambig_qa`):**

| File | What it is |
|---|---|
| `ambigqa_raw_train.jsonl` | AmbigQA `train` split (10,036 items), `multipleQAs` items serialized one record per line |
| `ambigqa_raw_validation.jsonl` | AmbigQA `validation` split (2,002 items), same format |

With those two files fetched and the reconstructed lever-1 list renamed
to `kau_extended.jsonl` (or `KAU_EXT` pointed at it),
`build_dataset.py` regenerates `expanded_dataset.json` exactly.

**Verified.** This regeneration was executed end-to-end (public
`sewon/ambig_qa` "full" splits + the reconstructed lever-1 list,
2026-07-09): `build_dataset.py` reproduced 1,128 candidates with
`id_set_sha256_first16 = 66cb0a494ead1143`, identical to the frozen
artifact.

**Why the reconstruction is sufficient.** `build_dataset.py` consumes
only the lever-1 rows with `condition == "A"`, and from each row only
`id` and `question`; disambiguations are re-derived from raw AmbigQA by
id, and all filters are deterministic. The original intermediate file
was a superset containing rows that those deterministic filters
discarded; the reconstruction contains exactly the 128 surviving rows in
their original order (recovered from `source_lever == 1` records in the
frozen artifact), which yields an identical lever-1 selection. Lever 2
is a seeded sample (`seed = 0`) over the remaining eligible questions
and is unaffected.
