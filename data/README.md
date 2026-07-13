# data/ — the dataset

This directory contains the complete dataset used in the paper, surfaced
here for direct inspection. The files are byte-identical to the copies
under `artifacts/` that the pipeline reads (SHA-256 pinned in
`repro.txt`).

| File | What it is | Size |
|---|---|---|
| `detected_pairs.json` | **The evaluation set**: 448 ambiguous questions whose greedy continuation commits to an annotated answer, with 1,103 (question, disambiguation) self-pairs total | 448 records |
| `expanded_dataset.json` | The 1,128-candidate pool the evaluation set was slot-detected from (128 lever-1 + 1,000 seeded-sample lever-2 candidates) | 1,128 records |
| `patching_pairs.jsonl` | The 51-pair seed set used during early slot-detection development | 51 records |
| `raw/` | Instructions (and a reconstructed intermediate) for regenerating `expanded_dataset.json` from raw AmbigQA | — |

## Schema (`detected_pairs.json`, one record per ambiguous question)

```json
{
  "id": "-4469503464110108318",
  "A_question": "When did the simpsons first air on television?",
  "source_lever": 1,
  "disambigs": [
    {"question": "When did the Simpsons first air ... on the Tracey Ullman Show?",
     "answer": "April 19, 1987",
     "answer_variants": ["April 19, 1987"],
     "first_token_variants": [4623, 11645]},
    ...
  ],
  "A_gen_text": "December 17, 1989",
  "match_strategy": "exact",
  "match_cand_idx": 1,
  "matched_text": "december 17, 1989"
}
```

`first_token_variants` are Gemma-2 tokenizer ids of each answer variant's
first token (used for hit@1). `A_gen_text` / `match_*` record the slot
detection that admitted the question.

## Provenance and license

Questions, disambiguated rewrites, and answers derive from **AmbigQA**
(Min et al., EMNLP 2020), which is distributed under **CC BY-SA 3.0**;
AmbigQA itself builds on Natural Questions. The derived files here carry
the same CC BY-SA 3.0 terms. The repository's MIT license covers the
code only.
