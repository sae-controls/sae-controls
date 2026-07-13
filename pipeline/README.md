# pipeline/

One script per analysis. Each writes a frozen output to `artifacts/`;
the paper cites those frozen artifacts directly. `src/` provides the
shared library the scripts call.

## Data

| Script | What it writes |
|---|---|
| `build_dataset.py` | `artifacts/expanded_dataset.json` — candidate ambiguous questions (1,128) before slot detection |
| `slot_detection_sensitivity.py` | `artifacts/slot_detection/` — slot-detection retention under varied matchers/prompts |

## Main pipeline

| Script | What it writes |
|---|---|
| `main_ablation.py` | `artifacts/results_main.json` + `detected_pairs.json` + `sae_encodings_L37.npz` — six-condition ablation pipeline at the legacy reference layer L37 |
| `recompute_headline_tables_L41.py` | `artifacts/reference_layer/corrected_headline_tables_L41.json` — recomputes the L41 six-condition headline (Table 1) and pairwise contrasts (Table 2) directly from the canonical per-pair artifact |
| `polysemy_partition.py` | `artifacts/wikitext_position_mode.json` — the `pct_pos0` polysemy partition at L37 |

## Decompositions

| Script | What it writes |
|---|---|
| `unique_vs_shared.py` | `artifacts/unique_vs_shared/` — cluster-shared vs answer-unique partition + ablation rows |
| `multimetric.py` | `artifacts/multimetric/` — KL, logit-difference, generation-flip alongside hit@1 |
| `four_way_decomposition.py` | `artifacts/four_way/` — joint (unique/shared) × (content/position) decomposition |
| `per_feature_equivalence.py` | `artifacts/per_feature_equivalence/` — single-feature (uc-top vs sc-top) head-to-head |
| `per_pair_audit.py` | `artifacts/per_pair_audit/` — per-pair regression on baseline hit@1 + covariates |
| `overlap_shells.py` | `artifacts/overlap_shells/` — T3 trajectory across max-overlap shells (K=0..10) |

## Robustness

| Script | What it writes |
|---|---|
| `layer_sweep.py` | `artifacts/layer_sweep/L26..L40/` — multi-layer T3 + decomposition |
| `layer_bookends.py` | `artifacts/layer_bookends/L20/`, `artifacts/layer_bookends/L41/` + `trajectory_summary.json` — endpoint layers; L41 is the paper's reference layer |
| `l0_sweep.py` | `artifacts/l0_sweep/L37/`, `artifacts/l0_sweep/L41/` + `l0_sweep_summary.json` — T3 across the public SAE L0 family |
| `reference_layer_analysis.py` | `artifacts/reference_layer/` — polysemy + multi-metric (KL, gen-flip) + 2×2 joint + Cramér's V at the reference layer (L41); writes the headline `L41_summary_table.json` |
| `score_weight_count_sensitivity.py` | stdout / JSON — shared/unique partition composition vs the score coefficient `a` (Eq. 1); model-free sweep on the saved L41 encodings |

## Cross-axis replication

| Script | What it writes |
|---|---|
| `replication_gemma2b.py` | `artifacts/replication_gemma2b/summary.json` — Gemma-2-2B-IT @ L25 / AmbigQA |
| `replication_triviaqa.py` | `artifacts/replication_triviaqa/summary.json` — TriviaQA-Web aliases @ Gemma-2-9B-IT @ L41 |
| `replication_llama.py` | `artifacts/replication_llama/summary.json` — Llama-3.1-8B-Instruct @ L31 / AmbigQA |

## Notes

- All outputs are frozen and cited as-is by the paper. Re-running requires
  a GPU with ≥48 GB and the model + SAE downloads listed in `repro.txt`.
- The paper's figures are computed from the artifacts written here; the
  figure-generation code ships in `paper_figures/` at the release root.
- Replication scripts use the same `src/` library; only the model loader
  and tokenizer template differ between architectures.
