# Sibling-disambiguation controls for SAE causal mediation

Code and frozen artifacts for the paper *Sibling-Disambiguation Controls
Reveal Cluster-Shared and Answer-Unique Components in SAE Causal Mediation*.

The paper proposes a six-condition control protocol for sparse-autoencoder (SAE)
causal-mediation studies on paired-output datasets, and a unique-vs-shared
partition that decomposes the targeted feature set into a cluster-shared and an
answer-unique component. The protocol is applied to Gemma-2-9B-IT at its last
residual-stream layer (L41) on 1,103 disambiguation self-pairs from AmbigQA, and
replicated across five robustness axes (layer depth, SAE sparsity, model scale,
substrate, architecture).

## Layout

```
src/              Library modules: model loading, SAE I/O, feature scoring,
                  forward hooks, statistical tests. Reusable beyond this paper.
pipeline/         One script per analysis. Each writes a frozen output to
                  artifacts/. See pipeline/README.md for the per-script map.
artifacts/        Frozen pipeline outputs cited by the paper (the published
                  numbers). JSON results + .npz SAE-encoding / residual caches.
data/
  README.md               DATASET CARD — start here for the dataset.
  detected_pairs.json     The evaluation set: 448 questions / 1,103 self-pairs.
  expanded_dataset.json   The 1,128-candidate pool it was detected from.
  patching_pairs.jsonl    Seed 51-pair set used by slot detection.
  raw/                    Regenerating the pool from raw AmbigQA, incl. the
                          reconstructed lever-1 list (see data/raw/README.md).
paper_figures/    make_figures.py + numbers.json — regenerate every
                  paper figure from artifacts/.
REPRO_MAP.md      Headline number → frozen artifact → producing script.
requirements.txt  Pinned Python dependencies.
repro.txt         Pinned versions, model + SAE checkpoint SHA-256, dataset
                  references, frozen-artifact hashes, and random seeds.
METHODOLOGY.md    Plain-language description of what the pipeline does and why.
LICENSE           MIT (code); dataset files derive from AmbigQA, CC BY-SA 3.0.
```

## Quickstart (no GPU)

The published numbers are already in `artifacts/`; nothing needs to run to read
them.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

This package ships the `artifacts/` JSON that every paper table and figure is
computed from; `paper_figures/make_figures.py` re-renders the figures from it.
Any table can be re-derived on CPU from the saved SAE encodings by re-running
the relevant analysis script (see the map below). A GPU is only required to
regenerate the SAE activations from the model weights.

## Script → artifact → paper result

`src/` is the shared library; the scripts below are the entry points. Each row
lists what the script writes under `artifacts/` and the paper result it backs.
L41 is the paper's reference layer; L37 is the legacy reference reported in the
robustness section.

| Script | Writes to `artifacts/` | Paper result |
|---|---|---|
| `build_dataset.py` | `expanded_dataset.json` | Sec. 3 Data — candidate pool (1,128 questions) |
| `main_ablation.py` | `results_main.json`, `detected_pairs.json`, `sae_encodings_L37.npz` | Sec. 3 Data (448 questions, n=1,103 self-pairs); six-condition pipeline at legacy L37 |
| `recompute_headline_tables_L41.py` | `reference_layer/corrected_headline_tables_L41.json` | Sec. 4.1 six-condition headline + pairwise contrasts (Tables 1–2) |
| `layer_bookends.py` | `layer_bookends/{L20,L41}/`, `trajectory_summary.json` | Sec. 4.1 L41 headline (Table 1); Sec. 4.5 7-layer T3 trajectory (Fig 2) |
| `reference_layer_analysis.py` | `reference_layer/` (`L41_summary_table.json`, `multimetric/`, `polysemy/`) | Sec. 4.2 content-vs-position (Table 3, top panel); Sec. 4.4 multi-metric (Table 4); Sec. 4.4/App. B Cramér's V + 2×2 joint; generation-flip confusion (Fig 4); threshold sweep (Fig 6) |
| `polysemy_partition.py` | `wikitext_position_mode.json` | Sec. 3.5 / Sec. 4.2 position-mode polysemy partition (L37) |
| `unique_vs_shared.py` | `unique_vs_shared/` | Sec. 4.3 cluster-shared vs answer-unique (Table 3, bottom panel) |
| `multimetric.py` | `multimetric/` | Sec. 4.4 KL / logit-difference / generation-flip at legacy L37 |
| `four_way_decomposition.py` | `four_way/` | Sec. 4.5 joint (unique/shared) × (content/position) |
| `per_feature_equivalence.py` | `per_feature_equivalence/` | Sec. 4.5 single-feature uc-top vs sc-top |
| `layer_sweep.py` | `layer_sweep/L{26,30,34,40}/` | Sec. 4.5 per-layer T3 + decomposition |
| `l0_sweep.py` | `l0_sweep/{L37,L41}/`, `l0_sweep_summary.json` | Sec. 4.5 SAE-L0 robustness (Fig 3) |
| `replication_gemma2b.py` | `replication_gemma2b/summary.json` | Sec. 4.6 Gemma-2-2B-IT @ L25 (Table 5) |
| `replication_triviaqa.py` | `replication_triviaqa/summary.json` | Sec. 4.6 TriviaQA aliases @ L41 (Table 5) |
| `replication_llama.py` | `replication_llama/summary.json` | Sec. 4.6 Llama-3.1-8B-Instruct @ L31 (Table 5) |
| `overlap_shells.py` | `overlap_shells/` | Sec. 7 Limitations: overlap-shell T3 trajectory (Fig 5) |
| `per_pair_audit.py` | `per_pair_audit/` | Sec. 7 Limitations: per-pair regression on baseline hit@1 |
| `slot_detection_sensitivity.py` | `slot_detection/` | release-only robustness check (not reported in the paper) |
| `score_weight_count_sensitivity.py` | stdout / JSON | Sec. 7 Limitations: score-coefficient (`a`) dependence |
| `matched_controls_select.py` | `matched_controls/selection.json` | App. F selection: activation-matched, SA-DQ donors, unrelated-answer W |
| `matched_controls_run.py` | `matched_controls/matched_controls_results.json` | App. F ablation runs (GPU) |
| `matched_controls_analyze.py` | `matched_controls/summary.json` | Sec. 4.1 + App. F (Table 10) paired contrasts |

## Reproducing the pipeline from scratch (GPU)

Requires a single ≥48 GB GPU and the gated model + SAE downloads listed in
`repro.txt`. Each script writes the corresponding subdirectory of `artifacts/`;
the scripts are independent once the data and slot-detection stages have produced
their inputs.

```bash
# Reference-layer pipeline (the headline numbers)
python pipeline/main_ablation.py
python pipeline/layer_sweep.py
python pipeline/layer_bookends.py
python pipeline/reference_layer_analysis.py
python pipeline/recompute_headline_tables_L41.py

# Robustness axes
python pipeline/l0_sweep.py
python pipeline/replication_gemma2b.py
python pipeline/replication_triviaqa.py
python pipeline/replication_llama.py

# Decompositions and supporting analyses
python pipeline/polysemy_partition.py
python pipeline/unique_vs_shared.py
python pipeline/multimetric.py
python pipeline/four_way_decomposition.py
python pipeline/per_feature_equivalence.py
python pipeline/per_pair_audit.py
python pipeline/overlap_shells.py
python pipeline/slot_detection_sensitivity.py
python pipeline/score_weight_count_sensitivity.py
```

End-to-end runtime on a single A40 is approximately 24 hours, dominated by the
cross-axis replication scripts (each loads a separate model). The reference-layer
pipeline alone is roughly 6 hours. `repro.txt` records the SHA-256 of every cited
artifact; comparing those to your re-run output is the first sanity check.

## Data entry point (`build_dataset.py`)

`build_dataset.py` builds `artifacts/expanded_dataset.json` (the 1,128-candidate
pool) from raw AmbigQA. **Its raw inputs are not bundled** — see
`data/raw/README.md` for what to place there and where to obtain it (AmbigQA is
public on the Hugging Face Hub). The frozen `artifacts/expanded_dataset.json`
(SHA-256 in `repro.txt`) is provided and is the entry point that every downstream
script reads, so the pipeline runs end-to-end from it **without** re-running
`build_dataset.py`.

## Reproducibility notes

- Random seeds (defined in `src/config.py`): `SEED = 0`, `SHUFFLE_SEED = 42`,
  `RANDOM_BASELINE_SEED = 1`.
- Compute device is read from the `CWX_DEVICE` environment variable (default
  `cuda`).
- Pinned dependencies in `requirements.txt`; model/SAE SHA-256, dataset
  references, frozen-artifact hashes, and determinism notes in `repro.txt`.

## License

Code is released under the MIT License (see `LICENSE`). The frozen artifacts are
derivative outputs of (i) Gemma-2 / Llama-3.1 model weights and (ii) Gemma Scope /
Llama Scope SAE checkpoints, each subject to their respective (gated) licenses and
to be downloaded directly from Hugging Face.
