# REPRO_MAP — every headline number → frozen artifact → producing script

All artifacts below ship in this package (hashes in `repro.txt`).
"Recompute" scripts are pure post-processing over frozen per-pair rows —
no GPU needed. GPU stages are marked.

## Main results (paper Sec. 4.1, Tables 1–2)

| Paper number | Artifact | Script |
|---|---|---|
| Baseline hit@1 0.2928; Targeted −13.24 pp; Sibling −9.05; Shuffled-AmbigQA −1.18; WikiText-shuffled 0.00; Random 0.00 | `artifacts/layer_bookends/L41/results_main.json` | GPU: `pipeline/layer_bookends.py` · recompute: `pipeline/recompute_headline_tables_L41.py` |
| T1–T4 contrasts (Table 2) | same | `pipeline/recompute_headline_tables_L41.py` |
| T5 Targeted-vs-Sibling +4.19 pp (Table 2) | same | `pipeline/matched_controls_analyze.py` (`T5_Targeted_vs_Sibling`) |

## Decompositions (paper Sec. 4.2–4.3, Table 3)

| Paper number | Artifact | Script |
|---|---|---|
| Content −10.06 / Position −2.54 / interaction −0.63 (76/19/5 %) | `artifacts/reference_layer/L41_summary_table.json` (`polysemy_at_080`) | GPU: `pipeline/reference_layer_analysis.py` |
| Shared −7.98 / Unique −4.62 / residual −0.63 (60/35/5 %) | `artifacts/layer_bookends/L41/unique_shared_decomp.json` | GPU: `pipeline/unique_vs_shared.py` |

## Multi-metric, orthogonality, robustness (paper Sec. 4.4–4.5)

| Paper number | Artifact | Script |
|---|---|---|
| Table 4 (hit@1 / KL / gen-flip percentages) | `artifacts/reference_layer/multimetric/` | GPU: `pipeline/reference_layer_analysis.py` |
| Cramér's V = 0.060; contingency (App. B) | `artifacts/reference_layer/cross_decomp_L41.json` | GPU: `pipeline/reference_layer_analysis.py` |
| Layer trajectory (Fig. 2, App. E) | `artifacts/layer_bookends/trajectory_summary.json` | GPU: `pipeline/layer_sweep.py`, `pipeline/layer_bookends.py` |
| L0 sweep (Fig. 3, App. E) | `artifacts/l0_sweep/l0_sweep_summary.json` | GPU: `pipeline/l0_sweep.py` |

## Replications (paper Sec. 4.6, Table 5)

| Cell | Artifact | Script |
|---|---|---|
| Gemma-2-2B @ L25 | `artifacts/replication_gemma2b/` | GPU: `pipeline/replication_gemma2b.py` |
| TriviaQA aliases @ L41 | `artifacts/replication_triviaqa/` | GPU: `pipeline/replication_triviaqa.py` |
| Llama-3.1-8B @ L31 | `artifacts/replication_llama/` | GPU: `pipeline/replication_llama.py` |

## Matched / donor-constrained controls (paper Sec. 4.1 + App. F, Table 10)

| Result | Artifact | Script |
|---|---|---|
| Selection sets + pool/mass diagnostics | `artifacts/matched_controls/selection.json` | `pipeline/matched_controls_select.py` (offline) |
| Per-row ablation outcomes + baseline parity gate | `artifacts/matched_controls/matched_controls_results.json` | GPU: `pipeline/matched_controls_run.py` |
| T6–T9 contrasts | `artifacts/matched_controls/summary.json` | `pipeline/matched_controls_analyze.py` (offline) |

## Figures

`paper_figures/make_figures.py` regenerates every paper figure from
the artifacts above; `paper_figures/numbers.json` is the frozen
snapshot of paper numbers extracted from these artifacts.
