# Methodology

Plain-language description of what the pipeline does, why each step
exists, and how the pieces fit together. The paper's Sec. 3 covers the same
ground formally; this document is for someone reading the code.

## The question

Causal-mediation studies on SAE features typically compare a *targeted*
ablation (drop a small set of features chosen for one prompt) against a
*random* ablation (drop a random set). The contrast is real but coarse:
random features are usually inactive on the prompt, so the comparison
collapses two distinct null hypotheses — *content-specific mediation* and
*generic distributional contribution* — into one.

We refine the contrast in two ways:

1. **A graded control ladder** (six conditions) between Targeted and Random. We
   interpose a *sibling-disambiguation* control (features picked for an
   alternative correct answer to the same ambiguous question), an
   *in-distribution shuffle* (features picked for an unrelated AmbigQA
   question), and an *out-of-distribution shuffle* (features active on a
   random WikiText paragraph). These let us decompose the targeted
   effect into a content component, an in-distribution residual, and an
   out-of-distribution residual.
2. **A unique-vs-shared partition of the targeted feature set itself.**
   We split the top-K features picked for disambiguation $D_i$ into
   *cluster-shared* (features that also appear in the top-K for some
   sibling $D_j$) and *answer-unique* (features that appear only for
   $D_i$). Ablating these two subsets independently decomposes the
   targeted effect into a cluster-shared component and an answer-unique
   residual.

## The substrate

AmbigQA provides ambiguous questions paired with two-or-more
human-annotated answers, each with a disambiguated rewrite. This is the
right shape: each ambiguous parent $A$ has multiple disambiguations
$D_1, D_2, \ldots$ with distinct annotated answers, so within-question
sibling controls are immediately available. The pipeline filters down
to questions where the model commits (under greedy decode) to one of
the annotated disambiguations. We end up with 1,103 self-pairs over
448 ambiguous parents.

## What the scripts do, in order

### Data preparation (`pipeline/build_dataset.py`)

Reads AmbigQA train + validation, keeps multipleQA questions whose
length is 15–140 chars and whose annotation contains ≥ 2 distinct
short-answer disambiguations. Combines a relaxed-collision-filter
seed pool (128 questions) with a uniform random sample (1,000) of
remaining eligible questions. Output: `artifacts/expanded_dataset.json`,
1,128 candidates.

### Slot detection (inside `pipeline/main_ablation.py`)

For each candidate, greedy-decodes 15 tokens from Gemma-2-9B-IT under a
chat-templated prompt and checks whether the continuation matches any
annotated disambiguation answer (case-insensitive substring or
distinctive-token match — the same matcher used in earlier
exploration). Drops candidates that don't commit to any annotated
answer. Survivors: 448 ambiguous questions, 1,103 self-pair
ablation targets.

### Feature scoring (`src/features.py`)

For each $(A, D_i)$ self-pair, encodes the last-prompt-token residual
through the SAE to obtain $z_A$ and $z_{D_i}$. Picks the top-10
features by

$$
\text{score}(f, i) = z_{D_i}(f) \;-\; \tfrac{1}{|J|}\sum_{j \in J} z_{D_j}(f) \;+\; \tfrac{1}{2}\, z_A(f)
$$

restricted to $f$ with $z_{D_i}(f) > 0$. The first two terms enforce
*specificity to $D_i$* (active on $D_i$, less active on siblings). The
third term down-weights features that are unconditionally inactive on
the parent — these would be "free wins" in the ablation comparison.

### Ablation (`src/hooks.py`)

All ablations apply an *error-preserving SAE feature splice* at the
last-prompt-token of $A$'s forward pass through the target layer:

$$
h' = h - \mathrm{decode}(z_K)
$$

where $K$ is the set of feature indices being ablated and $z_K$ is the
restriction of $z = \mathrm{encode}(h)$ to those indices. With $K =
\varnothing$ the splice is the identity, so the empty-feature ablation
matches the unmodified forward pass exactly — used as a sanity check.
The splice preserves the SAE's reconstruction error term, isolating the
ablation effect to the contribution of $K$.

### Six conditions (`pipeline/main_ablation.py`)

For each self-pair we run:

1. **Baseline** — no intervention.
2. **Targeted** — ablate the top-10 picked for $D_i$.
3. **Sibling** — ablate the top-10 picked for $D_j$ (some $j \neq i$).
4. **Shuffled-AmbigQA** — ablate the top-10 picked for an unrelated
   ambiguous question $A'$ (3 random draws per pair).
5. **WikiText-shuffled** — ablate the top-10 features most active on a
   random WikiText paragraph's last token (3 random draws per pair).
6. **Random** — ablate 10 random feature indices (3 random draws).

We measure hit@1 against the first-token-variants of $D_i$'s annotated
answer. The shuffle controls average across draws. The contrast
between conditions 3 and 5 — *Sibling vs WikiText-shuffled* — is the
load-bearing answer-specificity test, T3 in the paper.

### Polysemy correction (`pipeline/polysemy_partition.py`)

A subset of disambiguation-derived features fire dominantly on
*paragraph-initial* tokens of generic prose — a structural rather than
content-driven activation pattern. We test each feature on 800
WikiText-2 raw paragraphs and compute, across all (paragraph, position)
pairs with $z_f > 0$, the fraction at position 0. Features with
$\text{pct}_{\text{pos}0} \geq 0.80$ and at least 100 nonzero
appearances are flagged *position-suspect*. Splitting each top-10 set
into content vs position subsets lets us decompose the targeted effect
along an orthogonal axis.

### Unique-vs-shared decomposition (`pipeline/unique_vs_shared.py`)

For each $(A, D_i)$, we define

$$
\mathcal{F}_i^{\mathrm{shared}} = \mathcal{F}_i \cap \bigcup_{j \neq i} \mathcal{F}_j, \qquad
\mathcal{F}_i^{\mathrm{unique}} = \mathcal{F}_i \setminus \mathcal{F}_i^{\mathrm{shared}}
$$

and run **Shared-only** and **Unique-only** ablations independently.
The two effects roughly add to the joint Targeted effect; the residual
captures the shared × unique interaction.

### Robustness scripts

- **Layer trajectory** (`layer_sweep.py` + `layer_bookends.py`): re-runs
  the six-condition protocol at L20, L26, L30, L34, L37, L40, L41. The signal
  grows monotonically with depth and peaks at the last residual layer.
- **L0 sweep** (`l0_sweep.py`): re-runs at L37 and L41 across the
  publicly released SAE L0 family. T3 collapses with L0 at L37; remains
  stable at L41. This is why L41 is the reference layer.
- **Reference-layer analysis** (`reference_layer_analysis.py`): produces
  the headline polysemy + multi-metric (KL, generation-flip) +
  2×2-joint + Cramér's V analysis at L41, plus the cross-substrate
  / cross-architecture summary table.
- **Cross-axis replication** (`replication_gemma2b.py`,
  `replication_triviaqa.py`, `replication_llama.py`): re-runs the
  protocol with a different model (Gemma-2-2B-IT @ L25), a different
  substrate (TriviaQA-Web answer aliases), and a different architecture
  (Llama-3.1-8B-Instruct @ L31, with Llama Scope SAE).

### Supporting / disclosure analyses

- `multimetric.py`: KL divergence, logit-difference, generation-flip
  alongside hit@1.
- `four_way_decomposition.py`: joint (unique/shared) × (content/position)
  partition.
- `per_feature_equivalence.py`: single-feature ablation of the
  highest-magnitude content feature in the unique vs shared subset.
- `per_pair_audit.py`: regression of the headline outcomes on per-pair
  covariates (baseline hit@1, $K$, etc.).
- `overlap_shells.py`: T3 across max-overlap shells $K = 0, 1, \ldots,
  10$, used in the Limitations discussion.
- `slot_detection_sensitivity.py`: slot-retention under varied matchers.

## What's not in the pipeline

- Hyperparameter search. The score function's $a$-weight (= 0.5), the
  top-K (= 10), and the polysemy threshold (= 0.80) are all set in
  `src/config.py`. We report a sweep of the polysemy threshold in the
  paper as a robustness check; the $a$-weight and top-K were chosen
  before the published runs and are not swept.
- Adversarial probing. The protocol is designed to surface a specific
  decomposition; whether the SAE features are robust under adversarial
  ablation patterns is out of scope.

## Key library modules

- `src/model.py` — Gemma-2 / Llama-3 loading with eager attention
  (required for forward-hook interventions).
- `src/sae.py` — Gemma Scope and Llama Scope SAE I/O. Llama Scope's
  `dataset_average_activation_norm` field is informational only at
  inference; the published weights are raw-input-ready (see the
  Limitations-section disclosure on the Llama base→Instruct shift).
- `src/hooks.py` — last-token residual capture, all-position residual
  capture (used for WikiText), and the error-preserving SAE feature
  splice.
- `src/features.py` — the score function and its variants.
- `src/slot_detection.py` — substring + distinctive-token matcher.
- `src/prompts.py` — chat-templated prompt formatter shared across
  Gemma; `pipeline/replication_llama.py` defines the Llama-3 variant
  inline because the chat templates differ.
- `src/analysis.py` — bootstrap CIs, McNemar, paired Wilcoxon, per-pair
  means.
- `src/wikitext.py` — WikiText-2 raw test loader for shuffle controls
  and polysemy.
