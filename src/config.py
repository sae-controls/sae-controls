"""Constants and paths for the paper pipeline.

Every constant in this file is referenced by paper text. Changing one without
re-running the pipeline will produce numbers that disagree with the paper.
"""
from __future__ import annotations

import os
from pathlib import Path

# ============================================================================
# Model
# ============================================================================
MODEL_ID = "google/gemma-2-9b-it"
DTYPE_STR = "bfloat16"
ATTN_IMPL = "eager"          # required for forward-hook-based interventions
FINAL_LOGIT_SOFTCAP = 30.0   # informational only; applied inside Gemma2ForCausalLM

# ============================================================================
# SAE — Gemma Scope, layer 37, width 16k
# ============================================================================
LAYER = 37
SAE_WIDTH = "16k"
SAE_L0_SEL = "canonical"     # request label; resolves to average_l0_11 in practice
SAE_REPOS = (
    "google/gemma-scope-9b-pt-res-canonical",
    "google/gemma-scope-9b-pt-res",
)
# Pinned: SHA-256 of the resolved checkpoint as observed in every run
SAE_CHECKPOINT_SHA256 = (
    "71491ae42b5c36760cf685ccd117c6cff81c573eda8dfd6d9baa21d8dc3a521e"
)
SAE_RESOLVED_SUBPATH = "layer_37/width_16k/average_l0_11/params.npz"

# ============================================================================
# Methodology constants
# ============================================================================
TOP_K_FEATURES = 10          # features ablated per (P, D_i)
SCORE_A_WEIGHT = 0.5         # weight on z_A in the score function
RANDOM_CONTROLS = 3          # random-feature draws per self-pair
N_SHUFFLES_PER_PAIR = 3      # AmbigQA-shuffle and WikiText-shuffle draws per self-pair
TOP_LOGITS_K = 10            # top-k decoded logits stored per forward pass

# ============================================================================
# Position-mode classification (polysemy correction, paper Sec. 3.5)
# ============================================================================
POSITION_NONZERO_FLOOR = 100         # minimum n_nonzero on WikiText for a feature to be tested
POSITION_PCT_THRESHOLD = 0.80        # minimum pct_at_pos0 for a feature to be flagged

# ============================================================================
# WikiText (used for OOD shuffle control + feature characterization + position mode)
# ============================================================================
WIKITEXT_DATASET = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"
WIKITEXT_SPLIT = "test"
WIKITEXT_N_PARAGRAPHS = 800
WIKITEXT_MIN_TOKENS = 50
WIKITEXT_MAX_TOKENS = 512

# ============================================================================
# Seeds
# ============================================================================
SEED = 0                # main run / random-feature controls / dataset sample / feature sampling
SHUFFLE_SEED = 42       # WikiText-shuffle draws (intentionally distinct so streams don't correlate)
RANDOM_BASELINE_SEED = 1 # 5 random features for the interpretability baseline

# ============================================================================
# Prompt formatting
# ============================================================================
SYSTEM_PREAMBLE = "Answer the question with a single name or short phrase."

# ============================================================================
# Paths
# ============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
REPORTS_DIR = REPO_ROOT / "reports"
PAIRS_PATH = DATA_DIR / "patching_pairs.jsonl"


def device() -> str:
    """Compute device. Override with CWX_DEVICE env var; default cuda."""
    return os.environ.get("CWX_DEVICE", "cuda")
