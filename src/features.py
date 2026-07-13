"""Feature scoring — picks the top-K SAE features specific to disambiguation D_i
within an ambiguous-question pair P (paper Sec. 3.2.2).

Score formula:

    score(f, i) = z_{D_i}(f) - mean_{j ≠ i} z_{D_j}(f) + 0.5 · z_A(f)

Mask to features with z_{D_i}(f) > 0 (only features the disambiguation activates
positively can be specific to it). Take the top-K by score; if fewer than K
features are positive, return however many are.
"""
from __future__ import annotations

from .config import SCORE_A_WEIGHT, TOP_K_FEATURES


def score_marginal(z_i, top_k: int = TOP_K_FEATURES) -> list[int]:
    """Non-discriminative selector, for the selector-ablation check.

    Picks top-K features by raw z_{D_i} (post-JumpReLU activation), with no
    subtraction of D_j and no z_A boost. Mask to z_{D_i} > 0; clamp k to the
    number of positives. Used to test whether the discriminative structure of
    the published score function (z_{D_i} - mean_{j != i} z_{D_j} + a*z_A) is
    doing the work behind the answer-specificity gap, or whether any
    "active-on-D_i" selector would suffice.
    """
    import torch
    mask = z_i > 0
    score_masked = torch.where(mask, z_i, torch.full_like(z_i, -1e6))
    n_pos = int(mask.sum().item())
    k = min(top_k, n_pos)
    if k <= 0:
        return []
    return torch.topk(score_masked, k=k).indices.cpu().tolist()


def score_specific_features(z_A, z_D_list, top_k: int = TOP_K_FEATURES,
                            a_weight: float = SCORE_A_WEIGHT) -> list[list[int]]:
    """
    Args:
      z_A:      (d_sae,) post-JumpReLU activation for the ambiguous prompt
      z_D_list: list of (d_sae,) tensors, one per disambiguation in order
      top_k:    features to keep per disambig (clamped to # of positives)
      a_weight: multiplier on z_A in the score (0.5 in the published methodology)

    Returns:
      list of length len(z_D_list); entry i is the top-K feature ids for D_i.
    """
    import torch
    out: list[list[int]] = []
    for i, z_i in enumerate(z_D_list):
        others = [z_D_list[j] for j in range(len(z_D_list)) if j != i]
        mean_other = (
            torch.stack(others, dim=0).mean(0) if others else torch.zeros_like(z_i)
        )
        score = (z_i - mean_other) + a_weight * z_A
        mask = z_i > 0
        score_masked = torch.where(mask, score, torch.full_like(score, -1e6))
        n_pos = int(mask.sum().item())
        k = min(top_k, n_pos)
        if k <= 0:
            out.append([])
            continue
        topk = torch.topk(score_masked, k=k)
        out.append(topk.indices.cpu().tolist())
    return out
