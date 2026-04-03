"""
score_fusion.py
===============
Score-level (fractional layer) fusion methods (paper §3.4).

Two fusion strategies:

1. Modality Evaluation (Equation 4)
   ----------------------------------
   D_t = Σ_q s[q]

   Sum of all comparison scores produced by a single modality classifier
   for a given claimant.  The modality with the highest D_t is trusted.

2. Rank-1 Evaluation (Equation 5)
   --------------------------------
   D_t = s[rank1] - Σ_{q=2}^{C} s[q]

   Difference between the top-ranked candidate's score and the sum of all
   other candidates' scores.  The modality with the highest D_t wins.

Both methods operate at inference time only — no additional training is
required.  They select among the individual modality classifiers that were
already trained (e.g., as branches of FeatureFusionModel).

Score conventions
-----------------
  Higher score = better match (cosine similarity in [0, 1] after L2-norm).
  Scores array shape: [N_claimants, C]  where C = candidate pool size.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

# ── Scoring functions ─────────────────────────────────────────────────────────


def modality_evaluation_score(scores: np.ndarray) -> float:
    """
    Equation (4): D_t = Σ_q s[q]

    Parameters
    ----------
    scores : np.ndarray  [C]
        Comparison scores for all C candidates for one claimant.

    Returns
    -------
    float
        Scalar score value D_t.
    """
    return float(np.sum(scores))


def rank1_evaluation_score(scores: np.ndarray) -> float:
    """
    Equation (5): D_t = s[rank1] - Σ_{q≠rank1} s[q]

    Parameters
    ----------
    scores : np.ndarray  [C]
        Comparison scores for all C candidates for one claimant.

    Returns
    -------
    float
        Scalar score value D_t.
    """
    sorted_scores = np.sort(scores)[::-1]
    rank1_score = sorted_scores[0]
    rest_sum = float(np.sum(sorted_scores[1:]))
    return float(rank1_score) - rest_sum


# ── Score fusion class ────────────────────────────────────────────────────────


class ScoreFusion:
    """
    Inference-time score-level fusion.

    For each claimant, the method:
      1. Computes D_t for every modality using the chosen scoring function.
      2. Selects the modality with the highest D_t.
      3. Uses that modality's top-1 candidate as the final decision.

    Parameters
    ----------
    method : str
        ``'rank1'`` (Eq. 5) or ``'modality'`` (Eq. 4).

    Example
    -------
    >>> sf = ScoreFusion(method='rank1')
    >>> # scores_dict maps modality name → similarity matrix [N_q, N_gallery]
    >>> results = sf.fuse(scores_dict)
    >>> # results[i] = index of the selected gallery candidate for query i
    """

    def __init__(self, method: str = "rank1") -> None:
        if method not in ("rank1", "modality"):
            raise ValueError(f"method must be 'rank1' or 'modality', got '{method}'")
        self.method = method
        self._score_fn = (
            rank1_evaluation_score if method == "rank1" else modality_evaluation_score
        )

    def fuse(
        self,
        scores_dict: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """
        Fuse scores from multiple modalities.

        Parameters
        ----------
        scores_dict : dict
            Maps modality name (str) to similarity matrix of shape
            ``[N_claimants, C]`` where C is the gallery size.

        Returns
        -------
        np.ndarray  [N_claimants]
            Index of the selected gallery candidate for each claimant.
        """
        modalities = list(scores_dict.keys())
        n_claimants = next(iter(scores_dict.values())).shape[0]

        selected_candidates = np.empty(n_claimants, dtype=np.int64)

        for i in range(n_claimants):
            best_D = -np.inf
            best_scores = None

            for mod in modalities:
                s = scores_dict[mod][i]  # shape [C]
                D = self._score_fn(s)
                if D > best_D:
                    best_D = D
                    best_scores = s

            # Select the gallery candidate with the highest score
            selected_candidates[i] = int(np.argmax(best_scores))

        return selected_candidates

    def fuse_with_labels(
        self,
        scores_dict: Dict[str, np.ndarray],
        gallery_labels: np.ndarray,
    ) -> np.ndarray:
        """
        Convenience wrapper: returns the predicted *labels* rather than
        gallery indices.

        Parameters
        ----------
        scores_dict : dict
            As in ``fuse()``.
        gallery_labels : np.ndarray  [C]
            Identity label for each gallery entry.

        Returns
        -------
        np.ndarray  [N_claimants]
            Predicted identity label for each claimant.
        """
        indices = self.fuse(scores_dict)
        return gallery_labels[indices]

    # ── Utility: build score matrices from embedding arrays ──────────────────

    @staticmethod
    def build_score_matrix(
        query_embeddings: np.ndarray,
        gallery_embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query and gallery embeddings.

        Parameters
        ----------
        query_embeddings   : np.ndarray  [N_q, D]
        gallery_embeddings : np.ndarray  [N_g, D]

        Returns
        -------
        np.ndarray  [N_q, N_g]  — cosine similarity in [-1, 1]
        """
        # L2-normalise (should already be done by the branch, but be safe)
        q_norm = query_embeddings / (
            np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-12
        )
        g_norm = gallery_embeddings / (
            np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-12
        )
        return q_norm @ g_norm.T  # [N_q, N_g]

    @staticmethod
    def split_query_gallery(
        embeddings: np.ndarray,
        labels: np.ndarray,
        query_ratio: float = 0.20,
        seed: int = 42,
    ):
        """
        Randomly split embeddings into query and gallery sets.

        Returns
        -------
        (q_emb, q_lbl, g_emb, g_lbl)
        """
        rng = np.random.default_rng(seed)
        n = len(labels)
        query_mask = rng.random(n) < query_ratio

        q_emb = embeddings[query_mask]
        q_lbl = labels[query_mask]
        g_emb = embeddings[~query_mask]
        g_lbl = labels[~query_mask]
        return q_emb, q_lbl, g_emb, g_lbl
