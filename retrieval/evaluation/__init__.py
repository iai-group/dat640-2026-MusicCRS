"""Evaluation metrics for the retrieval module.

Provides nDCG (primary metric) and diversity metrics, adapted from the
RecSys 2026 MusicCRS challenge evaluator.
"""
from .metrics import compute_ndcg_metrics, get_ndcg
from .diversity import compute_catalog_diversity, compute_lexical_diversity
from .make_ground_truth import make_ground_truth

__all__ = [
    "compute_ndcg_metrics",
    "get_ndcg",
    "compute_catalog_diversity",
    "compute_lexical_diversity",
    "make_ground_truth",
]
