"""Diversity metrics for recommendations and generated responses."""
from collections.abc import Sequence


def compute_catalog_diversity(
    recommended_track_ids: Sequence[str], catalog_size: int
) -> float:
    """Catalog diversity: (# unique recommended tracks) / (catalog size)."""
    if catalog_size <= 0:
        return 0.0
    return len(set(recommended_track_ids)) / float(catalog_size)


def compute_lexical_diversity(responses: Sequence[str], n: int = 2) -> float:
    """Lexical diversity via Distinct-n: unique n-grams / total n-grams."""
    ngrams = set()
    total_ngrams = 0
    for response in responses:
        tokens = response.lower().split()
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngrams.add(tuple(tokens[i : i + n]))
            total_ngrams += 1
    if total_ngrams == 0:
        return 0.0
    return len(ngrams) / total_ngrams
