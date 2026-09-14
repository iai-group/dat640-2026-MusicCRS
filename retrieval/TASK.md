# Task: Item Ranking

## Overview

Given a conversation between a user and a music recommender system, your task is to
**rank the track catalog and return the top tracks the user is most likely looking
for at the current turn**. This is the retrieval component of the MusicCRS pipeline:
given the conversation so far, produce a ranked list of candidate track IDs from the
~47k-track catalog.

You are given a working BM25 baseline (see [`README.md`](README.md)) as your
starting point. Your task is to build a retrieval method that **outperforms it** on
the final score defined below.

## Input and output

- **Input:** a conversation turn, consisting of the dialogue history up to and
  including the user's current message (see `run_bm25_baseline.py` for how the
  baseline builds its query from this history).
- **Output:** a ranked list of up to 20 track IDs from the catalog
  (`talkpl-ai/TalkPlayData-Challenge-Track-Metadata`), most relevant first.

Implement your method as a subclass of `RetrievalModule` (see `base.py`):

```python
def text_to_item_retrieval(self, query: str, topk: int) -> list[str]: ...
def batch_text_to_item_retrieval(self, queries: list[str], topk: int) -> list[list[str]]: ...
```

This keeps your method a drop-in replacement for `BM25Retriever` everywhere it's
used (prediction generation, evaluation).

## Data available to you

- Track metadata (name, artist, album, tags, popularity, release date)
- Track and user embeddings, precomputed with multiple encoders (audio, image,
  collaborative filtering, and text) — see the top-level
  [README's Dataset section](../README.md#dataset) for the full list
- The conversation dataset itself, including user profiles and conversation goals

You are free to use any combination of these. Some directions to consider:
dense/semantic retrieval using the precomputed embeddings, hybrid ranking
(combining BM25 with dense similarity), query expansion or rewriting,
personalization using user profiles, or reranking with a cross-encoder.

## Evaluation

Your method will be evaluated on the `test` split (the devset) using:

```bash
python3 -m retrieval.run_bm25_baseline --split test --topk 20 --output predictions.json   # or your own runner
python3 -m retrieval.evaluation.evaluate --predictions predictions.json --ground_truth ground_truth.json --catalog_size 47071
```

### Final score

Your ranking is scored on a single combined metric:

```
final_score = 0.8 * nDCG@20 + 0.2 * catalog_diversity
```

- **nDCG@20** — how highly you rank the ground truth track within your top 20
  predictions, averaged over all turns. This is the primary signal: getting the
  right track ranked highly matters most.
- **catalog_diversity** — the fraction of the track catalog covered by your
  recommendations across all turns (unique recommended tracks / catalog size).
  This penalizes methods that recommend the same small set of popular tracks to
  everyone, regardless of the query.

Use `retrieval/evaluation/evaluate.py` (or `retrieval.evaluation.evaluate()` from
Python) to compute this yourself during development — this is the exact function
used for grading.

### Baseline to beat

| Metric | BM25 baseline |
|---|---:|
| nDCG@20 | 0.0828 |
| Catalog diversity | 0.3906 |
| **Final score** | **0.1444** |

## Notes

- You do not need to generate a text response for this task — leave
  `predicted_response` empty in your predictions; it isn't scored here.
- Ground truth for the `test` split can be extracted yourself with
  `retrieval.evaluation.make_ground_truth` for local development — see
  `README.md`. Do not use it to fit your method (e.g. hard-coding gold
  tracks); it's for evaluation only.
