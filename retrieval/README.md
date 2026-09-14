# Retrieval

Text-to-item retrieval over the track metadata catalog used in the RecSys 2026
MusicCRS challenge, plus evaluation utilities for scoring predictions.
Adapted from [`music-crs-baselines`](https://github.com/nlp4musa/music-crs-baselines)
and [`music-crs-evaluator`](https://github.com/nlp4musa/music-crs-evaluator).

## Structure

```
retrieval/
├── base.py               # RetrievalModule abstract interface
├── data_loader.py        # MusicCatalogLoader — loads track metadata from Hugging Face
├── bm25.py                # BM25Retriever — BM25 baseline
├── run_bm25_baseline.py   # generates BM25 predictions over a dialogue dataset split
└── evaluation/
    ├── metrics.py             # nDCG
    ├── diversity.py           # item (catalog) diversity
    ├── make_ground_truth.py   # extract ground truth from the challenge dialogue dataset
    └── evaluate.py            # evaluate() + CLI runner — nDCG, diversity, and final_score
```

See the top-level [README's Dataset section](../README.md#dataset) for what
data is available in the RecSys 2026 MusicCRS challenge collection.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Retrieval

`BM25Retriever` builds (and caches to `./cache/bm25/<corpus_name>`) a BM25 index
over selected metadata fields for `talkpl-ai/TalkPlayData-Challenge-Track-Metadata`,
then ranks track IDs for a query.

```python
from retrieval import BM25Retriever

retriever = BM25Retriever()  # downloads + indexes on first run, cached after
retriever.text_to_item_retrieval("upbeat pop song", topk=5)
retriever.batch_text_to_item_retrieval(["upbeat pop song", "sad piano ballad"], topk=5)
```

To add a new retrieval method, implement `RetrievalModule` (`base.py`):
`text_to_item_retrieval(query, topk)` and `batch_text_to_item_retrieval(queries, topk)`.

## Evaluation

### Ground truth

`talkpl-ai/TalkPlayData-Challenge-Dataset` has `train` (15,199 sessions) and
`test` (1,000 sessions, used as the devset) splits. Each session has 8 turns;
each turn embeds the gold track ID and gold response directly in its
`conversations` field. Extract these into the format `evaluate.py` expects:

```bash
python3 -m retrieval.evaluation.make_ground_truth --split test --output path/to/ground_truth.json
```

### Generating BM25 predictions

`run_bm25_baseline.py` retrieves top-k track IDs for every session x turn in a
dialogue dataset split. The query is the full conversation history up to and
including that turn's user message (prior `music` turns are expanded to
track metadata), matching the retrieval input built in
`music-crs-baselines/mcrs/crs_baseline.py`. There is no LLM query rewriting
or response generation — retrieval only, so `predicted_response` is left
empty:

```bash
python3 -m retrieval.run_bm25_baseline --split test --topk 20 --output path/to/predictions.json
```

### Scoring predictions

`evaluate()` computes macro-averaged nDCG@{1,10,20}, item (catalog)
diversity, and a combined **final score**, given predictions and ground
truth in the challenge format:

  - Predictions: list of `{session_id, turn_number, predicted_track_ids, predicted_response}`
  - Ground truth: list of `{session_id, turn_number, ground_truth_track_id}`

```bash
python3 -m retrieval.evaluation.evaluate \
  --predictions path/to/predictions.json \
  --ground_truth path/to/ground_truth.json \
  --catalog_size 47071 \
  --output path/to/results.json
```

Or from Python:

```python
from retrieval.evaluation import compute_ndcg_metrics, compute_catalog_diversity
```

#### Final score

Submissions for the item ranking task are ranked by a single combined score:

```
final_score = 0.8 * nDCG@20 + 0.2 * catalog_diversity
```

nDCG@20 rewards ranking the correct track highly; catalog diversity
(unique recommended tracks / catalog size) rewards not just recommending
the same popular handful of tracks to everyone. `evaluate()` returns this
as `final_score` alongside the individual metrics.

### BM25 baseline results (test split, 1,000 sessions x 8 turns)

| Metric | Value |
| --- | ---: |
| nDCG@1 | 0.0101 |
| nDCG@10 | 0.0644 |
| nDCG@20 | 0.0828 |
| Catalog diversity | 0.3906 |
| **Final score** | **0.1444** |
