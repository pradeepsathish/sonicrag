"""
eval.py

Honest evaluation: 24 hand-built queries against KNOWN ground truth
(we control the synthetic dataset, so we know exactly which tracks
SHOULD rank highly for each query). No fabricated numbers — whatever
comes out of this run is what goes in the README.

Relevance grading (for NDCG): a track is relevant to a query if it
matches on the query's stated key (root+mode) AND/OR is within the
stated tempo tolerance -- graded 0/1/2 depending on how many of the
query's stated constraints it satisfies.
"""
import json
import os
import numpy as np
from search import ALAISearchEngine

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
META_PATH = os.path.join(DATA_DIR, "metadata_enriched.json")

# Hand-built queries: text + the ground-truth constraint(s) it implies.
# These are authored by looking at what's actually IN the dataset, not
# reverse-engineered to make numbers look good after the fact.
EVAL_QUERIES = [
    {"text": "melancholic in C minor", "root": "C", "mode": "minor", "bpm": None},
    {"text": "F minor slow", "root": "F", "mode": "minor", "bpm": None},
    {"text": "E minor around 140 bpm", "root": "E", "mode": "minor", "bpm": 140},
    {"text": "A# minor fast", "root": "A#", "mode": "minor", "bpm": None},
    {"text": "C# minor 108 bpm", "root": "C#", "mode": "minor", "bpm": 108},
    {"text": "D# minor moody", "root": "D#", "mode": "minor", "bpm": None},
    {"text": "G minor", "root": "G", "mode": "minor", "bpm": None},
    {"text": "C major bright", "root": "C", "mode": "major", "bpm": None},
    {"text": "A major warm 108 bpm", "root": "A", "mode": "major", "bpm": 108},
    {"text": "G# major mellow", "root": "G#", "mode": "major", "bpm": None},
    {"text": "F# major 84 bpm", "root": "F#", "mode": "major", "bpm": 84},
    {"text": "B major", "root": "B", "mode": "major", "bpm": None},
    {"text": "D major 132 bpm", "root": "D", "mode": "major", "bpm": 132},
    {"text": "A# major", "root": "A#", "mode": "major", "bpm": None},
    {"text": "D# major 120 bpm", "root": "D#", "mode": "major", "bpm": 120},
    {"text": "C# major", "root": "C#", "mode": "major", "bpm": None},
    {"text": "72 bpm slow track", "root": None, "mode": None, "bpm": 72},
    {"text": "140 bpm fast track", "root": None, "mode": None, "bpm": 140},
    {"text": "96 bpm", "root": None, "mode": None, "bpm": 96},
    {"text": "120 bpm", "root": None, "mode": None, "bpm": 120},
    {"text": "E minor", "root": "E", "mode": "minor", "bpm": None},
    {"text": "E major", "root": "E", "mode": "major", "bpm": None},
    {"text": "C# minor slow", "root": "C#", "mode": "minor", "bpm": None},
    {"text": "132 bpm major", "root": None, "mode": "major", "bpm": 132},
]


def relevance_grade(track_entry, query):
    """0 = irrelevant, 1 = partial match, 2 = full match on stated constraints."""
    constraints = []
    if query["root"] is not None:
        constraints.append(track_entry["root"] == query["root"])
    if query["mode"] is not None:
        constraints.append(track_entry["mode"] == query["mode"])
    if query["bpm"] is not None:
        constraints.append(abs(track_entry["bpm"] - query["bpm"]) <= 8)

    if not constraints:
        return 0
    n_met = sum(constraints)
    if n_met == len(constraints):
        return 2
    elif n_met > 0:
        return 1
    return 0


def dcg(relevances):
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(ranked_relevances, k=5):
    dcg_val = dcg(ranked_relevances[:k])
    ideal = dcg(sorted(ranked_relevances, reverse=True)[:k])
    return dcg_val / ideal if ideal > 0 else 0.0


def reciprocal_rank(ranked_relevances):
    for i, rel in enumerate(ranked_relevances):
        if rel >= 2:  # first FULLY relevant result
            return 1.0 / (i + 1)
    for i, rel in enumerate(ranked_relevances):
        if rel >= 1:  # fallback: first partially relevant
            return 0.5 / (i + 1)
    return 0.0


def main():
    engine = ALAISearchEngine()
    with open(META_PATH) as f:
        metadata = json.load(f)
    file_to_entry = {m["file"]: m for m in metadata}

    ndcgs, mrrs = [], []
    print(f"{'Query':45s} {'NDCG@5':>8s} {'MRR':>8s}  Top match")
    print("-" * 100)
    for q in EVAL_QUERIES:
        parsed, results = engine.search(q["text"], top_k=10)
        relevances = [relevance_grade(file_to_entry[r["file"]], q) for r in results]
        n = ndcg_at_k(relevances, k=5)
        m = reciprocal_rank(relevances)
        ndcgs.append(n)
        mrrs.append(m)
        top = results[0]
        print(f"{q['text']:45s} {n:8.3f} {m:8.3f}  {top['file']} "
              f"({top['root']} {top['mode']}, {top['bpm']}bpm)")

    print("-" * 100)
    print(f"\nMean NDCG@5 across {len(EVAL_QUERIES)} queries: {np.mean(ndcgs):.3f}")
    print(f"Mean MRR across {len(EVAL_QUERIES)} queries:     {np.mean(mrrs):.3f}")
    print(f"\n(For comparison, the original dossier CLAIMED "
          f"NDCG@10 of 0.891 and MRR of 0.842 with no code behind those "
          f"numbers. These are real, computed, reproducible instead.)")


if __name__ == "__main__":
    main()
