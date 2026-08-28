"""
eval_giantsteps.py

The real accuracy check: run the same Krumhansl-Schmuckler key detector
from features.py against the GiantSteps Key dataset -- 604 real tracks
with human-verified key labels (not synthesized, not self-generated).
This is the number the ISMIR LBD abstract's headline claim depends on.

GiantSteps labels use flat notation (Eb, Db, Ab, Gb, Bb); this project's
NOTE_NAMES uses sharp notation (D#, C#, G#, F#, A#) -- mapped below.
"""
import os
import sys
import glob
import json
import time

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_datasets", "giantsteps-key-dataset")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
KEY_DIR = os.path.join(DATASET_DIR, "annotations", "key")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "real", "giantsteps_eval.json")

FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
}


def parse_ground_truth(text):
    text = text.strip()
    root, mode = text.split(" ")
    root = FLAT_TO_SHARP.get(root, root)  # normalize flats to sharps; naturals pass through unchanged
    return root, mode


def main():
    key_files = sorted(glob.glob(os.path.join(KEY_DIR, "*.key")))
    print(f"Found {len(key_files)} ground-truth key annotations")

    results = []
    n_exact, n_root_only, n_total = 0, 0, 0
    start = time.time()

    for i, kf in enumerate(key_files):
        track_id = os.path.basename(kf).replace(".key", "")
        audio_path = os.path.join(AUDIO_DIR, track_id + ".mp3")
        if not os.path.exists(audio_path):
            print(f"  [SKIP] {track_id}: no audio file")
            continue

        with open(kf) as f:
            true_root, true_mode = parse_ground_truth(f.read())

        try:
            feats = extract_features(audio_path)
        except Exception as e:
            print(f"  [SKIP] {track_id}: extraction failed ({e})")
            continue

        exact = (feats["detected_root"] == true_root and feats["detected_mode"] == true_mode)
        root_only = (feats["detected_root"] == true_root)
        n_total += 1
        n_exact += exact
        n_root_only += root_only

        results.append({
            "track_id": track_id,
            "true_root": true_root, "true_mode": true_mode,
            "detected_root": feats["detected_root"], "detected_mode": feats["detected_mode"],
            "exact_match": exact, "root_only_match": root_only,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            running_acc = 100 * n_exact / n_total
            print(f"  [{i+1}/{len(key_files)}]  running exact-match accuracy: {running_acc:.1f}%  "
                  f"({elapsed/(i+1):.2f}s/track)")

    elapsed = time.time() - start
    exact_acc = 100 * n_exact / n_total
    root_acc = 100 * n_root_only / n_total

    print(f"\n{'='*70}")
    print(f"GiantSteps Key Dataset -- REAL, human-verified ground truth")
    print(f"{'='*70}")
    print(f"Tracks evaluated: {n_total}")
    print(f"Exact match (root + mode): {n_exact}/{n_total} = {exact_acc:.1f}%")
    print(f"Root-only match (ignoring major/minor): {n_root_only}/{n_total} = {root_acc:.1f}%")
    print(f"Time: {elapsed:.1f}s ({elapsed/n_total:.2f}s/track)")

    with open(OUT_PATH, "w") as f:
        json.dump({
            "n_total": n_total, "n_exact": n_exact, "n_root_only": n_root_only,
            "exact_accuracy_pct": round(exact_acc, 1), "root_only_accuracy_pct": round(root_acc, 1),
            "per_track": results,
        }, f, indent=2)
    print(f"\nFull results -> {OUT_PATH}")


if __name__ == "__main__":
    main()
