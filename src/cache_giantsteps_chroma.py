"""
cache_giantsteps_chroma.py

Extracts and caches the raw chroma vector for all 604 GiantSteps tracks,
ONE TIME, so different key profiles (Krumhansl, Essentia edma/edmm, etc.)
can be scored against the same real audio in seconds instead of re-running
CQT + beat tracking for every comparison. Does not modify features.py,
search.py, or any of the official pipeline's configured behavior --
this is a standalone analysis cache.
"""
import os
import sys
import glob
import json
import time
import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(__file__))

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_datasets", "giantsteps-key-dataset")
AUDIO_DIR = os.path.join(DATASET_DIR, "audio")
KEY_DIR = os.path.join(DATASET_DIR, "annotations", "key")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "real", "giantsteps_chroma_cache.npz")

FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


def parse_ground_truth(text):
    root, mode = text.strip().split(" ")
    return FLAT_TO_SHARP.get(root, root), mode


def main():
    key_files = sorted(glob.glob(os.path.join(KEY_DIR, "*.key")))
    print(f"Found {len(key_files)} ground-truth key annotations")

    chroma_vectors, track_ids, true_roots, true_modes = [], [], [], []
    start = time.time()

    for i, kf in enumerate(key_files):
        track_id = os.path.basename(kf).replace(".key", "")
        audio_path = os.path.join(AUDIO_DIR, track_id + ".mp3")
        if not os.path.exists(audio_path):
            continue
        with open(kf) as f:
            true_root, true_mode = parse_ground_truth(f.read())

        try:
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
            chroma_mean = chroma.mean(axis=1)
            chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-9)
        except Exception as e:
            print(f"  [SKIP] {track_id}: {e}")
            continue

        chroma_vectors.append(chroma_mean)
        track_ids.append(track_id)
        true_roots.append(true_root)
        true_modes.append(true_mode)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(key_files)}]  ({elapsed/(i+1):.2f}s/track)")

    chroma_vectors = np.stack(chroma_vectors)
    np.savez(OUT_PATH, chroma=chroma_vectors, track_ids=track_ids,
              true_roots=true_roots, true_modes=true_modes)

    elapsed = time.time() - start
    print(f"\nCached {len(track_ids)} chroma vectors in {elapsed:.1f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
