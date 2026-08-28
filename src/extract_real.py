"""
extract_real.py

Runs the same extraction logic as features.py, but on real audio from
FMA-small instead of the synthesized clips. No ground-truth key/tempo/
timbre labels exist for these tracks (that needs GiantSteps or similar,
a separate step) -- this just proves real DSP + real search work on
real music, using the search engine's own detected values.

Samples a spread of tracks across the dataset rather than the first N,
since track IDs aren't randomly distributed.
"""
import numpy as np
import json
import os
import sys
import glob
import time

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features

FMA_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_datasets", "fma_small")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real")
FEATURES_PATH = os.path.join(OUT_DIR, "features.npz")
META_PATH = os.path.join(OUT_DIR, "metadata.json")

N_SAMPLE = 8000


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(FMA_AUDIO_DIR, "*", "*.mp3")))
    print(f"Found {len(all_files)} real tracks in {FMA_AUDIO_DIR}")

    step = max(1, len(all_files) // N_SAMPLE)
    sample = all_files[::step][:N_SAMPLE]
    print(f"Sampling {len(sample)} tracks (every {step}th file) for first real-data pass")

    vectors, chroma_vectors, metadata = [], [], []
    start = time.time()
    for i, path in enumerate(sample):
        try:
            feats = extract_features(path)
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(path)}: {e}")
            continue
        vectors.append(feats["vector"])
        chroma_vectors.append(feats["chroma"])
        rel_path = os.path.relpath(path, FMA_AUDIO_DIR).replace("\\", "/")
        metadata.append({
            "id": i,
            "file": rel_path,
            "detected_root": feats["detected_root"],
            "detected_mode": feats["detected_mode"],
            "detected_tempo": round(feats["tempo"], 1),
            "spectral_centroid": round(feats["centroid"], 1),
        })
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            rate = elapsed / (i + 1)
            print(f"  [{i+1}/{len(sample)}] {rel_path}  "
                  f"key={feats['detected_root']} {feats['detected_mode']}  "
                  f"bpm={feats['tempo']:.1f}  ({rate:.2f}s/track)")

    vectors = np.stack(vectors)
    chroma_vectors = np.stack(chroma_vectors)
    np.savez(FEATURES_PATH, vectors=vectors, chroma=chroma_vectors)
    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start
    print(f"\nExtracted {len(metadata)} real tracks in {elapsed:.1f}s "
          f"({elapsed/len(metadata):.2f}s/track average)")
    print(f"Features -> {FEATURES_PATH}")
    print(f"Metadata -> {META_PATH}")
    print(f"\nAt this rate, all {len(all_files)} tracks would take "
          f"~{elapsed/len(metadata)*len(all_files)/60:.0f} minutes.")


if __name__ == "__main__":
    main()
