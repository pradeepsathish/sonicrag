"""
cache_wtc_chroma.py

Same idea as cache_giantsteps_chroma.py, but for Bach's Well-Tempered
Clavier Book 1 -- the non-EDM, public-domain contrast dataset. Ground
truth is parsed directly from the filenames, which explicitly state the
key (e.g. "... in C-sharp minor, BWV 849") since each of the 48 pieces
is, by design, one prelude+fugue pair per key across all 24 keys.
"""
import os
import re
import glob
import json
import time
import numpy as np
import librosa

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_datasets", "wtc_book1")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "real", "wtc_chroma_cache.npz")

NOTE_MAP = {
    "C-sharp": "C#", "D-sharp": "D#", "F-sharp": "F#", "G-sharp": "G#", "A-sharp": "A#",
    "D-flat": "C#", "E-flat": "D#", "G-flat": "F#", "A-flat": "G#", "B-flat": "A#",
}


def parse_key_from_filename(fname):
    m = re.search(r"in ([A-G](?:-sharp|-flat)?) (major|minor)", fname)
    if not m:
        return None, None
    root_raw, mode = m.group(1), m.group(2)
    root = NOTE_MAP.get(root_raw, root_raw)
    return root, mode


def main():
    files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.mp3")))
    print(f"Found {len(files)} WTC audio files")

    chroma_vectors, track_ids, true_roots, true_modes = [], [], [], []
    start = time.time()

    for i, path in enumerate(files):
        fname = os.path.basename(path)
        root, mode = parse_key_from_filename(fname)
        if root is None:
            print(f"  [SKIP] could not parse key from: {fname}")
            continue

        try:
            y, sr = librosa.load(path, sr=22050, mono=True)
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
            chroma_mean = chroma.mean(axis=1)
            chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-9)
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")
            continue

        chroma_vectors.append(chroma_mean)
        track_ids.append(fname)
        true_roots.append(root)
        true_modes.append(mode)
        print(f"  [{i+1}/{len(files)}] {fname[-50:]:50s} -> true={root} {mode}")

    chroma_vectors = np.stack(chroma_vectors)
    np.savez(OUT_PATH, chroma=chroma_vectors, track_ids=track_ids,
              true_roots=true_roots, true_modes=true_modes)

    elapsed = time.time() - start
    print(f"\nCached {len(track_ids)} WTC chroma vectors in {elapsed:.1f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
