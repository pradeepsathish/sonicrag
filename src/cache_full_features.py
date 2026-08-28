"""
cache_full_features.py

Re-extracts the FULL feature vector (chroma + mel-spectrogram + tempo +
spectral shape -- everything features.py computes, not just chroma) for
GiantSteps and WTC, so the FMA-trained genre classifier (which expects
that full feature space) can be applied to them. The earlier chroma-only
caches (cache_giantsteps_chroma.py, cache_wtc_chroma.py) were built for
the profile comparison and don't include what a genre classifier needs.
"""
import os
import sys
import glob
import re
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from features import extract_features

GIANTSTEPS_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_datasets", "giantsteps-key-dataset", "audio")
GIANTSTEPS_KEY_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_datasets", "giantsteps-key-dataset", "annotations", "key")
WTC_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw_datasets", "wtc_book1")

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real")

FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}
NOTE_MAP = {
    "C-sharp": "C#", "D-sharp": "D#", "F-sharp": "F#", "G-sharp": "G#", "A-sharp": "A#",
    "D-flat": "C#", "E-flat": "D#", "G-flat": "F#", "A-flat": "G#", "B-flat": "A#",
}


def cache_giantsteps():
    key_files = sorted(glob.glob(os.path.join(GIANTSTEPS_KEY_DIR, "*.key")))
    vectors, track_ids, true_roots, true_modes = [], [], [], []
    start = time.time()
    for i, kf in enumerate(key_files):
        track_id = os.path.basename(kf).replace(".key", "")
        audio_path = os.path.join(GIANTSTEPS_AUDIO_DIR, track_id + ".mp3")
        if not os.path.exists(audio_path):
            continue
        with open(kf) as f:
            root, mode = f.read().strip().split(" ")
            root = FLAT_TO_SHARP.get(root, root)
        try:
            feats = extract_features(audio_path)
        except Exception as e:
            print(f"  [SKIP] {track_id}: {e}")
            continue
        vectors.append(feats["vector"])
        track_ids.append(track_id)
        true_roots.append(root)
        true_modes.append(mode)
        if (i + 1) % 50 == 0:
            print(f"  giantsteps [{i+1}/{len(key_files)}]  ({(time.time()-start)/(i+1):.2f}s/track)")
    vectors = np.stack(vectors)
    np.savez(os.path.join(OUT_DIR, "giantsteps_full_features.npz"),
              vectors=vectors, track_ids=track_ids, true_roots=true_roots, true_modes=true_modes)
    print(f"GiantSteps: cached {len(track_ids)} full feature vectors in {time.time()-start:.1f}s")


def cache_wtc():
    files = sorted(glob.glob(os.path.join(WTC_AUDIO_DIR, "*.mp3")))
    vectors, track_ids, true_roots, true_modes = [], [], [], []
    start = time.time()
    for i, path in enumerate(files):
        fname = os.path.basename(path)
        m = re.search(r"in ([A-G](?:-sharp|-flat)?) (major|minor)", fname)
        if not m:
            continue
        root, mode = NOTE_MAP.get(m.group(1), m.group(1)), m.group(2)
        try:
            feats = extract_features(path)
        except Exception as e:
            print(f"  [SKIP] {fname}: {e}")
            continue
        vectors.append(feats["vector"])
        track_ids.append(fname)
        true_roots.append(root)
        true_modes.append(mode)
        print(f"  wtc [{i+1}/{len(files)}]  ({(time.time()-start)/(i+1):.2f}s/track)")
    vectors = np.stack(vectors)
    np.savez(os.path.join(OUT_DIR, "wtc_full_features.npz"),
              vectors=vectors, track_ids=track_ids, true_roots=true_roots, true_modes=true_modes)
    print(f"WTC: cached {len(track_ids)} full feature vectors in {time.time()-start:.1f}s")


if __name__ == "__main__":
    cache_wtc()
    cache_giantsteps()
