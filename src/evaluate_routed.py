"""
evaluate_routed.py

The decisive end-to-end test: apply the FMA-trained genre classifier to
GiantSteps and WTC (both real, held-out -- the classifier has never seen
either), route each track to krumhansl (if predicted not-electronic) or
essentia_edmm (if predicted electronic), and measure real key-detection
accuracy using PREDICTED routing, not oracle genre labels.

Compared against:
  - always krumhansl (the original baseline)
  - always essentia_edmm
  - oracle routing (using true genre -- the theoretical ceiling)
"""
import os
import numpy as np
import joblib

from compare_profiles import PROFILES, score_all

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real")
MODEL_PATH = os.path.join(DATA_DIR, "genre_classifier.joblib")

DATASETS = {
    "giantsteps (EDM, true genre=electronic)": ("giantsteps_full_features.npz", True),
    "wtc-book1 (Bach, true genre=NOT electronic)": ("wtc_full_features.npz", False),
}


def key_accuracy(chroma_vectors, true_roots, true_modes, profile_choices):
    """profile_choices: list of 'krumhansl' or 'essentia_edmm' per track."""
    n_exact, n_root = 0, 0
    for i in range(len(chroma_vectors)):
        profile = PROFILES[profile_choices[i]]
        root, mode = score_all(chroma_vectors[i], profile["major"], profile["minor"], "correlation")
        if root == true_roots[i] and mode == true_modes[i]:
            n_exact += 1
        if root == true_roots[i]:
            n_root += 1
    n = len(chroma_vectors)
    return n_exact / n * 100, n_root / n * 100


def main():
    clf = joblib.load(MODEL_PATH)

    for label, (fname, true_is_electronic) in DATASETS.items():
        data = np.load(os.path.join(DATA_DIR, fname), allow_pickle=True)
        full_vectors = data["vectors"]       # 144-dim, for the classifier
        chroma = full_vectors[:, :12]        # first 12 dims are chroma, for key scoring
        true_roots = list(data["true_roots"])
        true_modes = list(data["true_modes"])
        n = len(chroma)

        pred_electronic = clf.predict(full_vectors).astype(bool)
        pred_pct = 100 * pred_electronic.mean()

        print(f"=== {label}, n={n} ===")
        print(f"Classifier predicted electronic for {pred_electronic.sum()}/{n} tracks ({pred_pct:.1f}%)")

        # routed: use the classifier's own prediction
        routed_choices = ["essentia_edmm" if p else "krumhansl" for p in pred_electronic]
        routed_exact, routed_root = key_accuracy(chroma, true_roots, true_modes, routed_choices)

        # oracle: use true genre (ceiling)
        oracle_choices = ["essentia_edmm" if true_is_electronic else "krumhansl"] * n
        oracle_exact, oracle_root = key_accuracy(chroma, true_roots, true_modes, oracle_choices)

        # fixed baselines
        always_ks = ["krumhansl"] * n
        always_edmm = ["essentia_edmm"] * n
        ks_exact, ks_root = key_accuracy(chroma, true_roots, true_modes, always_ks)
        edmm_exact, edmm_root = key_accuracy(chroma, true_roots, true_modes, always_edmm)

        print(f"{'Strategy':30s} {'Exact match':>12s} {'Root-only':>12s}")
        print("-" * 56)
        print(f"{'always krumhansl':30s} {ks_exact:11.1f}% {ks_root:11.1f}%")
        print(f"{'always essentia_edmm':30s} {edmm_exact:11.1f}% {edmm_root:11.1f}%")
        print(f"{'oracle routing (true genre)':30s} {oracle_exact:11.1f}% {oracle_root:11.1f}%")
        print(f"{'CLASSIFIER-ROUTED (real)':30s} {routed_exact:11.1f}% {routed_root:11.1f}%")
        print()


if __name__ == "__main__":
    main()
