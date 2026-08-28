"""
compare_profiles.py

Stage 1 of the genre-adaptive key-detection idea: does a genre-specific
key profile beat the generic Krumhansl-Schmuckler one, on real EDM data,
once genre is already known (every GiantSteps track IS EDM)?

Compares three profile sets x two similarity measures against the same
cached chroma vectors (from cache_giantsteps_chroma.py) -- no re-extraction,
no changes to features.py/search.py's actual configured behavior. Also
tries a simple ensemble (majority vote across profile sets).

Profile sources:
  - Krumhansl-Schmuckler (1982): perceptual listener-experiment profiles,
    what this project currently uses in features.py.
  - Essentia 'edma': corpus-derived from real EDM audio (automatic
    extraction). Per Essentia's own source comments: "normally perform
    better than Sha'ath's [KeyFinder's]".
  - Essentia 'edmm': corpus-derived + manually tweaked. Major profile is
    deliberately FLAT (uniform) -- an explicit design choice given how
    rare major keys are in EDM, trading mode-nuance for overall accuracy.
  Values pulled directly from Essentia's source (src/algorithms/tonal/key.cpp),
  not from memory.
"""
import os
import numpy as np

CACHES = {
    "giantsteps (EDM, n=604)": os.path.join(os.path.dirname(__file__), "..", "data", "real", "giantsteps_chroma_cache.npz"),
    "wtc-book1 (Bach, n=48)": os.path.join(os.path.dirname(__file__), "..", "data", "real", "wtc_chroma_cache.npz"),
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

PROFILES = {
    "krumhansl": {
        "major": np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
        "minor": np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
    },
    "essentia_edma": {
        "major": np.array([1.00, 0.29, 0.50, 0.40, 0.60, 0.56, 0.32, 0.80, 0.31, 0.45, 0.42, 0.39]),
        "minor": np.array([1.00, 0.31, 0.44, 0.58, 0.33, 0.49, 0.29, 0.78, 0.43, 0.29, 0.53, 0.32]),
    },
    "essentia_edmm": {
        "major": np.array([0.083] * 12),  # deliberately flat -- see module docstring
        "minor": np.array([0.17235348, 0.04, 0.0761009, 0.12, 0.05621498, 0.08527853,
                            0.0497915, 0.13451001, 0.07458916, 0.05003023, 0.09187879, 0.05545106]),
    },
}


def score_all(chroma_vec, major_profile, minor_profile, measure):
    best_score, best_root, best_mode = -np.inf, None, None
    for i in range(12):
        maj = np.roll(major_profile, i)
        minr = np.roll(minor_profile, i)
        if measure == "correlation":
            maj_score = np.corrcoef(chroma_vec, maj)[0, 1]
            min_score = np.corrcoef(chroma_vec, minr)[0, 1]
        else:  # cosine
            maj_score = np.dot(chroma_vec, maj) / (np.linalg.norm(chroma_vec) * np.linalg.norm(maj) + 1e-9)
            min_score = np.dot(chroma_vec, minr) / (np.linalg.norm(chroma_vec) * np.linalg.norm(minr) + 1e-9)
        if maj_score > best_score:
            best_score, best_root, best_mode = maj_score, NOTE_NAMES[i], "major"
        if min_score > best_score:
            best_score, best_root, best_mode = min_score, NOTE_NAMES[i], "minor"
    return best_root, best_mode


def evaluate(chroma, true_roots, true_modes, profile_name, measure):
    profile = PROFILES[profile_name]
    n_exact, n_root_only = 0, 0
    predictions = []
    for i in range(len(chroma)):
        root, mode = score_all(chroma[i], profile["major"], profile["minor"], measure)
        predictions.append((root, mode))
        if root == true_roots[i] and mode == true_modes[i]:
            n_exact += 1
        if root == true_roots[i]:
            n_root_only += 1
    n = len(chroma)
    return n_exact / n * 100, n_root_only / n * 100, predictions


def evaluate_ensemble_vote(chroma, true_roots, true_modes, profile_names, measure):
    """Majority vote across profile sets; ties broken by krumhansl."""
    all_preds = [evaluate(chroma, true_roots, true_modes, p, measure)[2] for p in profile_names]
    n_exact, n_root_only = 0, 0
    n = len(chroma)
    for i in range(n):
        votes = [all_preds[p_idx][i] for p_idx in range(len(profile_names))]
        # majority vote on (root, mode) pair; fallback to first profile (krumhansl) on no majority
        from collections import Counter
        counts = Counter(votes)
        winner, count = counts.most_common(1)[0]
        if count == 1:  # no agreement at all -- fall back to krumhansl's own answer
            winner = votes[0]
        root, mode = winner
        if root == true_roots[i] and mode == true_modes[i]:
            n_exact += 1
        if root == true_roots[i]:
            n_root_only += 1
    return n_exact / n * 100, n_root_only / n * 100


def main():
    for cache_label, cache_path in CACHES.items():
        if not os.path.exists(cache_path):
            print(f"[SKIP] {cache_label}: no cache found at {cache_path}\n")
            continue
        data = np.load(cache_path, allow_pickle=True)
        chroma = data["chroma"]
        true_roots = list(data["true_roots"])
        true_modes = list(data["true_modes"])
        n = len(chroma)

        print(f"=== {cache_label} ===\n")
        print(f"{'Profile':20s} {'Measure':12s} {'Exact match':>12s} {'Root-only':>12s}")
        print("-" * 60)
        for profile_name in PROFILES:
            for measure in ["correlation", "cosine"]:
                exact_pct, root_pct, _ = evaluate(chroma, true_roots, true_modes, profile_name, measure)
                print(f"{profile_name:20s} {measure:12s} {exact_pct:11.1f}% {root_pct:11.1f}%")

        print("-" * 60)
        ens_exact, ens_root = evaluate_ensemble_vote(
            chroma, true_roots, true_modes,
            ["krumhansl", "essentia_edma", "essentia_edmm"], "correlation"
        )
        print(f"{'ensemble (3-way vote)':20s} {'correlation':12s} {ens_exact:11.1f}% {ens_root:11.1f}%")
        print()


if __name__ == "__main__":
    main()
