"""
train_genre_classifier.py

Trains a binary genre classifier (electronic vs. not) on the 300 real
FMA tracks already extracted, using the exact same 144-dim feature
vector features.py already computes (chroma + mel-spectrogram + tempo +
spectral shape) -- no new DSP, just a classifier on top of existing
infrastructure. Genre labels come from FMA's own metadata (genres.csv's
top_level hierarchy: genre_id 15 = Electronic; any track whose genre
tree includes it is labeled electronic).

Evaluated with cross-validation on FMA itself (honest, since FMA is
never used as a key-detection test set) -- NOT on GiantSteps/WTC, which
are reserved for testing whether the classifier's PREDICTIONS (not
ground truth) improve downstream key-detection routing.
"""
import os
import ast
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

RAW_TRACKS_PATH = r"C:\Users\hii\Documents\GitHub\sonicrag\data\raw_datasets\fma_metadata_extracted\fma_metadata\raw_tracks.csv"
GENRES_PATH = r"C:\Users\hii\Documents\GitHub\sonicrag\data\raw_datasets\fma_metadata_extracted\fma_metadata\genres.csv"
FMA_FEATURES_PATH = r"C:\Users\hii\Documents\GitHub\sonicrag\data\real\features.npz"
FMA_METADATA_PATH = r"C:\Users\hii\Documents\GitHub\sonicrag\data\real\metadata.json"
MODEL_OUT_PATH = r"C:\Users\hii\Documents\GitHub\sonicrag\data\real\genre_classifier.joblib"

ELECTRONIC_TOP_LEVEL = 15


def build_genre_labels():
    genres = pd.read_csv(GENRES_PATH)
    genre_id_to_top = dict(zip(genres["genre_id"], genres["top_level"]))

    tracks = pd.read_csv(RAW_TRACKS_PATH, low_memory=False)
    tracks["track_id"] = tracks["track_id"].astype(int)

    def is_electronic(genre_list_str):
        try:
            genre_list = ast.literal_eval(genre_list_str)
        except (ValueError, SyntaxError):
            return None
        if not genre_list:
            return None
        for g in genre_list:
            top = genre_id_to_top.get(int(g["genre_id"]))
            if top == ELECTRONIC_TOP_LEVEL:
                return True
        return False

    tracks["is_electronic"] = tracks["track_genres"].apply(is_electronic)
    return dict(zip(tracks["track_id"], tracks["is_electronic"]))


def main():
    label_map = build_genre_labels()

    data = np.load(FMA_FEATURES_PATH)
    vectors = data["vectors"]
    with open(FMA_METADATA_PATH) as f:
        metadata = json.load(f)

    X, y, track_files = [], [], []
    for i, entry in enumerate(metadata):
        track_id = int(entry["file"].split("/")[-1].replace(".mp3", ""))
        label = label_map.get(track_id)
        if label is None:
            continue
        X.append(vectors[i])
        y.append(1 if label else 0)
        track_files.append(entry["file"])

    X = np.stack(X)
    y = np.array(y)
    print(f"Labeled samples: {len(y)} total, {y.sum()} electronic ({100*y.mean():.1f}%), {len(y)-y.sum()} non-electronic")

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    print(f"\n5-fold cross-validated accuracy on FMA (held-out folds): {scores.mean()*100:.1f}% (+/- {scores.std()*100:.1f}%)")
    print(f"Per-fold: {[f'{s*100:.1f}%' for s in scores]}")

    baseline = max(y.mean(), 1 - y.mean()) * 100
    print(f"(Baseline -- always predicting the majority class: {baseline:.1f}%)")

    # Fit on all labeled FMA data for the final model, used downstream on GiantSteps/WTC
    clf.fit(X, y)
    joblib.dump(clf, MODEL_OUT_PATH)
    print(f"\nFinal model (trained on all {len(y)} labeled FMA tracks) saved -> {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
