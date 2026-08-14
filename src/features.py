"""
features.py

Real DSP feature extraction with librosa. No shortcuts here — this is
the part of the pipeline that's genuinely "just" signal processing,
and it runs the same on synthetic or real audio.

For each track we extract:
  - 12-bin chroma vector (key/harmonic content) -> HPCP-equivalent
  - Tempo estimate (BPM) via beat tracking
  - Spectral centroid (brightness/warmth proxy)
  - Spectral bandwidth + rolloff (timbral shape)
  - Mel-spectrogram summary (mean per band, log-compressed)

These get concatenated into one feature vector per track and L2-normalized,
which is what gets indexed for similarity search.
"""
import numpy as np
import librosa
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
AUDIO_DIR = os.path.join(DATA_DIR, "audio")
META_PATH = os.path.join(DATA_DIR, "metadata.json")
FEATURES_PATH = os.path.join(DATA_DIR, "features.npz")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles — perceptual weightings of how strongly
# each scale degree implies a tonic, derived from music-cognition research.
# Raw chroma argmax picks whichever pitch class is loudest/most-played,
# which is often NOT the tonic (e.g. a melody leaning on the 5th degree).
# Correlating the observed chroma against these profiles for all 24
# major/minor rotations is the standard fix, and is genuinely more correct.
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_key_krumhansl(chroma_mean):
    """Correlate observed chroma against all 24 rotated major/minor profiles."""
    best_score, best_root, best_mode = -np.inf, None, None
    for i in range(12):
        maj_profile = np.roll(KS_MAJOR, i)
        min_profile = np.roll(KS_MINOR, i)
        maj_corr = np.corrcoef(chroma_mean, maj_profile)[0, 1]
        min_corr = np.corrcoef(chroma_mean, min_profile)[0, 1]
        if maj_corr > best_score:
            best_score, best_root, best_mode = maj_corr, NOTE_NAMES[i], "major"
        if min_corr > best_score:
            best_score, best_root, best_mode = min_corr, NOTE_NAMES[i], "minor"
    return best_root, best_mode


def extract_features(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)

    # --- Harmonic Pitch-Class Profile (chroma) ---
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512)
    chroma_mean = chroma.mean(axis=1)
    chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-9)
    detected_root, detected_mode = detect_key_krumhansl(chroma_mean)

    # --- Tempo (autocorrelation-based beat tracking) ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # --- Spectral shape (brightness / warmth proxies) ---
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()

    # --- Mel-spectrogram summary (128 bands -> mean log energy per band) ---
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_summary = mel_db.mean(axis=1)  # 128-dim

    # Assemble full feature vector: chroma(12) + mel_summary(128) + [tempo, centroid, bandwidth, rolloff]
    scalar_feats = np.array([
        tempo / 200.0,          # rough normalization
        centroid / (sr / 2),
        bandwidth / (sr / 2),
        rolloff / (sr / 2),
    ])
    vec = np.concatenate([chroma_mean, mel_summary / 80.0, scalar_feats])  # 128 db range normalization
    vec = vec / (np.linalg.norm(vec) + 1e-9)

    return {
        "vector": vec.astype(np.float32),
        "detected_root": detected_root,
        "detected_mode": detected_mode,
        "tempo": tempo,
        "centroid": float(centroid),
        "chroma": chroma_mean,
    }


def main():
    with open(META_PATH) as f:
        metadata = json.load(f)

    vectors = []
    chroma_vectors = []
    enriched_meta = []
    for entry in metadata:
        path = os.path.join(AUDIO_DIR, entry["file"])
        feats = extract_features(path)
        vectors.append(feats["vector"])
        chroma_vectors.append(feats["chroma"])
        entry = dict(entry)
        entry["detected_root"] = feats["detected_root"]
        entry["detected_mode"] = feats["detected_mode"]
        entry["detected_tempo"] = round(feats["tempo"], 1)
        entry["spectral_centroid"] = round(feats["centroid"], 1)
        enriched_meta.append(entry)
        correct = "OK " if (feats["detected_root"] == entry["root"]
                             and feats["detected_mode"] == entry["mode"]) else "MISS"
        print(f"[{correct}] {entry['file']:45s} true={entry['root']:2s} {entry['mode']:5s} "
              f"detected={feats['detected_root']:2s} {feats['detected_mode']:5s} "
              f"true_bpm={entry['bpm']:3d} detected_bpm={feats['tempo']:.1f}")

    vectors = np.stack(vectors)
    chroma_vectors = np.stack(chroma_vectors)
    np.savez(FEATURES_PATH, vectors=vectors, chroma=chroma_vectors)
    with open(os.path.join(DATA_DIR, "metadata_enriched.json"), "w") as f:
        json.dump(enriched_meta, f, indent=2)

    n_correct_key = sum(
        1 for e in enriched_meta
        if e["detected_root"] == e["root"] and e["detected_mode"] == e["mode"]
    )
    n_correct_root_only = sum(1 for e in enriched_meta if e["detected_root"] == e["root"])
    print(f"\nFeature extraction complete: {len(enriched_meta)} tracks")
    print(f"Key detection (root+mode exact) accuracy: "
          f"{n_correct_key}/{len(enriched_meta)} = {100*n_correct_key/len(enriched_meta):.1f}%")
    print(f"Root-only accuracy (ignoring major/minor): "
          f"{n_correct_root_only}/{len(enriched_meta)} = {100*n_correct_root_only/len(enriched_meta):.1f}%")
    print(f"Feature vectors saved -> {FEATURES_PATH}")


if __name__ == "__main__":
    main()
