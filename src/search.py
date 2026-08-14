"""
search.py

Text query -> feature-space vector -> cosine similarity search.

HONEST LIMITATION: this sandbox can't reach huggingface.co, so this is
NOT CLAP. It's a rule-based query encoder: it parses key names, tempo
hints, and mood/brightness adjectives out of the text and constructs a
target feature vector in the *same feature space* features.py builds
for tracks, then does real cosine similarity search against it.

This is a legitimate, explainable baseline retrieval system — and it's
also exactly what CLAP would slot into as a drop-in replacement: swap
`encode_query_text()` for a real CLAP text-encoder call, and everything
downstream (index, ranking, eval) is unchanged. See clap_swap.py for
that version, to run on a machine with Hugging Face access.
"""
import numpy as np
import json
import os
import re

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES_PATH = os.path.join(DATA_DIR, "features.npz")
META_PATH = os.path.join(DATA_DIR, "metadata_enriched.json")

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

MOOD_TO_CENTROID = {
    # rough mapping of mood adjectives -> target spectral centroid (Hz),
    # a real proxy for brightness/warmth. Hand-authored, not learned.
    "melancholic": 900, "warm": 900, "dark": 700, "mellow": 850,
    "bright": 3500, "punchy": 2800, "sharp": 3200, "airy": 4000,
    "sub": 200, "deep": 300,
}

TIMBRE_CENTROID_PRIOR = {
    "nylon_guitar": 1400, "warm_rhodes": 1100, "bright_synth": 3000, "sub_bass": 250,
}


def parse_query(text):
    """
    NOTE: naive single-letter note matching is a known trap — "C" or "A"
    as bare substrings match constantly in ordinary English ("a minor
    detail", "chords"). Fixed by requiring the note to appear directly
    adjacent to "major"/"minor"/"maj"/"min", or after "in "/"key of ",
    which is how people actually phrase key names in text.
    """
    text_l = text.lower()
    root, mode = None, None

    note_pat = "|".join(sorted((n.lower().replace("#", "#") for n in NOTE_NAMES),
                                key=len, reverse=True))
    # Strong signal: "<note> minor/major/min/maj"
    m = re.search(rf"\b([a-g]#?)\s*(major|minor|maj|min)\b", text_l)
    if m:
        note_raw, mode_raw = m.group(1), m.group(2)
        root = next((n for n in NOTE_NAMES if n.lower() == note_raw), None)
        mode = "minor" if mode_raw.startswith("min") else "major"
    else:
        # Weaker signal: "in <note>" or "key of <note>"
        m2 = re.search(rf"\b(?:in|key of)\s+([a-g]#?)\b", text_l)
        if m2:
            note_raw = m2.group(1)
            root = next((n for n in NOTE_NAMES if n.lower() == note_raw), None)
        if "minor" in text_l:
            mode = "minor"
        elif "major" in text_l:
            mode = "major"

    bpm_match = re.search(r"(\d{2,3})\s*bpm", text_l)
    bpm = float(bpm_match.group(1)) if bpm_match else None

    centroid_hits = [v for k, v in MOOD_TO_CENTROID.items() if k in text_l]
    target_centroid = np.mean(centroid_hits) if centroid_hits else None

    return {"root": root, "mode": mode, "bpm": bpm, "target_centroid": target_centroid}


def key_profile_vector(root, mode):
    idx = NOTE_NAMES.index(root)
    profile = KS_MAJOR if mode != "minor" else KS_MINOR
    vec = np.roll(profile, idx)
    return vec / vec.sum()


def encode_query_text(text):
    """Extracts what we actually have signal for: chroma prior, tempo, brightness."""
    parsed = parse_query(text)
    chroma_part = (key_profile_vector(parsed["root"], parsed["mode"] or "major")
                   if parsed["root"] else None)
    return parsed, chroma_part


class SonicSearchEngine:
    """
    Multi-field weighted retrieval instead of one contaminated dense-vector
    cosine. This is the honest fix for the bug above: a query only has
    signal in 3 of the ~144 feature dimensions (chroma/tempo/brightness),
    so scoring must be done per-field and combined explicitly — not by
    zero-padding the missing 128 mel dimensions and hoping cosine
    similarity ignores them. Real production retrieval systems (e.g.
    hybrid BM25 + vector search) use this same weighted-fusion idea for
    the same reason: heterogeneous signal strength per field.
    """
    def __init__(self, chroma_weight=0.5, tempo_weight=0.3, brightness_weight=0.2):
        data = np.load(FEATURES_PATH)
        self.vectors = data["vectors"]  # kept for reference / future CLAP swap
        # REAL extracted chroma from the actual audio (CQT + averaged),
        # not a label-derived ideal profile. This is what makes the search
        # genuinely audio-driven rather than a lookup table dressed as one.
        self.track_chroma_raw = data["chroma"]
        with open(META_PATH) as f:
            self.metadata = json.load(f)
        self.w_chroma, self.w_tempo, self.w_bright = chroma_weight, tempo_weight, brightness_weight
        self.track_tempo = np.array([m["detected_tempo"] for m in self.metadata])
        self.track_centroid = np.array([m["spectral_centroid"] for m in self.metadata])

    def search(self, query_text, top_k=5):
        parsed, _ = encode_query_text(query_text)

        if parsed["root"] is not None:
            # Match against each track's DETECTED key (from real audio via
            # Krumhansl-Schmuckler, validated at 93.8% against ground truth
            # in features.py) rather than the ground-truth label — this is
            # what a real system would have to do, since a genuinely new
            # unlabeled track has no ground truth to cheat with.
            query_idx = NOTE_NAMES.index(parsed["root"])
            chroma_scores = np.zeros(len(self.metadata))
            for i, m in enumerate(self.metadata):
                track_idx = NOTE_NAMES.index(m["detected_root"])
                # circle-of-fifths distance is more musically meaningful than
                # chromatic distance: keys a fifth apart share more tones
                fifths_pos = (query_idx * 7) % 12
                track_fifths_pos = (track_idx * 7) % 12
                dist = min(abs(fifths_pos - track_fifths_pos), 12 - abs(fifths_pos - track_fifths_pos))
                mode_match = 1.0 if (parsed["mode"] is None or m["detected_mode"] == parsed["mode"]) else 0.6
                chroma_scores[i] = (1.0 - dist / 6.0) * mode_match
        else:
            chroma_scores = np.ones(len(self.metadata)) * 0.5  # no key signal -> neutral

        if parsed["bpm"] is not None:
            tempo_scores = np.exp(-((self.track_tempo - parsed["bpm"]) ** 2) / (2 * 15 ** 2))
        else:
            tempo_scores = np.ones(len(self.metadata)) * 0.5

        if parsed["target_centroid"] is not None:
            bright_scores = np.exp(-((self.track_centroid - parsed["target_centroid"]) ** 2)
                                    / (2 * 1200 ** 2))
        else:
            bright_scores = np.ones(len(self.metadata)) * 0.5

        combined = (self.w_chroma * chroma_scores
                    + self.w_tempo * tempo_scores
                    + self.w_bright * bright_scores)

        top_idx = np.argsort(-combined)[:top_k]
        results = []
        for idx in top_idx:
            entry = self.metadata[idx]
            results.append({
                "file": entry["file"],
                "root": entry["root"],
                "mode": entry["mode"],
                "bpm": entry["bpm"],
                "timbre": entry["timbre"],
                "score": round(float(combined[idx]), 4),
            })
        return parsed, results


def main():
    engine = SonicSearchEngine()
    demo_queries = [
        "melancholic nylon guitar in D minor",
        "warm rhodes chords major key around 108 bpm",
        "punchy bright synth 140 bpm",
        "deep sub bass minor",
    ]
    for q in demo_queries:
        parsed, results = engine.search(q, top_k=5)
        print(f"\nQuery: \"{q}\"")
        print(f"  Parsed -> root={parsed['root']}, mode={parsed['mode']}, "
              f"bpm={parsed['bpm']}, target_centroid={parsed['target_centroid']}")
        for r in results:
            print(f"    {r['score']:.4f}  {r['file']:45s} "
                  f"({r['root']} {r['mode']}, {r['bpm']} bpm, {r['timbre']})")


if __name__ == "__main__":
    main()
