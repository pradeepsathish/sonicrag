"""
generate_dataset.py

Synthesizes a small ground-truth audio dataset using additive synthesis.
No external downloads required — this is what lets the rest of the
pipeline actually run in a network-restricted environment. On your own
machine, swap this step out for real audio (FMA, MTG-Jamendo, or your
own library) — everything downstream (features, index, eval) works
the same way on real files.

Each clip has KNOWN ground truth (key, mode, tempo, brightness) because
we generated it — that's what lets eval.py compute honest numbers
instead of made-up ones.
"""
import numpy as np
import soundfile as sf
import json
import os
import random

SR = 22050
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "audio")
META_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "metadata.json")

NOTE_FREQS = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13, "E": 329.63,
    "F": 349.23, "F#": 369.99, "G": 392.00, "G#": 415.30, "A": 440.00,
    "A#": 466.16, "B": 493.88,
}

MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]

TIMBRES = {
    # (harmonic amplitude decay, brightness) -> rough instrument character
    "nylon_guitar": dict(n_harmonics=6, decay=0.55, attack=0.01, pluck=True),
    "warm_rhodes":  dict(n_harmonics=4, decay=0.7, attack=0.02, pluck=False),
    "bright_synth": dict(n_harmonics=10, decay=0.35, attack=0.005, pluck=True),
    "sub_bass":     dict(n_harmonics=2, decay=0.9, attack=0.01, pluck=False),
}


def semitone_freq(root_freq, steps):
    return root_freq * (2 ** (steps / 12.0))


def synth_note(freq, duration, timbre, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    n_harm = timbre["n_harmonics"]
    decay = timbre["decay"]
    signal = np.zeros_like(t)
    for h in range(1, n_harm + 1):
        amp = decay ** (h - 1) / h
        signal += amp * np.sin(2 * np.pi * freq * h * t)
    # envelope
    attack_n = int(timbre["attack"] * sr)
    env = np.ones_like(t)
    if timbre["pluck"]:
        env = np.exp(-3.0 * t / duration)
    if attack_n > 0:
        env[:attack_n] *= np.linspace(0, 1, attack_n)
    return signal * env


def synth_progression(root_note, mode, bpm, timbre_name, duration=8.0, seed=0):
    rng = random.Random(seed)
    timbre = TIMBRES[timbre_name]
    steps = MAJOR_STEPS if mode == "major" else MINOR_STEPS
    root_freq = NOTE_FREQS[root_note] / 2  # drop an octave, more musical range
    beat_dur = 60.0 / bpm
    n_beats = int(duration / beat_dur)

    audio = np.zeros(int(duration * SR) + SR)
    t_cursor = 0.0
    # simple chord-tone walk over the scale to keep it musically coherent
    degree_sequence = [0, 2, 4, 2, 0, 4, 3, 4]
    for i in range(n_beats):
        degree = degree_sequence[i % len(degree_sequence)]
        note_freq = semitone_freq(root_freq, steps[degree % len(steps)])
        note_dur = beat_dur * rng.choice([1.0, 1.0, 2.0])
        note = synth_note(note_freq, note_dur, timbre)
        start = int(t_cursor * SR)
        end = start + len(note)
        if end > len(audio):
            note = note[: len(audio) - start]
            end = len(audio)
        audio[start:end] += note
        t_cursor += note_dur
        if t_cursor >= duration:
            break

    audio = audio[: int(duration * SR)]
    peak = np.max(np.abs(audio)) + 1e-9
    audio = audio / peak * 0.9
    return audio.astype(np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = random.Random(42)

    roots = list(NOTE_FREQS.keys())
    configs = []
    clip_id = 0

    # Build a deliberately diverse, labeled set spanning key/mode/tempo/timbre
    for mode in ["major", "minor"]:
        for timbre_name in TIMBRES:
            for _ in range(6):
                root = rng.choice(roots)
                bpm = rng.choice([72, 84, 96, 108, 120, 132, 140])
                configs.append((root, mode, bpm, timbre_name))

    metadata = []
    for root, mode, bpm, timbre_name in configs:
        fname = f"clip_{clip_id:03d}_{root.replace('#','s')}_{mode}_{bpm}bpm_{timbre_name}.wav"
        path = os.path.join(OUT_DIR, fname)
        audio = synth_progression(root, mode, bpm, timbre_name, duration=8.0, seed=clip_id)
        sf.write(path, audio, SR)
        metadata.append({
            "id": clip_id,
            "file": fname,
            "root": root,
            "mode": mode,
            "bpm": bpm,
            "timbre": timbre_name,
        })
        clip_id += 1

    with open(META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Generated {clip_id} clips -> {OUT_DIR}")
    print(f"Ground-truth metadata -> {META_PATH}")


if __name__ == "__main__":
    main()
