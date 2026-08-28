# ALAI — Working Prototype

> "ALAI" (Tamil for "wave") is the public/brand name; "SonicRAG" is retained as the internal name for the retrieval architecture specifically.

A real, runnable audio retrieval pipeline: text query → parsed intent →
weighted similarity search → ranked results, evaluated against ground
truth. Built to replace the original SonicRAG dossier's unmeasured
benchmark table with numbers that actually came from running code —
and, as of this update, from running that code against real, human-verified
audio, not just synthesized test data.

## Live interface

Open `index.html` directly, or serve the repo root (`python3 -m http.server`
from the project folder, or GitHub Pages) for audio playback to work. It
runs the exact same weighted-scoring logic as `src/search.py`, ported to
JS, against the exact same real extracted features — not a mockup. **The
live demo currently still plays the 48 synthesized clips** described below;
the real-audio work has been done offline (see "Real-data results") and
not yet wired into this interface.

## What's real vs. simulated (read this first)

| Component | Status |
|---|---|
| Audio dataset (live demo) | **Synthesized.** 48 clips, additive synthesis, known ground-truth key/tempo/timbre. Used by `index.html` only. |
| Audio dataset (offline pipeline) | **Real.** [FMA-small](https://github.com/mdeff/fma) (7,994 of 8,000 tracks, CC-licensed) fully feature-extracted; [GiantSteps Key](https://github.com/GiantSteps/giantsteps-key-dataset) (604 real, human-verified EDM tracks) used as the key-detection benchmark; Bach's *Well-Tempered Clavier* Book 1 (48 tracks, public domain) used as a second, genre-opposite benchmark. |
| DSP feature extraction (chroma, tempo, spectral centroid, mel-spectrogram) | **Real.** `librosa`, actually computed, actually run on actually-real audio. |
| Key detection | **Real, and benchmarked against published academic baselines**, not just an internal number — see below. |
| Genre-adaptive key detection | **Real, working, honestly validated.** A classifier (trained only on FMA) detects whether a track is electronic/EDM and routes to a genre-matched key profile, tested end-to-end on two held-out real datasets it never saw during training. |
| Text query understanding | **Rule-based**, not CLAP. Parses key names, tempo, and mood adjectives via regex + hand-built mappings. Honestly labeled as a placeholder in the code. |
| Retrieval / ranking | **Real.** Weighted multi-field scoring (chroma-key match + tempo closeness + brightness closeness) over real extracted features. |
| Evaluation (NDCG@5, MRR) | **Real, but still only run on the synthesized set** — real-data retrieval evaluation (as opposed to key-detection evaluation) hasn't been done yet. |
| CLAP integration | **Not run here** — `clap_swap.py` is the exact drop-in code to run on a machine with Hugging Face access. |
| Kafka/Ray/10M-vector production serving claims from the original dossier | **Removed.** Not something a laptop benchmark can honestly support. |

## Real-data results

**Key detection accuracy, benchmarked against real, human-verified ground truth:**

```
GiantSteps Key dataset (604 real EDM tracks, Beatport):
  Fixed Krumhansl-Schmuckler profile:           44.5% exact match
  Genre-adaptive (classifier-routed):           52.6% exact match
  Oracle (perfect genre routing) ceiling:       53.8% exact match
  -> captures 87% of the available routing benefit

Bach Well-Tempered Clavier Book 1 (48 tracks, public domain, non-EDM control):
  Fixed Krumhansl-Schmuckler profile:           79.2% exact match
  Genre-adaptive (classifier-routed):           79.2% exact match  (matches oracle exactly)
```

For context: on the GiantSteps benchmark specifically, this system's plain
Krumhansl-Schmuckler baseline (44.5%) already beats the two academic
baselines reported in the dataset's own paper (Knees et al., ISMIR 2015) —
QM-Key (39.4%) and Essentia's default key extractor (30.5%) — and the
genre-adaptive version (52.6%) closes most of the remaining gap to
KeyFinder (45.4%) and approaches the proprietary commercial tools
(Mixed-In-Key 67.2%, Rekordbox 71.9%), which have the advantage of
undisclosed, presumably genre-tuned internals.

**How the genre-adaptive system was built and validated**, in short: a
binary electronic/non-electronic classifier was trained *only* on FMA
(using FMA's own genre metadata for labels) and then evaluated on
GiantSteps and the Bach dataset — both held out completely from training.
This strict three-corpus separation means the accuracy numbers above
reflect real generalization, not overfitting to the test set.

**Retrieval-quality evaluation (NDCG@5, MRR) is still only run on the
synthesized 48-clip set** — see below — and hasn't yet been extended to
real audio.

```
Mean NDCG@5 across 24 hand-built eval queries (synthesized set): 0.927
Mean MRR across 24 hand-built eval queries (synthesized set):    0.879
```

Reproduce the real-data results with:
```
cd src
python3 extract_real.py              # extract features for all real FMA tracks
python3 eval_giantsteps.py           # key detection accuracy vs GiantSteps ground truth
python3 cache_full_features.py       # full feature caches for GiantSteps + WTC
python3 train_genre_classifier.py    # train the genre classifier on FMA only
python3 evaluate_routed.py           # end-to-end genre-adaptive accuracy on held-out data
```

Reproduce the original synthesized-set numbers with:
```
cd src
python3 generate_dataset.py   # synthesize 48 labeled clips
python3 features.py           # DSP feature extraction + key detection accuracy
python3 search.py             # demo text queries against the index
python3 eval.py               # full eval harness, prints NDCG@5 / MRR
```

## What actually broke during development (kept in, on purpose)

- **Chroma-argmax key detection was unreliable.** Root cause: raw chroma
  picks the most-*played* pitch class over a clip, not the tonic — melodies
  frequently lean on the 5th degree almost as often as the root. Fixed by
  switching to Krumhansl-Schmuckler key-profile correlation (correlating
  observed chroma against all 24 rotated major/minor profiles), the
  standard MIR approach for exactly this reason.
- **A single fixed key profile is not genre-general.** The same
  Krumhansl-Schmuckler profile that's strongest on classical material
  (79.2% on Bach) is measurably weaker on EDM (44.5% on GiantSteps) than
  a genre-specific profile (53.8%) — and the reverse is true just as
  sharply (a profile tuned for EDM drops to 43.8% on Bach). Neither fixed
  choice is right for both; this motivated the genre-adaptive system above.
- **Naive text parsing matched single letters as notes inside random
  words** (e.g. "chords" got parsed as key C). Fixed by requiring the
  note to appear adjacent to "major/minor" or after "in "/"key of ".
- **Zero-padding the CLAP-shaped 144-dim vector for queries let the
  128 zeroed mel dimensions dominate cosine similarity**, drowning out
  the 3 real signals (chroma/tempo/brightness) the query actually had.
  Fixed by scoring per-field and combining with explicit weights.
- **The chroma matcher was initially built from ground-truth labels**,
  not the actual extracted audio chroma. Fixed to match against each
  track's *detected* key instead.

## Honest failure case worth knowing about

On the synthesized set, the query `"C# minor 108 bpm"` doesn't retrieve
the true C# minor 108bpm track at rank 1 — because that exact track is
one the key detector mis-labeled. The error cascades: a detection mistake
in feature extraction shows up as a ranking mistake in search. That's a
real, traceable failure mode, not noise.

## Next steps

1. **Wire real audio into the live demo** — currently the offline pipeline
   uses real data, but `index.html` still plays the synthesized clips.
2. **Extend retrieval-quality evaluation (NDCG@5, MRR) to real audio** —
   currently only benchmarked on the synthesized set.
3. **Scale the genre-adaptive idea beyond binary electronic/non-electronic**
   — more genres, more profile options, as more labeled real data becomes
   available.
4. Run `clap_swap.py` on a machine with Hugging Face access — replaces
   the rule-based query parser with real cross-modal CLAP embeddings.
5. If pursuing a "production scale" story, that's a separate infra
   conversation (sharding, latency budgets under real load) — kept
   separate from retrieval-quality results.
