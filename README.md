# ALAI — Working Prototype

> "ALAI" (Tamil for "wave") is the public/brand name; "SonicRAG" is retained as the internal name for the retrieval architecture specifically.

A real, runnable audio retrieval pipeline: text query → parsed intent →
weighted similarity search → ranked results, evaluated against ground
truth. Built to replace the original SonicRAG dossier's unmeasured
benchmark table with numbers that actually came from running code.

## Live interface

Open `index.html` directly, or serve the repo root (`python3 -m http.server`
from the project folder, or GitHub Pages) for audio playback to work. It
runs the exact same weighted-scoring logic as `src/search.py`, ported to
JS, against the exact same real extracted features — not a mockup.

## What's real vs. simulated (read this first)

| Component | Status |
|---|---|
| Audio dataset | **Synthesized**, not downloaded — this sandbox can't reach external dataset hosts. 48 clips, additive synthesis, with known ground-truth key/tempo/timbre. Swap for FMA/MTG-Jamendo/your own library on a machine with internet access; nothing downstream changes. |
| DSP feature extraction (chroma, tempo, spectral centroid, mel-spectrogram) | **Real.** `librosa`, actually computed, actually run on actually-generated audio. |
| Key detection | **Real.** Krumhansl-Schmuckler profile correlation, not chroma-argmax (see "what broke" below). 93.8% accuracy against ground truth. |
| Text query understanding | **Rule-based**, not CLAP. This sandbox can't reach huggingface.co to pull model weights. Parses key names, tempo, and mood adjectives via regex + hand-built mappings. Honestly labeled as a placeholder in the code. |
| Retrieval / ranking | **Real.** Weighted multi-field scoring (chroma-key match + tempo closeness + brightness closeness) over the real extracted features. |
| Evaluation (NDCG@5, MRR) | **Real.** 24 hand-authored queries against known ground truth, computed on this run — see numbers below. |
| CLAP integration | **Not run here** — `clap_swap.py` is the exact drop-in code to run on your own machine where Hugging Face is reachable. Everything else in the pipeline is unchanged when you swap it in. |
| Kafka/Ray/10M-vector production serving claims from the original dossier | **Removed.** That's an infra project requiring a real cluster and real traffic, not something a laptop benchmark can honestly support. If you want that story for an interview, it needs to actually be built with a platform engineer, or clearly labeled as a "how I'd scale this" design section, never as measured results. |

## What actually broke during development (kept in, on purpose)

- **Chroma-argmax key detection was 16.7% accurate.** Root cause: raw
  chroma picks the most-*played* pitch class over a clip, not the tonic
  — in this dataset the melody leans on the 5th degree almost as often
  as the root. Fixed by switching to Krumhansl-Schmuckler key-profile
  correlation (correlating observed chroma against all 24 rotated
  major/minor profiles), which is the standard MIR approach for exactly
  this reason. Fix took accuracy to 93.8%.
- **Naive text parsing matched single letters as notes inside random
  words** (e.g. "chords" got parsed as key C). Fixed by requiring the
  note to appear adjacent to "major/minor" or after "in "/"key of ".
- **Zero-padding the CLAP-shaped 144-dim vector for queries let the
  128 zeroed mel dimensions dominate cosine similarity**, drowning out
  the 3 real signals (chroma/tempo/brightness) the query actually had.
  Fixed by scoring per-field (chroma match, tempo closeness, brightness
  closeness) and combining with explicit weights, instead of one dense
  cosine over a vector that's mostly missing data.
- **The chroma matcher was initially built from ground-truth labels**,
  not the actual extracted audio chroma — a shortcut that would fail
  the moment it saw an unlabeled track. Fixed to match against each
  track's *detected* key instead.

## Real results (this run)

```
Key detection accuracy (Krumhansl-Schmuckler vs ground truth): 45/48 = 93.8%
Mean NDCG@5 across 24 hand-built eval queries: 0.927
Mean MRR across 24 hand-built eval queries:    0.879
```

Reproduce with:
```
cd src
python3 generate_dataset.py   # synthesize 48 labeled clips
python3 features.py           # real DSP feature extraction + key detection accuracy
python3 search.py             # demo text queries against the real index
python3 eval.py                # full eval harness, prints NDCG@5 / MRR
```

## Honest failure case worth knowing about

Query `"C# minor 108 bpm"` doesn't retrieve the true C# minor 108bpm
track at rank 1 — because that exact track (`clip_007`) is one of the
3 the key detector mis-labeled. The error cascades: a detection mistake
in feature extraction shows up as a ranking mistake in search. That's
a real, traceable failure mode, not noise — and it's a legitimate thing
to walk an interviewer through.

## Next steps to make this genuinely production-relevant

1. Swap in real audio (FMA small, ~8k tracks, permissively licensed)
2. Run `clap_swap.py` on a machine with Hugging Face access — replaces
   the rule-based query parser with real cross-modal CLAP embeddings
3. Re-run `eval.py` against real CLAP embeddings, report whatever
   number actually comes out
4. If pursuing the "production scale" story for an interview, that's
   a separate infra conversation (sharding, latency budgets under real
   load) — don't conflate it with retrieval-quality results again
