"""
clap_swap.py

RUN THIS ON YOUR OWN MACHINE, NOT IN THE SANDBOX — it needs
huggingface.co access to pull model weights, which this sandbox
cannot reach.

This is the drop-in replacement for the rule-based encode_query_text()
in search.py. Everything else in the pipeline (features.py, the FAISS/
NearestNeighbors index, eval.py) is unchanged — only the text encoder
and the audio encoder for the *learned* embedding get swapped from
hand-built rules to a real CLAP dual-encoder.

    pip install transformers torch

Usage:
    python clap_swap.py
"""
import torch
import numpy as np
from transformers import ClapModel, ClapProcessor

MODEL_ID = "laion/clap-htsat-unfused"


class ClapEncoder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ClapModel.from_pretrained(MODEL_ID).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(MODEL_ID)

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt").to(self.device)
        emb = self.model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]  # 512-dim unit vector

    @torch.no_grad()
    def encode_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        inputs = self.processor(audios=[waveform], sampling_rate=sample_rate,
                                 return_tensors="pt").to(self.device)
        emb = self.model.get_audio_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy()[0]  # 512-dim unit vector


def build_real_index(audio_paths, out_path="clap_index.npz"):
    """
    Replaces the rule-based feature vectors in data/features.npz with
    real CLAP audio embeddings. Once you have this, cosine similarity
    search against encode_text() output is genuine cross-modal retrieval
    -- the actual thing the original SonicRAG dossier claimed to have.
    """
    import librosa
    encoder = ClapEncoder()
    vectors = []
    for path in audio_paths:
        y, sr = librosa.load(path, sr=48000, mono=True)  # CLAP expects 48kHz
        vectors.append(encoder.encode_audio(y, sr))
    vectors = np.stack(vectors)
    np.savez(out_path, vectors=vectors, paths=audio_paths)
    print(f"Real CLAP embeddings for {len(audio_paths)} tracks -> {out_path}")
    return vectors


if __name__ == "__main__":
    encoder = ClapEncoder()
    text_vec = encoder.encode_text("melancholic classical guitar in D minor")
    print("Text embedding shape:", text_vec.shape)
    print("First 8 dims:", text_vec[:8])
    print("\nNext step: point build_real_index() at data/audio/*.wav to get "
          "real audio embeddings in the same space, then reuse the "
          "NearestNeighbors search from search.py directly on these vectors.")
