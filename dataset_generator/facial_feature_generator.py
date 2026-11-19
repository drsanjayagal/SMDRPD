"""512-dimensional simulated facial-expression embedding generator.

Produces numerical vectors only. No photographs, no facial images, no
pretrained facial-recognition or emotion-recognition model is used. The
vectors are not equivalent to embeddings from any real model and carry no
physiological meaning.
"""
from __future__ import annotations

import numpy as np

from .latent_variable_model import EMOTION_NAMES

FACIAL_COLUMNS: tuple[str, ...] = tuple(f"F{i:03d}" for i in range(1, 513))


class FacialEmbeddingGenerator:
    """Block-structured linear projection + nonlinear interactions + noise."""

    def __init__(self, seed: int, n_features: int = 512, n_blocks: int = 8) -> None:
        if n_features % n_blocks != 0:
            raise ValueError("n_features must be divisible by n_blocks")
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.n_blocks = n_blocks
        self.block_size = n_features // n_blocks
        block_emotion = np.arange(n_blocks) % len(EMOTION_NAMES)

        weights = np.zeros((n_features, len(EMOTION_NAMES)))
        for block in range(n_blocks):
            lo = block * self.block_size
            hi = lo + self.block_size
            weights[lo:hi, block_emotion[block]] = 0.85
            weights[lo:hi, :] += 0.12 * rng.standard_normal((self.block_size, len(EMOTION_NAMES)))
        weights /= np.linalg.norm(weights, axis=1, keepdims=True)

        self.weights = weights.astype(np.float32)
        self.dominant_emotion_index = np.repeat(block_emotion, self.block_size)
        self.block_index = np.repeat(np.arange(n_blocks), self.block_size)
        self.feature_scale = np.exp(rng.normal(0.0, 0.25, n_features)).astype(np.float32)
        self.feature_offset = rng.normal(0.0, 0.15, n_features).astype(np.float32)
        self.severity_loading = (0.06 + 0.10 * rng.standard_normal(n_features)).astype(np.float32)
        self.noise_scale = (0.35 + 0.10 * rng.random(n_features)).astype(np.float32)

    def generate(
        self,
        rng: np.random.Generator,
        emotions: np.ndarray,
        severity: np.ndarray,
        shift: float = 0.0,
    ) -> np.ndarray:
        """Return an (n, 512) float32 simulated embedding matrix."""
        emotions = np.asarray(emotions, dtype=np.float32)
        severity = np.asarray(severity, dtype=np.float32)
        n = emotions.shape[0]

        base = emotions @ self.weights.T                                  # (n, 512)
        compressed = 0.6 * np.tanh(base) + 0.4 * base                     # nonlinear compression
        interaction = 0.25 * base * emotions[:, self.dominant_emotion_index]
        severity_term = severity[:, None] * self.severity_loading[None, :]
        shift_term = shift * (0.5 + 0.2 * self.weights[:, 1])[None, :]
        noise = rng.standard_normal((n, self.n_features)).astype(np.float32) * self.noise_scale

        out = compressed + interaction + severity_term + shift_term + noise
        out = out * self.feature_scale + self.feature_offset
        return out.astype(np.float32)

    def describe_features(self) -> list[dict[str, object]]:
        return [
            {
                "feature_id": column,
                "block": int(self.block_index[i]),
                "dominant_latent_dimension": EMOTION_NAMES[int(self.dominant_emotion_index[i])],
                "description": (
                    f"Simulated facial-expression embedding dimension {column} "
                    f"(block {int(self.block_index[i])}, dominant latent dimension "
                    f"{EMOTION_NAMES[int(self.dominant_emotion_index[i])]})"
                ),
            }
            for i, column in enumerate(FACIAL_COLUMNS)
        ]