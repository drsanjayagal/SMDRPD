"""Joint latent-variable model.

Samples one scalar synthetic severity factor and seven latent emotional
dimensions from a multivariate Gaussian with a documented correlation
structure. All quantities are simulated; nothing here corresponds to a real
person, a real photograph, or a clinical measurement.
"""
from __future__ import annotations

import numpy as np

EMOTION_NAMES = (
    "happiness",
    "sadness",
    "neutrality",
    "anger",
    "fear",
    "surprise",
    "emotional_variability",
)

# Documented correlation between synthetic latent severity and each emotion.
SEVERITY_EMOTION_CORR = np.array([-0.32, 0.38, 0.05, 0.22, 0.26, 0.02, 0.30])

# Three-factor loading structure of the seven emotional dimensions.
EMOTION_FACTORS = np.array(
    [
        [0.85, 0.00, 0.10],   # happiness
        [-0.55, 0.75, 0.05],  # sadness
        [0.00, 0.05, 0.90],   # neutrality
        [-0.20, 0.80, 0.05],  # anger
        [-0.15, 0.78, 0.10],  # fear
        [0.10, 0.20, 0.85],   # surprise
        [-0.30, 0.70, 0.15],  # emotional variability
    ]
)

_CHOLESKY: np.ndarray | None = None


def _nearest_correlation(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix onto the nearest positive-definite correlation matrix."""
    matrix = (matrix + matrix.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, floor, None)
    matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    scale = np.sqrt(np.diag(matrix))
    return matrix / np.outer(scale, scale)


def emotion_correlation_matrix() -> np.ndarray:
    return _nearest_correlation(EMOTION_FACTORS @ EMOTION_FACTORS.T)


def joint_correlation_matrix() -> np.ndarray:
    """8x8 correlation matrix of [severity, happiness, ..., emotional_variability]."""
    corr = np.eye(8)
    corr[1:, 1:] = emotion_correlation_matrix()
    corr[0, 1:] = SEVERITY_EMOTION_CORR
    corr[1:, 0] = SEVERITY_EMOTION_CORR
    return _nearest_correlation(corr)


def _cholesky() -> np.ndarray:
    global _CHOLESKY
    if _CHOLESKY is None:
        _CHOLESKY = np.linalg.cholesky(joint_correlation_matrix())
    return _CHOLESKY


def sample_latents(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (severity[n], emotions[n, 7]) drawn from the joint latent model."""
    z = rng.standard_normal((n, 8)) @ _cholesky().T
    return z[:, 0].astype(np.float64), z[:, 1:].astype(np.float64)