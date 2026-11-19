# visualization/dimensionality_reduction.py
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_pca(frame, columns, target_column: str, out_path: str | Path, seed: int = 0) -> Path:
    pca = PCA(n_components=2, random_state=seed)
    components = pca.fit_transform(frame[columns].to_numpy())
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(components[:, 0], components[:, 1], c=frame[target_column],
                         cmap="viridis", s=6, alpha=0.6)
    fig.colorbar(scatter, label="synthetic risk class")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA of the 522-dimensional synthetic representation")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path