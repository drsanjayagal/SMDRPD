# visualization/correlation_heatmap.py
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_heatmap(frame: pd.DataFrame, columns: list[str], out_path: str | Path,
                             max_features: int = 60) -> Path:
    subset = columns[:max_features]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(frame[subset].corr(), cmap="vlag", center=0.0, ax=ax,
                cbar_kws={"label": "Pearson r"})
    ax.set_title(f"Feature correlation (first {len(subset)} features)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path