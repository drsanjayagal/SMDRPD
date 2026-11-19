# visualization/feature_importance.py
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_feature_importance(importance: pd.DataFrame, out_path: str | Path,
                            feature_col: str = "feature",
                            value_col: str = "importance_mean",
                            top_n: int = 25) -> Path:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(top))))
    ax.barh(top[feature_col], top[value_col], color="#3b6ea5")
    ax.set_xlabel(value_col)
    ax.set_title(f"Top {len(top)} features (association, not causation)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path