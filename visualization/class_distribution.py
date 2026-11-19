# visualization/class_distribution.py
"""Plot the synthetic class distribution."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dataset_generator.target_generator import CLASS_LABELS


def plot_class_distribution(frame: pd.DataFrame, out_path: str | Path, target: str = "depression_risk") -> Path:
    counts = frame[target].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([CLASS_LABELS[int(i)] for i in counts.index], counts.values, color="#3b6ea5")
    ax.set_ylabel("Records")
    ax.set_title("SMDRPD synthetic class distribution")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path