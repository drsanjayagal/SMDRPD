# evaluation/benchmark_report.py
"""Aggregate benchmark JSON into markdown and CSV."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def build_report(results_path: str | Path, out_dir: str | Path = "results/reports") -> Path:
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = pd.json_normalize([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in results])
    lines = [
        "# SMDRPD Benchmark Report",
        "",
        "> All figures below were computed on synthetic SMDRPD data. They are not",
        "> evidence of clinical validity or real-world predictive performance.",
        "",
        "| Modality | Model | Features | Macro F1 | Accuracy | Balanced Acc. | MCC |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, row in flat.iterrows():
        lines.append(
            f"| {row.get('modality','-')} | {row.get('model','-')} | {int(row.get('n_features',0))} | "
            f"{row.get('f1_macro',float('nan')):.4f} | {row.get('accuracy',float('nan')):.4f} | "
            f"{row.get('balanced_accuracy',float('nan')):.4f} | {row.get('mcc',float('nan')):.4f} |"
        )
    path = out_dir / "benchmark_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path