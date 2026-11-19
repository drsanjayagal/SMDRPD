# benchmarking/runner.py
"""Subject-grouped experiment runner. Results are computed, never hardcoded."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from dataset_generator.multimodal_fusion import (
    FACIAL_COLUMNS,
    MULTIMODAL_COLUMNS,
    PSYCHOMETRIC_COLUMNS,
    TARGET_COLUMN,
)
from dataset_generator.utils import get_logger
from evaluation.metrics import Timer, compute_metrics

LOGGER = get_logger()

MODALITY_COLUMNS = {
    "psychometric": list(PSYCHOMETRIC_COLUMNS),
    "facial": list(FACIAL_COLUMNS),
    "multimodal": list(MULTIMODAL_COLUMNS),
}


def load_modality(dataset_dir: str | Path, modality: str) -> pd.DataFrame:
    if modality not in MODALITY_COLUMNS:
        raise ValueError(f"Unknown modality: {modality}")
    columns = MODALITY_COLUMNS[modality] + [TARGET_COLUMN, "synthetic_subject_id"]
    frame = pd.read_parquet(Path(dataset_dir) / "multimodal_features.parquet", columns=columns)
    return frame


def run_experiment(
    dataset_dir: str | Path,
    modality: str,
    model_name: str,
    seed: int = 0,
    test_size: float = 0.2,
    max_train_samples: int | None = 20000,
) -> dict[str, Any]:
    from benchmarking.models import build_model

    frame = load_modality(dataset_dir, modality)
    features = MODALITY_COLUMNS[modality]
    X = frame[features].to_numpy(dtype=np.float32)
    y = frame[TARGET_COLUMN].to_numpy()
    groups = frame["synthetic_subject_id"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))

    if max_train_samples is not None and len(train_idx) > max_train_samples:
        rng = np.random.default_rng(seed)
        train_idx = rng.choice(train_idx, size=max_train_samples, replace=False)

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    estimator = build_model(model_name, seed=seed)
    with Timer() as train_timer:
        estimator.fit(X_train, y_train)
    with Timer() as infer_timer:
        y_pred = estimator.predict(X_test)
        y_proba = (
            estimator.predict_proba(X_test)
            if hasattr(estimator, "predict_proba")
            else None
        )

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics.update(
        {
            "modality": modality,
            "model": model_name,
            "n_features": len(features),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "train_seconds": float(train_timer.elapsed),
            "inference_seconds": float(infer_timer.elapsed),
            "seed": seed,
            "data_provenance": "synthetic SMDRPD data - not evidence of real-world performance",
        }
    )
    LOGGER.info(
        "%s | %s | macro-F1=%.3f | accuracy=%.3f",
        modality, model_name, metrics["f1_macro"], metrics["accuracy"],
    )
    return metrics


def save_results(results: list[dict[str, Any]], out_dir: str | Path = "results/tables") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "benchmark_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    flat = pd.json_normalize(
        [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in results]
    )
    flat.to_csv(out_dir / "benchmark_results.csv", index=False)
    return path