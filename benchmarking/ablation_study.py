"""Ablation study: modality subsets, fusion variants and dimensionality reduction.

Runs on a bounded sample (default 20,000 records) so the study completes in
seconds on ordinary hardware even for the 1M-record dataset. Override
--max-samples to use more records. Results are written to
results/tables/ablation_results.{json,csv} and a markdown summary.

All metrics are computed on synthetic SMDRPD data only. They are not evidence
of clinical validity or real-world predictive performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from benchmarking.models import build_model
from benchmarking.runner import load_modality
from dataset_generator.multimodal_fusion import FACIAL_COLUMNS, PSYCHOMETRIC_COLUMNS
from dataset_generator.utils import get_logger
from evaluation.metrics import compute_metrics

LOGGER = get_logger()

CONFIGURATIONS: dict[str, list[str]] = {
    "psychometric_only": list(PSYCHOMETRIC_COLUMNS),
    "facial_only": list(FACIAL_COLUMNS),
    "early_fusion": list(PSYCHOMETRIC_COLUMNS) + list(FACIAL_COLUMNS),
}


def _subsample(frame: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    """Deterministic stratified subsample, preserving class proportions."""
    if len(frame) <= max_samples:
        return frame
    parts = []
    per_class = max_samples // frame["depression_risk"].nunique()
    for _, group in frame.groupby("depression_risk"):
        take = min(per_class, len(group))
        parts.append(group.sample(n=take, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def _evaluate(X_tr, X_te, y_tr, y_te, model_name: str, seed: int) -> dict:
    model = build_model(model_name, seed=seed)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te) if hasattr(model, "predict_proba") else None
    return compute_metrics(y_te, model.predict(X_te), proba)


def run_ablation(
    dataset_dir: str | Path,
    model_name: str = "logistic_regression",
    seed: int = 0,
    max_samples: int = 20_000,
    pca_components: int = 64,
) -> list[dict]:
    LOGGER.info("loading multimodal dataset from %s", dataset_dir)
    frame = load_modality(dataset_dir, "multimodal")
    LOGGER.info("loaded %d records; subsampling to %d", len(frame), max_samples)
    frame = _subsample(frame, max_samples, seed)
    y = frame["depression_risk"].to_numpy()
    LOGGER.info("working set: %d records, class counts %s",
                len(frame), dict(zip(*np.unique(y, return_counts=True))))

    results: list[dict] = []

    for name, columns in CONFIGURATIONS.items():
        LOGGER.info("running configuration: %s (%d features)", name, len(columns))
        X = frame[columns].to_numpy(dtype=np.float32)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        metrics = _evaluate(X_tr, X_te, y_tr, y_te, model_name, seed)
        metrics.update({
            "configuration": name,
            "model": model_name,
            "n_features": len(columns),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
        })
        results.append(metrics)
        LOGGER.info("  -> macro-F1=%.4f accuracy=%.4f",
                    metrics["f1_macro"], metrics["accuracy"])

    # PCA on the fused representation.
    LOGGER.info("running configuration: pca_%d on early_fusion", pca_components)
    X = frame[CONFIGURATIONS["early_fusion"]].to_numpy(dtype=np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    n_components = min(pca_components, X_tr.shape[1], X_tr.shape[0])
    pca = PCA(n_components=n_components, random_state=seed)
    X_tr_p = pca.fit_transform(X_tr)
    X_te_p = pca.transform(X_te)
    metrics = _evaluate(X_tr_p, X_te_p, y_tr, y_te, model_name, seed)
    metrics.update({
        "configuration": f"pca_{n_components}",
        "model": model_name,
        "n_features": int(n_components),
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
    })
    results.append(metrics)
    LOGGER.info("  -> macro-F1=%.4f (explained variance %.3f)",
                metrics["f1_macro"], metrics["explained_variance"])

    return results


def save_results(results: list[dict], out_dir: str | Path = "results/tables") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ablation_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    flat = pd.json_normalize(
        [{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in results]
    )
    flat.to_csv(out_dir / "ablation_results.csv", index=False)

    report_dir = out_dir.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SMDRPD Ablation Study",
        "",
        "> Metrics computed on synthetic SMDRPD data only. Not evidence of clinical",
        "> validity or real-world predictive performance.",
        "",
        "| Configuration | Features | Macro F1 | Accuracy | Balanced Acc. | MCC |",
        "|---|---|---|---|---|---|",
    ]
    for r in flat.itertuples():
        lines.append(
            f"| {r.configuration} | {int(r.n_features)} | {r.f1_macro:.4f} | "
            f"{r.accuracy:.4f} | {r.balanced_accuracy:.4f} | {r.mcc:.4f} |"
        )
    (report_dir / "ablation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir / "ablation_results.json"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SMDRPD ablation study")
    parser.add_argument("--dataset-dir", default="data/processed")
    parser.add_argument("--model", default="logistic_regression",
                        choices=["logistic_regression", "svm", "random_forest",
                                 "gradient_boosting", "mlp", "xgboost"])
    parser.add_argument("--max-samples", type=int, default=20000,
                        help="records used for train/test (default: 20000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pca-components", type=int, default=64)
    args = parser.parse_args(argv)

    results = run_ablation(
        args.dataset_dir,
        model_name=args.model,
        seed=args.seed,
        max_samples=args.max_samples,
        pca_components=args.pca_components,
    )
    save_results(results)
    LOGGER.info("ablation results saved to results/tables/ablation_results.{json,csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())