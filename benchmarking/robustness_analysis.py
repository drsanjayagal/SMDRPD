"""Experiment D: missing features, missing modalities, Gaussian noise, imbalance."""
from __future__ import annotations

import argparse

import numpy as np

from benchmarking.models import build_model
from benchmarking.runner import load_modality
from evaluation.metrics import compute_metrics


def mask_modality(X: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    X = X.copy()
    rows = rng.random(X.shape[0]) < fraction
    X[rows, :] = 0.0
    return X


def inject_gaussian_noise(X: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return X + rng.normal(0.0, sigma, size=X.shape)


def drop_features(X: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    X = X.copy()
    n_drop = int(round(X.shape[1] * fraction))
    if n_drop:
        columns = rng.choice(X.shape[1], size=n_drop, replace=False)
        X[:, columns] = 0.0
    return X


def perturbation_experiment(dataset_dir, model_name="logistic_regression", seed=0) -> list[dict]:
    from sklearn.model_selection import train_test_split

    frame = load_modality(dataset_dir, "multimodal")
    features = [c for c in frame.columns if c not in ("depression_risk", "synthetic_subject_id")]
    X = frame[features].to_numpy(dtype=np.float32)
    y = frame["depression_risk"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = build_model(model_name, seed=seed)
    model.fit(X_train, y_train)
    rng = np.random.default_rng(seed)

    conditions = {
        "clean": X_test,
        "gaussian_noise_0.1": inject_gaussian_noise(X_test, 0.1, rng),
        "gaussian_noise_0.5": inject_gaussian_noise(X_test, 0.5, rng),
        "missing_features_10pct": drop_features(X_test, 0.10, rng),
        "missing_features_50pct": drop_features(X_test, 0.50, rng),
        "missing_modality_20pct": mask_modality(X_test, 0.20, rng),
    }
    output = []
    for name, X_cond in conditions.items():
        proba = model.predict_proba(X_cond) if hasattr(model, "predict_proba") else None
        metrics = compute_metrics(y_test, model.predict(X_cond), proba)
        metrics.update({"condition": name, "model": model_name})
        output.append(metrics)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/processed")
    args = parser.parse_args()
    for row in perturbation_experiment(args.dataset_dir):
        print(f"{row['condition']:<24} macro-F1={row['f1_macro']:.4f}")