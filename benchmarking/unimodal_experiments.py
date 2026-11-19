"""CLI: Experiment A (psychometric, 10 features) and B (facial, 512 features)."""
from __future__ import annotations

import argparse

from benchmarking.models import SUPPORTED_MODELS
from benchmarking.runner import run_experiment, save_results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/processed")
    parser.add_argument("--models", default="logistic_regression,random_forest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=20000)
    args = parser.parse_args(argv)
    models = [m for m in args.models.split(",") if m in SUPPORTED_MODELS]
    results = []
    for modality in ("psychometric", "facial"):
        for model in models:
            results.append(
                run_experiment(args.dataset_dir, modality, model, args.seed,
                               max_train_samples=args.max_train_samples)
            )
    save_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())