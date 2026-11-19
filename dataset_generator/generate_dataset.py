"""CLI entry point: generate the SMDRPD synthetic multimodal dataset."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .facial_feature_generator import FACIAL_COLUMNS, FacialEmbeddingGenerator
from .io_utils import make_writer
from .latent_variable_model import sample_latents
from .metadata_generator import MetadataGenerator
from .multimodal_fusion import (
    MULTIMODAL_COLUMNS,
    PSYCHOMETRIC_COLUMNS,
    TARGET_COLUMN,
    build_multimodal,
    with_ids,
)
from .psychometric_generator import describe_features, generate_psychometric
from .target_generator import CLASS_LABELS, CLASS_SCENARIOS, TargetGenerator
from .utils import batch_rngs, ensure_dir, get_logger, write_sha256sums

LOGGER = get_logger()

DEFAULT_CONFIG: dict[str, Any] = {
    "project": {
        "name": "SMDRPD",
        "version": "1.0.0",
        "dataset_name": "Synthetic Multimodal Depression Risk Prediction Dataset",
    },
    "generation": {
        "base_seed": 20240115,
        "n_records": 100000,
        "batch_size": 10000,
        "observations_per_subject": 1,
        "distribution_shift": 0.0,
        "modality_dropout_rate": 0.0,
        "class_scenario": "moderate_imbalance",
    },
    "features": {"n_psychometric": 10, "n_facial": 512, "n_blocks": 8},
    "target": {"calibration_size": 200000, "nonlinear_strength": 0.45, "noise_sd": 0.55},
    "output": {
        "formats": ["parquet"],
        "redundant_modalities": True,
        "sample_records": 500,
        "statistics_sample_rows": 200000,
    },
    "paths": {
        "output_dir": "data/processed",
        "sample_dir": "data/sample",
        "metadata_dir": "data/metadata",
        "results_dir": "results",
    },
}


def deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None and Path(path).exists():
        with open(path, "r", encoding="utf-8") as handle:
            user_config = yaml.safe_load(handle) or {}
        deep_update(config, user_config)
    return config


class StatsBuffer:
    """Deterministic bounded buffer used for the published descriptive statistics."""

    def __init__(self, columns: list[str], max_rows: int) -> None:
        self.columns = columns
        self.max_rows = max_rows
        self._chunks: list[np.ndarray] = []
        self._rows = 0

    def add(self, values: np.ndarray) -> None:
        if self._rows >= self.max_rows:
            return
        take = min(self.max_rows - self._rows, values.shape[0])
        self._chunks.append(np.asarray(values[:take], dtype=np.float64).copy())
        self._rows += take

    @property
    def rows(self) -> int:
        return self._rows

    def frame(self) -> pd.DataFrame:
        if not self._chunks:
            return pd.DataFrame(columns=self.columns)
        return pd.DataFrame(np.vstack(self._chunks), columns=self.columns)


def _feature_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": frame.columns,
            "mean": frame.mean().values,
            "median": frame.median().values,
            "std": frame.std().values,
            "variance": frame.var().values,
            "min": frame.min().values,
            "max": frame.max().values,
            "skewness": frame.skew().values,
            "kurtosis": frame.kurtosis().values,
        }
    )


def _data_dictionary(facial_generator: FacialEmbeddingGenerator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in ("synthetic_subject_id", "synthetic_observation_id"):
        rows.append(
            {
                "column_name": column,
                "modality": "metadata",
                "data_type": "string",
                "valid_range": "synthetic identifier",
                "description": "Synthetic identifier; encodes no target information.",
            }
        )
    for item in describe_features():
        rows.append(
            {
                "column_name": item["feature_id"],
                "modality": "psychometric",
                "data_type": "float32",
                "valid_range": "0-3 (ordinal)",
                "description": item["description"],
            }
        )
    for item in facial_generator.describe_features():
        rows.append(
            {
                "column_name": item["feature_id"],
                "modality": "facial_expression",
                "data_type": "float32",
                "valid_range": "unbounded simulated embedding",
                "description": item["description"],
            }
        )
    rows.append(
        {
            "column_name": TARGET_COLUMN,
            "modality": "target",
            "data_type": "int8",
            "valid_range": "0, 1, 2",
            "description": "Synthetic depression-risk category (simulated research label).",
        }
    )
    return pd.DataFrame(rows)


def generate_dataset(
    config: dict[str, Any],
    output_dir: str | Path,
    n_records: int | None = None,
    seed: int | None = None,
    batch_size: int | None = None,
    scenario: str | None = None,
    formats: tuple[str, ...] | None = None,
    sample_records: int | None = None,
) -> dict[str, Any]:
    """Generate the dataset in batches and write all artefacts. Returns a summary dict."""
    generation = config["generation"]
    n_records = int(n_records if n_records is not None else generation["n_records"])
    seed = int(seed if seed is not None else generation["base_seed"])
    batch_size = int(batch_size if batch_size is not None else generation["batch_size"])
    scenario = scenario or generation["class_scenario"]
    formats = tuple(formats or config["output"]["formats"])
    sample_records = int(
        sample_records if sample_records is not None else config["output"]["sample_records"]
    )
    if scenario not in CLASS_SCENARIOS:
        raise ValueError(f"Unknown class scenario: {scenario}")
    if n_records < 1 or batch_size < 1:
        raise ValueError("n_records and batch_size must be positive")

    output_dir = ensure_dir(output_dir)
    sample_dir = ensure_dir(output_dir / "sample")
    version = str(config["project"]["version"])

    facial_generator = FacialEmbeddingGenerator(
        seed=seed,
        n_features=int(config["features"]["n_facial"]),
        n_blocks=int(config["features"]["n_blocks"]),
    )
    target_generator = TargetGenerator(
        base_seed=seed,
        scenario=scenario,
        calibration_size=int(config["target"]["calibration_size"]),
        nonlinear_strength=float(config["target"]["nonlinear_strength"]),
        noise_sd=float(config["target"]["noise_sd"]),
    )
    metadata_generator = MetadataGenerator(version=version)

    writers: dict[str, Any] = {}
    for fmt in formats:
        suffix = {"parquet": ".parquet", "csv_gz": ".csv.gz", "csv": ".csv"}[fmt]
        writers[f"psychometric_{fmt}"] = make_writer(output_dir / f"psychometric_features{suffix}", fmt)
        writers[f"facial_{fmt}"] = make_writer(
            output_dir / f"facial_expression_features{suffix}", fmt
        )
        if config["output"].get("redundant_modalities", True):
            writers[f"multimodal_{fmt}"] = make_writer(
                output_dir / f"multimodal_features{suffix}", fmt
            )
        writers[f"metadata_{fmt}"] = make_writer(
            output_dir / f"synthetic_metadata{suffix}", fmt
        )

    stats_buffer = StatsBuffer(list(MULTIMODAL_COLUMNS), int(config["output"]["statistics_sample_rows"]))
    class_counts = np.zeros(3, dtype=np.int64)
    sample_store: dict[str, list[pd.DataFrame]] = {"psych": [], "facial": [], "multi": []}
    samples_collected = 0

    observations_per_subject = max(int(generation["observations_per_subject"]), 1)
    dropout_rate = float(generation["modality_dropout_rate"])
    shift = float(generation["distribution_shift"])
    n_batches = int(np.ceil(n_records / batch_size))

    subject_offset = 0
    observation_offset = 0
    generated = 0

    for batch_index in range(n_batches):
        n = min(batch_size, n_records - generated)
        rng_latent, rng_psych, rng_facial, rng_target, rng_meta = batch_rngs(seed, batch_index, 5)
        n_subjects = int(np.ceil(n / observations_per_subject))

        severity_subject, emotion_subject = sample_latents(rng_latent, n_subjects)
        subject_index = np.repeat(np.arange(n_subjects), observations_per_subject)[:n]

        severity = 0.9 * severity_subject[subject_index] + 0.35 * rng_latent.standard_normal(n)
        emotions = (
            0.9 * emotion_subject[subject_index]
            + 0.35 * rng_latent.standard_normal((n, emotion_subject.shape[1]))
        )

        psychometric = generate_psychometric(rng_psych, severity)
        facial = facial_generator.generate(rng_facial, emotions, severity, shift=shift)
        risk = target_generator.assign(rng_target, severity, emotions, psychometric)

        metadata = metadata_generator.generate(
            rng_meta,
            subject_index=subject_index,
            n_subjects=n_subjects,
            subject_offset=subject_offset,
            observation_offset=observation_offset,
            risk=risk,
            seed=seed,
            dropout_rate=dropout_rate,
        )
        ids = metadata[["synthetic_subject_id", "synthetic_observation_id"]]

        psych_frame = with_ids(ids, psychometric, PSYCHOMETRIC_COLUMNS)
        facial_frame = with_ids(ids, facial, FACIAL_COLUMNS)
        multimodal_frame = with_ids(ids, build_multimodal(psychometric, facial), MULTIMODAL_COLUMNS)
        multimodal_frame[TARGET_COLUMN] = risk.astype(np.int8)

        for fmt in formats:
            writers[f"psychometric_{fmt}"].write(psych_frame)
            writers[f"facial_{fmt}"].write(facial_frame)
            if config["output"].get("redundant_modalities", True):
                writers[f"multimodal_{fmt}"].write(multimodal_frame)
            writers[f"metadata_{fmt}"].write(metadata)

        stats_buffer.add(multimodal_frame[list(MULTIMODAL_COLUMNS)].to_numpy())
        class_counts += np.bincount(risk, minlength=3)

        if samples_collected < sample_records:
            take = min(sample_records - samples_collected, n)
            sample_store["psych"].append(psych_frame.head(take))
            sample_store["facial"].append(facial_frame.head(take))
            sample_store["multi"].append(multimodal_frame.head(take))
            samples_collected += take

        subject_offset += n_subjects
        observation_offset += n
        generated += n
        LOGGER.info("batch %d/%d generated (%d records)", batch_index + 1, n_batches, generated)

    for writer in writers.values():
        writer.close()

    statistics_frame = stats_buffer.frame()
    feature_statistics = _feature_statistics(statistics_frame)
    feature_statistics.to_csv(output_dir / "feature_statistics.csv", index=False)

    proportions = class_counts / max(class_counts.sum(), 1)
    class_distribution = pd.DataFrame(
        {
            "class_value": [0, 1, 2],
            "class_label": [CLASS_LABELS[i] for i in range(3)],
            "count": class_counts,
            "proportion": proportions,
        }
    )
    class_distribution.to_csv(output_dir / "class_distribution.csv", index=False)

    _data_dictionary(facial_generator).to_csv(output_dir / "data_dictionary.csv", index=False)

    with open(output_dir / "generation_config.yaml", "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    for key, frames in sample_store.items():
        if frames:
            suffix = ".parquet"
            pd.concat(frames, ignore_index=True).to_parquet(sample_dir / f"sample_{key}{suffix}", index=False)

    artefact_paths = sorted(
        p for p in output_dir.iterdir() if p.is_file() and p.suffix in {".parquet", ".gz", ".csv", ".yaml"}
    )
    hashes = write_sha256sums(artefact_paths, output_dir / "SHA256SUMS.txt")

    descriptor = {
        "dataset_name": config["project"]["dataset_name"],
        "dataset_identifier": config["project"]["name"],
        "version": version,
        "generator_version": __version__,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": int(generated),
        "batches": n_batches,
        "batch_size": batch_size,
        "seed": seed,
        "class_scenario": scenario,
        "target_proportions_requested": CLASS_SCENARIOS[scenario],
        "class_distribution_observed": {
            CLASS_LABELS[i]: int(class_counts[i]) for i in range(3)
        },
        "observations_per_subject": observations_per_subject,
        "distribution_shift": shift,
        "n_psychometric_features": 10,
        "n_facial_features": 512,
        "n_multimodal_features": 522,
        "formats": list(formats),
        "files_sha256": hashes,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "statistics_computed_on_rows": int(stats_buffer.rows),
        "disclaimer": (
            "Entirely synthetic data. No real patients, records, photographs or "
            "identifiable information. Labels are simulated research labels and are "
            "not clinical diagnoses."
        ),
    }
    with open(output_dir / "dataset_metadata.json", "w", encoding="utf-8") as handle:
        json.dump(descriptor, handle, indent=2)

    LOGGER.info("generation complete: %d records -> %s", generated, output_dir)
    return descriptor


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the SMDRPD synthetic dataset")
    parser.add_argument("--records", type=int, default=None, help="number of records")
    parser.add_argument("--seed", type=int, default=None, help="base random seed")
    parser.add_argument("--batch-size", type=int, default=None, help="records per batch")
    parser.add_argument("--class-scenario", default=None, choices=sorted(CLASS_SCENARIOS))
    parser.add_argument("--config", default="config/dataset_config.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--formats", default=None, help="comma separated: parquet,csv_gz")
    parser.add_argument("--sample-records", type=int, default=None)
    parser.add_argument("--no-redundant-modalities", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    LOGGER.setLevel(args.log_level.upper())
    config = load_config(args.config)
    if args.no_redundant_modalities:
        config["output"]["redundant_modalities"] = False
    output_dir = args.output_dir or config["paths"]["output_dir"]
    formats = tuple(f.strip() for f in args.formats.split(",")) if args.formats else None
    generate_dataset(
        config,
        output_dir,
        n_records=args.records,
        seed=args.seed,
        batch_size=args.batch_size,
        scenario=args.class_scenario,
        formats=formats,
        sample_records=args.sample_records,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())