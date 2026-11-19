"""Independent structural, statistical and integrity validation of SMDRPD."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dataset_generator.multimodal_fusion import (
    ID_COLUMNS,
    MULTIMODAL_COLUMNS,
    PSYCHOMETRIC_COLUMNS,
    TARGET_COLUMN,
)
from dataset_generator.utils import get_logger, sha256_file

LOGGER = get_logger()


def _iter_sample(path: Path, columns: list[str] | None, max_rows: int) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(path)
    chunks, collected = [], 0
    for batch in parquet_file.iter_batches(batch_size=min(50000, max_rows), columns=columns):
        chunks.append(batch.to_pandas())
        collected += batch.num_rows
        if collected >= max_rows:
            break
    if not chunks:
        return pd.DataFrame(columns=columns)
    return pd.concat(chunks, ignore_index=True).head(max_rows)


def _read_checksums(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        mapping[name.strip()] = digest
    return mapping


def validate_dataset(
    dataset_dir: str | Path,
    max_rows: int = 200_000,
    expected_records: int | None = None,
    write_reports: bool = True,
    report_dir: str | Path | None = None,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, passed: bool, detail: Any = None) -> None:
        checks[name] = {"passed": bool(passed), "detail": detail}

    multimodal_path = dataset_dir / "multimodal_features.parquet"
    psych_path = dataset_dir / "psychometric_features.parquet"
    facial_path = dataset_dir / "facial_expression_features.parquet"
    metadata_path = dataset_dir / "synthetic_metadata.parquet"

    for name, path in [
        ("file_multimodal_exists", multimodal_path),
        ("file_psychometric_exists", psych_path),
        ("file_facial_exists", facial_path),
        ("file_metadata_exists", metadata_path),
        ("file_descriptor_exists", dataset_dir / "dataset_metadata.json"),
    ]:
        record(name, path.exists(), str(path))

    if not multimodal_path.exists():
        report = {"dataset_dir": str(dataset_dir), "checks": checks, "passed": False}
        return report

    multimodal_meta = pq.ParquetFile(multimodal_path).metadata
    n_rows = multimodal_meta.num_rows
    record("record_count_positive", n_rows > 0, n_rows)
    if expected_records is not None:
        record("record_count_matches_expected", n_rows == expected_records, n_rows)

    sample = _iter_sample(multimodal_path, None, max_rows)
    record(
        "multimodal_feature_count",
        all(c in sample.columns for c in MULTIMODAL_COLUMNS),
        int(sum(c in sample.columns for c in MULTIMODAL_COLUMNS)),
    )
    record("target_column_present", TARGET_COLUMN in sample.columns, TARGET_COLUMN)
    record("no_missing_values", bool(sample.notna().all().all()), int(sample.isna().sum().sum()))

    feature_block = sample[list(MULTIMODAL_COLUMNS)]
    record(
        "feature_dtypes_numeric",
        bool(all(np.issubdtype(t, np.number) for t in feature_block.dtypes)),
        [str(t) for t in feature_block.dtypes.unique()],
    )
    psych_block = feature_block[list(PSYCHOMETRIC_COLUMNS)]
    record(
        "psychometric_range_0_3",
        bool(psych_block.min().min() >= 0 and psych_block.max().max() <= 3),
        [float(psych_block.min().min()), float(psych_block.max().max())],
    )
    record(
        "psychometric_integer_valued",
        bool(np.allclose(psych_block.to_numpy(), np.round(psych_block.to_numpy()))),
        None,
    )
    record(
        "facial_features_finite",
        bool(np.isfinite(feature_block.to_numpy()).all()),
        None,
    )
    record(
        "target_values_valid",
        bool(set(sample[TARGET_COLUMN].unique()) <= {0, 1, 2}),
        sorted(int(v) for v in sample[TARGET_COLUMN].unique()),
    )

    if ID_COLUMNS[1] in sample.columns:
        duplicates = int(sample[ID_COLUMNS[1]].duplicated().sum())
        record("no_duplicate_observation_ids", duplicates == 0, duplicates)

    if psych_path.exists() and facial_path.exists():
        psych_ids = _iter_sample(psych_path, list(ID_COLUMNS), 5000)
        facial_ids = _iter_sample(facial_path, list(ID_COLUMNS), 5000)
        record(
            "modality_alignment_head",
            bool(
                len(psych_ids) == len(facial_ids)
                and (psych_ids[ID_COLUMNS[1]].to_numpy() == facial_ids[ID_COLUMNS[1]].to_numpy()).all()
            ),
            None,
        )
        record(
            "psychometric_file_dimensions",
            pq.ParquetFile(psych_path).metadata.num_columns == 12,
            pq.ParquetFile(psych_path).metadata.num_columns,
        )
        record(
            "facial_file_dimensions",
            pq.ParquetFile(facial_path).metadata.num_columns == 514,
            pq.ParquetFile(facial_path).metadata.num_columns,
        )

    statistics = pd.DataFrame(
        {
            "feature": feature_block.columns,
            "mean": feature_block.mean().values,
            "median": feature_block.median().values,
            "std": feature_block.std().values,
            "min": feature_block.min().values,
            "max": feature_block.max().values,
            "skewness": feature_block.skew().values,
            "kurtosis": feature_block.kurtosis().values,
        }
    )
    record("all_features_have_variance", bool((statistics["std"] > 0).all()), None)

    correlation = feature_block.corr()
    off_diagonal = correlation.to_numpy()[~np.eye(len(correlation), dtype=bool)]
    record("correlation_matrix_finite", bool(np.isfinite(off_diagonal).all()), None)

    class_distribution = (
        sample[TARGET_COLUMN].value_counts(normalize=True).sort_index().to_dict()
    )
    record("all_three_classes_present", len(class_distribution) == 3, class_distribution)

    checksums = _read_checksums(dataset_dir / "SHA256SUMS.txt")
    if checksums:
        mismatches = [
            name
            for name, digest in checksums.items()
            if (dataset_dir / name).exists() and sha256_file(dataset_dir / name) != digest
        ]
        record("sha256_checksums_match", len(mismatches) == 0, mismatches)
    else:
        record("sha256_checksums_present", False, "SHA256SUMS.txt missing")

    passed = all(item["passed"] for item in checks.values())
    report = {
        "dataset_dir": str(dataset_dir),
        "records_total": int(n_rows),
        "records_checked": int(len(sample)),
        "checks": checks,
        "class_distribution_sampled": {str(k): float(v) for k, v in class_distribution.items()},
        "passed": passed,
        "disclaimer": (
            "Validation concerns internal consistency of synthetic data only. It is NOT "
            "validation against real clinical populations."
        ),
    }

    if write_reports:
        out_dir = Path(report_dir) if report_dir else dataset_dir.parent.parent / "results" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        statistics.to_csv(out_dir / "validation_feature_statistics.csv", index=False)
        correlation.to_csv(out_dir / "validation_correlation_matrix.csv")
        lines = ["# SMDRPD Data Validation Report", "", f"Dataset: `{dataset_dir}`", ""]
        lines.append(f"Records: **{n_rows}** (checked sample: {len(sample)})")
        lines.append("")
        lines.append("| Check | Result |")
        lines.append("|---|---|")
        for name, item in checks.items():
            lines.append(f"| {name} | {'PASS' if item['passed'] else 'FAIL'} |")
        lines.append("")
        lines.append(f"**Overall: {'PASS' if passed else 'FAIL'}**")
        lines.append("")
        lines.append(
            "Validation covers synthetic-data integrity only and makes no claim about "
            "clinical validity or real-world generalisation."
        )
        (out_dir / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    LOGGER.info("validation %s (%d/%d checks passed)", "PASSED" if passed else "FAILED",
                sum(c["passed"] for c in checks.values()), len(checks))
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a generated SMDRPD dataset")
    parser.add_argument("--dataset-dir", default="data/processed")
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--expected-records", type=int, default=None)
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args(argv)
    report = validate_dataset(
        args.dataset_dir,
        max_rows=args.max_rows,
        expected_records=args.expected_records,
        report_dir=args.report_dir,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())