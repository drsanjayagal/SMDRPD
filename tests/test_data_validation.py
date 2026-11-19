# tests/test_data_validation.py
from preprocessing.data_validation import validate_dataset


def test_validation_passes(small_dataset, tmp_path):
    out_dir, _ = small_dataset
    report = validate_dataset(
        out_dir, max_rows=1500, expected_records=1500, report_dir=tmp_path
    )
    failed = {name: item for name, item in report["checks"].items() if not item["passed"]}
    assert report["passed"], f"Failed checks: {failed}"
    assert (tmp_path / "validation_report.json").exists()


def test_validation_detects_corruption(small_dataset, tmp_path):
    out_dir, _ = small_dataset
    corrupted = tmp_path / "corrupted"
    corrupted.mkdir()
    report = validate_dataset(corrupted, write_reports=False)
    assert report["passed"] is False