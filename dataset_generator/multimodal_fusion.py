"""Alignment and concatenation of the psychometric and facial modalities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .facial_feature_generator import FACIAL_COLUMNS
from .psychometric_generator import PSYCHOMETRIC_COLUMNS

ID_COLUMNS = ("synthetic_subject_id", "synthetic_observation_id")
TARGET_COLUMN = "depression_risk"
MULTIMODAL_COLUMNS: tuple[str, ...] = PSYCHOMETRIC_COLUMNS + FACIAL_COLUMNS
META_COLUMNS = ID_COLUMNS + (
    "synthetic_age_group",
    "synthetic_gender_category",
    "generation_seed",
    "generation_version",
    "modality_availability",
    TARGET_COLUMN,
)

assert len(MULTIMODAL_COLUMNS) == 522, "Multimodal schema must stay at 522 dimensions"


def build_multimodal(psychometric: np.ndarray, facial: np.ndarray) -> np.ndarray:
    """Concatenate to the 522-dimensional integrated representation X = [P, F]."""
    if psychometric.shape[1] != 10 or facial.shape[1] != 512:
        raise ValueError("Expected 10 psychometric and 512 facial features")
    if psychometric.shape[0] != facial.shape[0]:
        raise ValueError("Modality row counts are misaligned")
    return np.hstack([psychometric, facial]).astype(np.float32)


def with_ids(ids: pd.DataFrame, values: np.ndarray, columns: tuple[str, ...]) -> pd.DataFrame:
    """Attach identifiers in front of a feature matrix, preserving row order."""
    feature_frame = pd.DataFrame(values, columns=list(columns))
    return pd.concat([ids.reset_index(drop=True), feature_frame], axis=1)


def reconstruct_multimodal(psychometric_path, facial_path) -> pd.DataFrame:
    """Rebuild the integrated dataset from the two modality files and verify alignment."""
    psych = pd.read_parquet(psychometric_path)
    facial = pd.read_parquet(facial_path)
    merged = psych.merge(facial, on=list(ID_COLUMNS), how="inner", validate="one_to_one")
    if len(merged) != len(psych) or len(merged) != len(facial):
        raise ValueError("Modality files could not be aligned on identifiers")
    ordered = list(ID_COLUMNS) + list(MULTIMODAL_COLUMNS)
    return merged[ordered]