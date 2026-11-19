"""Synthetic metadata generation.

Age groups and gender categories are arbitrary synthetic strata (AG1-AG6,
G1-G3). They are NOT real demographic distributions and must never be read as
such. Identifiers are sequential counters and therefore encode no target
information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

AGE_GROUPS = ("AG1", "AG2", "AG3", "AG4", "AG5", "AG6")
AGE_PROBS = (0.16, 0.22, 0.20, 0.17, 0.14, 0.11)
GENDER_CATEGORIES = ("G1", "G2", "G3")
GENDER_PROBS = (0.49, 0.49, 0.02)

MODALITY_BOTH = "psychometric+facial"
MODALITY_PSYCH = "psychometric_only"
MODALITY_FACIAL = "facial_only"


class MetadataGenerator:
    def __init__(self, version: str) -> None:
        self.version = version

    def generate(
        self,
        rng: np.random.Generator,
        subject_index: np.ndarray,
        n_subjects: int,
        subject_offset: int,
        observation_offset: int,
        risk: np.ndarray,
        seed: int,
        dropout_rate: float = 0.0,
    ) -> pd.DataFrame:
        n = subject_index.shape[0]
        subject_ids = np.arange(subject_offset, subject_offset + n_subjects)
        age = rng.choice(len(AGE_GROUPS), size=n_subjects, p=AGE_PROBS)
        gender = rng.choice(len(GENDER_CATEGORIES), size=n_subjects, p=GENDER_PROBS)

        if dropout_rate > 0.0:
            draw = rng.random(n)
            availability = np.where(
                draw < dropout_rate / 2,
                MODALITY_PSYCH,
                np.where(draw < dropout_rate, MODALITY_FACIAL, MODALITY_BOTH),
            )
        else:
            availability = np.full(n, MODALITY_BOTH, dtype=object)

        return pd.DataFrame(
            {
                "synthetic_subject_id": np.array(
                    [f"S{i:07d}" for i in subject_ids[subject_index]], dtype="<U8"
                ),
                "synthetic_observation_id": np.array(
                    [f"O{i:08d}" for i in np.arange(observation_offset, observation_offset + n)],
                    dtype="<U9",
                ),
                "synthetic_age_group": np.asarray(AGE_GROUPS, dtype=object)[age[subject_index]],
                "synthetic_gender_category": np.asarray(GENDER_CATEGORIES, dtype=object)[
                    gender[subject_index]
                ],
                "generation_seed": np.full(n, int(seed), dtype=np.int64),
                "generation_version": np.full(n, self.version, dtype=object),
                "modality_availability": availability,
                "depression_risk": risk.astype(np.int8),
            }
        )