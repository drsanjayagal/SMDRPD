# Changelog

All notable changes to the SMDRPD project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-19

### Added

- Synthetic multimodal dataset generator producing 10 psychometric features
  (`P01`-`P10`, ordinal 0-3) and 512 simulated facial-expression embedding
  dimensions (`F001`-`F512`), yielding a 522-dimensional integrated
  representation.
- Latent-variable model driving both modalities from a shared synthetic
  severity factor and seven correlated emotional dimensions.
- Probabilistic three-class target generator (`depression_risk` in {0, 1, 2})
  with configurable class scenarios: `balanced`, `moderate_imbalance`,
  `high_imbalance`.
- Synthetic metadata generator producing subject and observation identifiers
  (`synthetic_subject_id`, `synthetic_observation_id`), synthetic age groups,
  synthetic gender categories, generation seed, generation version and
  modality availability.
- Independent validation framework with 23 structural, statistical,
  cross-modality and integrity checks; produces `validation_report.json`
  and `validation_report.md`.
- Reproducibility framework: fixed seeds, deterministic per-batch RNG
  derivation, SHA-256 checksums, automated tests.
- Optional benchmarking layer: SVM, Random Forest, Logistic Regression,
  XGBoost, Gradient Boosting, MLP; unimodal, multimodal, robustness and
  ablation experiments; multiclass metrics including macro F1, MCC,
  balanced accuracy, ROC-AUC and per-class sensitivity/specificity.
- Optional explainability layer: SHAP, permutation importance, modality-level
  contribution analysis.
- Visualization utilities: correlation heatmap, class distribution, PCA
  projection, confusion matrix, feature importance.
- Documentation: `DATASET_CARD.md`, `DATA_DICTIONARY.md`,
  `GENERATION_METHODOLOGY.md`, `VALIDATION_REPORT.md`,
  `REPRODUCIBILITY.md`, `ETHICAL_CONSIDERATIONS.md`,
  `ZENODO_METADATA.md`.
- 15 automated tests covering dataset generation, feature dimensions,
  data validation, reproducibility and data integrity.

### Known limitations

- The dataset is entirely synthetic. It carries no clinical validity and
  must not be used for diagnosis, screening, triage, treatment decisions
  or any other medical purpose.
- The 512-dimensional facial-expression features are simulated. They are
  not extracted from photographs and are not equivalent to embeddings from
  any pretrained facial-recognition or facial-emotion-recognition model.
- Synthetic age groups and gender categories are arbitrary strata. They do
  not represent real demographic distributions.
- Benchmark and ablation metrics reported by the optional benchmarking
  layer are computed on synthetic data only. They are demonstration
  infrastructure, not evidence of real-world predictive performance.

### Not included in this release

- No real patient records, no real photographs, no identifiable personal
  information.
- No pre-written VALIDATION_REPORT.pdf. The authoritative report is
  generated at run time by `preprocessing.data_validation`.

[1.0.0]: https://github.com/drsanjayagal/SMDRPD/releases/tag/v1.0.0