"""SMDRPD synthetic multimodal dataset generation package.

Contains the psychometric generator, facial-expression embedding generator,
latent-variable model, target generator, metadata generator, multimodal fusion
helper and the CLI entry point.

All data produced by this package is entirely synthetic. No real patient data,
medical records, photographs or identifiable information are used or produced.
"""

__version__ = "1.0.0"
__dataset_name__ = "SMDRPD"
__dataset_full_name__ = (
    "Synthetic Multimodal Depression Risk Prediction Dataset"
)

__all__ = ["__version__", "__dataset_name__", "__dataset_full_name__"]