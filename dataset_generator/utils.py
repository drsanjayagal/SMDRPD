"""Shared utilities: logging, deterministic RNG derivation, checksums."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterable

import numpy as np

LOGGER_NAME = "smdrpd"


def get_logger(name: str = LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def derive_rng(base_seed: int, *keys: int) -> np.random.Generator:
    """Deterministic independent generator for a (seed, key...) tuple."""
    return np.random.default_rng(np.random.SeedSequence(entropy=[int(base_seed), *map(int, keys)]))


def batch_rngs(base_seed: int, batch_index: int, n_streams: int = 5) -> list[np.random.Generator]:
    """Independent reproducible RNG streams for one generation batch."""
    ss = np.random.SeedSequence([int(base_seed), int(batch_index)])
    return [np.random.default_rng(child) for child in ss.spawn(n_streams)]


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def write_sha256sums(paths: Iterable[Path], out_file: Path) -> dict[str, str]:
    """Write `hash  filename` lines (GNU coreutils compatible)."""
    result = {}
    lines = []
    for path in paths:
        digest = sha256_file(path)
        result[path.name] = digest
        lines.append(f"{digest}  {path.name}")
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path