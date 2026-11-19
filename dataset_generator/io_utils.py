"""Streaming writers that keep memory bounded for large record counts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetBatchWriter:
    """Append successive batches as row groups to one parquet file."""

    def __init__(self, path: str | Path, compression: str = "snappy") -> None:
        self.path = Path(path)
        self.compression = compression
        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, table.schema, compression=self.compression)
            self._schema = table.schema
        elif table.schema != self._schema:
            table = table.cast(self._schema)
        self._writer.write_table(table)
        self.rows += table.num_rows

    def close(self) -> int:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return self.rows


class CsvBatchWriter:
    """Append successive batches to a single (optionally gzip) CSV file."""

    def __init__(self, path: str | Path, compression: str | None = "gzip") -> None:
        self.path = Path(path)
        self.compression = compression
        self._started = False
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        frame.to_csv(
            self.path,
            mode="a",
            header=not self._started,
            index=False,
            compression=self.compression,
        )
        self._started = True
        self.rows += len(frame)

    def close(self) -> int:
        return self.rows


def make_writer(path: Path, fmt: str):
    if fmt == "parquet":
        return ParquetBatchWriter(path)
    if fmt == "csv_gz":
        return CsvBatchWriter(path, compression="gzip")
    if fmt == "csv":
        return CsvBatchWriter(path, compression=None)
    raise ValueError(f"Unsupported output format: {fmt}")