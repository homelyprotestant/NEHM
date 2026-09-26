"""Manifest.tsv and Material_Database.xlsx alignment (Image Index ↔ pipeline row)."""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np


def load_manifest_keys(manifest_path: Path) -> List[str]:
    """First column (record key / Image Index) per row; skip header."""
    lines = Path(manifest_path).read_text(encoding="utf-8").strip().splitlines()
    out: List[str] = []
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            out.append(parts[0].strip())
    return out


def material_column_aligned_to_manifest(
    material_xlsx: Path,
    manifest_path: Path,
    *,
    column: str,
    sheet_name: str = "Sheet2",
) -> np.ndarray:
    """
    One label per **manifest row**, joined on ``Image Index`` (manifest key).

    Row ``i`` of the pipeline bundle matches ``load_manifest_keys(manifest_path)[i]``,
    which must be the integer ``Image Index`` in the spreadsheet.
    """
    import pandas as pd

    df = pd.read_excel(Path(material_xlsx), sheet_name=sheet_name)
    if "Image Index" not in df.columns or column not in df.columns:
        raise ValueError(f"Expected 'Image Index' and '{column}' in {material_xlsx} sheet {sheet_name!r}")
    ii = df["Image Index"].astype(int)
    if ii.duplicated().any():
        dup = ii[ii.duplicated()].tolist()
        raise ValueError(f"Duplicate Image Index values in spreadsheet: {dup[:10]!r}...")
    by_idx = df.set_index(ii)

    keys = load_manifest_keys(manifest_path)
    sheet_idx = set(ii.tolist())
    manifest_idx = []
    for i, k in enumerate(keys):
        if not str(k).isdigit():
            raise ValueError(f"Manifest row {i}: key {k!r} must be an integer Image Index")
        ik = int(k)
        manifest_idx.append(ik)
        if ik not in by_idx.index:
            raise KeyError(f"Manifest row {i}: Image Index {ik} not found in spreadsheet")

    if set(manifest_idx) != sheet_idx:
        raise ValueError(
            f"Manifest Image Index set != spreadsheet set | "
            f"only in sheet: {sorted(sheet_idx - set(manifest_idx))[:20]}... "
            f"only in manifest: {sorted(set(manifest_idx) - sheet_idx)[:20]}..."
        )
    if len(keys) != len(df):
        raise ValueError(f"Manifest rows {len(keys)} != spreadsheet rows {len(df)}")

    return np.array([int(by_idx.loc[int(k), column]) for k in keys], dtype=np.int64)
