"""Download official research data and validate anonymous four-factor CSVs."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import pandas as pd
from src.config import DATA_PATH, DATA_URL, SCHEMAS


def download_uci(path: Path = DATA_PATH) -> None:
    """Save exact official mathematics CSV bytes; no synthetic fallback."""
    with urlopen(DATA_URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    if "student.zip" in archive.namelist():
        archive = zipfile.ZipFile(io.BytesIO(archive.read("student.zip")))
    member = next(name for name in archive.namelist() if name.endswith("student-mat.csv"))
    payload = archive.read(member)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.with_suffix(".source.json").write_text(json.dumps({
        "source_url": DATA_URL, "dataset": "UCI Student Performance, mathematics only",
        "doi": "https://doi.org/10.24432/C5TG7T", "license": "CC BY 4.0",
        "synthetic": False, "sha256": hashlib.sha256(payload).hexdigest(),
    }, indent=2), encoding="utf-8")


def validate_data(frame: pd.DataFrame, schema: str, pass_mark: float | None = None):
    """Keep only declared anonymous numeric columns and enforce a usable target."""
    spec = SCHEMAS[schema]
    pass_mark = spec["pass_mark"] if pass_mark is None else float(pass_mark)
    if not np.isfinite(pass_mark) or not 0 < pass_mark <= spec["target_max"]:
        raise ValueError(f"Pass mark must be above 0 and at most {spec['target_max']}.")
    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names are not allowed.")
    columns = spec["features"] + [spec["target"]]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}.")
    # Remove exact source duplicates before discarding non-predictive columns.
    # Different students with identical predictor values remain distinct observations.
    original_count = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    duplicates = original_count - len(frame)
    frame = frame.loc[:, columns].copy()
    for col in columns:
        try:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{col} must contain numbers, with blanks only for missing inputs.") from exc
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("Infinite values are not allowed.")
    if frame[spec["target"]].isna().any():
        raise ValueError("Final marks cannot be missing: they define the outcome.")
    for col, (lo, hi) in zip(columns, spec["bounds"] + [(0, spec["target_max"])]):
        values = frame[col].dropna()
        if ((values < lo) | (values > hi)).any():
            raise ValueError(f"{col} must be between {lo} and {hi}.")
        if schema == "uci" and (values != np.floor(values)).any():
            raise ValueError(f"{col} must contain whole numbers.")
        if values.empty:
            raise ValueError(f"{col} cannot be entirely missing.")
    y = frame[spec["target"]].ge(pass_mark).astype(int).rename("pass")
    if len(frame) < 50 or len(y.unique()) != 2 or y.value_counts().min() < 15:
        raise ValueError("Provide at least 50 rows, including at least 15 passes and 15 fails after duplicate removal.")
    if len(frame) > 10000:
        raise ValueError("Use at most 10,000 rows for this interactive research app.")
    return frame, y, {
        "schema": schema, "rows": len(frame), "duplicates_removed": duplicates,
        "missing_inputs": int(frame[spec["features"]].isna().sum().sum()),
        "pass_mark": pass_mark, "pass_count": int(y.sum()), "fail_count": int((1-y).sum()),
        "target_definition": f"Pass = {spec['target']} >= {pass_mark:g}; Fail = below {pass_mark:g}.",
    }


def load_data(path: Path = DATA_PATH, schema: str = "uci", pass_mark: float | None = None):
    path = Path(path)
    if not path.exists():
        if schema == "uci" and path == DATA_PATH:
            download_uci(path)
        else:
            raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    clean, y, metadata = validate_data(frame, schema, pass_mark)
    metadata["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata["source"] = ("Official UCI mathematics dataset; real observational records"
                          if schema == "uci" else "User-provided CSV; provenance must be verified by the owner")
    return clean, y, metadata


def load_uploaded(payload: bytes, pass_mark: float = 50):
    if len(payload) > 5 * 1024 * 1024:
        raise ValueError("CSV must be 5 MB or smaller.")
    try:
        frame = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig", sep=None, engine="python")
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise ValueError("Could not read the CSV. Use a UTF-8 comma-separated file with the template headers.") from exc
    frame, y, meta = validate_data(frame, "custom", pass_mark)
    meta["sha256"] = hashlib.sha256(payload).hexdigest()
    meta["source"] = "User-provided CSV; source and collection timing are not independently verified"
    return frame, y, meta
