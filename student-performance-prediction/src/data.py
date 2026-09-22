"""Download official UCI records, validate the schema, and split reproducibly."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, DATA_URL, SEED, STAGES, ZIP_URL


def validate_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Reject invalid labels/schema; normalize headers and retain feature NaNs."""
    frame = frame.copy()
    frame.columns = (frame.columns.str.strip()
                     .str.replace("1st sem (", "1st semester (", regex=False)
                     .str.replace("2nd sem (", "2nd semester (", regex=False))
    if frame.columns.duplicated().any():
        raise ValueError("Duplicate column names found after stripping whitespace.")
    required = STAGES["first_semester"] + ["Gender", "Target"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if frame.empty or frame["Target"].isna().any():
        raise ValueError("Dataset must contain rows with non-missing outcome labels.")
    unknown = set(frame["Target"]) - {"Dropout", "Graduate", "Enrolled"}
    if unknown:
        raise ValueError(f"Unexpected target labels: {sorted(unknown)}")
    for column in required[:-1]:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Expected numeric values in '{column}'.") from exc
        if np.isinf(frame[column].dropna().to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite values found in '{column}'.")
    if not set(frame["Gender"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Gender must use the source's codes 0 or 1, or be missing.")
    if frame[STAGES["first_semester"]].isna().all().any():
        raise ValueError("A required predictor is entirely missing; median imputation is impossible.")
    return frame


def download_data(path: Path = DATA_PATH) -> Path:
    """Try the supplied CSV, then the official ZIP; never substitute synthetic data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in (DATA_URL, ZIP_URL):
        try:
            with urlopen(Request(url, headers={"User-Agent": "StudentRiskResearch/1.0"}), timeout=60) as response:
                payload = response.read()
            if url == ZIP_URL:
                with ZipFile(io.BytesIO(payload)) as archive:
                    names = [name for name in archive.namelist() if name.endswith("data.csv")]
                    if len(names) != 1:
                        raise ValueError("Official ZIP did not contain one data.csv.")
                    payload = archive.read(names[0])
            validate_data(pd.read_csv(io.BytesIO(payload), sep=None, engine="python", encoding="utf-8-sig"))
            temporary = path.with_suffix(".download")
            temporary.write_bytes(payload)
            temporary.replace(path)
            path.with_suffix(".source.json").write_text(json.dumps({"url": url, "synthetic": False}, indent=2), encoding="utf-8")
            return path
        except (OSError, ValueError, KeyError, BadZipFile) as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download the official UCI dataset. Check your internet connection.\n" + "\n".join(errors))


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Load cached official data, deduplicate before splitting, and report provenance."""
    path = Path(path)
    if not path.exists():
        download_data(path)
    frame = validate_data(pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig"))
    rows_before = len(frame)
    frame = frame.drop_duplicates().reset_index(drop=True)
    y = frame["Target"].eq("Dropout").astype(int).rename("at_risk")
    if y.nunique() != 2 or y.value_counts().min() < 20:
        raise ValueError("Insufficient records in both classes for nested five-fold calibration.")
    source_file = path.with_suffix(".source.json")
    source = json.loads(source_file.read_text(encoding="utf-8")) if source_file.exists() else {"url": DATA_URL, "provenance": "cached file; verify origin"}
    metadata = {"rows_before": rows_before, "rows": len(frame), "duplicates_removed": rows_before - len(frame),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "source": source,
                "original_target_counts": {str(k): int(v) for k, v in frame["Target"].value_counts().items()},
                "binary_target_counts": {str(k): int(v) for k, v in y.value_counts().items()},
                "missing_predictor_values": int(frame[STAGES["first_semester"]].isna().sum().sum())}
    return frame, y, metadata


def split_indices(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """One shared 80/20 stratified split for a paired comparison of both stages."""
    return train_test_split(np.arange(len(y)), test_size=0.20, stratify=y, random_state=SEED)
