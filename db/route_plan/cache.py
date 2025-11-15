"""Local cache helpers used by route planning APIs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Mapping, Optional

import pandas as pd

from .logger import LOGGER


def _coerce_decimal_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ``Decimal`` objects to ``float`` for parquet compatibility."""
    if df is None or df.empty:
        return df
    for column in df.columns:
        series = df[column]
        if series.dtype == "object" and series.map(lambda value: hasattr(value, "as_tuple"), na_action="ignore").any():
            df[column] = series.map(lambda value: float(value) if hasattr(value, "as_tuple") else value)
    return df


def _df_to_parquet(df: pd.DataFrame, path: Path) -> None:
    """Persist ``df`` as parquet with a pickle fallback when pyarrow is missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialisable = _coerce_decimal_to_float(df.copy())
    try:
        serialisable.to_parquet(path, index=False)
    except Exception as exc:  # pragma: no cover - triggered when pyarrow is missing
        LOGGER.debug("Parquet export failed (%s), falling back to pickle", exc)
        serialisable.to_pickle(path.with_suffix(".pkl"))


def _df_from_parquet(path: Path) -> pd.DataFrame:
    """Load a dataframe stored via :func:`_df_to_parquet`."""
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:  # pragma: no cover - rely on pickle fallback
            fallback = path.with_suffix(".pkl")
            if fallback.exists():
                return pd.read_pickle(fallback)
            raise
    fallback = path.with_suffix(".pkl")
    if fallback.exists():
        return pd.read_pickle(fallback)
    raise FileNotFoundError(path)


def _schema_fingerprint(frames: Mapping[str, pd.DataFrame]) -> str:
    """Compute a lightweight fingerprint based on dataframe column schema."""
    import hashlib
    import json

    payload = []
    for name, frame in frames.items():
        if not isinstance(frame, pd.DataFrame):
            continue
        payload.append((name, tuple((column, str(frame[column].dtype)) for column in frame.columns)))
    return hashlib.md5(json.dumps(payload, ensure_ascii=False).encode("utf-8")).hexdigest()


@dataclass
class CacheManifest:
    """Metadata stored alongside cached tables."""

    n34_id: str
    saved_at: str
    tables: Dict[str, Dict[str, str]]
    pandas_version: str
    schema_fingerprint: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "n34_id": self.n34_id,
            "saved_at": self.saved_at,
            "tables": self.tables,
            "pandas_version": self.pandas_version,
            "schema_fingerprint": self.schema_fingerprint,
        }


def save_tables_local(n34_id: str, tables: Mapping[str, pd.DataFrame], cache_dir: str = "cache_vrp") -> Path:
    """Persist ``tables`` under ``cache_dir/n34_id``.

    Each dataframe is stored as parquet.  Metadata about the stored tables is
    written to ``manifest.json`` so we can reload them later.
    """

    import json

    root = Path(cache_dir) / str(n34_id)
    root.mkdir(parents=True, exist_ok=True)

    table_entries: Dict[str, Dict[str, str]] = {}
    for name, frame in tables.items():
        if frame is None:
            continue
        output = root / f"{name}.parquet"
        _df_to_parquet(frame, output)
        table_entries[name] = {"file": output.name, "type": "parquet"}

    manifest = CacheManifest(
        n34_id=str(n34_id),
        saved_at=datetime.now().isoformat(timespec="seconds"),
        tables=table_entries,
        pandas_version=pd.__version__,
        schema_fingerprint=_schema_fingerprint({k: v for k, v in tables.items() if isinstance(v, pd.DataFrame)}),
    )

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return root


def save_tables_or_str_local(
    n34_id: str,
    tables: Mapping[str, object],
    cache_dir: str = "cache_vrp",
) -> Path:
    """Persist a mixture of dataframes, strings and picklable objects."""

    import json
    import pickle

    root = Path(cache_dir) / str(n34_id)
    root.mkdir(parents=True, exist_ok=True)

    frame_tables: Dict[str, pd.DataFrame] = {}
    string_tables: Dict[str, str] = {}
    object_tables: Dict[str, object] = {}

    for name, value in tables.items():
        if value is None:
            continue
        if isinstance(value, pd.DataFrame):
            frame_tables[name] = value
        elif isinstance(value, str):
            string_tables[name] = value
        else:
            object_tables[name] = value

    manifest_tables: Dict[str, Dict[str, str]] = {}

    for name, frame in frame_tables.items():
        output = root / f"{name}.parquet"
        _df_to_parquet(frame, output)
        manifest_tables[name] = {"file": output.name, "type": "parquet"}

    for name, text in string_tables.items():
        output = root / f"{name}.txt"
        output.write_text(text, encoding="utf-8")
        manifest_tables[name] = {"file": output.name, "type": "text"}

    for name, obj in object_tables.items():
        output = root / f"{name}.pkl"
        with output.open("wb") as handle:
            pickle.dump(obj, handle)
        manifest_tables[name] = {"file": output.name, "type": "binary"}

    manifest = CacheManifest(
        n34_id=str(n34_id),
        saved_at=datetime.now().isoformat(timespec="seconds"),
        tables=manifest_tables,
        pandas_version=pd.__version__,
        schema_fingerprint=_schema_fingerprint(frame_tables) if frame_tables else None,
    )

    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return root


def load_tables_local(
    n34_id: str,
    cache_dir: str = "cache_vrp",
    *,
    strict: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load previously persisted tables from :func:`save_tables_local`."""

    import json

    root = Path(cache_dir) / str(n34_id)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest for n34_id={n34_id!r} under {root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    tables: Dict[str, pd.DataFrame] = {}
    for name, meta in manifest.get("tables", {}).items():
        file_name = meta["file"] if isinstance(meta, dict) else meta
        try:
            tables[name] = _df_from_parquet(root / file_name)
        except Exception as exc:  # pragma: no cover - user controlled inputs
            if strict:
                raise
            LOGGER.warning("Failed to load cached table %s: %s", name, exc)
    return tables


def load_tables_or_str_local(
    n34_id: str,
    cache_dir: str = "cache_vrp",
    *,
    strict: bool = False,
) -> Dict[str, object]:
    """Load artefacts persisted via :func:`save_tables_or_str_local`."""

    import json
    import pickle

    root = Path(cache_dir) / str(n34_id)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest for n34_id={n34_id!r} under {root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    results: Dict[str, object] = {}

    for name, meta in manifest.get("tables", {}).items():
        if isinstance(meta, str):
            file_name = meta
            file_type = "parquet"
        else:
            file_name = meta.get("file")
            file_type = meta.get("type", "parquet")
        try:
            path = root / file_name
            if file_type == "parquet":
                results[name] = _df_from_parquet(path)
            elif file_type == "text":
                results[name] = path.read_text(encoding="utf-8")
            elif file_type == "binary":
                with path.open("rb") as handle:
                    results[name] = pickle.load(handle)
            else:
                raise ValueError(f"Unsupported cache file type: {file_type}")
        except Exception as exc:  # pragma: no cover - user controlled inputs
            if strict:
                raise
            LOGGER.warning("Failed to load cached artefact %s (%s): %s", name, file_type, exc)
    return results


__all__ = [
    "CacheManifest",
    "load_tables_local",
    "load_tables_or_str_local",
    "save_tables_local",
    "save_tables_or_str_local",
]
