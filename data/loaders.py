"""Dataset loading utilities for CSV/JSON social post datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict

import pandas as pd


REQUIRED_FIELDS = {"post_id", "text"}


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "post_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "post_id"})
        else:
            df["post_id"] = [str(i) for i in range(1, len(df) + 1)]

    if "text" not in df.columns:
        raise ValueError("Input dataset must contain a 'text' column.")

    df = df[["post_id", "text"]].copy()
    df["post_id"] = df["post_id"].astype(str)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    return df


def load_dataset(dataset_path: str | Path) -> List[Dict[str, str]]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("data", [])
        df = pd.DataFrame(payload)
    else:
        raise ValueError("Unsupported dataset format. Use CSV or JSON.")

    df = _standardize_columns(df)
    return df.to_dict(orient="records")


def _standardize_gold(df: pd.DataFrame) -> pd.DataFrame:
    if "post_id" not in df.columns:
        if "id" in df.columns:
            df = df.rename(columns={"id": "post_id"})
        else:
            df["post_id"] = [str(i) for i in range(1, len(df) + 1)]
    if "text" not in df.columns:
        raise ValueError("Gold set must contain a 'text' column.")
    if "label" in df.columns:
        pass
    elif "gold" in df.columns:
        df = df.rename(columns={"gold": "label"})
    else:
        raise ValueError("Gold set must include a 'label' or 'gold' column with expected class name.")
    keep = [c for c in ("post_id", "text", "label", "subtype", "slice_type") if c in df.columns]
    df = df[keep].copy()
    df["post_id"] = df["post_id"].astype(str)
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""]
    return df


def load_labeled_goldset(dataset_path: str | Path) -> List[Dict[str, str]]:
    """Like load_dataset but requires *label* (or *gold*); kept for evaluation."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            payload = payload.get("data", [])
        df = pd.DataFrame(payload)
    else:
        raise ValueError("Unsupported format. Use CSV or JSON.")
    df = _standardize_gold(df)
    return df.to_dict(orient="records")
