from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ["id", "date", "text", "source", "brand", "sentiment"]

RAW_CGM_COLUMNS = {
    "Post ID": "id",
    "Published Date (GMT-04:00) New York": "date",
    "Sound Bite Text": "text",
    "Source Type": "source",
    "Sentiment": "sentiment",
}

RAW_TEXT_COLUMNS = ["Title", "Sound Bite Text"]

SENTIMENT_MAP = {
    "positives": "Positive",
    "positive": "Positive",
    "negatives": "Negative",
    "negative": "Negative",
    "neutrals": "Neutral",
    "neutral": "Neutral",
    "mixed": "Mixed",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON case-study config."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_table(path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name or 0)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def load_case_data(path: str | Path, *, sheet_name: str | None = None) -> pd.DataFrame:
    """Load data already shaped to the dashboard schema."""
    frame = _read_table(path, sheet_name=sheet_name)
    return validate_case_frame(frame)


def load_cgm_excel(path: str | Path, *, sheet_name: str = "Stream") -> pd.DataFrame:
    """Load the local CGM Excel export without retaining author or URL fields."""
    raw = pd.read_excel(path, sheet_name=sheet_name)
    missing = [column for column in RAW_CGM_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Raw CGM export is missing expected columns: {missing}")

    frame = raw.rename(columns=RAW_CGM_COLUMNS)
    title = raw.get("Title", "").fillna("").astype(str)
    body = raw.get("Sound Bite Text", "").fillna("").astype(str)
    frame["text"] = (title.str.strip() + ". " + body.str.strip()).str.strip(". ")
    frame["brand"] = "Unknown"
    frame = frame[REQUIRED_COLUMNS]
    return validate_case_frame(frame)


def validate_case_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the reusable case-study schema."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = frame[REQUIRED_COLUMNS].copy()
    out["id"] = out["id"].astype(str)
    try:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", format="mixed")
    except TypeError:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["text"] = out["text"].fillna("").astype(str).str.strip()
    out["source"] = out["source"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    out["brand"] = out["brand"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    out["sentiment"] = (
        out["sentiment"]
        .fillna("Neutral")
        .astype(str)
        .str.strip()
        .str.lower()
        .map(SENTIMENT_MAP)
        .fillna("Neutral")
    )
    out = out[out["text"].str.len() > 0].drop_duplicates(subset="id").reset_index(drop=True)
    return out


def pii_columns_present(frame: pd.DataFrame) -> list[str]:
    """Return sensitive columns that should never appear in public samples."""
    risky_terms = ("author", "handle", "url", "location", "id ", "karma", "follower", "subscriber")
    return [column for column in frame.columns if any(term in column.lower() for term in risky_terms)]
