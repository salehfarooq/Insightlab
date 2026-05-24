from __future__ import annotations

import re
from collections.abc import Mapping

import pandas as pd

BRAND_PATTERNS = {
    "Dexcom": [r"\bdexcom\b", r"\bg6\b", r"\bg7\b"],
    "Freestyle Libre": [r"\blibre\b", r"\bfreestyle\b", r"\bfreestyle libre\b"],
}


def infer_brand(text: str, existing: str = "Unknown") -> str:
    """Infer brand mentions from text while respecting explicit labels."""
    explicit = str(existing).strip()
    if explicit and explicit.lower() not in {"unknown", "other", "nan"}:
        return explicit

    lower = str(text).lower()
    matches = [
        brand
        for brand, patterns in BRAND_PATTERNS.items()
        if any(re.search(pattern, lower) for pattern in patterns)
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        return "Both"
    return "Other / Unspecified"


def add_brand_tags(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["brand"] = [infer_brand(text, brand) for text, brand in zip(out["text"], out["brand"])]
    return out


def _keyword_regex(keywords: list[str]) -> re.Pattern[str]:
    escaped = [re.escape(keyword.lower()) for keyword in keywords]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b")


def add_taxonomy_flags(frame: pd.DataFrame, taxonomy: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    """Add one boolean need-category column per taxonomy entry."""
    out = frame.copy()
    if "clean_text" not in out.columns:
        out["clean_text"] = out["text"].astype(str).str.lower()

    for category, spec in taxonomy.items():
        keywords = list(spec.get("keywords", []))
        pattern = _keyword_regex(keywords)
        column = category_column(category)
        out[column] = out["clean_text"].map(lambda text: bool(pattern.search(str(text))))
    return out


def category_column(category: str) -> str:
    safe = re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
    return f"need_{safe}"


def category_summary(frame: pd.DataFrame, taxonomy: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    rows = []
    for category in taxonomy:
        column = category_column(category)
        count = int(frame[column].sum()) if column in frame else 0
        rows.append(
            {
                "category": category,
                "mentions": count,
                "share": count / max(len(frame), 1),
                "urgency": float(taxonomy[category].get("urgency", 0.5)),
            }
        )
    return pd.DataFrame(rows).sort_values(["mentions", "urgency"], ascending=False)
