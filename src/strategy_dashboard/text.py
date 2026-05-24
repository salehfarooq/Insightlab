from __future__ import annotations

import re

import pandas as pd

URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
NON_WORD_RE = re.compile(r"[^a-z0-9\s]")
SPACE_RE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    """Lowercase and normalize noisy social/customer text."""
    text = URL_RE.sub(" ", str(value).lower())
    text = NON_WORD_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()


def add_clean_text(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["clean_text"] = out["text"].map(clean_text)
    return out
