from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .taxonomy import category_column

DEFAULT_WEIGHTS = {
    "volume": 0.25,
    "sentiment_risk": 0.30,
    "urgency": 0.25,
    "brand_gap": 0.20,
}

SENTIMENT_RISK = {
    "Positive": 0.0,
    "Neutral": 0.25,
    "Mixed": 0.65,
    "Negative": 1.0,
}


def normalize_weights(weights: Mapping[str, float] | None = None) -> dict[str, float]:
    merged = DEFAULT_WEIGHTS | dict(weights or {})
    total = sum(max(value, 0) for value in merged.values())
    if total <= 0:
        return DEFAULT_WEIGHTS.copy()
    return {key: max(value, 0) / total for key, value in merged.items()}


def opportunity_scores(
    frame: pd.DataFrame,
    taxonomy: Mapping[str, Mapping[str, object]],
    *,
    brands: list[str],
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Rank brand-category opportunities through consulting-weighted signals."""
    weights = normalize_weights(weights)
    rows = []
    max_mentions = 1

    for brand in brands:
        brand_frame = frame[frame["brand"].eq(brand)]
        for category, spec in taxonomy.items():
            column = category_column(category)
            mentions = int(brand_frame[column].sum()) if column in brand_frame else 0
            max_mentions = max(max_mentions, mentions)
            issue_frame = brand_frame[brand_frame[column]] if column in brand_frame else brand_frame.iloc[0:0]
            sentiment_risk = issue_frame["sentiment"].map(SENTIMENT_RISK).mean()
            rows.append(
                {
                    "brand": brand,
                    "category": category,
                    "mentions": mentions,
                    "share_of_brand": mentions / max(len(brand_frame), 1),
                    "sentiment_risk": float(sentiment_risk) if pd.notna(sentiment_risk) else 0.25,
                    "urgency": float(spec.get("urgency", 0.5)),
                }
            )

    scored = pd.DataFrame(rows)
    scored["volume_score"] = scored["mentions"] / max_mentions

    gap_values = []
    for _, row in scored.iterrows():
        competitors = scored[(scored["category"].eq(row["category"])) & (~scored["brand"].eq(row["brand"]))]
        competitor_floor = competitors["share_of_brand"].min() if not competitors.empty else 0
        gap_values.append(max(row["share_of_brand"] - competitor_floor, 0))
    scored["brand_gap"] = gap_values
    max_gap = scored["brand_gap"].max()
    scored["brand_gap"] = scored["brand_gap"] / max_gap if max_gap else 0.0

    scored["opportunity_score"] = (
        weights["volume"] * scored["volume_score"]
        + weights["sentiment_risk"] * scored["sentiment_risk"]
        + weights["urgency"] * scored["urgency"]
        + weights["brand_gap"] * scored["brand_gap"]
    ) * 100
    scored["priority"] = pd.cut(
        scored["opportunity_score"],
        bins=[-1, 45, 65, 100],
        labels=["Monitor", "Improve", "Priority"],
    ).astype(str)
    return scored.sort_values("opportunity_score", ascending=False).reset_index(drop=True)


def brand_benchmark(
    frame: pd.DataFrame,
    taxonomy: Mapping[str, Mapping[str, object]],
    *,
    brands: list[str],
) -> pd.DataFrame:
    rows = []
    for brand in brands:
        sub = frame[frame["brand"].eq(brand)]
        row = {
            "brand": brand,
            "posts": len(sub),
            "positive_share": sub["sentiment"].eq("Positive").mean() if len(sub) else 0,
            "negative_share": sub["sentiment"].eq("Negative").mean() if len(sub) else 0,
            "mixed_or_negative_share": sub["sentiment"].isin(["Mixed", "Negative"]).mean() if len(sub) else 0,
        }
        issue_columns = [category_column(category) for category in taxonomy]
        row["issue_burden"] = sub[issue_columns].sum(axis=1).mean() if len(sub) else 0
        for category in taxonomy:
            column = category_column(category)
            row[category] = sub[column].mean() if len(sub) else 0
        rows.append(row)
    return pd.DataFrame(rows)
