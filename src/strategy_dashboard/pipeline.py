from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .data import validate_case_frame
from .modeling import derive_segments, derive_topics
from .recommendations import executive_brief
from .scoring import brand_benchmark, opportunity_scores
from .taxonomy import add_brand_tags, add_taxonomy_flags, category_summary
from .text import add_clean_text


def run_case_analysis(
    frame: pd.DataFrame,
    config: Mapping[str, object],
    *,
    weights: Mapping[str, float] | None = None,
    include_models: bool = True,
) -> dict[str, pd.DataFrame | dict[str, str]]:
    """Run the full consulting intelligence pipeline."""
    taxonomy = config["taxonomy"]
    brands = list(config.get("brands", []))
    default_weights = dict(config.get("default_weights", {}))
    active_weights = default_weights | dict(weights or {})

    data = validate_case_frame(frame)
    data = add_clean_text(data)
    data = add_brand_tags(data)
    data = add_taxonomy_flags(data, taxonomy)

    topics = pd.DataFrame()
    segments = pd.DataFrame()
    if include_models and len(data) >= 3:
        data, topics = derive_topics(data)
        data, segments = derive_segments(data)

    scores = opportunity_scores(data, taxonomy, brands=brands, weights=active_weights)
    benchmark = brand_benchmark(data, taxonomy, brands=brands)
    needs = category_summary(data, taxonomy)
    brief = executive_brief(scores, benchmark)

    return {
        "data": data,
        "topics": topics,
        "segments": segments,
        "scores": scores,
        "benchmark": benchmark,
        "needs": needs,
        "brief": brief,
    }
