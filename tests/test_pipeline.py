from pathlib import Path

import pandas as pd

from strategy_dashboard.data import load_case_data, load_config, pii_columns_present
from strategy_dashboard.pipeline import run_case_analysis
from strategy_dashboard.scoring import opportunity_scores
from strategy_dashboard.taxonomy import add_brand_tags, add_taxonomy_flags
from strategy_dashboard.text import add_clean_text, clean_text

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT / "configs" / "cgm_healthcare.json")
SAMPLE = ROOT / "data" / "sample" / "synthetic_cgm_posts.csv"


def test_sample_data_matches_public_schema_without_pii_columns():
    frame = pd.read_csv(SAMPLE)
    assert set(CONFIG["schema"]).issubset(frame.columns)
    assert pii_columns_present(frame) == []


def test_load_case_data_normalizes_sentiment_and_dates():
    frame = load_case_data(SAMPLE)
    assert len(frame) == 36
    assert frame["date"].notna().all()
    assert set(frame["sentiment"]).issubset({"Positive", "Neutral", "Mixed", "Negative"})


def test_clean_text_removes_urls_and_punctuation():
    assert clean_text("Dexcom app crashed! See https://example.com") == "dexcom app crashed see"


def test_brand_and_taxonomy_tagging_detects_core_signals():
    frame = load_case_data(SAMPLE).head(4)
    frame = add_clean_text(add_brand_tags(frame))
    frame = add_taxonomy_flags(frame, CONFIG["taxonomy"])
    assert "Dexcom" in set(frame["brand"])
    assert "Freestyle Libre" in set(frame["brand"])
    assert frame.filter(like="need_").sum().sum() > 0


def test_opportunity_scores_are_ranked_and_weighted():
    frame = load_case_data(SAMPLE)
    frame = add_clean_text(add_brand_tags(frame))
    frame = add_taxonomy_flags(frame, CONFIG["taxonomy"])
    scores = opportunity_scores(frame, CONFIG["taxonomy"], brands=CONFIG["brands"])
    assert scores["opportunity_score"].is_monotonic_decreasing
    assert scores["opportunity_score"].between(0, 100).all()
    assert {"brand", "category", "priority"}.issubset(scores.columns)


def test_full_pipeline_returns_consulting_outputs():
    frame = load_case_data(SAMPLE)
    analysis = run_case_analysis(frame, CONFIG, include_models=True)
    assert not analysis["scores"].empty
    assert not analysis["benchmark"].empty
    assert not analysis["segments"].empty
    assert "recommendation" in analysis["brief"]
