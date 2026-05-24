from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strategy_dashboard.data import load_case_data, load_cgm_excel, load_config
from strategy_dashboard.pipeline import run_case_analysis

CONFIG_PATH = ROOT / "configs" / "cgm_healthcare.json"
SAMPLE_PATH = ROOT / "data" / "sample" / "synthetic_cgm_posts.csv"

st.set_page_config(page_title="Strategy Intelligence Dashboard", layout="wide")


@st.cache_data(show_spinner=False)
def load_analysis(source_path: str, source_type: str, weights: dict[str, float]) -> dict:
    config = load_config(CONFIG_PATH)
    if source_type == "raw_cgm":
        frame = load_cgm_excel(source_path)
    else:
        frame = load_case_data(source_path)
    return run_case_analysis(frame, config, weights=weights, include_models=True)


def metric_row(data: pd.DataFrame, benchmark: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Posts", f"{len(data):,}")
    c2.metric("Sources", f"{data['source'].nunique():,}")
    c3.metric("Brands", f"{benchmark['brand'].nunique():,}")
    c4.metric("Mixed/Negative", f"{data['sentiment'].isin(['Mixed', 'Negative']).mean() * 100:.1f}%")


def sidebar_weights() -> dict[str, float]:
    st.sidebar.header("Scenario weights")
    return {
        "volume": st.sidebar.slider("Market size", 0.0, 1.0, 0.25, 0.05),
        "sentiment_risk": st.sidebar.slider("Sentiment risk", 0.0, 1.0, 0.30, 0.05),
        "urgency": st.sidebar.slider("Customer urgency", 0.0, 1.0, 0.25, 0.05),
        "brand_gap": st.sidebar.slider("Brand gap", 0.0, 1.0, 0.20, 0.05),
    }


def load_source_selector() -> tuple[str, str]:
    st.sidebar.header("Data source")
    raw_path = os.environ.get("STRATEGY_DASHBOARD_RAW_XLSX", "")
    use_raw = raw_path and Path(raw_path).exists()
    source_label = "Local raw CGM Excel" if use_raw else "Public synthetic demo"
    st.sidebar.caption(f"Active source: {source_label}")
    if use_raw:
        return raw_path, "raw_cgm"
    return str(SAMPLE_PATH), "sample"


def executive_summary(analysis: dict) -> None:
    data = analysis["data"]
    scores = analysis["scores"]
    benchmark = analysis["benchmark"]
    brief = analysis["brief"]
    metric_row(data, benchmark)
    st.subheader("Executive readout")
    st.info(brief["headline"])
    st.write(brief["recommendation"])

    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(
            px.bar(
                scores.head(8),
                x="opportunity_score",
                y="category",
                color="brand",
                orientation="h",
                title="Top ranked opportunity areas",
                labels={"opportunity_score": "Opportunity score", "category": ""},
            ),
            use_container_width=True,
        )
    with right:
        st.dataframe(
            benchmark[["brand", "posts", "positive_share", "mixed_or_negative_share", "issue_burden"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "positive_share": st.column_config.ProgressColumn("Positive", format="%.0f%%", min_value=0, max_value=1),
                "mixed_or_negative_share": st.column_config.ProgressColumn("Risk", format="%.0f%%", min_value=0, max_value=1),
            },
        )


def market_signals(analysis: dict) -> None:
    data = analysis["data"].copy()
    data["month"] = data["date"].dt.to_period("M").dt.to_timestamp()
    st.subheader("Market signals")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(data.groupby("month").size().reset_index(name="posts"), x="month", y="posts", title="Conversation volume"), use_container_width=True)
    with c2:
        st.plotly_chart(px.histogram(data, x="source", color="sentiment", title="Source and sentiment mix"), use_container_width=True)
    st.plotly_chart(px.bar(analysis["needs"], x="mentions", y="category", orientation="h", title="Patient need frequency"), use_container_width=True)


def segment_explorer(analysis: dict) -> None:
    st.subheader("Customer segments")
    segments = analysis["segments"]
    topics = analysis["topics"]
    if segments.empty:
        st.warning("Not enough rows to build segments.")
        return
    left, right = st.columns([1, 1])
    with left:
        st.plotly_chart(px.bar(segments, x="size", y="persona", orientation="h", color="dominant_brand", title="Segment size and dominant brand"), use_container_width=True)
    with right:
        st.dataframe(segments, hide_index=True, use_container_width=True)
    if not topics.empty:
        st.subheader("Topic model")
        st.dataframe(topics, hide_index=True, use_container_width=True)


def brand_benchmark(analysis: dict) -> None:
    st.subheader("Brand benchmark")
    benchmark = analysis["benchmark"]
    score_cols = ["brand", "posts", "positive_share", "negative_share", "mixed_or_negative_share", "issue_burden"]
    st.dataframe(benchmark[score_cols], hide_index=True, use_container_width=True)
    melted = benchmark.drop(columns=["posts", "positive_share", "negative_share", "mixed_or_negative_share", "issue_burden"]).melt("brand", var_name="need", value_name="share")
    st.plotly_chart(px.bar(melted, x="need", y="share", color="brand", barmode="group", title="Need coverage by brand"), use_container_width=True)


def opportunity_matrix(analysis: dict) -> None:
    st.subheader("Opportunity matrix")
    scores = analysis["scores"]
    st.plotly_chart(
        px.scatter(
            scores,
            x="share_of_brand",
            y="sentiment_risk",
            size="mentions",
            color="brand",
            hover_name="category",
            title="Issue burden vs sentiment risk",
            labels={"share_of_brand": "Share of brand conversation", "sentiment_risk": "Sentiment risk"},
        ),
        use_container_width=True,
    )
    st.dataframe(scores, hide_index=True, use_container_width=True)


def recommendation_brief(analysis: dict) -> None:
    st.subheader("Recommendation brief")
    brief = analysis["brief"]
    st.markdown(f"**Headline:** {brief['headline']}")
    st.markdown(f"**Priority moves:** {brief['priority_moves']}")
    st.markdown(f"**Recommendation:** {brief['recommendation']}")
    st.markdown(f"**Risk note:** {brief['risk_note']}")
    st.download_button(
        "Download brief",
        "\n\n".join([brief["headline"], brief["priority_moves"], brief["recommendation"], brief["risk_note"]]),
        file_name="strategy_brief.txt",
    )


def main() -> None:
    st.title("Strategy Intelligence Dashboard")
    st.caption("A consulting analytics suite that turns customer and market text into executive decisions.")
    source_path, source_type = load_source_selector()
    weights = sidebar_weights()
    analysis = load_analysis(source_path, source_type, weights)

    page = st.sidebar.radio(
        "View",
        ["Executive Summary", "Market Signals", "Segment Explorer", "Brand Benchmark", "Opportunity Matrix", "Recommendation Brief"],
    )
    if page == "Executive Summary":
        executive_summary(analysis)
    elif page == "Market Signals":
        market_signals(analysis)
    elif page == "Segment Explorer":
        segment_explorer(analysis)
    elif page == "Brand Benchmark":
        brand_benchmark(analysis)
    elif page == "Opportunity Matrix":
        opportunity_matrix(analysis)
    else:
        recommendation_brief(analysis)


if __name__ == "__main__":
    main()
