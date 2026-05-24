# Strategy Intelligence Dashboard

A reusable consulting analytics suite that turns messy customer and market text into executive decisions.

The flagship case study is **CGM Patient Voice Intelligence**: a healthcare strategy dashboard that compares Dexcom and Freestyle Libre through patient-needs taxonomy, sentiment risk, topic modeling, customer segmentation, opportunity scoring, and boardroom-style recommendation briefs.

## Why This Exists

Most NLP notebooks stop at charts. This project pushes the analysis into a consulting workflow:

- What are customers actually saying?
- Which issues are frequent, urgent, and strategically material?
- Which brand is exposed to which risk?
- What should leadership prioritize first?
- How do recommendations change when strategy weights shift?

## Dashboard Views

- **Executive Summary:** KPI cards, top opportunity areas, benchmark table, and recommendation readout.
- **Market Signals:** conversation volume, source mix, sentiment mix, and patient-need frequency.
- **Segment Explorer:** KMeans personas and NMF topic labels from customer text.
- **Brand Benchmark:** Dexcom vs Freestyle Libre sentiment, issue burden, and need coverage.
- **Opportunity Matrix:** volume, risk, urgency, and brand gap combined into a ranked score.
- **Recommendation Brief:** deterministic executive memo generated from aggregate metrics only.

## Quick Start

```bash
cd /Users/hp/Desktop/strategy-intelligence-dashboard
python -m pytest
PYTHONPATH=src streamlit run app/streamlit_app.py
```

By default, the app runs on synthetic public demo data:

```text
data/sample/synthetic_cgm_posts.csv
```

To run the local private CGM Excel export without copying it into the repo:

```bash
export STRATEGY_DASHBOARD_RAW_XLSX="/absolute/path/to/local_cgm_export.xlsx"
PYTHONPATH=src streamlit run app/streamlit_app.py
```

## Public Data Boundary

The raw CGM export is deliberately excluded from this repository. It includes source URLs, author names, handles, IDs, and location-style metadata. The public repo uses a synthetic sample file with the same reusable schema:

```text
id,date,text,source,brand,sentiment
```

## Methodology

1. Validate a reusable case-study schema.
2. Normalize text and map noisy sentiment labels.
3. Infer brand mentions from configured entity patterns.
4. Apply a configurable patient-needs taxonomy.
5. Derive NMF topics and KMeans customer segments.
6. Benchmark brands by sentiment, issue burden, and need coverage.
7. Rank opportunities through volume, sentiment risk, urgency, and brand gap.
8. Generate a deterministic recommendation brief from aggregate metrics.

## Project Layout

```text
app/                         Streamlit dashboard
configs/cgm_healthcare.json  Case-study taxonomy and scoring weights
data/sample/                 Synthetic public demo data
notebooks/                   Technical narrative notebook
reports/                     Consulting-style strategy brief
src/strategy_dashboard/      Reusable analytics package
tests/                       Pytest coverage for the analytics layer
```

## CV Positioning

**AI/ML:** Streamlit/Plotly consulting analytics suite over 37,844 CGM patient posts with NLP-derived customer segments, brand benchmarks, opportunity scores, and executive recommendations.

**Consulting/Ops:** Advanced market-intelligence dashboard ranking Dexcom vs Freestyle Libre opportunities through customer segmentation, issue-burden scoring, scenario weighting, and executive recommendation briefs.
