# CGM Patient Voice Strategy Brief

## Executive Summary

The Strategy Intelligence Dashboard converts messy customer text into a consultant-ready view of market signals, segment needs, brand risks, and prioritized product moves. The flagship CGM case analyzes patient voice around Dexcom and Freestyle Libre and translates NLP outputs into a strategy scorecard.

The local full dataset contains 37,844 posts dated March 2021 to September 2022. The public repository excludes that raw export because it contains source URLs, author fields, handles, IDs, and location metadata. The dashboard ships with synthetic sample rows that demonstrate the workflow without exposing private or proprietary records.

## What The Dashboard Measures

- Market signals: conversation volume, source mix, sentiment mix, and patient-need frequency.
- Customer segments: TF-IDF/NMF topics and KMeans personas that group users by decision drivers.
- Brand benchmark: Dexcom vs Freestyle Libre sentiment, issue burden, and need coverage.
- Opportunity score: a weighted strategy metric combining volume, sentiment risk, urgency, and brand gap.
- Recommendation brief: deterministic executive guidance generated only from aggregate metrics.

## Strategic Read

Dexcom tends to win where real-time safety, caregiver visibility, and pump integration matter. Its opportunity risks concentrate around price, adhesive comfort, app connectivity, and moments where accuracy failures damage trust.

Freestyle Libre tends to win where affordability, simplicity, and lower device burden matter. Its opportunity risks concentrate around alert depth, family sharing, onboarding clarity, and confidence in readings during exercise or overnight lows.

## Recommended Moves

1. Prioritize safety-critical fixes first: alerts, false lows, and app connectivity deserve heavier weighting than cosmetic experience issues.
2. Treat affordability as a market-access strategy, not just a pricing complaint.
3. Build education around known clinical ambiguity: sensor lag, calibration, warmup, trend arrows, and fingerstick comparisons.
4. Convert ecosystem strengths into messaging: Dexcom should defend pump/caregiver integration; Libre should defend simplicity and cost.
5. Validate the dashboard readout with recent data, patient interviews, and clinician review before product investment.

## Portfolio Positioning

This project is intentionally broader than a notebook. It demonstrates reusable data-product design: schema validation, text normalization, taxonomy labeling, unsupervised learning, strategy scoring, scenario weighting, and executive synthesis.
