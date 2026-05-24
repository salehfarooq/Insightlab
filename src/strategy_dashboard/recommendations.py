from __future__ import annotations

import pandas as pd


def executive_brief(scores: pd.DataFrame, benchmark: pd.DataFrame) -> dict[str, str]:
    """Create a deterministic boardroom-style brief from aggregate outputs."""
    top = scores.head(3)
    leader = benchmark.sort_values(["positive_share", "posts"], ascending=False).head(1)
    leader_name = leader["brand"].iat[0] if not leader.empty else "the leading brand"

    priorities = [
        f"{row.brand}: {row.category} ({row.opportunity_score:.1f})"
        for row in top.itertuples(index=False)
    ]
    priority_sentence = "; ".join(priorities)

    return {
        "headline": f"{leader_name} leads the current signal set, but the highest-value moves sit in targeted issue reduction.",
        "priority_moves": priority_sentence,
        "recommendation": (
            "Treat the top-ranked categories as a 90-day product and messaging sprint: fix the operational friction, "
            "publish clearer education, and reinforce the brand proof points where customer language is already strongest."
        ),
        "risk_note": (
            "This dashboard uses public-style customer text as directional strategy evidence; decisions should be validated "
            "with recent market data, interviews, and domain expert review before investment."
        ),
    }
