from pathlib import Path

import numpy as np
import pandas as pd

from brightstar.scoring.weights import compute_dynamic_weights


def test_dynamic_weights_basic_behavior():
    # Build a small quality frame for two entities, one with strong quality
    dfq = pd.DataFrame(
        [
            {  # Strong signal: complete, fresh, strong trend
                "entity_id": "E1",
                "metric": "Units",
                "Week_Order": 10,
                "completeness": 1.0,
                "recency_weeks": 0.0,
                "variance": 1.0,
                "trend_quality": 1.5,
            },
            {  # Weak signal: sparse, stale, weak trend
                "entity_id": "E2",
                "metric": "Units",
                "Week_Order": 10,
                "completeness": 0.2,
                "recency_weeks": 8.0,
                "variance": 5.0,
                "trend_quality": 0.1,
            },
        ]
    )

    base = {"Units": 0.4}
    out = compute_dynamic_weights(dfq, base, lookback_weeks=8, min_multiplier=0.5, max_multiplier=1.25)

    assert set(out.columns) >= {
        "entity_id",
        "metric",
        "Week_Order",
        "weight_effective",
        "weight_multiplier",
        "base_weight",
        "quality_completeness",
        "quality_recency_factor",
        "quality_trend_norm",
    }

    # E1 should have higher multiplier than E2
    mul = out.set_index("entity_id")["weight_multiplier"]
    assert mul.loc["E1"] > mul.loc["E2"]

    # Multipliers should respect bounds
    assert (mul >= 0.5 - 1e-9).all() and (mul <= 1.25 + 1e-9).all()

    # Effective weights tie to base weight
    eff = out.set_index("entity_id")["weight_effective"]
    assert np.isclose(eff.loc["E1"], 0.4 * mul.loc["E1"])  # base * multiplier
