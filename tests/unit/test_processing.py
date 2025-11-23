from pathlib import Path

import numpy as np
import pandas as pd

from brightstar.scoring.processing import (
    add_grain_columns,
    compute_derived_metrics,
    detect_outliers_mad_iqr,
    winsorize_by_limits,
    add_rolling_features,
    compute_signal_quality,
)


def make_base_df() -> pd.DataFrame:
    data = [
        {"vendor_code": "V1", "vendor_name": "Vendor One", "asin": "A1", "metric": "Shipped Revenue", "value": 200.0, "week_label": "2025W40"},
        {"vendor_code": "V1", "vendor_name": "Vendor One", "asin": "A1", "metric": "Shipped COGS", "value": 120.0, "week_label": "2025W40"},
        {"vendor_code": "V1", "vendor_name": "Vendor One", "asin": "A1", "metric": "Units", "value": 10.0, "week_label": "2025W40"},
        {"vendor_code": "V1", "vendor_name": "Vendor One", "asin": "A1", "metric": "Shipped Revenue", "value": 300.0, "week_label": "2025W41"},
        {"vendor_code": "V1", "vendor_name": "Vendor One", "asin": "A1", "metric": "Shipped COGS", "value": 150.0, "week_label": "2025W41"},
        {"vendor_code": "V1", "vendor_name": "Vendor One", "asin": "A1", "metric": "Units", "value": 15.0, "week_label": "2025W41"},
    ]
    return pd.DataFrame(data)


def test_compute_derived_metrics_basic():
    df = make_base_df()
    derived = compute_derived_metrics(df)
    assert not derived.empty
    # Expect CP, ASP, CM %, Net PPM % present
    metrics_present = set(derived["metric"].unique())
    assert {"CP", "ASP", "CM %", "Net PPM %"}.issubset(metrics_present)

    # Validate simple math on one week
    week40 = derived[derived["week_label"] == "2025W40"].set_index("metric")["value"]
    assert np.isclose(week40["CP"], 200 - 120)
    assert np.isclose(week40["ASP"], 200 / 10)
    assert np.isclose(week40["CM %"], (200 - 120) / 200)


def test_add_grain_columns_week_label_only():
    df = make_base_df()
    enriched = add_grain_columns(df)
    assert "year" in enriched.columns
    assert int(enriched["year"].dropna().unique()[0]) == 2025


def test_outlier_detection_and_winsorization():
    df = pd.DataFrame(
        {
            "metric": ["Units"] * 6 + ["Shipped Revenue"] * 6,
            "value": [1, 1, 1, 1, 1, 100, 10, 10, 10, 10, 10, 1000],
        }
    )
    # MAD flags extreme values
    units_flags = detect_outliers_mad_iqr(df[df["metric"] == "Units"]["value"], method="mad")
    assert units_flags.any()
    # Winsorize per metric
    limits = {"Units": (0.05, 0.95), "Shipped Revenue": (0.05, 0.95)}
    w = winsorize_by_limits(df.assign(value=pd.to_numeric(df["value"])), limits)
    assert w["value"].max() <= df["value"].quantile(0.95) + 1e-9


def test_rolling_features_and_signal_quality():
    # Build a small long frame with Week_Order
    base = make_base_df()
    # Map two weeks to Week_Order
    base["Week_Order"] = base["week_label"].map({"2025W40": 1, "2025W41": 2})
    base["entity_id"] = base["vendor_code"] + "_" + base["asin"]
    # Focus on one metric to simplify
    m = base[base["metric"] == "Shipped Revenue"][
        ["entity_id", "metric", "Week_Order", "value"]
    ].copy()
    feats = add_rolling_features(m, window=2)
    assert {"volatility", "trend_slope", "trend_strength"}.issubset(set(feats.columns))
    qual = compute_signal_quality(m, lookback_weeks=2)
    tail_row = qual.sort_values("Week_Order").iloc[-1]
    assert "completeness" in tail_row.index and not pd.isna(tail_row["completeness"])  # should be 1.0
