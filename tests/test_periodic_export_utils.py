from __future__ import annotations

from pathlib import Path

import pandas as pd

from brightstar.periodic_export_utils import (
    derive_period_column,
    get_recent_periods,
    filter_period,
    export_periods_to_tabs,
    export_periods_to_files,
)


def _sample_scorecard() -> pd.DataFrame:
    # Minimal columns typical for scorecards plus a date column
    return pd.DataFrame(
        {
            "vendor_code": ["V1", "V1", "V2", "V2", "V3"],
            "metric": ["GMS($)", "GMS($)", "GMS($)", "GMS($)", "GMS($)"],
            "Composite_Score": [800, 820, 600, 610, 900],
            "week_start": [
                "2025-10-06",  # ISO week 41
                "2025-10-13",  # ISO week 42
                "2025-10-13",
                "2025-11-03",  # ISO week 45, month 2025-11
                "2025-11-10",  # ISO week 46, month 2025-11
            ],
        }
    )


def test_weekly_period_extraction():
    df = _sample_scorecard()
    out = derive_period_column(df, frequency="weekly")
    assert "period_key" in out.columns
    assert out.loc[0, "period_key"].startswith("2025-W")
    # Check two known rows map to specific iso weeks
    assert out.loc[1, "period_key"].endswith("42")
    assert out.loc[4, "period_key"].endswith("46")


def test_monthly_period_extraction():
    df = _sample_scorecard()
    out = derive_period_column(df, frequency="monthly")
    assert "period_key" in out.columns
    # Expect YYYY-MM format
    assert out.loc[0, "period_key"] == "2025-10"
    assert out.loc[3, "period_key"] == "2025-11"


def test_get_recent_periods_limiting():
    df = _sample_scorecard()
    out = derive_period_column(df, frequency="weekly")
    periods = get_recent_periods(out, max_periods=2)
    # Should return at most 2 periods, sorted desc
    assert len(periods) == 2
    assert periods == sorted(periods, reverse=True)


def test_filter_period_roundtrip():
    df = derive_period_column(_sample_scorecard(), frequency="weekly")
    periods = get_recent_periods(df, max_periods=10)
    assert periods, "Expected at least one period"
    p = periods[0]
    sub = filter_period(df, p)
    assert not sub.empty
    assert set(sub["period_key"].unique()) == {p}


def test_export_tabs(tmp_path: Path):
    df = derive_period_column(_sample_scorecard(), frequency="weekly")
    periods = get_recent_periods(df, max_periods=3)
    out_dir = tmp_path / "exports"
    path = export_periods_to_tabs(df, periods, out_dir)
    assert path is not None
    assert Path(path).exists()


def test_export_files(tmp_path: Path):
    df = derive_period_column(_sample_scorecard(), frequency="monthly")
    periods = get_recent_periods(df, max_periods=5)
    out_dir = tmp_path / "exports_files"
    paths = export_periods_to_files(df, periods, out_dir)
    assert len(paths) == len(periods)
    for p in paths:
        assert p.exists()
