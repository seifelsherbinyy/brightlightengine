"""Unit tests for Brightlight forecasting helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from brightlight.forecasting import Standardizer, StandardizerConfig, run_prophet_forecast


def _calendar_fixture(tmp_path: Path, weeks: int = 6) -> Path:
    """Create a simple unified calendar fixture for testing."""

    start = pd.Timestamp("2024-01-01")
    rows = []
    for i in range(weeks):
        week_start = start + pd.Timedelta(days=7 * i)
        rows.append({"canonical_week_id": f"2024W{i+1:02d}", "week_start_date": week_start})

    path = tmp_path / "unified_calendar_map.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_gap_filling(tmp_path: Path) -> None:
    """Ensure missing calendar weeks are present and filled with NaN values."""

    calendar_path = _calendar_fixture(tmp_path, weeks=4)
    data = pd.DataFrame(
        {
            "asin_id": ["A1", "A1"],
            "canonical_week_id": ["2024W01", "2024W03"],
            "week_start_date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-15")],
            "gms": [10.0, 30.0],
        }
    )
    cfg = StandardizerConfig(calendar_path=calendar_path, history_weeks=4, metrics=["gms"])

    standardized = Standardizer.create_weekly_index(data, cfg)

    assert standardized["canonical_week_id"].tolist() == ["2024W01", "2024W02", "2024W03", "2024W04"]
    missing_week = standardized.loc[standardized["canonical_week_id"] == "2024W02", "value"]
    assert missing_week.isna().all(), "Gap-filled week should be NaN"


def test_rolling_window_trim(tmp_path: Path) -> None:
    """Verify that only the most recent calendar window is retained."""

    calendar_path = _calendar_fixture(tmp_path, weeks=8)
    data = pd.DataFrame(
        {
            "asin_id": ["A1"] * 6,
            "canonical_week_id": [f"2024W{i:02d}" for i in range(1, 7)],
            "week_start_date": [pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * (i - 1)) for i in range(1, 7)],
            "units": [1, 2, 3, 4, 5, 6],
        }
    )
    cfg = StandardizerConfig(calendar_path=calendar_path, history_weeks=4, metrics=["units"])

    standardized = Standardizer.create_weekly_index(data, cfg)

    assert standardized["canonical_week_id"].tolist() == ["2024W03", "2024W04", "2024W05", "2024W06"]
    assert standardized["value"].tolist() == [3.0, 4.0, 5.0, 6.0]


def test_forecast_output_shape(tmp_path: Path) -> None:
    """Confirm forecast output contains expected columns and rows."""

    calendar_path = _calendar_fixture(tmp_path, weeks=8)
    data = pd.DataFrame(
        {
            "asin_id": ["A1"] * 5,
            "canonical_week_id": [f"2024W0{i}" for i in range(1, 6)],
            "week_start_date": [pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * (i - 1)) for i in range(1, 6)],
            "units": [2, 3, 4, 5, 6],
        }
    )
    cfg = StandardizerConfig(calendar_path=calendar_path, history_weeks=5, metrics=["units"])

    standardized = Standardizer.create_weekly_index(data, cfg)
    forecast = run_prophet_forecast(standardized, horizon_weeks=4)

    assert set(forecast.columns) == {"asin_id", "metric", "forecast_week", "forecast_value", "lower_bound", "upper_bound"}
    assert len(forecast) == 4
    assert (forecast["asin_id"] == "A1").all()
    assert (forecast["metric"] == "units").all()
    assert forecast[["forecast_value", "lower_bound", "upper_bound"]].apply(np.isfinite).all().all()
