"""Tests for Phase 4 forecasting engine."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import pytest

from brightstar.phase4_forecasting import run_phase4
from brightstar.forecasting_utils import load_metadata


@pytest.fixture(scope="module")
def forecasting_results() -> Dict[str, pd.DataFrame]:
    return run_phase4("config.yaml")


def test_phase4_generates_forecasts(forecasting_results: Dict[str, pd.DataFrame]) -> None:
    vendor = forecasting_results["vendor"]
    asin = forecasting_results["asin"]

    assert not vendor.empty, "Vendor forecasts should not be empty for sample dataset"
    assert not asin.empty, "ASIN forecasts should not be empty for sample dataset"

    required_columns = {
        "entity_type",
        "vendor_code",
        "metric",
        "forecast_week",
        "forecast_value",
        "potential_gain",
        "methods",
        "status",
        "forecast_horizon_weeks",
        "delta_value",
        "delta_pct",
        "risk_gain_flag",
        "Forecast_Risk_Flag",
        "week_offset",
    }
    assert required_columns.issubset(vendor.columns)
    assert required_columns.issubset(asin.columns)
    assert vendor["Forecast_Risk_Flag"].dropna().between(0, 4).all()


def test_phase4_metadata_written(tmp_path: Path, forecasting_results: Dict[str, pd.DataFrame]) -> None:
    metadata = load_metadata("config.yaml")
    assert metadata.get("horizons_weeks") == [4, 6, 12]
    assert metadata.get("records_vendor", 0) >= len(forecasting_results["vendor"])
    assert "timestamp" in metadata
    assert "methods_used" in metadata and "fallback" in metadata["methods_used"]


def test_phase4_directional_gain_non_negative(forecasting_results: Dict[str, pd.DataFrame]) -> None:
    vendor = forecasting_results["vendor"]
    so_roos = vendor[vendor["metric"] == "SoROOS(%)"]
    if not so_roos.empty:
        assert (so_roos["potential_gain"] >= -1e-9).all()

