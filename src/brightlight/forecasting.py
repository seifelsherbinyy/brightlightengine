from __future__ import annotations

"""Forecasting utilities for Brightlight Phase 4.

This module provides helpers to standardize weekly time series aligned to the
unified calendar and produce baseline forecasts with Prophet or a lightweight
linear regression fallback.
"""

import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:  # Optional dependency
    from prophet import Prophet
except Exception:  # pragma: no cover - Prophet is optional
    Prophet = None  # type: ignore[misc]

try:  # Lightweight fallback
    from sklearn.linear_model import LinearRegression
except Exception:  # pragma: no cover - allow pure-numpy fallback
    LinearRegression = None  # type: ignore[misc]

logger = logging.getLogger(__name__)

CalendarPath = Union[str, Path]
SourceType = Union[str, Path, pd.DataFrame]


@dataclass
class StandardizerConfig:
    """Configuration options for weekly standardization.

    Attributes
    ----------
    week_column:
        Column representing the canonical week identifier produced during
        earlier phases.
    date_column:
        Column containing week start dates aligned to the unified calendar.
    asin_column:
        Column containing the ASIN identifier.
    metrics:
        Optional list of metric columns to include. If ``None``, metrics are
        inferred from numeric columns excluding identifiers and calendar fields.
    calendar_path:
        Path to the unified calendar map CSV (e.g., ``unified_calendar_map.csv``).
    history_weeks:
        Number of trailing weeks to retain for model input. Defaults to ``52``
        to align with a one-year rolling history window.
    """

    week_column: str = "canonical_week_id"
    date_column: str = "week_start_date"
    asin_column: str = "asin_id"
    metrics: Optional[Sequence[str]] = None
    calendar_path: CalendarPath = Path("data/reference/unified_calendar_map.csv")
    history_weeks: int = 52


class Standardizer:
    """Standardize ASIN × metric time series to the unified weekly calendar."""

    @staticmethod
    def _load_calendar(calendar_path: CalendarPath) -> pd.DataFrame:
        """Load the unified calendar map.

        Parameters
        ----------
        calendar_path:
            Path to the unified calendar CSV containing ``canonical_week_id`` and
            ``week_start_date``.

        Returns
        -------
        pd.DataFrame
            Calendar rows sorted by week start date.
        """

        calendar = pd.read_csv(calendar_path)
        if "week_start_date" in calendar.columns:
            calendar["week_start_date"] = pd.to_datetime(calendar["week_start_date"])
        calendar = calendar.sort_values("week_start_date")
        return calendar

    @staticmethod
    def _load_source(source: SourceType) -> pd.DataFrame:
        """Load a Phase 1/2 artifact into a ``DataFrame``.

        Parameters
        ----------
        source:
            DataFrame instance, parquet path, or directory containing Phase 2 outputs.

        Returns
        -------
        pd.DataFrame
            Loaded dataframe with at least ASIN, metric, and calendar fields.
        """

        if isinstance(source, pd.DataFrame):
            return source.copy()

        path = Path(source)
        if path.is_dir():
            for name in ("scoring_detailed.parquet", "validated.parquet"):
                candidate = path / name
                if candidate.exists():
                    return pd.read_parquet(candidate)
            raise FileNotFoundError(f"No Phase 1/2 artifact found in {path}")

        if path.suffix in {".parquet", ".pq"} and path.exists():
            return pd.read_parquet(path)

        raise FileNotFoundError(f"Unsupported source: {source}")

    @classmethod
    def _detect_metrics(cls, df: pd.DataFrame, asin_column: str, week_column: str) -> List[str]:
        """Infer metric columns by excluding identifier and calendar fields."""

        exclude = {asin_column, week_column, "week_start_date", "vendor_id", "canonical_week_id"}
        numeric_cols = [col for col, dtype in df.dtypes.items() if np.issubdtype(dtype, np.number)]
        return [col for col in numeric_cols if col not in exclude]

    @classmethod
    def create_weekly_index(cls, source: SourceType, config: Optional[StandardizerConfig] = None) -> pd.DataFrame:
        """Build a gap-filled weekly index aligned to the unified calendar.

        The method performs the following steps:
        - Loads Phase 2 outputs or ``validated.parquet``.
        - Expands to a global week index using the unified calendar map.
        - Trims to the most recent ``history_weeks`` window.
        - Fills missing weeks with ``NaN`` (never zeros).

        Parameters
        ----------
        source:
            DataFrame, parquet path, or directory containing Phase 2 outputs.
        config:
            Optional :class:`StandardizerConfig` controlling column names, metrics,
            calendar path, and history window.

        Returns
        -------
        pd.DataFrame
            Long-format dataframe with columns ``asin_id``, ``metric``,
            ``canonical_week_id``, ``week_start_date``, and ``value``.
        """

        cfg = config or StandardizerConfig()
        df = cls._load_source(source)
        calendar = cls._load_calendar(cfg.calendar_path)

        metrics = list(cfg.metrics) if cfg.metrics else cls._detect_metrics(df, cfg.asin_column, cfg.week_column)

        if cfg.date_column not in df.columns and cfg.week_column in df.columns:
            df = df.merge(calendar[[cfg.week_column, cfg.date_column]], how="left", on=cfg.week_column)

        df = df[[cfg.asin_column, cfg.week_column, cfg.date_column] + metrics].copy()
        df[cfg.date_column] = pd.to_datetime(df[cfg.date_column])
        df = df.dropna(subset=[cfg.week_column, cfg.date_column])

        # Determine rolling window using calendar alignment
        calendar = calendar.dropna(subset=[cfg.date_column])
        calendar = calendar.sort_values(cfg.date_column).reset_index(drop=True)
        max_date = df[cfg.date_column].max()
        calendar_dates = calendar[cfg.date_column]
        match_idx = calendar_dates[calendar_dates == max_date]
        if not match_idx.empty:
            anchor_idx = int(match_idx.index[0])
        else:
            anchor_idx = int(calendar_dates.searchsorted(max_date, side="right") - 1)
            anchor_idx = max(anchor_idx, 0)

        start_idx = max(0, anchor_idx - cfg.history_weeks + 1)
        end_idx = min(len(calendar), start_idx + cfg.history_weeks)
        week_index = calendar.loc[start_idx:end_idx - 1, [cfg.week_column, cfg.date_column]].reset_index(drop=True)

        melted = df.melt(
            id_vars=[cfg.asin_column, cfg.week_column, cfg.date_column],
            value_vars=metrics,
            var_name="metric",
            value_name="value",
        )

        # Attach full index per ASIN × metric
        unique_keys = melted[[cfg.asin_column, "metric"]].drop_duplicates()
        cartesian = (
            unique_keys.assign(key=1)
            .merge(week_index.assign(key=1), on="key")
            .drop(columns="key")
        )

        merged = cartesian.merge(
            melted,
            how="left",
            on=[cfg.asin_column, "metric", cfg.week_column, cfg.date_column],
            suffixes=("", "_orig"),
        )

        merged = merged[[cfg.asin_column, "metric", cfg.week_column, cfg.date_column, "value"]].sort_values(
            [cfg.asin_column, "metric", cfg.date_column]
        )
        return merged


def _fit_linear_regression(history: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    """Simple linear-regression forecast used when Prophet is unavailable.

    Parameters
    ----------
    history:
        Historical frame with columns ``week_start_date`` and ``value``.
    horizon_weeks:
        Number of weeks to forecast into the future.

    Returns
    -------
    pd.DataFrame
        Forecast frame containing ``forecast_week``, ``forecast_value``,
        ``lower_bound``, and ``upper_bound``.
    """

    history = history.dropna(subset=["value"]).reset_index(drop=True)
    if history.empty:
        return pd.DataFrame(columns=["forecast_value", "lower_bound", "upper_bound", "forecast_week"])

    x = np.arange(len(history)).reshape(-1, 1)
    y = history["value"].to_numpy()
    if LinearRegression is not None:
        model = LinearRegression()
        model.fit(x, y)
        preds = model.predict(np.arange(len(history), len(history) + horizon_weeks).reshape(-1, 1))
        fitted = model.predict(x)
    else:  # numpy polyfit fallback
        slope, intercept = np.polyfit(np.arange(len(history)), y, 1)
        preds = intercept + slope * np.arange(len(history), len(history) + horizon_weeks)
        fitted = intercept + slope * np.arange(len(history))

    residuals = y - fitted
    std = float(np.std(residuals)) if residuals.size else 0.0
    lower = preds - 1.96 * std
    upper = preds + 1.96 * std

    start_date = history["week_start_date"].max()
    forecast_weeks = [start_date + timedelta(days=7 * i) for i in range(1, horizon_weeks + 1)]
    return pd.DataFrame(
        {
            "forecast_week": forecast_weeks,
            "forecast_value": preds,
            "lower_bound": lower,
            "upper_bound": upper,
        }
    )


def run_prophet_forecast(
    standardized: pd.DataFrame,
    horizon_weeks: int = 4,
    date_column: str = "week_start_date",
) -> pd.DataFrame:
    """Run a baseline forecast per ASIN × metric using Prophet with fallback.

    Parameters
    ----------
    standardized:
        Output of :func:`Standardizer.create_weekly_index` containing
        ``asin_id``, ``metric``, ``week_start_date``, and ``value`` columns.
    horizon_weeks:
        Number of future weeks to forecast.
    date_column:
        Column containing week start datetime values.

    Returns
    -------
    pd.DataFrame
        Forecast rows with columns ``asin_id``, ``metric``, ``forecast_week``,
        ``forecast_value``, ``lower_bound``, ``upper_bound``.
    """

    results: List[pd.DataFrame] = []
    for (asin, metric), group in standardized.groupby(["asin_id", "metric"]):
        history = group.dropna(subset=[date_column]).sort_values(date_column)
        history = history[[date_column, "value"]].rename(columns={date_column: "ds", "value": "y"})

        if Prophet is not None and not history.empty:
            try:
                model = Prophet(interval_width=0.95, weekly_seasonality=False, daily_seasonality=False)
                model.fit(history)
                future = model.make_future_dataframe(periods=horizon_weeks, freq="W-MON", include_history=False)
                forecast = model.predict(future)
                segment = pd.DataFrame(
                    {
                        "forecast_week": forecast["ds"],
                        "forecast_value": forecast["yhat"],
                        "lower_bound": forecast["yhat_lower"],
                        "upper_bound": forecast["yhat_upper"],
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Prophet failed for asin=%s metric=%s: %s", asin, metric, exc)
                temp_history = history.rename(columns={"ds": date_column, "y": "value"})
                segment = _fit_linear_regression(temp_history, horizon_weeks)
        else:
            temp_history = history.rename(columns={"ds": date_column, "y": "value"})
            segment = _fit_linear_regression(temp_history, horizon_weeks)

        segment["asin_id"] = asin
        segment["metric"] = metric
        results.append(segment)

    if not results:
        return pd.DataFrame(columns=["asin_id", "metric", "forecast_week", "forecast_value", "lower_bound", "upper_bound"])

    output = pd.concat(results, ignore_index=True)
    cols = ["asin_id", "metric", "forecast_week", "forecast_value", "lower_bound", "upper_bound"]
    return output[cols]
