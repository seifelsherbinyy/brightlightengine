"""Utility helpers supporting Phase 4 forecasting workflows."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .ingestion_utils import ensure_directory, load_config
from .path_utils import get_phase1_normalized_path


LOGGER_NAME = "brightstar.phase4"


@dataclass
class ForecastContext:
    """Configuration bundle required to generate forecasts."""

    horizons: Sequence[int]
    min_history: int
    frequency: str
    metrics: Sequence[str]
    ensemble_weights: Dict[str, float]
    metric_directions: Dict[str, str]
    uncertainty_buffer: float
    accuracy_backtest: int
    risk_gain_thresholds: Dict[str, float]


# =============================
# Topic 4: Simple, offline forecasting helpers
# =============================

def build_time_series(
    normalized_df: pd.DataFrame,
    metric_list: Sequence[str],
    min_history_weeks: int,
) -> pd.DataFrame:
    """Build tidy weekly time series per entity and metric.

    Input is normalized Phase 1 data with columns at least:
      vendor_code, asin (optional), metric, value, week_start

    Returns a DataFrame with columns:
      entity_type, entity_id, vendor_code, asin, metric, week_start_date, value, history_weeks

    Entities with fewer than ``min_history_weeks`` non-null observations are dropped.
    The function is empty-safe and will return an empty DataFrame on missing input.
    """

    if normalized_df is None or normalized_df.empty:
        return pd.DataFrame(
            columns=[
                "entity_type",
                "entity_id",
                "vendor_code",
                "asin",
                "metric",
                "week_start_date",
                "value",
                "history_weeks",
            ]
        )

    df = normalized_df.copy()
    # Coerce date
    if "week_start" not in df.columns:
        # Try common alternatives; if not found, return empty
        for cand in ("Week_Start", "week_label", "Week_Label"):
            if cand in df.columns:
                if cand.lower().endswith("label"):
                    # Best effort: parse label as date start
                    df["week_start"] = pd.to_datetime(df[cand], errors="coerce")
                else:
                    df["week_start"] = pd.to_datetime(df[cand], errors="coerce")
                break
    df["week_start"] = pd.to_datetime(df.get("week_start"), errors="coerce")
    if "week_start" not in df.columns or df["week_start"].isna().all():
        return pd.DataFrame(
            columns=[
                "entity_type",
                "entity_id",
                "vendor_code",
                "asin",
                "metric",
                "week_start_date",
                "value",
                "history_weeks",
            ]
        )

    # Filter metrics
    metrics = set(map(str, metric_list or []))
    if metrics:
        df = df[df["metric"].astype(str).isin(metrics)]

    # Ensure identifiers
    if "asin" not in df.columns:
        df["asin"] = pd.NA

    # Build vendor-level series
    vendor = (
        df.groupby(["vendor_code", "metric", "week_start"], as_index=False)["value"].mean()
        .assign(entity_type="vendor", asin=pd.NA)
        .rename(columns={"week_start": "week_start_date"})
    )
    vendor["entity_id"] = vendor["vendor_code"].astype(str)

    # Build ASIN-level series (only rows with asin present)
    asin_df = df.dropna(subset=["asin"]).copy()
    asin = (
        asin_df.groupby(["vendor_code", "asin", "metric", "week_start"], as_index=False)["value"].mean()
        .assign(entity_type="asin")
        .rename(columns={"week_start": "week_start_date"})
    )
    asin["entity_id"] = asin["vendor_code"].astype(str) + "_" + asin["asin"].astype(str)

    combined = pd.concat([vendor, asin], ignore_index=True, sort=False)
    if combined.empty:
        return combined.assign(history_weeks=pd.Series(dtype=int))

    # Compute history count per series and filter
    combined.sort_values(["entity_type", "entity_id", "metric", "week_start_date"], inplace=True)
    counts = (
        combined.groupby(["entity_type", "entity_id", "metric"])['value']
        .apply(lambda s: s.dropna().shape[0])
        .reset_index(name="history_weeks")
    )
    combined = combined.merge(counts, on=["entity_type", "entity_id", "metric"], how="left")
    if min_history_weeks and int(min_history_weeks) > 0:
        combined = combined[combined["history_weeks"] >= int(min_history_weeks)]
    return combined.reset_index(drop=True)


def apply_exp_smoothing(series: pd.DataFrame, horizons_weeks: Sequence[int]) -> pd.DataFrame:
    """Apply a very simple exponential smoothing to a single series.

    Expects ``series`` columns: week_start_date, value, entity_type, entity_id, vendor_code, asin, metric.
    Produces one row per requested horizon with columns:
      entity_type, entity_id, vendor_code, asin, metric,
      forecast_week, forecast_value, forecast_horizon_weeks, method_name
    """

    if series is None or series.empty:
        return pd.DataFrame(
            columns=[
                "entity_type",
                "entity_id",
                "vendor_code",
                "asin",
                "metric",
                "forecast_week",
                "forecast_value",
                "forecast_horizon_weeks",
                "method_name",
            ]
        )

    s = series.sort_values("week_start_date").copy()
    values = pd.to_numeric(s["value"], errors="coerce")
    # Fixed alpha for transparency; avoids auto-tuning
    alpha = 0.3
    # EWM: last smoothed value as level
    smoothed = values.ewm(alpha=alpha, adjust=False).mean()
    level = float(smoothed.iloc[-1]) if not smoothed.empty else float(values.iloc[-1])
    # Naive drift using last 4 deltas
    deltas = values.diff().dropna()
    drift = float(deltas.tail(4).mean()) if not deltas.empty else 0.0

    last_week = pd.to_datetime(s["week_start_date"].iloc[-1])
    out_rows: List[Dict[str, object]] = []
    for h in sorted(set(int(x) for x in horizons_weeks if int(x) > 0)):
        forecast_week = last_week + pd.to_timedelta(7 * h, unit="D")
        forecast_value = level + drift * h
        out_rows.append(
            {
                "entity_type": s["entity_type"].iloc[0],
                "entity_id": s["entity_id"].iloc[0],
                "vendor_code": s["vendor_code"].iloc[0],
                "asin": s.get("asin", pd.Series([pd.NA])).iloc[0],
                "metric": s["metric"].iloc[0],
                "forecast_week": forecast_week,
                "forecast_value": float(forecast_value),
                "forecast_horizon_weeks": int(h),
                "method_name": "exp_smoothing_v1",
            }
        )

    return pd.DataFrame(out_rows)


def estimate_reliability(series: pd.DataFrame, residuals: Optional[pd.Series] = None) -> str:
    """Estimate a simple reliability label based on history length and volatility.

    Rules (transparent, hand-crafted):
      - history >= 52 and coefvar <= 0.2 -> high
      - history >= 24 and coefvar <= 0.35 -> medium
      - else -> low
    """
    if series is None or series.empty:
        return "low"

    values = pd.to_numeric(series["value"], errors="coerce").dropna()
    n = int(series.get("history_weeks", pd.Series([len(values)])).iloc[0]) if not series.empty else len(values)
    mean = float(values.mean()) if len(values) > 0 else 0.0
    std = float(values.std(ddof=0)) if len(values) > 1 else 0.0
    coefvar = (std / mean) if mean else 1.0

    if n >= 52 and coefvar <= 0.2:
        return "high"
    if n >= 24 and coefvar <= 0.35:
        return "medium"
    return "low"


def setup_logging(config: Dict) -> logging.Logger:
    """Configure logging for the forecasting phase."""

    log_dir = ensure_directory(config["paths"]["logs_dir"])
    file_name = config.get("logging", {}).get("forecasting_file_name", "phase4_forecasting.log")
    log_path = log_dir / file_name

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, config.get("logging", {}).get("forecasting_level", "INFO")))
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.debug("Forecasting logger initialised at %s", log_path)
    return logger


def load_normalized_dataset(config: Dict, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Load the normalized dataset created during Phase 1."""

    forecasting_config = config.get("forecasting", {})
    # 1) Canonical path from Phase 1 helper
    try:
        canonical = get_phase1_normalized_path(config)
        if canonical.exists():
            return pd.read_parquet(canonical)
    except Exception as exc:
        if logger:
            logger.debug("Canonical normalized parquet not available yet (%s). Trying fallbacks.", exc)

    # 2) Legacy/config-driven fallbacks
    candidate_paths: List[Path] = []
    preferred = forecasting_config.get("normalized_input")
    if preferred:
        candidate_paths.append(Path(preferred))

    default_normalized = config["paths"].get("normalized_output")
    if default_normalized:
        candidate_paths.append(Path(default_normalized))

    candidate_paths.append(Path(config["paths"]["processed_dir"]) / "normalized_phase1.csv")

    for path in candidate_paths:
        if path.exists():
            if path.suffix.lower() == ".parquet":
                try:
                    return pd.read_parquet(path)
                except Exception as exc:  # pragma: no cover - depends on parquet engines
                    if logger:
                        logger.warning(
                            "Unable to read parquet file %s (%s). Falling back to CSV if available.",
                            path,
                            exc,
                        )
                    continue
            if path.suffix.lower() in {".csv", ".txt"}:
                return pd.read_csv(path)

    raise FileNotFoundError(
        "Unable to locate normalized dataset. Ensure Phase 1 output exists before running Phase 4."
    )


def build_context(config: Dict) -> ForecastContext:
    """Assemble the forecasting context from configuration settings."""

    forecasting_config = config.get("forecasting", {})
    horizons_config = forecasting_config.get("horizons_weeks")
    if horizons_config:
        horizons = [int(value) for value in horizons_config if int(value) > 0]
    else:
        horizon = int(forecasting_config.get("horizon_weeks", 4))
        horizons = [horizon]

    if not horizons:
        raise ValueError("At least one forecast horizon must be configured.")

    min_history = int(forecasting_config.get("min_history_weeks", 3))
    frequency = str(forecasting_config.get("frequency", "W-MON"))
    metrics = list(forecasting_config.get("metrics", []))
    if not metrics:
        raise ValueError("No metrics configured for forecasting.")

    ensemble_weights = forecasting_config.get("ensemble_weights", {})
    if not ensemble_weights:
        ensemble_weights = {"fallback": 1.0}

    metric_directions = forecasting_config.get("metric_directions") or config.get("scoring", {}).get(
        "metric_directions", {}
    )
    uncertainty_buffer = float(forecasting_config.get("uncertainty_buffer_pct", 0.15))
    accuracy_backtest = int(forecasting_config.get("accuracy_backtest_weeks", 4))
    risk_gain_thresholds = {
        "gain_pct": float(forecasting_config.get("risk_gain_thresholds", {}).get("gain_pct", 0.05)),
        "risk_pct": float(forecasting_config.get("risk_gain_thresholds", {}).get("risk_pct", -0.05)),
        "gain_absolute": float(forecasting_config.get("risk_gain_thresholds", {}).get("gain_absolute", 0.0)),
    }

    return ForecastContext(
        horizons=horizons,
        min_history=min_history,
        frequency=frequency,
        metrics=metrics,
        ensemble_weights={str(k): float(v) for k, v in ensemble_weights.items()},
        metric_directions={str(k): str(v) for k, v in metric_directions.items()},
        uncertainty_buffer=uncertainty_buffer,
        accuracy_backtest=accuracy_backtest,
        risk_gain_thresholds=risk_gain_thresholds,
    )


def _prepare_series(df: pd.DataFrame, entity_columns: List[str]) -> pd.DataFrame:
    columns = entity_columns + ["metric", "week_start", "value"]
    return df[columns].copy()


def prepare_entity_frames(normalized_df: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    """Aggregate normalized data into time series per entity and metric."""

    frame = normalized_df.copy()
    # Tolerant parsing: allow various date formats without raising
    frame["week_start"] = pd.to_datetime(frame["week_start"], errors="coerce")  # ensure datetime handling

    if entity_type == "vendor":
        grouped = (
            frame.groupby(["vendor_code", "metric", "week_start"], as_index=False)["value"].sum()
        )
        grouped["entity_type"] = "vendor"
    else:
        grouped = (
            frame.groupby(["vendor_code", "asin", "metric", "week_start"], as_index=False)["value"].sum()
        )
        grouped["entity_type"] = "asin"

    return grouped.sort_values("week_start").reset_index(drop=True)


def _compute_future_index(last_week: pd.Timestamp, horizon: int, frequency: str) -> pd.DatetimeIndex:
    offset = pd.tseries.frequencies.to_offset(frequency)
    start = last_week + offset
    return pd.date_range(start=start, periods=horizon, freq=frequency)


def _prophet_forecast(history: pd.DataFrame, horizon: int, frequency: str, logger: Optional[logging.Logger]) -> Optional[pd.DataFrame]:
    try:  # pragma: no cover - prophet is optional in test environments
        from prophet import Prophet
    except Exception as exc:  # pragma: no cover - optional dependency
        if logger:
            logger.debug("Prophet unavailable (%s); skipping Prophet component.", exc)
        return None

    df = history.rename(columns={"week_start": "ds", "value": "y"})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=horizon, freq=frequency)
    forecast = model.predict(future).tail(horizon)
    result = pd.DataFrame(
        {
            "forecast_week": pd.to_datetime(forecast["ds"]),
            "forecast_value": forecast["yhat"],
            "lower_bound": forecast["yhat_lower"],
            "upper_bound": forecast["yhat_upper"],
        }
    )
    result["method"] = "prophet"
    return result


def _lstm_forecast(history: pd.DataFrame, horizon: int, frequency: str, logger: Optional[logging.Logger]) -> Optional[pd.DataFrame]:
    try:  # pragma: no cover - tensorflow is optional in test environments
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except Exception as exc:  # pragma: no cover - optional dependency
        if logger:
            logger.debug("TensorFlow unavailable (%s); skipping LSTM component.", exc)
        return None

    values = history["value"].astype(float).to_numpy()
    if len(values) < 6:
        if logger:
            logger.debug("Insufficient history (%s points) for TensorFlow forecasting.", len(values))
        return None

    sequence_length = min(6, len(values))
    X, y = [], []
    for i in range(len(values) - sequence_length):
        X.append(values[i : i + sequence_length])
        y.append(values[i + sequence_length])

    if not X:
        if logger:
            logger.debug("Not enough sequence windows for LSTM model.")
        return None

    X = np.array(X).reshape((-1, sequence_length, 1))
    y = np.array(y)

    model = models.Sequential(
        [
            layers.Input(shape=(sequence_length, 1)),
            layers.LSTM(16, activation="tanh"),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=100, verbose=0)

    history_values = values[-sequence_length:].tolist()
    predictions: List[float] = []
    for _ in range(horizon):
        input_arr = np.array(history_values[-sequence_length:]).reshape((1, sequence_length, 1))
        pred = float(model.predict(input_arr, verbose=0)[0][0])
        predictions.append(pred)
        history_values.append(pred)

    future_index = _compute_future_index(history["week_start"].iloc[-1], horizon, frequency)
    std = float(np.std(values)) if len(values) > 1 else 0.0
    result = pd.DataFrame(
        {
            "forecast_week": future_index,
            "forecast_value": predictions,
            "lower_bound": np.array(predictions) - std,
            "upper_bound": np.array(predictions) + std,
        }
    )
    result["method"] = "lstm"
    return result


def _baseline_forecast(history: pd.DataFrame, horizon: int, frequency: str, buffer_pct: float) -> pd.DataFrame:
    values = history["value"].astype(float).to_numpy()
    if len(values) >= 2:
        diffs = np.diff(values)
        trend = float(np.mean(diffs))
    else:
        trend = 0.0

    last_value = float(values[-1]) if len(values) else 0.0
    predictions = []
    current = last_value
    for _ in range(horizon):
        current = current + trend
        predictions.append(current)

    future_index = _compute_future_index(history["week_start"].iloc[-1], horizon, frequency)
    std = float(np.std(values)) if len(values) > 1 else abs(trend)
    buffer = [abs(val) * buffer_pct for val in predictions]
    lower = [val - std - buf for val, buf in zip(predictions, buffer)]
    upper = [val + std + buf for val, buf in zip(predictions, buffer)]
    result = pd.DataFrame(
        {
            "forecast_week": future_index,
            "forecast_value": predictions,
            "lower_bound": lower,
            "upper_bound": upper,
        }
    )
    result["method"] = "fallback"
    return result


def _combine_forecasts(
    forecasts: Dict[str, pd.DataFrame],
    future_index: pd.DatetimeIndex,
    weights: Dict[str, float],
) -> pd.DataFrame:
    if not forecasts:
        raise ValueError("At least one forecast method is required to combine results.")

    combined = pd.DataFrame({"forecast_week": future_index})
    requested_weights = {name: float(weights.get(name, 0.0)) for name in forecasts}
    if not any(weight > 0 for weight in requested_weights.values()):
        requested_weights = {name: 1.0 for name in forecasts}

    total_weight = 0.0
    value_accumulator = np.zeros(len(future_index))
    lower_accumulator = np.zeros(len(future_index))
    upper_accumulator = np.zeros(len(future_index))

    for name, forecast in forecasts.items():
        weight = requested_weights.get(name, 0.0)
        if weight <= 0:
            continue
        aligned = forecast.set_index("forecast_week").reindex(future_index).ffill().bfill()
        value_accumulator += aligned["forecast_value"].to_numpy() * weight
        lower_accumulator += aligned["lower_bound"].to_numpy() * weight
        upper_accumulator += aligned["upper_bound"].to_numpy() * weight
        total_weight += weight

    if total_weight == 0:
        total_weight = float(len(forecasts))
        for forecast in forecasts.values():
            aligned = forecast.set_index("forecast_week").reindex(future_index).ffill().bfill()
            value_accumulator += aligned["forecast_value"].to_numpy()
            lower_accumulator += aligned["lower_bound"].to_numpy()
            upper_accumulator += aligned["upper_bound"].to_numpy()

    combined["forecast_value"] = value_accumulator / total_weight
    combined["lower_bound"] = lower_accumulator / total_weight
    combined["upper_bound"] = upper_accumulator / total_weight
    return combined


def _directional_potential_gain(
    last_value: float,
    forecast_value: float,
    direction: str,
) -> float:
    direction_normalized = str(direction).lower()
    if direction_normalized in {"down", "desc", "lower", "decrease"}:
        return max(last_value - forecast_value, 0.0)
    return max(forecast_value - last_value, 0.0)


def _evaluate_method_accuracy(
    history: pd.DataFrame,
    builder: Callable[[pd.DataFrame, int], Optional[pd.DataFrame]],
    window: int,
    logger: Optional[logging.Logger] = None,
) -> Optional[float]:
    if window <= 0 or len(history) <= 1:
        return None

    history = history.sort_values("week_start")
    errors: List[float] = []

    for offset in range(window, 0, -1):
        training = history.iloc[: len(history) - offset]
        if len(training) < 2:
            break
        forecast = builder(training, 1)
        if forecast is None or forecast.empty:
            return None
        predicted = float(forecast.iloc[0]["forecast_value"])
        actual = history.iloc[len(history) - offset]["value"]
        if pd.isna(actual):
            continue
        errors.append(abs(float(actual) - predicted))

    if not errors:
        if logger:
            logger.debug("Unable to compute accuracy window; defaulting to configured weights.")
        return None

    return float(np.mean(errors))


def _derive_accuracy_weights(
    history: pd.DataFrame,
    builders: Dict[str, Callable[[pd.DataFrame, int], Optional[pd.DataFrame]]],
    forecasts: Dict[str, pd.DataFrame],
    context: ForecastContext,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    available_builders = {name: builders[name] for name in forecasts if name in builders}
    window = min(context.accuracy_backtest, max(0, len(history) - 1))
    errors: Dict[str, float] = {}

    for name, builder in available_builders.items():
        error = _evaluate_method_accuracy(history, builder, window, logger)
        if error is not None:
            errors[name] = error

    if not errors:
        return {name: context.ensemble_weights.get(name, 0.0) for name in forecasts}

    inverse_errors = {name: 1.0 / (error + 1e-6) for name, error in errors.items() if error >= 0}
    if not inverse_errors:
        return {name: context.ensemble_weights.get(name, 0.0) for name in forecasts}

    total = sum(inverse_errors.values())
    if total == 0:
        return {name: context.ensemble_weights.get(name, 0.0) for name in forecasts}

    return {name: inverse_errors.get(name, 0.0) / total for name in forecasts}


def _risk_gain_flag(
    delta_value: float,
    delta_pct: Optional[float],
    direction: str,
    thresholds: Dict[str, float],
) -> str:
    gain_pct = thresholds.get("gain_pct", 0.05)
    risk_pct = thresholds.get("risk_pct", -0.05)
    gain_absolute = thresholds.get("gain_absolute", 0.0)

    direction_normalized = str(direction).lower()
    adjusted_delta = delta_value
    adjusted_pct = delta_pct

    if direction_normalized in {"down", "desc", "lower", "decrease"}:
        adjusted_delta = -delta_value
        adjusted_pct = -delta_pct if delta_pct is not None else None

    if adjusted_pct is not None:
        if adjusted_pct <= risk_pct:
            return "critical"
        if adjusted_pct <= risk_pct / 2:
            return "at_risk"
        if adjusted_pct >= gain_pct * 2:
            return "high_performing"
        if adjusted_pct >= gain_pct:
            return "growing"

    if adjusted_delta <= -abs(gain_absolute):
        return "at_risk"
    if adjusted_delta >= abs(gain_absolute) and adjusted_delta > 0:
        return "growing"

    return "steady"


def _risk_flag_to_score(flag: str) -> int:
    mapping = {
        "critical": 0,
        "at_risk": 1,
        "steady": 2,
        "growing": 3,
        "high_performing": 4,
    }
    return mapping.get(str(flag).lower(), 2)





def forecast_single_entity(
    history: pd.DataFrame,
    context: ForecastContext,
    metric: str,
    entity_descriptor: Dict[str, str],
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Generate ensemble forecasts for a single entity/metric time series."""

    history = history.sort_values("week_start")
    history_weeks = len(history)
    last_week = history["week_start"].iloc[-1]
    max_horizon = max(context.horizons)
    future_index = _compute_future_index(last_week, max_horizon, context.frequency)

    status_flags: List[str] = []
    if history_weeks < context.min_history:
        status_flags.append("insufficient_history")

    builder_map: Dict[str, Callable[[pd.DataFrame, int], Optional[pd.DataFrame]]] = {
        "prophet": lambda hist, horizon: _prophet_forecast(hist, horizon, context.frequency, logger),
        "lstm": lambda hist, horizon: _lstm_forecast(hist, horizon, context.frequency, logger),
        "fallback": lambda hist, horizon: _baseline_forecast(
            hist, horizon, context.frequency, context.uncertainty_buffer
        ),
    }

    forecasts: Dict[str, pd.DataFrame] = {}
    for name, builder in builder_map.items():
        result = builder(history, max_horizon)
        if result is not None and not result.empty:
            forecasts[name] = result

    if "fallback" not in forecasts:
        forecasts["fallback"] = builder_map["fallback"](history, max_horizon)

    if set(forecasts.keys()) == {"fallback"}:
        status_flags.append("fallback")

    weights = _derive_accuracy_weights(history, builder_map, forecasts, context, logger)
    if not any(weight > 0 for weight in weights.values()):
        weights = {name: context.ensemble_weights.get(name, 0.0) for name in forecasts}

    combined = _combine_forecasts(forecasts, future_index, weights)
    combined["methods"] = ",".join(sorted(forecasts.keys()))
    combined["status"] = ",".join(sorted(status_flags)) if status_flags else "ready"

    last_value = float(history["value"].iloc[-1])
    direction = context.metric_directions.get(metric, "up")

    combined["last_observed_week"] = last_week
    combined["last_observed_value"] = last_value
    combined["potential_gain"] = combined["forecast_value"].apply(
        lambda value: _directional_potential_gain(last_value, float(value), direction)
    )
    combined["metric"] = metric
    combined["history_weeks"] = history_weeks
    combined["week_offset"] = range(1, len(combined) + 1)

    for key, value in entity_descriptor.items():
        combined[key] = value

    summary_rows: List[pd.Series] = []
    for horizon in sorted(set(context.horizons)):
        if horizon <= 0 or horizon > len(combined):
            continue
        row = combined.iloc[horizon - 1].copy()
        row["forecast_horizon_weeks"] = horizon
        delta_value = float(row["forecast_value"]) - last_value
        row["delta_value"] = round(delta_value, 4)
        row["delta_pct"] = float(delta_value / last_value) if last_value else None
        flag = _risk_gain_flag(delta_value, row["delta_pct"], direction, context.risk_gain_thresholds)
        row["risk_gain_flag"] = flag
        row["Forecast_Risk_Flag"] = _risk_flag_to_score(flag)
        summary_rows.append(row)

    if not summary_rows and not combined.empty:
        fallback_row = combined.iloc[-1].copy()
        fallback_row["forecast_horizon_weeks"] = len(combined)
        delta_value = float(fallback_row["forecast_value"]) - last_value
        fallback_row["delta_value"] = round(delta_value, 4)
        fallback_row["delta_pct"] = float(delta_value / last_value) if last_value else None
        flag = _risk_gain_flag(delta_value, fallback_row["delta_pct"], direction, context.risk_gain_thresholds)
        fallback_row["risk_gain_flag"] = flag
        fallback_row["Forecast_Risk_Flag"] = _risk_flag_to_score(flag)
        summary_rows.append(fallback_row)

    if not summary_rows:
        return pd.DataFrame(
            columns=[
                "entity_type",
                "vendor_code",
                "asin",
                "metric",
                "forecast_week",
                "forecast_value",
                "lower_bound",
                "upper_bound",
                "last_observed_week",
                "last_observed_value",
                "potential_gain",
                "history_weeks",
                "week_offset",
                "forecast_horizon_weeks",
                "delta_value",
                "delta_pct",
                "risk_gain_flag",
                "Forecast_Risk_Flag",
                "methods",
                "status",
            ]
        )

    combined_summary = pd.DataFrame(summary_rows)

    ordered_cols = [
        "entity_type",
        "vendor_code",
        "asin",
        "metric",
        "forecast_week",
        "forecast_value",
        "lower_bound",
        "upper_bound",
        "last_observed_week",
        "last_observed_value",
        "potential_gain",
        "history_weeks",
        "week_offset",
        "forecast_horizon_weeks",
        "delta_value",
        "delta_pct",
        "risk_gain_flag",
        "Forecast_Risk_Flag",
        "methods",
        "status",
    ]
    for column in ordered_cols:
        if column not in combined_summary.columns:
            combined_summary[column] = None

    return combined_summary[ordered_cols]

def generate_forecasts(
    normalized_df: pd.DataFrame,
    context: ForecastContext,
    entity_type: str,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Generate forecasts for either vendor or ASIN entity levels."""

    base_columns = [
        "entity_type",
        "vendor_code",
        "asin",
        "metric",
        "forecast_week",
        "forecast_value",
        "lower_bound",
        "upper_bound",
        "last_observed_week",
        "last_observed_value",
        "potential_gain",
        "history_weeks",
        "week_offset",
        "forecast_horizon_weeks",
        "delta_value",
        "delta_pct",
        "risk_gain_flag",
        "Forecast_Risk_Flag",
        "methods",
        "status",
    ]

    if normalized_df.empty:
        return pd.DataFrame(columns=base_columns)

    prepared = prepare_entity_frames(normalized_df, entity_type)

    entity_columns = ["vendor_code"] if entity_type == "vendor" else ["vendor_code", "asin"]
    results: List[pd.DataFrame] = []

    for metric in context.metrics:
        subset = prepared[prepared["metric"] == metric]
        if subset.empty:
            if logger:
                logger.debug("No data available for metric %s at entity level %s", metric, entity_type)
            continue

        group_columns = entity_columns
        for entity_keys, history in subset.groupby(group_columns):
            if not isinstance(entity_keys, tuple):
                entity_keys = (entity_keys,)
            descriptor = {"entity_type": entity_type, "vendor_code": entity_keys[0]}
            if entity_type == "asin":
                descriptor["asin"] = entity_keys[1]
            else:
                descriptor["asin"] = None

            forecast = forecast_single_entity(history, context, metric, descriptor, logger=logger)
            results.append(forecast)

    if not results:
        return pd.DataFrame(columns=base_columns)

    combined_results = pd.concat(results, ignore_index=True)
    missing_cols = [col for col in base_columns if col not in combined_results.columns]
    for column in missing_cols:
        combined_results[column] = None

    return combined_results[base_columns]


def persist_forecast_outputs(df: pd.DataFrame, config: Dict, entity_type: str, logger: Optional[logging.Logger] = None) -> Tuple[Path, Optional[Path]]:
    """Write forecast outputs to CSV and Parquet if available."""

    paths_config = config.get("paths", {})
    ensure_directory(paths_config.get("forecasts_dir", "data/processed/forecasts"))

    if entity_type == "vendor":
        csv_path = Path(paths_config.get("forecast_vendor_csv", "data/processed/forecasts/vendor_forecasts.csv"))
        parquet_path = Path(
            paths_config.get("forecast_vendor_parquet", "data/processed/forecasts/vendor_forecasts.parquet")
        )
    else:
        csv_path = Path(paths_config.get("forecast_asin_csv", "data/processed/forecasts/asin_forecasts.csv"))
        parquet_path = Path(
            paths_config.get("forecast_asin_parquet", "data/processed/forecasts/asin_forecasts.parquet")
        )

    df_to_save = df.copy()
    df_to_save["forecast_week"] = df_to_save["forecast_week"].astype(str)
    df_to_save["last_observed_week"] = df_to_save["last_observed_week"].astype(str)

    df_to_save.to_csv(csv_path, index=False)

    parquet_written: Optional[Path] = None
    try:
        df_to_save.to_parquet(parquet_path, index=False)
        parquet_written = parquet_path
    except Exception as exc:  # pragma: no cover - depends on parquet engine availability
        if logger:
            logger.warning("Unable to write parquet output %s (%s).", parquet_path, exc)

    return csv_path, parquet_written


def write_metadata(metadata: Dict[str, object], config: Dict) -> Path:
    """Persist run metadata to disk."""

    paths_config = config.get("paths", {})
    metadata_path = Path(paths_config.get("forecast_metadata", "data/processed/forecasts/run_metadata.json"))
    ensure_directory(metadata_path.parent)
    with open(metadata_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, default=str)
    return metadata_path


def summarise_potential(df: pd.DataFrame, entity_column: str, top_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return top and bottom entities by potential gain."""

    if df.empty:
        empty = pd.DataFrame(columns=[entity_column, "potential_gain"])
        return empty, empty

    aggregated = df.groupby(entity_column)["potential_gain"].mean().reset_index()
    top = aggregated.sort_values("potential_gain", ascending=False).head(top_n)
    bottom = aggregated.sort_values("potential_gain", ascending=True).head(top_n)
    return top, bottom


def load_metadata(config_path: str) -> Dict:
    """Convenience wrapper used in tests to read metadata."""

    config = load_config(config_path)
    metadata_path = Path(config["paths"].get("forecast_metadata", "data/processed/forecasts/run_metadata.json"))
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as stream:
        return json.load(stream)
