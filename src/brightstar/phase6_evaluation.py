"""Phase 6: Forecast Evaluation & Governance (offline, interpretable).

Neural connection:
  Phase 4 → Forecasts
  Phase 6 → Evaluation of those forecasts vs actuals
  Phase 5 → Displays both forecasts and their health_state (optional)

This module computes granular forecast–actual error rows and aggregates
health metrics per metric/entity_type/horizon/method without feeding back
into forecasting or scoring. All operations degrade gracefully and never
interrupt the pipeline on empty data.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import logging
import pandas as pd

from .ingestion_utils import ensure_directory, load_config
from .scoring_utils import load_normalized_input


LOGGER_NAME = "brightstar.phase6"


def _logger(config: Dict) -> logging.Logger:
    log_dir = ensure_directory(config["paths"]["logs_dir"])
    file_name = config.get("logging", {}).get("evaluation_file_name", "phase6_evaluation.log")
    log_path = log_dir / file_name

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, config.get("logging", {}).get("forecasting_level", "INFO")))
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def _coerce_forecast_schema(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "entity_type",
                "entity_id",
                "vendor_code",
                "asin",
                "metric",
                "forecast_week_start_date",
                "forecast_value",
                "forecast_horizon_weeks",
                "method_name",
                "forecast_reliability",
            ]
        )

    out = df.copy()
    # Standardize date and id columns common from Phase 4 outputs
    if "forecast_week_start_date" not in out.columns:
        if "forecast_week" in out.columns:
            out.rename(columns={"forecast_week": "forecast_week_start_date"}, inplace=True)
        elif "week_start" in out.columns:
            out.rename(columns={"week_start": "forecast_week_start_date"}, inplace=True)
    for col in [
        "entity_type",
        "entity_id",
        "vendor_code",
        "asin",
        "metric",
        "forecast_week_start_date",
        "forecast_value",
        "forecast_horizon_weeks",
        "method_name",
        "forecast_reliability",
    ]:
        if col not in out.columns:
            out[col] = pd.NA
    return out[
        [
            "entity_type",
            "entity_id",
            "vendor_code",
            "asin",
            "metric",
            "forecast_week_start_date",
            "forecast_value",
            "forecast_horizon_weeks",
            "method_name",
            "forecast_reliability",
        ]
    ].copy()


def load_forecasts(config: Dict) -> pd.DataFrame:
    paths = config.get("paths", {})
    eval_cfg = config.get("evaluation", {})
    base = Path(eval_cfg.get("forecast_input_dir", paths.get("forecasts_dir", "data/processed/forecasts")))

    candidates = [
        Path(paths.get("forecast_vendor_parquet")) if paths.get("forecast_vendor_parquet") else base / "vendor_forecasts.parquet",
        base / "vendor_forecasts.csv",
        Path(paths.get("forecast_asin_parquet")) if paths.get("forecast_asin_parquet") else base / "asin_forecasts.parquet",
        base / "asin_forecasts.csv",
    ]
    frames = []
    for path in candidates:
        if path and path.exists():
            if path.suffix.lower() == ".parquet":
                try:
                    frames.append(pd.read_parquet(path))
                except Exception:
                    continue
            elif path.suffix.lower() in {".csv", ".txt"}:
                frames.append(pd.read_csv(path))
    if not frames:
        return _coerce_forecast_schema(pd.DataFrame())
    df = pd.concat(frames, ignore_index=True)
    return _coerce_forecast_schema(df)


def load_actuals_for_forecast_window(config: Dict, forecasts_df: pd.DataFrame) -> pd.DataFrame:
    if forecasts_df is None or forecasts_df.empty:
        return pd.DataFrame(columns=["entity_type", "entity_id", "vendor_code", "asin", "metric", "week_start_date", "actual_value"]) 

    normalized = load_normalized_input(config)
    if normalized is None or normalized.empty:
        return pd.DataFrame(columns=["entity_type", "entity_id", "vendor_code", "asin", "metric", "week_start_date", "actual_value"]) 

    fdates = pd.to_datetime(forecasts_df["forecast_week_start_date"], errors="coerce")
    min_wk, max_wk = fdates.min(), fdates.max()
    df = normalized.copy()
    df["week_start"] = pd.to_datetime(df.get("week_start"), errors="coerce")
    df = df[(df["week_start"] >= min_wk) & (df["week_start"] <= max_wk)]
    # Aggregate to vendor-week and asin-week per metric (mean to match common scoring aggregation)
    vendor = (
        df.groupby(["vendor_code", "metric", "week_start"], as_index=False)["value"].mean()
        .assign(entity_type="vendor", asin=pd.NA)
        .rename(columns={"week_start": "week_start_date", "value": "actual_value"})
    )
    asin_df = df.dropna(subset=["asin"]).copy()
    asin = (
        asin_df.groupby(["vendor_code", "asin", "metric", "week_start"], as_index=False)["value"].mean()
        .assign(entity_type="asin")
        .rename(columns={"week_start": "week_start_date", "value": "actual_value"})
    )
    vendor["entity_id"] = vendor["vendor_code"].astype(str)
    asin["entity_id"] = asin["vendor_code"].astype(str) + "_" + asin["asin"].astype(str)
    return pd.concat([vendor, asin], ignore_index=True, sort=False)


def align_forecast_actual(forecasts_df: pd.DataFrame, actuals_df: pd.DataFrame) -> pd.DataFrame:
    if forecasts_df is None or forecasts_df.empty or actuals_df is None or actuals_df.empty:
        return pd.DataFrame(
            columns=[
                "entity_type",
                "entity_id",
                "vendor_code",
                "asin",
                "metric",
                "week_start_date",
                "forecast_value",
                "actual_value",
                "forecast_horizon_weeks",
                "method_name",
                "forecast_reliability",
            ]
        )

    left = forecasts_df.copy()
    left["forecast_week_start_date"] = pd.to_datetime(left["forecast_week_start_date"], errors="coerce")
    right = actuals_df.copy()
    right["week_start_date"] = pd.to_datetime(right["week_start_date"], errors="coerce")

    merged = left.merge(
        right,
        on=["entity_type", "entity_id", "vendor_code", "asin", "metric"],
        how="inner",
        suffixes=("_f", "_a"),
    )
    # Align by week start
    merged = merged[merged["forecast_week_start_date"] == merged["week_start_date"]]
    return merged[
        [
            "entity_type",
            "entity_id",
            "vendor_code",
            "asin",
            "metric",
            "week_start_date",
            "forecast_value",
            "actual_value",
            "forecast_horizon_weeks",
            "method_name",
            "forecast_reliability",
        ]
    ].copy()


def compute_errors(aligned_df: pd.DataFrame) -> pd.DataFrame:
    if aligned_df is None or aligned_df.empty:
        return pd.DataFrame(columns=list(aligned_df.columns) + ["error", "abs_error", "error_pct", "smape", "bias"]) if isinstance(aligned_df, pd.DataFrame) else pd.DataFrame()

    df = aligned_df.copy()
    df["error"] = df["actual_value"] - df["forecast_value"]
    df["abs_error"] = df["error"].abs()
    # Guard division by zero
    df["error_pct"] = df.apply(lambda r: (r["error"] / r["actual_value"]) if r["actual_value"] not in (0, None, pd.NA) else pd.NA, axis=1)
    def _smape(a, f):
        denom = (abs(a) + abs(f))
        return (2 * abs(f - a) / denom) if denom else 0.0
    df["smape"] = df.apply(lambda r: _smape(float(r["actual_value"]), float(r["forecast_value"])) if pd.notna(r["actual_value"]) and pd.notna(r["forecast_value"]) else pd.NA, axis=1)
    df["bias"] = df["forecast_value"] - df["actual_value"]
    return df


def aggregate_health_metrics(errors_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    eval_cfg = config.get("evaluation", {})
    thresholds = eval_cfg.get("health_thresholds", {})
    smape_healthy = float(thresholds.get("smape_healthy", 0.20))
    smape_warning = float(thresholds.get("smape_warning", 0.35))
    min_pairs = int(thresholds.get("min_pairs", 10))

    if errors_df is None or errors_df.empty:
        return pd.DataFrame(
            columns=[
                "metric",
                "entity_type",
                "forecast_horizon_weeks",
                "method_name",
                "avg_smape",
                "avg_bias",
                "count_pairs",
                "health_state",
            ]
        )

    g = errors_df.groupby(["metric", "entity_type", "forecast_horizon_weeks", "method_name"], as_index=False)
    agg = g.agg(avg_smape=("smape", "mean"), avg_bias=("bias", "mean"), count_pairs=("smape", "count"))

    def _state(row):
        if row["count_pairs"] < min_pairs:
            return "Insufficient_Data"
        if row["avg_smape"] <= smape_healthy:
            return "Healthy"
        if row["avg_smape"] <= smape_warning:
            return "Mild_Drift"
        return "Severe_Drift"

    agg["health_state"] = agg.apply(_state, axis=1)
    return agg


def persist_evaluation_outputs(config: Dict, granular_df: pd.DataFrame, health_df: pd.DataFrame) -> Tuple[Path, Path]:
    out_dir = ensure_directory(Path(config.get("evaluation", {}).get("output_dir", "outputs/forecast_evaluation")))
    granular_path = out_dir / "forecast_actual_pairs.csv"
    health_path = out_dir / "health_summary.csv"
    granular_df.to_csv(granular_path, index=False)
    health_df.to_csv(health_path, index=False)
    # Try parquet when possible
    try:
        granular_df.to_parquet(out_dir / "forecast_actual_pairs.parquet", index=False)
        health_df.to_parquet(out_dir / "health_summary.parquet", index=False)
    except Exception:
        pass
    return granular_path, health_path


def run_phase6(config_path: str = "config.yaml") -> Dict[str, pd.DataFrame]:
    """Execute forecast evaluation; return dict with granular and health dataframes."""
    config = load_config(config_path)
    logger = _logger(config)

    if not bool(config.get("evaluation", {}).get("enabled", True)):
        logger.info("Evaluation disabled via configuration; skipping Phase 6.")
        return {"pairs": pd.DataFrame(), "health": pd.DataFrame()}

    forecasts = load_forecasts(config)
    if forecasts.empty:
        logger.warning("No forecasts found for evaluation; skipping.")
        return {"pairs": pd.DataFrame(), "health": pd.DataFrame()}

    actuals = load_actuals_for_forecast_window(config, forecasts)
    aligned = align_forecast_actual(forecasts, actuals)
    if aligned.empty:
        logger.info("No forecast–actual pairs available for evaluation.")
        return {"pairs": pd.DataFrame(), "health": pd.DataFrame()}

    errors = compute_errors(aligned)
    health = aggregate_health_metrics(errors, config)
    g_path, h_path = persist_evaluation_outputs(config, errors, health)
    logger.info(
        "Forecast evaluation complete: %d pairs, %d groups | outputs: %s, %s",
        len(errors), len(health), g_path, h_path,
    )
    return {"pairs": errors, "health": health}
