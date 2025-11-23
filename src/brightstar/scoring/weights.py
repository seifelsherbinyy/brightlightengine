"""Dynamic weighting utilities for Phase 2 scoring.

Compute per-entity, per-metric, per-week effective weights based on
signal quality inputs such as completeness, recency, variance, and
trend_quality computed in the processing layer.

All functions are pure and operate on DataFrames.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _normalize_trend_quality(tq: pd.Series, cap: float = 2.0) -> pd.Series:
    """Normalize trend_quality (|slope|/std) to [0, 1] by capping then scaling.

    Values above ``cap`` are set to ``cap`` and divided by ``cap``.
    NaNs produce NaN and should be handled by caller with fillna.
    """
    x = pd.to_numeric(tq, errors="coerce")
    x = x.clip(lower=0)
    x = x.where(x <= cap, cap)
    with np.errstate(divide="ignore", invalid="ignore"):
        return x / cap


def _recency_factor(recency_weeks: pd.Series, lookback_weeks: int) -> pd.Series:
    """Convert recency (0=most recent) into a [0,1] factor where 1 is freshest.

    Uses 1 / (1 + recency/lookback), then caps to [0,1]. NaNs become 0.5.
    """
    r = pd.to_numeric(recency_weeks, errors="coerce")
    with np.errstate(divide="ignore", invalid="ignore"):
        f = 1.0 / (1.0 + (r / max(1, lookback_weeks)))
    f = f.clip(lower=0, upper=1)
    return f


def compute_dynamic_weights(
    df_quality: pd.DataFrame,
    base_weights_by_metric: Dict[str, float],
    lookback_weeks: int = 8,
    min_multiplier: float = 0.5,
    max_multiplier: float = 1.25,
) -> pd.DataFrame:
    """Compute effective weights using signal quality.

    Args:
        df_quality: Long frame with columns ['entity_id','metric','Week_Order',
            'completeness','recency_weeks','variance','trend_quality'].
        base_weights_by_metric: Mapping of metric -> configured base weight.
        lookback_weeks: Lookback window used for quality snapshot.
        min_multiplier: Floor for weight multiplier (avoid zeroing a metric entirely).
        max_multiplier: Cap for weight multiplier (avoid exploding weights).

    Returns:
        DataFrame with columns ['entity_id','metric','Week_Order','weight_effective',
        'weight_multiplier','quality_completeness','quality_recency_factor','quality_trend_norm']
    """
    req = {"entity_id", "metric", "Week_Order"}
    if not req.issubset(df_quality.columns):
        missing = req - set(df_quality.columns)
        raise ValueError(f"compute_dynamic_weights missing columns: {missing}")

    df = df_quality.copy()

    # Build normalized quality components
    comp = pd.to_numeric(df.get("completeness"), errors="coerce").clip(lower=0, upper=1)
    rec = _recency_factor(df.get("recency_weeks"), lookback_weeks=lookback_weeks)
    tqn = _normalize_trend_quality(df.get("trend_quality"))

    # Default NaNs to neutral 0.75 to avoid over-penalizing sparse series
    comp = comp.fillna(0.75)
    rec = rec.fillna(0.75)
    tqn = tqn.fillna(0.75)

    # Combine via geometric mean for balanced effect (penalizes if any is weak)
    with np.errstate(invalid="ignore"):
        combined = (comp * rec * tqn) ** (1 / 3)

    # Map to multiplier range [min_multiplier, max_multiplier]
    mul = min_multiplier + (max_multiplier - min_multiplier) * combined
    mul = mul.clip(lower=min_multiplier, upper=max_multiplier)

    # Attach base weights by metric
    df["base_weight"] = df["metric"].map(base_weights_by_metric).fillna(0.0)
    df["weight_multiplier"] = mul
    df["weight_effective"] = df["base_weight"] * df["weight_multiplier"]

    # Return only necessary columns plus quality components for auditing
    out = df[[
        "entity_id",
        "metric",
        "Week_Order",
        "weight_effective",
        "weight_multiplier",
        "base_weight",
    ]].copy()
    out["quality_completeness"] = comp.values
    out["quality_recency_factor"] = rec.values
    out["quality_trend_norm"] = tqn.values

    return out
