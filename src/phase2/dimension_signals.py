from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


def _resolve_input_series(df: pd.DataFrame, input_name: str, metrics_cfg: Dict[str, Any]) -> pd.Series:
    """Resolve an input to an appropriate series, preferring normalized columns.

    Rules:
    - If input_name exists and endswith __norm, use it.
    - Else if input_name__norm exists, use that.
    - Else if input_name exists, use raw series.
    Applies directionality sign for raw series based on metrics.<metric>.higher_is_better when applicable.
    """
    idx = df.index
    # Already normalized
    if input_name in df.columns and str(input_name).endswith("__norm"):
        return pd.to_numeric(df[input_name], errors="coerce")

    base = input_name.replace("__norm", "")
    norm_col = f"{base}__norm"
    if norm_col in df.columns:
        return pd.to_numeric(df[norm_col], errors="coerce")

    if base in df.columns:
        s = pd.to_numeric(df[base], errors="coerce")
        # Directionality: invert if configured as higher_is_better == false and the series is not already a __norm
        m_cfg = metrics_cfg.get(base, {}) if isinstance(metrics_cfg.get(base, {}), dict) else {}
        hib = bool(m_cfg.get("higher_is_better", True))
        if not hib:
            s = -s
        return s

    # Fallback: empty series of NaN
    return pd.Series([np.nan] * len(idx), index=idx, dtype="float64")


def _weighted_signal(df: pd.DataFrame, inputs: List[str], weights: List[float], metrics_cfg: Dict[str, Any]) -> pd.Series:
    if not inputs:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    # Pad/trim weights to inputs length
    if not weights or len(weights) != len(inputs):
        weights = [1.0] * len(inputs)

    # Build matrix-like aggregation per row using only non-null values
    values = [
        _resolve_input_series(df, inp, metrics_cfg) for inp in inputs
    ]
    V = pd.concat(values, axis=1)
    V.columns = inputs

    W = np.array(weights, dtype=float)
    # For each row, use only available inputs
    def row_signal(row: pd.Series) -> float:
        mask = row.notna().values
        if not mask.any():
            return np.nan
        w = W[mask]
        v = row.values[mask].astype(float)
        wsum = float(w.sum())
        if wsum <= 0 or not np.isfinite(wsum):
            # equal weight fallback over available
            return float(np.mean(v))
        w_norm = w / wsum
        return float(np.dot(v, w_norm))

    return V.apply(row_signal, axis=1)


def _get_dim_inputs(cfg: Dict[str, Any], dim_key: str) -> Tuple[List[str], List[float]]:
    dims = (cfg or {}).get("dimensions", {}) or {}
    dcfg = dims.get(dim_key, {}) or {}
    inputs = list(dcfg.get("inputs", []) or [])
    weights = list(dcfg.get("weights", []) or [])
    return inputs, weights


def compute_profitability_signals(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    inputs, weights = _get_dim_inputs(config, "profitability")
    metrics_cfg = (config or {}).get("metrics", {})
    df = df.copy()
    df["profitability_signal"] = _weighted_signal(df, inputs, weights, metrics_cfg)
    return df


def compute_growth_signals(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    inputs, weights = _get_dim_inputs(config, "growth")
    metrics_cfg = (config or {}).get("metrics", {})
    df = df.copy()
    df["growth_signal"] = _weighted_signal(df, inputs, weights, metrics_cfg)
    return df


def compute_availability_signals(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    inputs, weights = _get_dim_inputs(config, "availability")
    metrics_cfg = (config or {}).get("metrics", {})
    df = df.copy()
    df["availability_signal"] = _weighted_signal(df, inputs, weights, metrics_cfg)
    return df


def compute_inventory_risk_signals(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    inputs, weights = _get_dim_inputs(config, "inventory_risk")
    metrics_cfg = (config or {}).get("metrics", {})
    df = df.copy()
    df["inventory_risk_signal"] = _weighted_signal(df, inputs, weights, metrics_cfg)
    return df


def compute_quality_signals(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    inputs, weights = _get_dim_inputs(config, "quality")
    metrics_cfg = (config or {}).get("metrics", {})
    df = df.copy()
    df["quality_signal"] = _weighted_signal(df, inputs, weights, metrics_cfg)
    return df
