from __future__ import annotations

from typing import Iterable, Tuple, Optional, List, Dict, Any

import numpy as np
import pandas as pd


def winsorize_series(series: pd.Series, limits: Tuple[float, float] = (0.01, 0.99)) -> pd.Series:
    """Winsorize a numeric series by clipping to given quantile limits.

    - Ignores NaNs when computing quantiles
    - If series has < 3 non-null values or zero IQR, returns original series (no-op)
    - Limits are quantile fractions in [0,1]
    """

    s = pd.to_numeric(series, errors="coerce")
    nonnull = s.dropna()
    if nonnull.shape[0] < 3:
        return s
    low_q, high_q = limits
    low = nonnull.quantile(low_q)
    high = nonnull.quantile(high_q)
    if pd.isna(low) or pd.isna(high) or low >= high:
        return s
    return s.clip(lower=low, upper=high)


def robust_scale_series(series: pd.Series) -> pd.Series:
    """Return robust z-score style scaling using median and MAD.

    z_robust = (x - median) / (1.4826 * MAD)
    Where MAD = median(|x - median|). If MAD==0 (constant), returns 0.
    """

    s = pd.to_numeric(series, errors="coerce")
    median = s.median(skipna=True)
    mad = (s - median).abs().median(skipna=True)
    denom = 1.4826 * mad if pd.notna(mad) else np.nan
    if not denom or denom == 0 or np.isclose(denom, 0):
        return pd.Series(np.where(s.notna(), 0.0, np.nan), index=s.index)
    return (s - median) / denom


def _pct_change(curr: pd.Series, prev: pd.Series) -> pd.Series:
    denom = prev.replace(0, np.nan)
    return (curr - prev) / denom


def compute_growth_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute WoW and rolling-window growth metrics in-place and return df.

    Expected columns: 'asin', 'vendor_id', 'week' (int YYYYWW), and numeric base metrics
    commonly including 'gms', 'units', 'gv' if present.

    Adds columns for each present base metric (m):
      - m_wow_abs, m_wow_pct
      - m_l4w_delta_abs, m_l4w_delta_pct (rolling 4-week sum vs previous 4-week sum)
      - m_l12w_delta_abs, m_l12w_delta_pct (rolling 12-week sum vs previous 12-week sum)
    """

    result = df.copy()
    if not {"asin", "vendor_id", "week"}.issubset(result.columns):
        return result

    base_metrics = [m for m in ("gms", "units", "gv") if m in result.columns]
    if not base_metrics:
        return result

    # Ensure proper sorting for group operations
    result = result.sort_values(["asin", "vendor_id", "week"]).reset_index(drop=True)

    def add_growth(group: pd.DataFrame) -> pd.DataFrame:
        for m in base_metrics:
            prev = group[m].shift(1)
            group[f"{m}_wow_abs"] = group[m] - prev
            group[f"{m}_wow_pct"] = _pct_change(group[m], prev)

            roll4 = group[m].rolling(window=4, min_periods=2).sum()
            prev4 = roll4.shift(4)
            group[f"{m}_l4w_delta_abs"] = roll4 - prev4
            group[f"{m}_l4w_delta_pct"] = _pct_change(roll4, prev4)

            roll12 = group[m].rolling(window=12, min_periods=4).sum()
            prev12 = roll12.shift(12)
            group[f"{m}_l12w_delta_abs"] = roll12 - prev12
            group[f"{m}_l12w_delta_pct"] = _pct_change(roll12, prev12)
        return group

    result = result.groupby(["asin", "vendor_id"], group_keys=False).apply(add_growth)
    return result


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean(skipna=True)
    sigma = s.std(skipna=True)
    if not sigma or np.isclose(sigma, 0):
        return pd.Series(np.where(s.notna(), 0.0, np.nan), index=s.index)
    return (s - mu) / sigma


def _percentile_rank(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    # Rank ignoring NaNs; normalize to [0,1]
    ranks = s.rank(method="average", na_option="keep")
    denom = ranks.max(skipna=True) - ranks.min(skipna=True)
    if pd.isna(denom) or np.isclose(float(denom), 0.0):
        return pd.Series(np.where(s.notna(), 0.5, np.nan), index=s.index)
    return (ranks - ranks.min(skipna=True)) / denom


def _minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mn = s.min(skipna=True)
    mx = s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or np.isclose(float(mx - mn), 0.0):
        return pd.Series(np.where(s.notna(), 0.5, np.nan), index=s.index)
    return (s - mn) / (mx - mn)


def normalize_metric(df: pd.DataFrame, metric_name: str, method: str) -> pd.Series:
    """Normalize a metric column using the specified method.

    Supported methods: 'zscore', 'robust_zscore', 'percentile', 'minmax'.
    Returns a Series aligned to df.index (NaN where source is NaN/missing).
    """

    if metric_name not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    s = df[metric_name]
    method = (method or "").strip().lower()
    if method == "zscore":
        return _zscore(s)
    if method == "robust_zscore":
        return robust_scale_series(s)
    if method == "percentile":
        return _percentile_rank(s)
    if method == "minmax":
        return _minmax(s)
    # Default: robust_zscore
    return robust_scale_series(s)


# -------------------------
# Spec-compliant thin wrappers
# -------------------------

def winsorize(series: pd.Series, limits: Tuple[float, float] = (0.01, 0.99)) -> pd.Series:
    """Spec wrapper for winsorize_series."""
    return winsorize_series(series, limits)


def robust_scale(series: pd.Series) -> pd.Series:
    """Spec wrapper for robust_scale_series."""
    return robust_scale_series(series)


def compute_deltas(df: pd.DataFrame, metric_list: Optional[List[str]] = None) -> pd.DataFrame:
    """Spec wrapper to compute growth deltas.

    Current engine computes WoW, L4W, L12W for base metrics present among {gms, units, gv}.
    metric_list is accepted for API compatibility but not strictly required.
    """
    return compute_growth_metrics(df)


# Additional thin spec wrappers requested by Session 2A
def zscore(series: pd.Series) -> pd.Series:
    return _zscore(series)


def robust_zscore(series: pd.Series) -> pd.Series:
    return robust_scale_series(series)


def percentile_scale(series: pd.Series) -> pd.Series:
    return _percentile_rank(series)


def minmax_scale(series: pd.Series) -> pd.Series:
    return _minmax(series)


def _get_cfg(d: Dict[str, Any], *keys, default=None):
    cur = d or {}
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, {}) if k != keys[-1] else cur.get(k, default)
    return cur if cur is not None else default


def apply_all_normalizations(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Bulk add metric__norm columns using config groupings.

    - Target list = unique ordered concat of config.metrics.numeric + percent + currency
    - For each metric present in df:
        * Winsorize using metrics.<m>.winsor_limits if present else config.transforms.winsor_limits
        * Choose method = metrics.<m>.method if present else config.metrics.normalization_method
        * If method == 'minmax' and transforms.enable_minmax is False, fall back to 'robust_zscore'
        * Compute normalization and write to m__norm (do not overwrite base columns)
        * If metrics.<m>.higher_is_better == false, multiply by -1
    """

    out = df.copy()

    metrics_cfg: Dict[str, Any] = (config or {}).get("metrics", {}) or {}
    transforms_cfg: Dict[str, Any] = (config or {}).get("transforms", {}) or {}

    grouping_numeric = list(metrics_cfg.get("numeric", []) or [])
    grouping_percent = list(metrics_cfg.get("percent", []) or [])
    grouping_currency = list(metrics_cfg.get("currency", []) or [])

    target_order: List[str] = []
    for m in grouping_numeric + grouping_percent + grouping_currency:
        if m not in target_order:
            target_order.append(m)

    default_limits = tuple(transforms_cfg.get("winsor_limits", [0.01, 0.99]))
    enable_minmax = bool(transforms_cfg.get("enable_minmax", False))
    default_method = (metrics_cfg.get("normalization_method") or "robust_zscore").strip().lower()

    for m in target_order:
        if m not in out.columns:
            continue
        # Determine per-metric overrides if present
        m_cfg = metrics_cfg.get(m, {}) if isinstance(metrics_cfg.get(m, {}), dict) else {}
        limits = tuple(m_cfg.get("winsor_limits", default_limits))
        method = (m_cfg.get("method", default_method) or "").strip().lower()
        higher_is_better = bool(m_cfg.get("higher_is_better", True))

        s = pd.to_numeric(out[m], errors="coerce")
        s_w = winsorize(s, (float(limits[0]), float(limits[1]))) if limits else s

        # Gate minmax
        if method == "minmax" and not enable_minmax:
            method = "robust_zscore"

        if method == "zscore":
            norm = zscore(s_w)
        elif method == "robust_zscore":
            norm = robust_zscore(s_w)
        elif method == "percentile":
            norm = percentile_scale(s_w)
        elif method == "minmax":
            norm = minmax_scale(s_w)
        else:
            norm = robust_zscore(s_w)

        if not higher_is_better:
            norm = -pd.to_numeric(norm, errors="coerce")

        out[f"{m}__norm"] = pd.to_numeric(norm, errors="coerce")

    return out
