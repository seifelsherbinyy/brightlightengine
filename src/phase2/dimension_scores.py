from __future__ import annotations

from typing import Dict, Any, Set, Optional, List

import numpy as np
import pandas as pd


def _weighted_mean_row(row: pd.Series, metrics_weights: Dict[str, float], ok_metrics: Optional[Set[str]] = None) -> float:
    values = []
    weights = []
    for metric, w in metrics_weights.items():
        # If coverage map provided, skip metrics that are not OK for this entity
        if ok_metrics is not None and metric not in ok_metrics:
            continue
        col = f"{metric}__norm"
        if col in row.index and pd.notna(row[col]):
            values.append(float(row[col]))
            weights.append(float(w))
    if not values:
        return np.nan
    wsum = float(np.sum(weights))
    if wsum <= 0:
        # Equal weight fallback
        return float(np.mean(values))
    norm_weights = [w / wsum for w in weights]
    return float(np.dot(values, norm_weights))


def _score_dimension(df: pd.DataFrame, config: Dict[str, Any], dim_name: str) -> pd.Series:
    dims_cfg = (config or {}).get("dimensions", {})
    dim_cfg = dims_cfg.get(dim_name, {})
    metrics_weights: Dict[str, float] = dim_cfg.get("metrics", {})
    if not metrics_weights:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    # Optional coverage support: config may include a mapping of asin -> set(metrics ok)
    ok_map = (config or {}).get("__metric_ok_by_asin") or {}
    asin_col = "asin" if "asin" in df.columns else ("asin_id" if "asin_id" in df.columns else None)

    def _score_row(r: pd.Series) -> float:
        ok_set: Optional[Set[str]] = None
        if asin_col is not None:
            key = r.get(asin_col)
            ok_set = ok_map.get(key)
        return _weighted_mean_row(r, metrics_weights, ok_set)

    return df.apply(_score_row, axis=1)


def score_profitability(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    return _score_dimension(df, config, "profitability")


def score_growth(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    return _score_dimension(df, config, "growth")


def score_availability(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    return _score_dimension(df, config, "availability")


def score_inventory_risk(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    return _score_dimension(df, config, "inventory_risk")


def score_quality(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    return _score_dimension(df, config, "quality")


def dimension_coverage_annotations(df: pd.DataFrame, config: Dict[str, Any], dim_name: str) -> pd.DataFrame:
    """Return per-row coverage annotations for a dimension.

    Produces columns:
      - <dim>_metric_count: number of configured metrics for the dimension
      - <dim>_metrics_used: comma-separated list actually used (after coverage mask)
      - <dim>_coverage_flag: 'OK' if >=1 metric used, 'NO_VALID_METRICS' if none, 'PARTIAL' if some filtered out

    When no coverage information is present, assumes all configured metrics are usable and marks 'OK'.
    """

    dims_cfg = (config or {}).get("dimensions", {})
    dim_cfg = dims_cfg.get(dim_name, {})
    metrics_weights: Dict[str, float] = dim_cfg.get("metrics", {}) or {}
    configured_metrics: List[str] = list(metrics_weights.keys())
    count_total = len(configured_metrics)

    ok_map = (config or {}).get("__metric_ok_by_asin") or {}
    asin_col = "asin" if "asin" in df.columns else ("asin_id" if "asin_id" in df.columns else None)

    used_list: List[List[str]] = []
    flag_list: List[str] = []

    if count_total == 0:
        # Nothing configured for this dimension
        return pd.DataFrame({
            f"{dim_name}_metric_count": [0] * len(df),
            f"{dim_name}_metrics_used": ["[]"] * len(df),
            f"{dim_name}_coverage_flag": ["NO_VALID_METRICS"] * len(df),
        }, index=df.index)

    for idx, row in df.iterrows():
        ok_set: Optional[Set[str]] = None
        if asin_col is not None:
            ok_set = ok_map.get(row.get(asin_col))
        if ok_set is None:
            # Assume all metrics usable
            used = configured_metrics
            flag = "OK" if used else "NO_VALID_METRICS"
        else:
            used = [m for m in configured_metrics if m in ok_set]
            if len(used) == 0:
                flag = "NO_VALID_METRICS"
            elif len(used) == count_total:
                flag = "OK"
            else:
                flag = "PARTIAL"
        used_list.append(used)
        flag_list.append(flag)

    used_str = ["[" + ",".join(u) + "]" for u in used_list]
    return pd.DataFrame({
        f"{dim_name}_metric_count": [count_total] * len(df),
        f"{dim_name}_metrics_used": used_str,
        f"{dim_name}_coverage_flag": flag_list,
    }, index=df.index)
