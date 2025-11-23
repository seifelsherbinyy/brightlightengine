from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from .metric_transforms import robust_scale_series


def _compute_entity_stats(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """Compute per-entity stats needed for readiness.

    Entity defined as (asin, vendor_id). Requires 'week' column for history length.
    Returns a DataFrame with one row per entity containing:
      asin, vendor_id, history_weeks, null_pct_<metric>, volatility_<metric>, anomaly_count_<metric>
    """

    if not {"asin", "vendor_id", "week"}.issubset(df.columns):
        # Return empty to signal no readiness possible
        return pd.DataFrame(columns=["asin", "vendor_id"])  

    # Prepare base
    base = (
        df[["asin", "vendor_id", "week"]].drop_duplicates().groupby(["asin", "vendor_id"]).agg(history_weeks=("week", "nunique")).reset_index()
    )

    # For each metric, compute null pct, volatility (MAD-based), and anomaly counts (|robust z|>5)
    pieces = [base]
    for m in metric_cols:
        if m not in df.columns:
            continue
        sub = df[["asin", "vendor_id", m]].copy()
        # Null pct
        agg = (
            sub.assign(is_null=sub[m].isna())
            .groupby(["asin", "vendor_id"]).agg(null_count=("is_null", "sum"), obs=(m, "size"))
            .reset_index()
        )
        agg[f"null_pct_{m}"] = np.where(agg["obs"] > 0, agg["null_count"] / agg["obs"], 1.0)
        agg = agg.drop(columns=["null_count", "obs"]).reset_index(drop=True)

        # Volatility and anomalies
        def _vol_anom(g: pd.DataFrame) -> Tuple[float, int]:
            s = pd.to_numeric(g[m], errors="coerce")
            if s.notna().sum() < 3:
                return float("nan"), 0
            rz = robust_scale_series(s)
            # Volatility as MAD of original values around median
            median = s.median(skipna=True)
            mad = (s - median).abs().median(skipna=True)
            # Anomalies: count robust z with |z| > 5
            anom = int((rz.abs() > 5).sum())
            return float(mad) if pd.notna(mad) else float("nan"), anom

        vol_anom = sub.groupby(["asin", "vendor_id"]).apply(_vol_anom).rename("tmp").reset_index()
        vol_anom[[f"volatility_{m}", f"anomaly_count_{m}"]] = pd.DataFrame(vol_anom["tmp"].tolist(), index=vol_anom.index)
        vol_anom = vol_anom.drop(columns=["tmp"])  

        # Merge
        merged = base.merge(agg, on=["asin", "vendor_id"], how="left").merge(vol_anom, on=["asin", "vendor_id"], how="left")
        pieces.append(merged.drop(columns=["history_weeks"]))

    # Combine all metric-specific pieces
    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on=["asin", "vendor_id"], how="left")
    return out


def compute_metric_readiness(df: pd.DataFrame, thresholds: Dict[str, Any]) -> pd.Series:
    """Compute readiness per ASIN/VENDOR pair and project back to row-index series.

    thresholds keys:
      - min_history_weeks: int
      - null_pct_max: float in [0,1]
      - volatility_max: float (MAD threshold)
      - anomaly_max: int
    Readiness is True if ALL considered metrics satisfy thresholds and history length is sufficient.
    """

    # Select numeric columns as candidate metrics, excluding identifiers and DQ flags
    ignore_prefixes = ("dq_",)
    ignore_cols = {"week", "asin", "vendor_id"}
    metric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and c not in ignore_cols and not any(str(c).startswith(p) for p in ignore_prefixes)
    ]

    stats = _compute_entity_stats(df, metric_cols)
    if stats.empty:
        return pd.Series([False] * len(df), index=df.index)

    # Build boolean per-entity
    min_weeks = int(thresholds.get("min_history_weeks", 6))
    null_max = float(thresholds.get("null_pct_max", 0.3))
    vol_max = float(thresholds.get("volatility_max", 1e9))
    anom_max = int(thresholds.get("anomaly_max", 10))

    cond = stats["history_weeks"] >= min_weeks
    # For each metric, apply constraints if the columns exist
    for c in list(stats.columns):
        if c.startswith("null_pct_"):
            cond &= (stats[c] <= null_max) | stats[c].isna()
        if c.startswith("volatility_"):
            cond &= (stats[c] <= vol_max) | stats[c].isna()
        if c.startswith("anomaly_count_"):
            cond &= (stats[c] <= anom_max) | stats[c].isna()

    # Map back to rows
    stats["is_ready"] = cond
    key = stats.set_index(["asin", "vendor_id"])["is_ready"]
    idx_key = list(zip(df["asin"].astype(str), df["vendor_id"].astype(str)))
    return pd.Series([bool(key.get(k, False)) for k in idx_key], index=df.index)


def filter_unreliable_metrics(df: pd.DataFrame, readiness_flags: pd.Series) -> pd.DataFrame:
    """Mask normalized metric columns where readiness is False.

    Any column ending with '__norm' will be nulled out for rows where readiness_flags is False.
    """

    out = df.copy()
    mask = ~readiness_flags.fillna(False)
    norm_cols = [c for c in out.columns if str(c).endswith("__norm")]
    if norm_cols:
        out.loc[mask, norm_cols] = np.nan
    return out


def apply_readiness_mask(df: pd.DataFrame, readiness_flags: pd.Series) -> pd.DataFrame:
    """Spec wrapper: mask unreliable metrics using readiness flags.

    Provided for API compatibility with the Phase 2 spec; delegates to filter_unreliable_metrics.
    """
    return filter_unreliable_metrics(df, readiness_flags)


def finalize_readiness(df: pd.DataFrame, readiness_config: Dict[str, Any]) -> pd.DataFrame:
    """Finalize readiness gating for Composite Score consumers (Session 2B).

    Rules per spec (config.readiness.final):
      - require_history_weeks: minimum unique weeks per entity (asin, vendor_id)
      - max_null_share: maximum fraction of nulls allowed across dimension signals
      - allow_missing_dimensions: if False, missing any dimension signal fails readiness

    Outputs:
      - readiness_flag (bool)
      - readiness_reason (list[str]) — semicolon-joined string for parquet friendliness
      - composite_score masked (NaN) where readiness_flag == False
    """

    df_out = df.copy()

    # Defaults
    fin = (readiness_config or {}).get("final", {}) if isinstance(readiness_config, dict) else {}
    require_weeks = int(fin.get("require_history_weeks", 6))
    max_null_share = float(fin.get("max_null_share", 0.25))
    allow_missing_dims = bool(fin.get("allow_missing_dimensions", False))

    # Dimension signal columns
    dim_cols = [
        c for c in [
            "profitability_signal",
            "growth_signal",
            "availability_signal",
            "inventory_risk_signal",
            "quality_signal",
        ] if c in df_out.columns
    ]

    # History weeks per entity
    if {"asin", "vendor_id", "week"}.issubset(df_out.columns):
        hist = (
            df_out[["asin", "vendor_id", "week"]]
            .drop_duplicates()
            .groupby(["asin", "vendor_id"]).agg(history_weeks=("week", "nunique"))
            .reset_index()
        )
    else:
        # If missing, mark zero history
        hist = pd.DataFrame(columns=["asin", "vendor_id", "history_weeks"])  

    # Null share across dimension signals (per entity across available weeks)
    if dim_cols:
        tmp = df_out[["asin", "vendor_id"] + dim_cols].copy()
        for c in dim_cols:
            tmp[f"isnull_{c}"] = tmp[c].isna()
        agg_null = (
            tmp.groupby(["asin", "vendor_id"]).agg(**{f"nulls_{c}": (f"isnull_{c}", "mean") for c in dim_cols}).reset_index()
        )
        # overall share: mean of per-dimension null shares
        null_cols = [c for c in agg_null.columns if c.startswith("nulls_")]
        if null_cols:
            agg_null["null_share_overall"] = agg_null[null_cols].mean(axis=1)
        else:
            agg_null["null_share_overall"] = 1.0
    else:
        agg_null = pd.DataFrame(columns=["asin", "vendor_id", "null_share_overall"]).assign(null_share_overall=1.0)

    # Missing dimension signals check (per row, then per entity if any missing across rows)
    if dim_cols and not allow_missing_dims:
        miss = (
            df_out[["asin", "vendor_id"] + dim_cols]
            .assign(missing_any=lambda d: d[dim_cols].isna().any(axis=1))
            .groupby(["asin", "vendor_id"]).agg(missing_any=("missing_any", "max"))
            .reset_index()
        )
    else:
        miss = pd.DataFrame(columns=["asin", "vendor_id", "missing_any"]).assign(missing_any=False)

    # Combine
    ent = (
        hist.merge(agg_null[["asin", "vendor_id", "null_share_overall"]], on=["asin", "vendor_id"], how="outer")
        .merge(miss, on=["asin", "vendor_id"], how="outer")
        .fillna({"history_weeks": 0, "null_share_overall": 1.0, "missing_any": False})
    )

    def _ready_row(r: pd.Series) -> tuple[bool, List[str]]:
        reasons: List[str] = []
        if int(r.get("history_weeks", 0)) < require_weeks:
            reasons.append("insufficient_history")
        if float(r.get("null_share_overall", 1.0)) > max_null_share:
            reasons.append("too_many_nulls")
        if bool(r.get("missing_any", False)) and not allow_missing_dims:
            reasons.append("missing_dimension_signal")
        return (len(reasons) == 0), reasons

    ent[["readiness_flag", "readiness_reason_list"]] = ent.apply(lambda x: pd.Series(_ready_row(x)), axis=1)
    ent["readiness_reason"] = ent["readiness_reason_list"].apply(lambda rs: ";".join(rs) if rs else "")
    ent = ent.drop(columns=["readiness_reason_list"])

    # Project back to row-level
    df_out = df_out.merge(ent, on=["asin", "vendor_id"], how="left")
    # Mask composite_score
    if "composite_score" in df_out.columns:
        df_out.loc[~df_out["readiness_flag"].fillna(False), "composite_score"] = np.nan

    return df_out
