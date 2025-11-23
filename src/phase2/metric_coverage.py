from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd


def _get_coverage_defaults(cfg: Dict[str, Any] | None) -> Dict[str, Any]:
    c = (cfg or {}).copy()
    c.setdefault("min_coverage_pct", 0.6)
    c.setdefault("min_history_weeks", 6)
    c.setdefault("treat_all_zero_as_inactive", True)
    c.setdefault("enforce_strict_for_dimensions", ["growth", "availability"])  # informational here
    c.setdefault("min_dimensions_for_composite", 2)
    return c


def _discover_metric_list(df: pd.DataFrame, registry: Dict[str, Any]) -> List[str]:
    """Return list of canonical metrics present in df.

    The registry may contain canonical names; we only consider those columns that exist in df
    and are numeric or can be coerced to numeric.
    """

    # Favor registry metrics if provided
    reg_metrics: List[str] = []
    if isinstance(registry, dict):
        for k, v in (registry.get("metrics", {}) or {}).items():
            name = v.get("canonical") if isinstance(v, dict) else k
            if name and isinstance(name, str):
                reg_metrics.append(name)
    # Fall back to numeric columns
    if not reg_metrics:
        reg_metrics = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    # Keep those that actually exist
    reg_metrics = [m for m in reg_metrics if m in df.columns]
    # De-duplicate while preserving order
    seen = set()
    ordered: List[str] = []
    for m in reg_metrics:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def compute_metric_coverage(df: pd.DataFrame, registry: Dict[str, Any] | None, coverage_config: Dict[str, Any] | None) -> pd.DataFrame:
    """Compute per-entity (ASIN) × metric coverage over available weeks.

    Output columns:
      - asin
      - metric
      - weeks_total, weeks_non_null, weeks_non_zero
      - coverage_pct = weeks_non_null / weeks_total
      - signal_pct = weeks_non_zero / weeks_total
      - metric_sparse (bool), metric_short_history (bool), metric_all_zero (bool)
      - metric_ok (bool)

    Notes:
    - Weeks are inferred from distinct values in 'week' column if present; otherwise
      from 'canonical_week_id'. Rows with missing week keys are ignored for coverage.
    - When treat_all_zero_as_inactive=True, a metric with all-zero non-null values is flagged as all_zero.
    - Vectorized groupby aggregations are used.
    """

    cfg = _get_coverage_defaults(coverage_config)

    data = df.copy()
    # Ensure asin and week keys exist
    if "asin" not in data.columns and "asin_id" in data.columns:
        data["asin"] = data["asin_id"].astype("string")
    if "asin" not in data.columns:
        # Cannot compute coverage without entity id
        return pd.DataFrame(columns=[
            "asin", "metric", "weeks_total", "weeks_non_null", "weeks_non_zero",
            "coverage_pct", "signal_pct", "metric_sparse", "metric_short_history",
            "metric_all_zero", "metric_ok",
        ])
    # Prefer integer week; fallback to canonical_week_id string
    if "week" in data.columns:
        wk = pd.to_numeric(data["week"], errors="coerce").astype("Int64")
        data = data.assign(_week_key=wk)
    elif "canonical_week_id" in data.columns:
        data = data.assign(_week_key=data["canonical_week_id"].astype("string"))
    else:
        # No week key → consider all rows as one pseudo-week; this leads to coverage=1 for non-null metrics
        data = data.assign(_week_key=1)

    metrics = _discover_metric_list(data, registry or {})
    if not metrics:
        return pd.DataFrame(columns=[
            "asin", "metric", "weeks_total", "weeks_non_null", "weeks_non_zero",
            "coverage_pct", "signal_pct", "metric_sparse", "metric_short_history",
            "metric_all_zero", "metric_ok",
        ])

    # Build long form for efficient aggregation: we do per-metric sequentially to reduce memory
    records: List[pd.DataFrame] = []
    # Distinct weeks per asin
    base_weeks = (
        data.dropna(subset=["asin", "_week_key"])
            .groupby(["asin"])["_week_key"].nunique()
            .rename("weeks_total_entity")
    )

    min_hist = int(cfg.get("min_history_weeks", 6))
    treat_zero = bool(cfg.get("treat_all_zero_as_inactive", True))

    for metric in metrics:
        s = pd.to_numeric(data[metric], errors="coerce")
        tmp = pd.DataFrame({
            "asin": data["asin"],
            "_week_key": data["_week_key"],
            "val": s,
        }).dropna(subset=["asin", "_week_key"])  # ignore rows without key

        # Aggregate per asin
        grp = tmp.groupby("asin")
        weeks_total = grp["_week_key"].nunique().rename("weeks_total")
        non_null_mask = tmp["val"].notna()
        weeks_non_null = tmp.loc[non_null_mask].groupby("asin")["_week_key"].nunique().rename("weeks_non_null")
        non_zero_mask = tmp["val"].notna() & (tmp["val"] != 0)
        weeks_non_zero = tmp.loc[non_zero_mask].groupby("asin")["_week_key"].nunique().rename("weeks_non_zero")

        dfm = pd.DataFrame({}).join(weeks_total, how="outer").join(weeks_non_null, how="outer").join(weeks_non_zero, how="outer")
        # Ensure integers
        for c in ("weeks_total", "weeks_non_null", "weeks_non_zero"):
            if c in dfm.columns:
                dfm[c] = dfm[c].fillna(0).astype(int)

        dfm["coverage_pct"] = (dfm["weeks_non_null"].astype(float) / dfm["weeks_total"].replace(0, np.nan)).fillna(0.0)
        dfm["signal_pct"] = (dfm["weeks_non_zero"].astype(float) / dfm["weeks_total"].replace(0, np.nan)).fillna(0.0)

        # Flags
        dfm["metric_short_history"] = dfm["weeks_total"] < max(min_hist, 1)
        dfm["metric_sparse"] = dfm["coverage_pct"] < float(cfg.get("min_coverage_pct", 0.6))
        if treat_zero:
            dfm["metric_all_zero"] = (dfm["weeks_non_zero"] == 0) & (dfm["weeks_non_null"] > 0)
        else:
            dfm["metric_all_zero"] = False

        # OK when history sufficient and not sparse and not all-zero
        dfm["metric_ok"] = ~(dfm["metric_short_history"] | dfm["metric_sparse"] | dfm["metric_all_zero"])
        dfm = dfm.reset_index().rename(columns={"index": "asin"})
        dfm["metric"] = metric
        records.append(dfm[[
            "asin", "metric", "weeks_total", "weeks_non_null", "weeks_non_zero",
            "coverage_pct", "signal_pct", "metric_sparse", "metric_short_history",
            "metric_all_zero", "metric_ok",
        ]])

    out = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    return out


def summarize_metric_coverage(coverage_df: pd.DataFrame, base_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Summarize coverage by vendor/category if available; else by vendor.

    Joins asin→vendor/category from base_df (Phase 2 working df) if provided.
    """

    if coverage_df is None or coverage_df.empty:
        return pd.DataFrame()
    cov = coverage_df.copy()
    # Attach attributes
    if base_df is not None and not base_df.empty:
        keys = [c for c in ["asin", "vendor_id", "category"] if c in base_df.columns]
        if keys and "asin" in keys:
            attr = base_df[keys].drop_duplicates("asin")
            cov = cov.merge(attr, on="asin", how="left")
    # Aggregates: percent of metrics ok and counts
    group_keys = [c for c in ["vendor_id", "category"] if c in cov.columns]
    if not group_keys:
        group_keys = ["vendor_id"] if "vendor_id" in cov.columns else []
    if group_keys:
        agg = cov.groupby(group_keys).agg(
            metrics_evaluated=("metric", "count"),
            metrics_ok=("metric_ok", "sum"),
            asins=("asin", pd.Series.nunique),
        ).reset_index()
        agg["ok_pct"] = (agg["metrics_ok"].astype(float) / agg["metrics_evaluated"].replace(0, np.nan)).fillna(0.0)
        return agg
    return pd.DataFrame()


def serialize_metric_coverage(coverage_df: pd.DataFrame, output_path: str) -> None:
    p = pd.Path(output_path) if hasattr(pd, 'Path') else None  # placeholder, not used
    # Caller provides full path with extension; assume Parquet
    if coverage_df is not None:
        try:
            coverage_df.to_parquet(output_path, index=False)
        except Exception:
            # Best-effort fallback to CSV (same stem)
            try:
                import os
                csv_path = os.path.splitext(output_path)[0] + ".csv"
                coverage_df.to_csv(csv_path, index=False)
            except Exception:
                pass


def serialize_metric_summary(summary_df: pd.DataFrame, output_path: str) -> None:
    if summary_df is not None:
        try:
            summary_df.to_parquet(output_path, index=False)
        except Exception:
            try:
                import os
                csv_path = os.path.splitext(output_path)[0] + ".csv"
                summary_df.to_csv(csv_path, index=False)
            except Exception:
                pass
