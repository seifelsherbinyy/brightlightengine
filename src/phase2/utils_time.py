from __future__ import annotations

from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import logging


def protect_zero_division(num: pd.Series | float, denom: pd.Series | float) -> pd.Series:
    """Safe division: return NaN where denom is 0 or NaN.

    Accepts scalars or Series; always returns a Series when either input is a Series.
    """
    num_s = pd.to_numeric(num, errors="coerce") if isinstance(num, (pd.Series, list)) else num
    den_s = pd.to_numeric(denom, errors="coerce") if isinstance(denom, (pd.Series, list)) else denom
    if isinstance(num_s, pd.Series) or isinstance(den_s, pd.Series):
        num_s = pd.Series(num_s) if not isinstance(num_s, pd.Series) else num_s
        den_s = pd.Series(den_s) if not isinstance(den_s, pd.Series) else den_s
        den_s = den_s.replace(0, np.nan)
        return num_s / den_s
    # Fallback scalar path
    try:
        return np.nan if den_s in (None, 0) or (isinstance(den_s, float) and np.isnan(den_s)) else num_s / den_s
    except Exception:
        return pd.Series([np.nan])


def _group_and_sort(df: pd.DataFrame) -> tuple[pd.core.groupby.generic.DataFrameGroupBy, str]:
    key_asin = "asin_id" if "asin_id" in df.columns else "asin"
    group_keys: List[str] = [key_asin]
    if "vendor_id" in df.columns:
        group_keys.append("vendor_id")

    # Determine order column
    order_col = None
    if "week_start_date" in df.columns:
        order_col = "week_start_date"
    elif "week" in df.columns:
        order_col = "week"
    else:
        raise ValueError("Missing calendar column for ordering: need week_start_date or week")

    df_sorted = df.sort_values(group_keys + [order_col]).reset_index(drop=True)
    return df_sorted.groupby(group_keys, group_keys=False), order_col


def compute_wow_delta(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute week-over-week deltas for a metric: adds *_wow_abs and *_wow_pct columns.
    """
    if metric not in df.columns:
        return df
    result = df.copy()
    g, _ = _group_and_sort(result)

    def add(group: pd.DataFrame) -> pd.DataFrame:
        prev = group[metric].shift(1)
        abs_change = pd.to_numeric(group[metric], errors="coerce") - pd.to_numeric(prev, errors="coerce")
        pct_change = protect_zero_division(abs_change, prev)
        group[f"{metric}_wow_abs"] = abs_change
        group[f"{metric}_wow_pct"] = pct_change
        return group

    return g.apply(add)


def compute_l4w_delta(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute last-4-weeks rolling sum delta vs previous 4 weeks: adds *_l4w_abs and *_l4w_pct.
    Also mirrors to legacy alias names *_l4w_delta_abs and *_l4w_delta_pct if not already present.
    """
    if metric not in df.columns:
        return df
    result = df.copy()
    g, _ = _group_and_sort(result)

    def add(group: pd.DataFrame) -> pd.DataFrame:
        s = pd.to_numeric(group[metric], errors="coerce")
        roll4 = s.rolling(window=4, min_periods=2).sum()
        prev4 = roll4.shift(4)
        abs_change = roll4 - prev4
        pct_change = protect_zero_division(abs_change, prev4)
        base = metric
        group[f"{base}_l4w_abs"] = abs_change
        group[f"{base}_l4w_pct"] = pct_change
        # legacy aliases
        group[f"{base}_l4w_delta_abs"] = group[f"{base}_l4w_abs"]
        group[f"{base}_l4w_delta_pct"] = group[f"{base}_l4w_pct"]
        return group

    out = g.apply(add)
    # Ensure legacy alias columns exist at top-level (setdefault above only affects per-group dict API; be explicit here)
    for suffix_src, suffix_alias in [("_l4w_abs", "_l4w_delta_abs"), ("_l4w_pct", "_l4w_delta_pct")]:
        src_col = f"{metric}{suffix_src}"
        alias_col = f"{metric}{suffix_alias}"
        if src_col in out.columns and alias_col not in out.columns:
            out[alias_col] = out[src_col]
    return out


def compute_l12w_delta(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute last-12-weeks rolling sum delta vs previous 12 weeks: adds *_l12w_abs and *_l12w_pct.
    Also mirrors to legacy alias names *_l12w_delta_abs and *_l12w_delta_pct if not already present.
    """
    if metric not in df.columns:
        return df
    result = df.copy()
    g, _ = _group_and_sort(result)

    def add(group: pd.DataFrame) -> pd.DataFrame:
        s = pd.to_numeric(group[metric], errors="coerce")
        roll12 = s.rolling(window=12, min_periods=4).sum()
        prev12 = roll12.shift(12)
        abs_change = roll12 - prev12
        pct_change = protect_zero_division(abs_change, prev12)
        base = metric
        group[f"{base}_l12w_abs"] = abs_change
        group[f"{base}_l12w_pct"] = pct_change
        # legacy aliases
        group[f"{base}_l12w_delta_abs"] = group[f"{base}_l12w_abs"]
        group[f"{base}_l12w_delta_pct"] = group[f"{base}_l12w_pct"]
        return group

    out = g.apply(add)
    for suffix_src, suffix_alias in [("_l12w_abs", "_l12w_delta_abs"), ("_l12w_pct", "_l12w_delta_pct")]:
        src_col = f"{metric}{suffix_src}"
        alias_col = f"{metric}{suffix_alias}"
        if src_col in out.columns and alias_col not in out.columns:
            out[alias_col] = out[src_col]
    return out


# -----------------------------
# Session 2b – Calendar utilities
# -----------------------------

SEASON_FIELDS_CANONICAL = [
    "season_label",
    "season_type",
    "event_code",
    "is_peak_week",
    "is_event_week",
    "quarter_label",
]


def ensure_week_key(df: pd.DataFrame, week_col: str = "canonical_week_id") -> None:
    """Validate presence of the canonical week column; raise clear error if missing.

    This is treated as critical by the scoring engine.
    """
    if week_col not in df.columns:
        raise ValueError(f"Required week key missing from dataset: {week_col}")


def _normalize_calendar_columns(cal_df: pd.DataFrame) -> pd.DataFrame:
    # normalize headers to lowercase/strip
    cal_df = cal_df.copy()
    cal_df.columns = [str(c).strip().lower() for c in cal_df.columns]
    return cal_df


def load_unified_calendar(path: str, join_key: str = "canonical_week_id", fields: Optional[List[str]] = None) -> pd.DataFrame:
    """Load unified calendar CSV as strings, validate keys and requested fields, and return normalized subset.

    - Reads with dtype=str to keep labels intact.
    - Ensures join_key exists; raises if missing.
    - If fields provided, ensures they exist; otherwise selects canonical season fields when present.
    """
    try:
        cal = pd.read_csv(path, dtype="string")
    except Exception as e:
        raise FileNotFoundError(f"Failed to read unified calendar file at {path}: {e}")

    cal = _normalize_calendar_columns(cal)

    if join_key not in cal.columns:
        raise ValueError(f"Unified calendar missing join key column: {join_key}")

    # Determine which fields to keep
    keep_fields = [join_key]
    requested = fields or SEASON_FIELDS_CANONICAL
    missing = [c for c in requested if c not in cal.columns]
    if requested and missing:
        # Not fatal for loader; return what exists and let enrichment fill defaults
        logging.getLogger(__name__).warning(
            "Unified calendar missing fields %s; they will be defaulted during enrichment.", missing,
        )
    for c in requested:
        if c in cal.columns and c not in keep_fields:
            keep_fields.append(c)

    return cal[keep_fields].copy()


def validate_season_uniqueness(calendar_df: pd.DataFrame, join_key: str) -> pd.DataFrame:
    """Ensure each join_key maps to exactly one season record; warn and drop duplicates deterministically (keep first)."""
    if calendar_df.empty:
        return calendar_df
    dup_mask = calendar_df.duplicated(subset=[join_key], keep=False)
    if bool(dup_mask.any()):
        logger = logging.getLogger(__name__)
        dup_keys = calendar_df.loc[dup_mask, join_key].astype("string").unique().tolist()
        logger.warning("Duplicate season records found for %d weeks; keeping first occurrences.", len(dup_keys))
        calendar_df = calendar_df.drop_duplicates(subset=[join_key], keep="first")
    return calendar_df


def enrich_with_seasonality(
    df: pd.DataFrame,
    calendar_df: Optional[pd.DataFrame],
    join_key: str,
    config_defaults: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Left-join seasonality columns onto df using join_key; fill missing fields with defaults.

    Ensures that all canonical season columns exist after enrichment.
    """
    defaults = {
        "season_label": "UNKNOWN_SEASON",
        "season_type": "BASELINE",
        "event_code": "NONE",
        "is_peak_week": False,
        "is_event_week": False,
        "quarter_label": None,
    }
    if config_defaults:
        defaults.update({k: config_defaults.get(k, defaults[k]) for k in defaults.keys()})

    out = df.copy()
    # Normalize join col type to string for safe join
    if join_key not in out.columns:
        raise ValueError(f"Cannot enrich with seasonality; join key missing in df: {join_key}")

    if calendar_df is not None and not calendar_df.empty:
        cal = _normalize_calendar_columns(calendar_df)
        cal = validate_season_uniqueness(cal, join_key)
        # Only keep join and season fields
        keep = [c for c in [join_key] + SEASON_FIELDS_CANONICAL if c in cal.columns]
        cal = cal[keep].copy()
        out = out.merge(cal, on=join_key, how="left")

    # Ensure columns and fill defaults
    for c in SEASON_FIELDS_CANONICAL:
        if c not in out.columns:
            out[c] = None
        # Fillna with defaults preserving dtype intent
        if c in ("is_peak_week", "is_event_week"):
            out[c] = out[c].astype("boolean")
            out[c] = out[c].fillna(bool(defaults[c]))
        else:
            out[c] = out[c].astype("string") if c != "quarter_label" else out[c].astype("string")
            default_val = defaults[c]
            out[c] = out[c].fillna(default_val if default_val is not None else pd.NA)

    return out
