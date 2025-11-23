from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def filter_high_impact_events(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    # Keep events which have severity above minimal threshold (if numeric)
    min_sev = float(config.get("filters", {}).get("min_severity", 0.15))
    if "severity" in df.columns:
        # coerce to numeric where possible
        df["severity_num"] = pd.to_numeric(df["severity"], errors="coerce")
        mask = df["severity_num"].fillna(0) >= min_sev
        df = df.loc[mask].drop(columns=["severity_num"]) if not df.empty else df
    return df


def dedupe_related_signals(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # Drop duplicates by vendor/asin/week/signal_type
    keys = [c for c in ["vendor_id", "asin_id", "canonical_week_id", "signal_type"] if c in df.columns]
    if not keys:
        return df
    # Keep the most severe within each group deterministically
    df = df.copy()
    df["_sev"] = pd.to_numeric(df.get("severity", 0), errors="coerce").fillna(0)
    df.sort_values(by=keys + ["_sev"], ascending=[True, True, True, False], inplace=True)
    out = df.drop_duplicates(subset=keys, keep="first").drop(columns=["_sev"])
    return out


def prioritize_by_materiality(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    # Score for priority: severity primarily; add tie-breakers
    df["_sev"] = pd.to_numeric(df.get("severity", 0), errors="coerce").fillna(0)
    df.sort_values(by=["vendor_id", "_sev", "signal_type", "asin_id", "canonical_week_id"], ascending=[True, False, True, True, False], inplace=True)

    # Cap per vendor between min and max
    caps = config.get("filters", {}).get("per_vendor", {})
    cap_min = int(caps.get("min", 3))
    cap_max = int(caps.get("max", 7))
    # Group and take top N per vendor
    out = (
        df.groupby("vendor_id", as_index=False, group_keys=False)
        .apply(lambda g: g.head(cap_max))
        .reset_index(drop=True)
    )

    # Ensure at least min per vendor if available (already satisfied by head); nothing to add deterministically
    out.drop(columns=["_sev"], inplace=True)
    return out
