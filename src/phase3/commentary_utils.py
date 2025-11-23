from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def _safe_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    return None


def _mk_event_base(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ["vendor_id", "asin_id", "canonical_week_id"] if c in df.columns]
    return df[keep].copy()


def detect_metric_movements(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    thresholds = config.get("thresholds", {}).get("metric_movements", {})
    candidates = [
        ("gms", "gms_wow_pct"),
        ("units", "units_wow_pct"),
        ("gv", "gv_wow_pct"),
        ("instock", "instock_pct_wow_pct"),
        ("fo_pct", "fo_pct_wow_pct"),
        ("returns", "returns_rate_wow_pct"),
        ("asp", "asp_wow_pct"),
    ]
    parts: list[pd.DataFrame] = []
    for metric, delta_col in candidates:
        col = _safe_col(df, [delta_col])
        if not col:
            continue
        thr = float(thresholds.get(metric, 0.15))
        tmp = df.loc[df[col].abs() >= thr].copy()
        if tmp.empty:
            continue
        ev = _mk_event_base(tmp)
        ev["signal_type"] = f"metric_movement:{metric}"
        ev["severity"] = (tmp[col].abs()).rank(pct=True)
        ev["metric"] = metric
        ev["delta_pct"] = tmp[col]
        parts.append(ev)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def detect_dimension_shifts(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    dims = ["profit", "growth", "availability", "inventory_risk", "quality"]
    thresholds = config.get("thresholds", {}).get("dimension_shifts", {})
    parts: list[pd.DataFrame] = []
    for d in dims:
        score = _safe_col(df, [f"{d}_score", f"score_{d}"])
        wow = _safe_col(df, [f"{d}_score_wow_pct", f"{d}_wow_pct", f"wow_{d}_score"])
        if not score or not wow:
            continue
        thr = float(thresholds.get(d, 0.1))
        tmp = df.loc[df[wow].abs() >= thr].copy()
        if tmp.empty:
            continue
        ev = _mk_event_base(tmp)
        ev["signal_type"] = f"dimension_shift:{d}"
        ev["severity"] = (tmp[wow].abs()).rank(pct=True)
        ev["dimension"] = d
        ev["delta_pct"] = tmp[wow]
        ev["score"] = tmp[score]
        parts.append(ev)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def detect_inventory_risks(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    flags = ["hcpo_present", "lcpo_only", "aging", "overstock", "oos_risk"]
    parts: list[pd.DataFrame] = []
    cols = {c.lower(): c for c in df.columns}
    for f in flags:
        if f in cols:
            col = cols[f]
            tmp = df.loc[df[col] == True].copy()  # noqa: E712
            if tmp.empty:
                continue
            ev = _mk_event_base(tmp)
            ev["signal_type"] = f"inventory_risk:{f}"
            ev["severity"] = 1.0
            parts.append(ev)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def detect_buyability_failures(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    flags = [
        "listing_blocked",
        "suppressed_from_website",
        "no_retail_offer",
        "no_inventory_offer",
    ]
    parts: list[pd.DataFrame] = []
    cols = {c.lower(): c for c in df.columns}
    for f in flags:
        if f in cols:
            col = cols[f]
            tmp = df.loc[df[col] == True].copy()  # noqa: E712
            if tmp.empty:
                continue
            ev = _mk_event_base(tmp)
            ev["signal_type"] = f"buyability_failure:{f}"
            ev["severity"] = 1.0
            parts.append(ev)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def detect_category_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Simple placeholder: catch extreme composite_score changes if available
    if df is None or df.empty:
        return pd.DataFrame()
    cols = {c.lower(): c for c in df.columns}
    comp = cols.get("composite_score")
    if not comp:
        return pd.DataFrame()
    # Rank by composite score; low or high extremes
    tmp = df.copy()
    tmp["_rank"] = tmp[comp].rank(pct=True)
    out = tmp.loc[(tmp["_rank"] <= 0.02) | (tmp["_rank"] >= 0.98)].copy()
    if out.empty:
        return pd.DataFrame()
    ev = _mk_event_base(out)
    ev["signal_type"] = "category_outlier:composite"
    ev["severity"] = (out["_rank"] - 0.5).abs()
    return ev
