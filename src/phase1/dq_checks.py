from __future__ import annotations

import re
from typing import Dict, Any, List, Tuple
from pathlib import Path
import yaml

import numpy as np
import pandas as pd


ASIN_RX = re.compile(r"^[A-Z0-9]{8,12}$")


def check_missing(df: pd.DataFrame) -> pd.Series:
    """Flag rows with missing core identifiers if present in the dataframe.

    Core identifiers considered: 'asin', 'vendor_id', 'week'. If a column does not
    exist, it is ignored in the check.
    """

    needed = [c for c in ("asin", "vendor_id", "week") if c in df.columns]
    if not needed:
        return pd.Series(False, index=df.index)
    return df[needed].isna().any(axis=1)


def _numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def check_numeric(df: pd.DataFrame) -> pd.Series:
    """Flag rows where any numeric column has obvious issues (NaN or negative).

    - Negative numbers in numeric columns are flagged.
    - NaN in numeric columns are not flagged here (left to missing), except when
      the column name suggests a required metric (heuristic minimal handling).
    """

    cols = _numeric_columns(df)
    if not cols:
        return pd.Series(False, index=df.index)
    negatives = (df[cols] < 0).any(axis=1)
    return negatives.fillna(False)


def check_ids(df: pd.DataFrame) -> pd.Series:
    """Flag rows with invalid identifiers: ASIN pattern, empty vendor_id, week=0.

    - ASIN must be 8-12 uppercase alphanumeric characters if column exists
    - vendor_id must be non-empty if column exists
    - week must be positive integer (YYYYWW) if column exists
    """

    flags = pd.Series(False, index=df.index)
    if "asin" in df.columns:
        asin_ok = df["asin"].fillna("").astype(str).str.fullmatch(ASIN_RX)
        flags |= ~asin_ok
    if "vendor_id" in df.columns:
        flags |= df["vendor_id"].isna() | (df["vendor_id"].astype(str).str.strip() == "")
    if "week" in df.columns:
        # week normalized as int YYYYWW; consider 0 invalid
        wk = pd.to_numeric(df["week"], errors="coerce")
        flags |= wk.isna() | (wk <= 0)
    return flags


def _percent_like_columns(df: pd.DataFrame) -> list[str]:
    names = [c for c in df.columns if isinstance(c, str)]
    percentish = [
        c
        for c in names
        if ("%" in c) or ("pct" in c.lower()) or ("percent" in c.lower())
    ]
    # Only keep numeric columns
    percentish = [c for c in percentish if pd.api.types.is_numeric_dtype(df[c])]
    return percentish


def _check_out_of_range_percentages(df: pd.DataFrame) -> pd.Series:
    cols = _percent_like_columns(df)
    if not cols:
        return pd.Series(False, index=df.index)
    # Treat 0..100 as valid domain; flag outside
    too_low = (df[cols] < 0).any(axis=1)
    too_high = (df[cols] > 100).any(axis=1)
    return (too_low | too_high).fillna(False)


def flag_missing(df: pd.DataFrame) -> pd.Series:
    """Missing core identifiers: asin/asin_id, vendor_id, canonical_week_id or week.

    Compatible with both legacy and new canonical names.
    """

    cols: List[str] = []
    for c in ("asin_id", "asin"):
        if c in df.columns:
            cols.append(c)
            break
    if "vendor_id" in df.columns:
        cols.append("vendor_id")
    for c in ("canonical_week_id", "week"):
        if c in df.columns:
            cols.append(c)
            break
    if not cols:
        return pd.Series(False, index=df.index)
    # Compute empties per column to avoid DataFrame.str usage
    is_na = df[cols].isna()
    as_str = df[cols].astype("string")
    is_empty = as_str.apply(lambda s: s.str.strip().eq(""))
    series = is_na | is_empty
    return series.any(axis=1)


def flag_invalid_ids(df: pd.DataFrame) -> pd.Series:
    """Invalid id formats: asin not matching pattern, week<=0, empty vendor."""

    return check_ids(df)


def flag_numeric_issues(df: pd.DataFrame, metric_config: Dict[str, Any] | None = None) -> pd.Series:
    """Identify impossible numeric values: negatives in count/value metrics.

    Uses metric_config numeric/currency fields if provided for a stricter check;
    otherwise falls back to all numeric columns.
    """

    if metric_config:
        cols = [
            *[c for c in metric_config.get("numeric_fields", []) if c in df.columns],
            *[c for c in metric_config.get("currency_fields", []) if c in df.columns],
        ]
        if cols:
            negatives = (df[cols] < 0).any(axis=1)
            return negatives.fillna(False)
    return check_numeric(df)


def flag_unmapped_weeks(df: pd.DataFrame) -> pd.Series:
    if "dq_unmapped_week" in df.columns:
        return df["dq_unmapped_week"].fillna(False)
    # If not present, infer from canonical_week_id
    if "canonical_week_id" in df.columns:
        return df["canonical_week_id"].isna() | (df["canonical_week_id"].astype(str).str.strip() == "")
    return pd.Series(False, index=df.index)


def apply_dq_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add DQ columns and a composite validity flag.

    New schema adds:
      - dq_missing
      - dq_invalid_ids
      - dq_numeric_issues
      - dq_unmapped_week
      - dq_impossible_values (alias for dq_numeric_issues to match spec wording)
      - dq_out_of_range (legacy percent range check)
      - dq_numeric_cast_fail (may be precomputed by metric_normalizer)
      - is_valid_row
    """

    result = df.copy()
    result["dq_missing"] = result.get("dq_missing", flag_missing(result))
    result["dq_invalid_ids"] = result.get("dq_invalid_ids", flag_invalid_ids(result))
    # numeric issues
    numeric_issues = check_numeric(result)
    result["dq_numeric_issues"] = result.get("dq_numeric_issues", numeric_issues)
    # impossible values aligns with numeric_issues per spec
    result["dq_impossible_values"] = result.get("dq_impossible_values", result["dq_numeric_issues"]) 
    # percent out-of-range legacy
    result["dq_out_of_range"] = result.get("dq_out_of_range", _check_out_of_range_percentages(result))
    # unmapped week
    result["dq_unmapped_week"] = result.get("dq_unmapped_week", flag_unmapped_weeks(result))

    # New flags with safe defaults; loaders may precompute and set them
    # dq_misplaced_id: default False
    if "dq_misplaced_id" not in result.columns:
        result["dq_misplaced_id"] = False
    else:
        result["dq_misplaced_id"] = result["dq_misplaced_id"].fillna(False).astype(bool)

    # dq_unbuyable: default False; can infer from is_buyable==False when present
    if "dq_unbuyable" not in result.columns:
        if "is_buyable" in result.columns:
            result["dq_unbuyable"] = result["is_buyable"].astype("boolean").fillna(False).eq(False)
        else:
            result["dq_unbuyable"] = False
    else:
        result["dq_unbuyable"] = result["dq_unbuyable"].fillna(False).astype(bool)

    # dq_buyability_missing: when buyability context is entirely absent
    if "dq_buyability_missing" not in result.columns:
        has_buy = any(c in result.columns for c in ("is_buyable", "listing_blocked", "suppressed_from_website", "no_retail_offer", "no_inventory_offer"))
        result["dq_buyability_missing"] = not has_buy
    else:
        result["dq_buyability_missing"] = result["dq_buyability_missing"].fillna(False).astype(bool)

    # dq_missing_item_name: default based on item_name empty when available
    if "dq_missing_item_name" not in result.columns:
        if "item_name" in result.columns:
            s = result["item_name"].astype("string").str.strip()
            result["dq_missing_item_name"] = s.eq("") | s.isna()
        else:
            result["dq_missing_item_name"] = False
    else:
        result["dq_missing_item_name"] = result["dq_missing_item_name"].fillna(False).astype(bool)

    # dq_category_mismatch: default False
    if "dq_category_mismatch" not in result.columns:
        result["dq_category_mismatch"] = False
    else:
        result["dq_category_mismatch"] = result["dq_category_mismatch"].fillna(False).astype(bool)

    # dq_week_coverage_low: default False
    if "dq_week_coverage_low" not in result.columns:
        result["dq_week_coverage_low"] = False
    else:
        result["dq_week_coverage_low"] = result["dq_week_coverage_low"].fillna(False).astype(bool)

    # dq_inferred_brand: default False
    if "dq_inferred_brand" not in result.columns:
        result["dq_inferred_brand"] = False
    else:
        result["dq_inferred_brand"] = result["dq_inferred_brand"].fillna(False).astype(bool)

    # dq_missing_brand: default based on brand/brand_name empty or UNKNOWN_BRAND when present
    if "dq_missing_brand" not in result.columns:
        if "brand" in result.columns or "brand_name" in result.columns:
            b = None
            if "brand" in result.columns:
                b = result["brand"].astype("string")
            else:
                b = result["brand_name"].astype("string")
            result["dq_missing_brand"] = b.isna() | b.str.strip().eq("") | b.str.upper().eq("UNKNOWN_BRAND")
        else:
            result["dq_missing_brand"] = False
    else:
        result["dq_missing_brand"] = result["dq_missing_brand"].fillna(False).astype(bool)

    # dq_inferred_category: default False
    if "dq_inferred_category" not in result.columns:
        result["dq_inferred_category"] = False
    else:
        result["dq_inferred_category"] = result["dq_inferred_category"].fillna(False).astype(bool)

    # dq_missing_category: default based on category empty or UNKNOWN_CATEGORY
    if "dq_missing_category" not in result.columns:
        if "category" in result.columns:
            c = result["category"].astype("string")
            result["dq_missing_category"] = c.isna() | c.str.strip().eq("") | c.str.upper().eq("UNKNOWN_CATEGORY")
        else:
            result["dq_missing_category"] = False
    else:
        result["dq_missing_category"] = result["dq_missing_category"].fillna(False).astype(bool)

    # New Session 3 flags: ensure safe defaults
    for col in [
        "dq_category_conflict",
        "dq_category_inferred",
        "dq_archetype_unassigned",
    ]:
        if col not in result.columns:
            result[col] = False
        else:
            result[col] = result[col].fillna(False).astype(bool)

    # Aliases for unknown brand/category (to match spec wording)
    if "dq_unknown_brand" not in result.columns:
        result["dq_unknown_brand"] = result.get("dq_missing_brand", pd.Series(False, index=result.index))
    else:
        result["dq_unknown_brand"] = result["dq_unknown_brand"].fillna(False).astype(bool)
    if "dq_unknown_category" not in result.columns:
        result["dq_unknown_category"] = result.get("dq_missing_category", pd.Series(False, index=result.index))
    else:
        result["dq_unknown_category"] = result["dq_unknown_category"].fillna(False).astype(bool)

    # dq_inventory_risk: roll-up if any known inventory risk flags present
    inv_flags = [c for c in ("inventory_risk_lcpo_only", "inventory_risk_overstock", "inventory_risk_oos", "inventory_risk_aging") if c in result.columns]
    if "dq_inventory_risk" not in result.columns:
        result["dq_inventory_risk"] = result[inv_flags].any(axis=1) if inv_flags else False
    else:
        result["dq_inventory_risk"] = result["dq_inventory_risk"].fillna(False).astype(bool)

    # dq_contra_mapping_broken: when agreement join failed but contra inputs indicated expected mapping
    if "dq_contra_mapping_broken" not in result.columns:
        # Heuristic: if a column hints contra was expected (e.g., contra_expected) but no active flag present
        exp = result.get("contra_expected", pd.Series(False, index=result.index))
        act = result.get("contra_active_flag", pd.Series(False, index=result.index))
        try:
            result["dq_contra_mapping_broken"] = exp.astype(bool) & (~act.astype(bool))
        except Exception:
            result["dq_contra_mapping_broken"] = False
    else:
        result["dq_contra_mapping_broken"] = result["dq_contra_mapping_broken"].fillna(False).astype(bool)

    # dq_invalid_vendor_code: vendor_id fails validation (basic heuristic here)
    if "dq_invalid_vendor_code" not in result.columns:
        if "vendor_id" in result.columns:
            s = result["vendor_id"].astype("string").str.strip()
            result["dq_invalid_vendor_code"] = s.eq("")
        else:
            result["dq_invalid_vendor_code"] = False
    else:
        result["dq_invalid_vendor_code"] = result["dq_invalid_vendor_code"].fillna(False).astype(bool)
    # numeric cast fail may be set by metric_normalizer; ensure boolean
    if "dq_numeric_cast_fail" in result.columns:
        result["dq_numeric_cast_fail"] = result["dq_numeric_cast_fail"].fillna(False).astype(bool)
    else:
        result["dq_numeric_cast_fail"] = False

    # Attach per-flag severity columns without altering boolean flags
    # Example: for flag 'dq_unmapped_week' add 'dq_unmapped_week_severity' with level when True else NA
    sev_map = get_dq_severity_map()
    result = attach_flag_severities(result, sev_map)
    # Compute overall dq_severity from per-flag severities (max)
    dq_sev = classify_row_severity(result, sev_map)
    result["dq_severity"] = dq_sev.astype("string")
    # Only CRITICAL rows are invalid
    crit_mask = result["dq_severity"].astype("string").str.upper().eq("CRITICAL")
    # If dq_severity is None/NA, treat as valid
    crit_mask = crit_mask.fillna(False)
    result["is_valid_row"] = ~crit_mask
    return result


# ---------------- Severity classification and escalation utilities ----------------

SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]


def _max_severity(a: str | None, b: str | None) -> str | None:
    if a is None:
        return b
    if b is None:
        return a
    ia = SEVERITY_ORDER.index(a) if a in SEVERITY_ORDER else -1
    ib = SEVERITY_ORDER.index(b) if b in SEVERITY_ORDER else -1
    return a if ia >= ib else b


def flag_severity_mapping() -> Dict[str, str]:
    """Return default mapping of DQ flags to severity classes per Phase 1e/1f spec.

    This is used as a fallback if config/dq_severity.yaml is not present.
    """

    base = {
        # Critical
        "dq_unmapped_week": "Critical",
        "dq_misplaced_id": "Critical",
        # High
        "dq_invalid_ids": "High",
        "dq_invalid_vendor_code": "High",
        "dq_buyability_missing": "High",
        "dq_unbuyable": "High",
        # Medium
        "dq_inventory_risk": "Medium",
        "dq_contra_mapping_broken": "Medium",
        "dq_week_coverage_low": "Medium",
        # Low
        "dq_missing_item_name": "Low",
        "dq_missing_brand": "Low",
        "dq_unknown_brand": "Low",
        "dq_unknown_category": "Low",
        "dq_numeric_issues": "Low",
        "dq_out_of_range": "Low",
        "dq_impossible_values": "Low",
        "dq_numeric_cast_fail": "Low",
    }
    # Add Session 3 category/archetype flags as Low severity by default (informative, non-fatal)
    base.update({
        "dq_category_conflict": "Low",
        "dq_category_inferred": "Low",
        "dq_archetype_unassigned": "Low",
    })
    return base


def attach_flag_severities(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Create per-flag severity string columns alongside boolean dq_* flags.

    For each mapped flag 'dq_x', creates 'dq_x_severity' that contains the normalized
    severity label when the boolean flag is True; otherwise NA. Does not modify the
    boolean flag column itself.
    """

    out = df.copy()
    for flag, level in (mapping or {}).items():
        sev_col = f"{flag}_severity"
        if flag in out.columns:
            mask = out[flag].fillna(False).astype(bool)
            col = pd.Series([pd.NA] * len(out), index=out.index, dtype="string")
            col = col.where(~mask, _normalize_level(level))
            out[sev_col] = col
        else:
            # Still create the column for schema stability (all NA)
            out[sev_col] = pd.Series([pd.NA] * len(out), index=out.index, dtype="string")
    return out


def _normalize_level(level: str | None) -> str | None:
    if level is None:
        return None
    s = str(level).strip().lower()
    if not s:
        return None
    if s.startswith("crit"):
        return "Critical"
    if s.startswith("high"):
        return "High"
    if s.startswith("med"):
        return "Medium"
    if s.startswith("low"):
        return "Low"
    # Unknown tokens -> Low (non-fatal) by default
    return "Low"


def get_dq_severity_map(config_path: str | None = None) -> Dict[str, str]:
    """Load dq severity mapping from YAML, falling back to defaults.

    YAML structure: flat mapping flag -> LEVEL (case-insensitive)
    Example keys per spec: dq_invalid_asin, dq_invalid_vendor_code, dq_misplaced_id, ...
    Unknown flags are accepted; LEVEL tokens are normalized to title-case.
    """

    # Start with defaults
    mapping = flag_severity_mapping()
    # Attempt to load override/extension
    path = Path(config_path) if config_path else Path("config") / "dq_severity.yaml"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            for flag, lvl in (cfg or {}).items():
                norm_level = _normalize_level(lvl)
                if norm_level is not None:
                    mapping[str(flag)] = norm_level
        except Exception:
            # Ignore config errors; keep defaults
            pass
    return mapping


def assign_dq_severity(row_dq_flags: Dict[str, Any] | pd.Series, mapping: Dict[str, str] | None = None) -> str | None:
    """Return the highest severity present in a row's DQ flags using mapping.

    row_dq_flags should be a Series or dict-like with dq_* boolean items.
    Returns one of {"Critical","High","Medium","Low"} or None if no flags True.
    """

    m = mapping or get_dq_severity_map()
    max_level: str | None = None
    for flag, level in m.items():
        try:
            active = bool(row_dq_flags.get(flag, False))
        except Exception:
            active = False
        if active:
            max_level = _max_severity(max_level, level)
    return max_level


def is_critical_failure(severity: str | None) -> bool:
    """Return True if severity string represents a CRITICAL failure."""
    return str(severity).strip().upper() == "CRITICAL"


def classify_row_severity(df: pd.DataFrame, mapping: Dict[str, str] | None = None) -> pd.Series:
    """Compute max severity per row from per-flag severities if present, else from booleans.

    Preference order:
      1) *_severity columns (e.g., dq_unmapped_week_severity)
      2) boolean dq_* columns with provided mapping
    Returns Series of severity labels or None per row.
    """

    m = mapping or get_dq_severity_map()
    # If *_severity columns exist, compute row max directly from them
    sev_cols = [f"{flag}_severity" for flag in m.keys() if f"{flag}_severity" in df.columns]
    if sev_cols:
        # Map textual levels to order values for max
        order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        mat = pd.DataFrame({c: df[c].astype("string").map(lambda s: order.get(str(s).title(), np.nan)) for c in sev_cols})
        max_idx = mat.max(axis=1, skipna=True)
        inv = {v: k for k, v in order.items()}
        return max_idx.map(lambda v: inv.get(v) if pd.notna(v) else None)

    # Fallback to boolean flags
    sev = pd.Series([None] * len(df), index=df.index, dtype="object")
    for flag, level in m.items():
        if flag in df.columns:
            mask = df[flag].fillna(False).astype(bool)
            # ensure we keep max severity when multiple flags apply
            for i in df.index[mask]:
                sev.at[i] = _max_severity(sev.at[i], level)
    return sev


def aggregate_severity_counts(df: pd.DataFrame, mapping: Dict[str, str] | None = None) -> Dict[str, int]:
    """Return counts of rows per severity bucket using dq_severity if present.

    Falls back to deriving from flags if dq_severity column is absent.
    """
    if "dq_severity" in df.columns:
        s = df["dq_severity"].astype("string").fillna("").str.title()
        return {k: int((s == k).sum()) for k in SEVERITY_ORDER}
    # Derive from flags
    sev_series = classify_row_severity(df, mapping)
    s = sev_series.astype("string").fillna("").str.title()
    return {k: int((s == k).sum()) for k in SEVERITY_ORDER}


def build_attention_rows(df: pd.DataFrame, mapping: Dict[str, str] | None = None) -> pd.DataFrame:
    """Return subset of rows that require attention (High or Medium)."""

    sev_series = classify_row_severity(df, mapping)
    out = df.copy()
    out["dq_severity"] = sev_series
    attn = out[out["dq_severity"].isin(["High", "Medium"])].copy()
    # small friendly set of columns
    keep = [c for c in ["vendor_id", "asin_id", "canonical_week_id", "dq_severity"] if c in attn.columns]
    # add a summary flags column
    dq_cols = [c for c in attn.columns if str(c).startswith("dq_")]
    if dq_cols:
        attn["dq_flags"] = attn[dq_cols].apply(lambda r: ",".join([c for c in dq_cols if bool(r.get(c, False))]) , axis=1)
    return attn[keep + (["dq_flags"] if "dq_flags" in attn.columns else [])]
