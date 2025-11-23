from __future__ import annotations

import math
import re
from typing import Any, Dict

import numpy as np
import pandas as pd


_CURRENCY_RX = re.compile(r"[\s$,£€‚ƒ₣₹¥₩₽₺,\u00a0]")


def strip_currency(value: str) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    if s == "":
        return None
    # remove currency symbols, spaces, and thousands separators
    s = _CURRENCY_RX.sub("", s)
    # handle parentheses for negatives e.g., (1,234.50)
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    try:
        val = float(s)
        return -val if neg else val
    except Exception:
        return None


def convert_percent(value: str) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    if s == "":
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1].replace(",", "").strip()) / 100.0
        except Exception:
            return None
    # Already maybe 0..1 number
    try:
        v = float(s.replace(",", ""))
        # If it's > 1 and <= 100 assume percent-like and convert
        if v > 1 and v <= 100:
            return v / 100.0
        return v
    except Exception:
        return None


def cast_numeric(value: str) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        return float(s.replace(",", ""))
    except Exception:
        return None


def normalize_metrics(df: pd.DataFrame, metric_config: Dict[str, Any]) -> pd.DataFrame:
    """Normalize metrics in-place according to metric_config.

    metric_config keys:
      - numeric_fields: list[str]
      - percent_fields: list[str]
      - currency_fields: list[str]

    Adds a boolean column 'dq_numeric_cast_fail' if any conversions fail per-row.
    Enforces float64 for all listed metric columns.
    """

    result = df.copy()
    numeric_fields = [c for c in metric_config.get("numeric_fields", []) if c in result.columns]
    percent_fields = [c for c in metric_config.get("percent_fields", []) if c in result.columns]
    currency_fields = [c for c in metric_config.get("currency_fields", []) if c in result.columns]

    cast_fail = pd.Series(False, index=result.index)

    for c in currency_fields:
        vals = result[c].map(strip_currency)
        cast_fail |= vals.isna() & result[c].notna() & (result[c].astype(str).str.strip() != "")
        result[c] = vals.astype("float64")

    for c in percent_fields:
        vals = result[c].map(convert_percent)
        cast_fail |= vals.isna() & result[c].notna() & (result[c].astype(str).str.strip() != "")
        result[c] = vals.astype("float64")

    for c in numeric_fields:
        vals = result[c].map(cast_numeric)
        cast_fail |= vals.isna() & result[c].notna() & (result[c].astype(str).str.strip() != "")
        result[c] = vals.astype("float64")

    # Enforce dtype float64 for all listed columns
    for c in set(numeric_fields + percent_fields + currency_fields):
        if c in result.columns:
            result[c] = pd.to_numeric(result[c], errors="coerce").astype("float64")

    result["dq_numeric_cast_fail"] = cast_fail.fillna(False)
    return result
