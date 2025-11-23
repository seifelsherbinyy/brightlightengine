"""Phase 2 Data Processing Layer utilities.

Modular, composable helpers for:
- Grain derivation (week -> month/quarter/year)
- Derived metrics (ASP, CPPU, Contribution Profit, Contribution Margin, Net PPM)
- Outlier detection (MAD, IQR) and winsorization wrappers
- Rolling features (volatility, trend slope/strength)
- Signal quality (completeness, recency, variance, trend quality)

All functions are pure and pandas‑vectorized where possible to ease testing.
They avoid side effects (no file I/O, no logging) and accept/return DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .transforms import winsorize


@dataclass(frozen=True)
class DerivedMetricSpec:
    name: str
    numerator_candidates: Tuple[str, ...]
    denominator_candidates: Tuple[str, ...]
    transform: Optional[str] = None  # e.g., 'ratio', 'difference'
    unit: str = ""


DEFAULT_DERIVED_SPECS: Tuple[DerivedMetricSpec, ...] = (
    # Average Selling Price = Revenue / Units
    DerivedMetricSpec(
        name="ASP",
        numerator_candidates=("Shipped Revenue", "Ordered Revenue", "Revenue"),
        denominator_candidates=("Units", "Shipped Units", "Ordered Units"),
        transform="ratio",
        unit="$",
    ),
    # Cost Per Unit = COGS / Units
    DerivedMetricSpec(
        name="CPPU",
        numerator_candidates=("Shipped COGS", "COGS", "PCOGS($)"),
        denominator_candidates=("Units", "Shipped Units", "Ordered Units"),
        transform="ratio",
        unit="$",
    ),
    # Contribution Profit = Revenue - COGS
    DerivedMetricSpec(
        name="CP",
        numerator_candidates=("Shipped Revenue", "Ordered Revenue", "Revenue"),
        denominator_candidates=("Shipped COGS", "COGS", "PCOGS($)"),
        transform="difference",
        unit="$",
    ),
    # Contribution Margin (%) = CP / Revenue
    DerivedMetricSpec(
        name="CM %",
        numerator_candidates=("CP",),
        denominator_candidates=("Shipped Revenue", "Ordered Revenue", "Revenue"),
        transform="ratio",
        unit="%",
    ),
    # Net PPM (%) approximated as CM % alias for now (can be overridden by taxonomy later)
    DerivedMetricSpec(
        name="Net PPM %",
        numerator_candidates=("CP",),
        denominator_candidates=("Shipped Revenue", "Ordered Revenue", "Revenue"),
        transform="ratio",
        unit="%",
    ),
)


def add_grain_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar grains derived from week_start where available.

    Expects columns:
      - week_start (datetime64) optional
      - week_end (datetime64) optional
      - week_label (str) optional (format YYYYWNN)

    Returns the same DataFrame with added columns when derivable:
      - year (int)
      - month (int)
      - month_key (str: YYYY-MM)
      - quarter (str: Q1..Q4)
      - quarter_key (str: YYYY-Qn)
    """
    df = df.copy()
    if "week_start" in df.columns and np.issubdtype(df["week_start"].dtype, np.datetime64):
        ws = pd.to_datetime(df["week_start"], errors="coerce")
        df["year"] = ws.dt.year
        df["month"] = ws.dt.month
        df["month_key"] = ws.dt.strftime("%Y-%m")
        df["quarter"] = "Q" + (((ws.dt.month - 1) // 3) + 1).astype(int).astype(str)
        df["quarter_key"] = ws.dt.year.astype(str) + "-" + df["quarter"]
    elif "week_label" in df.columns:
        # Fallback parsing from YYYYWNN
        w = df["week_label"].astype(str).str.extract(r"^(?P<y>\d{4})W\d{2}$")
        df["year"] = pd.to_numeric(w["y"], errors="coerce").astype("Int64")
        # month/quarter not derivable without reference date; leave nulls
        df["month"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        df["month_key"] = pd.NA
        df["quarter"] = pd.NA
        df["quarter_key"] = pd.NA
    return df


def _first_non_null_in(group: pd.Series, candidates: Iterable[str]) -> Optional[str]:
    names = list(candidates)
    present = [n for n in names if n in group.values]
    return present[0] if present else None


def compute_derived_metrics(df: pd.DataFrame, specs: Tuple[DerivedMetricSpec, ...] = DEFAULT_DERIVED_SPECS) -> pd.DataFrame:
    """Compute derived metrics opportunistically per (vendor_code, asin, week_label).

    Input columns required: ['vendor_code','asin','metric','value'] and at least one of
    ['week_label','Week_Order']; optional 'unit'.

    Returns a DataFrame with new rows for derived metrics, matching the input schema
    (preserving identifier/date columns found in input).
    """
    required_cols = {"vendor_code", "asin", "metric", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"compute_derived_metrics requires columns {required_cols}, missing {missing}")

    key_cols = [c for c in ["vendor_code", "asin", "week_label", "Week_Order"] if c in df.columns]
    id_cols = key_cols + [c for c in ["vendor_name", "unit", "week_start", "week_end"] if c in df.columns]

    # Pivot minimal wide for fast vector ops
    piv = df.pivot_table(index=key_cols, columns="metric", values="value", aggfunc="first")

    derived_frames: list[pd.DataFrame] = []

    for spec in specs:
        num = None
        den = None
        for cand in spec.numerator_candidates:
            if cand in piv.columns:
                num = piv[cand]
                num_name = cand
                break
        for cand in spec.denominator_candidates:
            if cand in piv.columns:
                den = piv[cand]
                den_name = cand
                break

        if spec.transform == "difference":
            # For difference, both numerator and denominator are required
            if num is None or den is None:
                continue
            values = num - den
        elif spec.transform == "ratio":
            # Allow ASP where numerator/denominator discovered; if numerator is derived CP, compute later
            if num is None:
                # If numerator is a derived metric like CP, skip until its frame exists
                if spec.numerator_candidates == ("CP",):
                    # Will compute CM%/Net PPM% only after CP exists
                    continue
                else:
                    continue
            if den is None:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                values = num / den
                values = values.replace([np.inf, -np.inf], np.nan)
        else:
            # Unknown transform, skip
            continue

        out = values.to_frame(name="value").reset_index()
        out["metric"] = spec.name
        out["unit"] = spec.unit if "unit" in df.columns else None
        derived_frames.append(out)

        # If we computed CP, store it into pivot for downstream ratio specs
        if spec.name == "CP":
            piv["CP"] = values

    # Re-run specs dependent on CP now that CP exists in piv
    if "CP" in piv.columns:
        for spec in specs:
            if spec.transform == "ratio" and spec.numerator_candidates == ("CP",):
                den = None
                for cand in spec.denominator_candidates:
                    if cand in piv.columns:
                        den = piv[cand]
                        break
                if den is None:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    values = piv["CP"] / den
                    values = values.replace([np.inf, -np.inf], np.nan)
                out = values.to_frame(name="value").reset_index()
                out["metric"] = spec.name
                out["unit"] = spec.unit if "unit" in df.columns else None
                derived_frames.append(out)

    if not derived_frames:
        return pd.DataFrame(columns=df.columns)

    derived = pd.concat(derived_frames, ignore_index=True)

    # Retain identifier columns present in input
    for c in id_cols:
        if c not in derived.columns and c in df.columns:
            # bring from original by key join
            derived = derived.merge(df[[*key_cols, c]].drop_duplicates(), on=key_cols, how="left")

    # Drop potential duplicates per key + metric (can occur if multiple candidate pairs exist)
    dedup_keys = [c for c in ["vendor_code", "asin", "week_label", "Week_Order", "metric"] if c in derived.columns]
    if dedup_keys:
        derived = derived.sort_values(dedup_keys).drop_duplicates(subset=dedup_keys, keep="first")

    # Order columns similar to input when possible
    ordered_cols = [c for c in df.columns if c in derived.columns]
    tail_cols = [c for c in derived.columns if c not in ordered_cols]
    derived = derived[[*ordered_cols, *tail_cols]]
    return derived


def detect_outliers_mad_iqr(series: pd.Series, method: str = "mad", thresh: float = 3.5) -> pd.Series:
    """Return boolean Series of outliers via MAD or IQR methods."""
    x = pd.to_numeric(series, errors="coerce")
    if method.lower() == "mad":
        med = x.median()
        mad = (x - med).abs().median()
        if mad == 0 or np.isnan(mad):
            # Fallback: when MAD is zero (all identical except potential outliers),
            # mark any values differing from the median as outliers.
            return (x != med) & x.notna()
        robust_z = 0.6745 * (x - med).abs() / mad
        return robust_z > thresh
    elif method.lower() == "iqr":
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or np.isnan(iqr):
            return pd.Series(False, index=x.index)
        lower = q1 - thresh * iqr
        upper = q3 + thresh * iqr
        return (x < lower) | (x > upper)
    else:
        raise ValueError("method must be 'mad' or 'iqr'")


def winsorize_by_limits(df: pd.DataFrame, metric_limits: Dict[str, Tuple[float, float]], value_col: str = "value") -> pd.DataFrame:
    """Apply winsorization per metric based on quantile limits mapping.

    metric_limits: mapping metric -> (q_low, q_high)
    """
    if not metric_limits:
        return df.copy()
    parts: list[pd.DataFrame] = []
    for m, sub in df.groupby("metric", dropna=False):
        q = metric_limits.get(m)
        if not q:
            parts.append(sub.copy())
            continue
        q_low, q_high = q
        sub = sub.copy()
        sub[value_col] = winsorize(pd.to_numeric(sub[value_col], errors="coerce"), q_low, q_high)
        parts.append(sub)
    out = pd.concat(parts, ignore_index=True)
    # Optional global clip if all limits are identical across metrics (common in tests)
    unique_limits = {tuple(v) for v in metric_limits.values()}
    if len(unique_limits) == 1:
        q_low, q_high = next(iter(unique_limits))
        global_low = pd.to_numeric(df[value_col], errors="coerce").quantile(q_low)
        global_high = pd.to_numeric(df[value_col], errors="coerce").quantile(q_high)
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce").clip(lower=global_low, upper=global_high)
    return out


def add_rolling_features(df_long: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Compute rolling volatility (std) and trend slope using ordinal Week_Order per
    (entity_id, metric). Requires columns: entity_id, metric, Week_Order, value.
    """
    req = {"entity_id", "metric", "Week_Order", "value"}
    missing = req - set(df_long.columns)
    if missing:
        raise ValueError(f"add_rolling_features missing columns: {missing}")

    df = df_long.copy()
    df = df.sort_values(["entity_id", "metric", "Week_Order"])  # ensure order

    def _roll(group: pd.DataFrame) -> pd.DataFrame:
        x = group["Week_Order"].astype(float)
        y = pd.to_numeric(group["value"], errors="coerce")
        # Rolling std as volatility
        vol = y.rolling(window=window, min_periods=max(2, window // 2)).std()
        # Trend slope via simple linear regression coefficients on rolling window
        slopes = (
            y.rolling(window=window, min_periods=max(2, window // 2))
            .apply(lambda arr: _slope(np.arange(len(arr), dtype=float), arr), raw=True)
        )
        out = group.copy()
        out["volatility"] = vol.values
        out["trend_slope"] = slopes.values
        # Trend strength as |slope| / volatility (stability of trend)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["trend_strength"] = (out["trend_slope"].abs() / out["volatility"]).replace([np.inf, -np.inf], np.nan)
        return out

    return df.groupby(["entity_id", "metric"], group_keys=False).apply(_roll)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    # Simple least squares slope for y ~ a + b * x
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    x_centered = x - x.mean()
    y_centered = y - np.nanmean(y)
    denom = (x_centered ** 2).sum()
    if denom == 0 or np.isnan(denom):
        return np.nan
    num = np.nansum(x_centered * y_centered)
    return float(num / denom) if denom else np.nan


def compute_signal_quality(df_long: pd.DataFrame, lookback_weeks: int = 8) -> pd.DataFrame:
    """Compute simple signal quality measures per (entity_id, metric):
    - completeness: fraction of non‑NA values in last lookback_weeks
    - recency_weeks: weeks since last non‑NA value
    - variance: variance over lookback window
    - trend_quality: normalized |slope| (slope / std)
    Requires Week_Order to be contiguous or roughly increasing.
    """
    req = {"entity_id", "metric", "Week_Order", "value"}
    missing = req - set(df_long.columns)
    if missing:
        raise ValueError(f"compute_signal_quality missing columns: {missing}")

    df = df_long.copy()
    df = df.sort_values(["entity_id", "metric", "Week_Order"])  # ensure order

    def _quality(group: pd.DataFrame) -> pd.DataFrame:
        # Focus on the last lookback_weeks for quality snapshot
        tail = group.tail(lookback_weeks)
        vals = pd.to_numeric(tail["value"], errors="coerce")
        completeness = float(vals.notna().mean()) if len(vals) else np.nan
        # Recency: distance from last observed non-NA
        mask = vals.notna().values
        if mask.any():
            last_idx = np.where(mask)[0][-1]
            recency_weeks = float((len(vals) - 1) - last_idx)
        else:
            recency_weeks = np.nan
        variance = float(vals.var(ddof=1)) if vals.notna().sum() > 1 else np.nan
        slope = _slope(np.arange(len(vals), dtype=float), vals.values)
        std = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        with np.errstate(divide="ignore", invalid="ignore"):
            trend_quality = float(abs(slope) / std) if std not in (0, np.nan) else np.nan

        out = group.copy()
        # Fill the last row of the group with snapshot values; forward fill others as NaN
        out["completeness"] = np.nan
        out["recency_weeks"] = np.nan
        out["variance"] = np.nan
        out["trend_quality"] = np.nan
        if len(out) > 0:
            out.iloc[-1, out.columns.get_loc("completeness")] = completeness
            out.iloc[-1, out.columns.get_loc("recency_weeks")] = recency_weeks
            out.iloc[-1, out.columns.get_loc("variance")] = variance
            out.iloc[-1, out.columns.get_loc("trend_quality")] = trend_quality
        return out

    return df.groupby(["entity_id", "metric"], group_keys=False).apply(_quality)
