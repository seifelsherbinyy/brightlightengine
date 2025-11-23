"""Pre-execution sanity checks for Phase 4 outputs readiness.

Runs a lightweight Phase 1 → Phase 2 (enhanced) pipeline on available input
and validates the presence of Patch-3 fields before proceeding to Phase 4.

No file I/O, no prints. Pure functions returning structured readiness state.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Optional, Any
import numpy as np

import pandas as pd

from ..ingestion_utils import load_config
from ..phase2_scoring_enhanced import build_wide_composite
from ..common.config_validator import load_and_validate_config, MetricConfig


PATCH3_LONG_REQUIRED: Set[str] = {
    # base
    "entity_id",
    "vendor_code",
    "Week_Order",
    "metric",
    "raw_value",
    "std_value",
    # rolling
    "volatility",
    "trend_slope",
    "trend_strength",
    # quality
    "completeness",
    "recency_weeks",
    "variance",
    "trend_quality",
    # weights
    "weight_effective",
    "weight_multiplier",
    "base_weight",
}


@dataclass(frozen=True)
class ReadyState:
    ok: bool
    errors: List[str]
    summary: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:  # convenience for metadata
        return {"ok": self.ok, "errors": list(self.errors), "summary": dict(self.summary)}


def _validate_long_fields(df_long: pd.DataFrame) -> List[str]:
    errors: List[str] = []
    missing = [c for c in PATCH3_LONG_REQUIRED if c not in df_long.columns]
    if missing:
        errors.append(f"Missing required long fields: {missing}")
    return errors


def _validate_wide_fields(df_wide: pd.DataFrame, metrics: List[str]) -> List[str]:
    errors: List[str] = []
    # Ensure composite and audit columns exist for at least one known metric
    if "score_composite" not in df_wide.columns:
        errors.append("Wide matrix missing 'score_composite'")
    for m in metrics:
        # if std column present, contrib/weight audit should also be present
        std_col = f"std_{m}"
        contrib_col = f"contrib_{m}"
        if std_col in df_wide.columns:
            for col in (contrib_col, f"weight_{m}", f"weightmul_{m}", f"base_weight_{m}"):
                if col not in df_wide.columns:
                    errors.append(f"Wide matrix missing audit column '{col}' for metric {m}")
    return errors


def _approx_equal(a: pd.Series, b: pd.Series, tol: float = 1e-6) -> pd.Series:
    import numpy as np
    a = pd.to_numeric(a, errors="coerce").fillna(0.0)
    b = pd.to_numeric(b, errors="coerce").fillna(0.0)
    return (a - b).abs() <= tol


def run_sanity_check(
    config_dict_or_path: Dict | str = "config.yaml",
    df_long: Optional[pd.DataFrame] = None,
    metrics_cfg: Optional[List[MetricConfig]] = None,
) -> ReadyState:
    """Execute a small Phase 2 run and validate Patch-3 readiness.

    Args:
        config_dict_or_path: Either loaded config dict or path to config.yaml

    Returns:
        ReadyState indicating whether the pipeline is ready to proceed to Phase 4.
    """
    # Load config (accept dict or path)
    if isinstance(config_dict_or_path, str):
        config = load_config(config_dict_or_path)
    else:
        config = config_dict_or_path

    scoring_config = load_and_validate_config(config)
    # Prefer provided metrics_cfg/df_long from caller (to avoid re-running Phase 2)
    if metrics_cfg is None:
        metrics_cfg = scoring_config.metrics
    metric_names = [m.name for m in metrics_cfg]

    if df_long is None:
        return ReadyState(False, ["Sanity check requires df_long from Phase 2 outputs"], {"rows": 0})
    errors: List[str] = []
    if df_long.empty:
        errors.append("Phase 2 long matrix is empty")
    else:
        errors.extend(_validate_long_fields(df_long))

    # Build wide composite locally to validate audit fields
    try:
        class _DummyLogger:
            def info(self, *args, **kwargs):
                return None
            def warning(self, *args, **kwargs):
                return None
            def error(self, *args, **kwargs):
                return None
        df_wide = build_wide_composite(df_long, metrics_cfg, scoring_config.weights_must_sum_to_1, logger=_DummyLogger())
    except Exception as ex:
        errors.append(f"Failed to build wide matrix for sanity validation: {ex}")
        df_wide = pd.DataFrame()

    if not df_wide.empty:
        errors.extend(_validate_wide_fields(df_wide, metric_names))
        # Logic validation: contrib = std * effective weight (per metric)
        for m in metric_names:
            std_col = f"std_{m}"
            contrib_col = f"contrib_{m}"
            weight_col = f"weight_{m}"
            if all(c in df_wide.columns for c in (std_col, contrib_col, weight_col)):
                ok_mask = _approx_equal(df_wide[contrib_col], df_wide[std_col] * df_wide[weight_col])
                if (~ok_mask).any():
                    errors.append(f"Phase-3 integrity check failed: {contrib_col} != {std_col} * {weight_col} for some rows")
    else:
        errors.append("Wide matrix is empty in sanity check")

    # Additional logic validations on long frame
    if df_long is not None and not df_long.empty:
        # completeness within [0,1]
        if not pd.to_numeric(df_long.get("completeness"), errors="coerce").between(0, 1).fillna(True).all():
            errors.append("completeness values must be within [0,1]")
        # recency_weeks non-negative and integer-ish
        rec = pd.to_numeric(df_long.get("recency_weeks"), errors="coerce")
        if (rec.dropna() < 0).any():
            errors.append("recency_weeks must be non-negative")
        # weights non-negative
        for col in ("weight_effective", "weight_multiplier", "base_weight"):
            if col in df_long.columns:
                s = pd.to_numeric(df_long[col], errors="coerce")
                if (s.dropna() < 0).any():
                    errors.append(f"{col} must be non-negative")
        # trend_strength consistency where volatility > 0
        if set(["trend_strength", "trend_slope", "volatility"]).issubset(df_long.columns):
            vs = pd.to_numeric(df_long["volatility"], errors="coerce")
            ts = pd.to_numeric(df_long["trend_strength"], errors="coerce")
            sl = pd.to_numeric(df_long["trend_slope"], errors="coerce").abs()
            mask = vs > 0
            if mask.any():
                calc = (sl[mask] / vs[mask]).replace([pd.NA, np.inf, -np.inf], np.nan)
                # allow small tolerance
                if not _approx_equal(ts[mask], calc, tol=1e-4).all():
                    errors.append("trend_strength must equal abs(trend_slope)/volatility where volatility>0")

    ok = len(errors) == 0
    summary = {
        "rows_long": int(len(df_long)) if df_long is not None else 0,
        "rows_wide": int(len(df_wide)) if df_wide is not None else 0,
        "metrics": metric_names,
    }
    return ReadyState(ok=ok, errors=errors, summary=summary)
