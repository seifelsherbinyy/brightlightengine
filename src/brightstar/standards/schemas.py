"""Schemas and validators for Brightstar pipeline phases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd


@dataclass(frozen=True)
class Schema:
    required: List[str]
    numeric: List[str]


# Minimal schemas based on current pipeline contracts
PHASE2_LONG = Schema(
    required=[
        "entity_id", "vendor_code", "Week_Order", "metric", "value", "raw_value", "std_value",
    ],
    numeric=[
        "value", "raw_value", "std_value",
    ],
)

PHASE4_WIDE = Schema(
    required=[
        "entity_id", "vendor_code", "Week_Order", "score_composite",
    ],
    numeric=[
        "score_composite",
    ],
)


def _ensure_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _ensure_numeric(df: pd.DataFrame, cols: List[str], name: str) -> None:
    for c in cols:
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"{name} column '{c}' must be numeric, got {df[c].dtype}")


def validate_phase_input(df: pd.DataFrame, phase: str) -> None:
    if phase == "phase2_long":
        _ensure_columns(df, PHASE2_LONG.required, phase)
        _ensure_numeric(df, PHASE2_LONG.numeric, phase)
    else:
        # For other phases we keep this minimal for now
        if df is None:
            raise ValueError(f"{phase}: input is None")


def validate_phase_output(df: pd.DataFrame, phase: str) -> None:
    if phase == "phase4_wide":
        _ensure_columns(df, PHASE4_WIDE.required, phase)
        _ensure_numeric(df, PHASE4_WIDE.numeric, phase)
    else:
        if df is None:
            raise ValueError(f"{phase}: output is None")
