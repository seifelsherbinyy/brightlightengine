"""Centralized naming utilities for metrics, identifiers, and week labels."""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

_PUNCT = re.compile(r"[\s\(\)\[\]%$€£,_/-]+")
_WEEK_PATTERNS = [
    re.compile(r"^(?P<y>\d{4})W(?P<w>\d{1,2})$", re.IGNORECASE),
    re.compile(r"^(?P<y>\d{4})-?W(?P<w>\d{1,2})$", re.IGNORECASE),
    re.compile(r"^(?P<y>\d{4})(?P<w>\d{2})$"),
]


def normalize_metric_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    s = _PUNCT.sub("", s)
    return s


def normalize_identifier(s: str) -> str:
    if s is None:
        return ""
    return _PUNCT.sub("", str(s).strip())


def normalize_week_label_value(val: str) -> str | None:
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    # Accept patterns like 2025W41 or 2025-W41 or 202541
    for pat in _WEEK_PATTERNS:
        m = pat.match(s)
        if m:
            y = int(m.group("y"))
            w = int(m.group("w")) if "w" in m.groupdict() else int(s[-2:])
            if 1 <= w <= 53:
                return f"{y}W{w:02d}"
    # Try to extract week number from tokens like "Week 41"
    m2 = re.search(r"(20\d{2}).*?(\d{1,2})", s)
    if m2:
        y = int(m2.group(1)); w = int(m2.group(2))
        if 1 <= w <= 53:
            return f"{y}W{w:02d}"
    return None


def normalize_week_label(series: pd.Series) -> pd.Series:
    return series.apply(normalize_week_label_value)
