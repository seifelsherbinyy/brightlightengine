"""Metric name aliasing and normalization utilities.

Provides a lightweight mapping layer to reconcile Phase‑1 metric names with
canonical scoring metric names configured in Phase‑2.

Rules:
- Load optional aliases from data/reference/metric_aliases.yaml (or path in config)
- Apply case‑insensitive, punctuation‑insensitive normalization for heuristic match
- Prefer explicit alias over heuristic
- Leave unmapped metrics unchanged (they can be filtered later)
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import re
import yaml


_PUNCT_RE = re.compile(r"[\s\(\)\[%\]$€£,+/_-]+")


def normalize_token(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = _PUNCT_RE.sub("", s)
    return s


def load_alias_map(path: str | Path | None) -> Dict[str, str]:
    """Load YAML alias map of observed->canonical metric names.

    YAML structure:
      aliases:
        Shipped Revenue: GMS($)
        Unit Session %: FillRate(%)
        OOS %: SoROOS(%)
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    aliases = data.get("aliases", {}) or {}
    # force str keys/values
    return {str(k): str(v) for k, v in aliases.items()}


def build_heuristic_map(observed: Iterable[str], canonical: Iterable[str]) -> Dict[str, str]:
    """Heuristically map observed metric names to canonical ones via normalization.

    If multiple canonical collide (unlikely), first match wins.
    """
    observed = list(observed)
    canonical = list(canonical)
    cano_by_norm = {normalize_token(c): c for c in canonical}
    mapping: Dict[str, str] = {}
    for o in observed:
        norm = normalize_token(o)
        if norm in cano_by_norm:
            mapping[o] = cano_by_norm[norm]
    return mapping


def build_metric_mapping(
    observed_metrics: Iterable[str],
    canonical_metrics: Iterable[str],
    alias_file: str | Path | None = None,
) -> Dict[str, str]:
    """Combine explicit alias map (if present) with heuristic normalization map.

    Explicit aliases override heuristics.
    """
    explicit = load_alias_map(alias_file)
    heuristic = build_heuristic_map(observed_metrics, canonical_metrics)
    # Overlay: explicit wins
    out = dict(heuristic)
    for k, v in explicit.items():
        out[k] = v
    return out


def apply_metric_mapping(
    series, mapping: Dict[str, str]
):
    """Map metric names in a pandas Series using the provided mapping.

    Unmapped values are left unchanged.
    """
    # lazy import to avoid hard pandas dependency here
    import pandas as pd

    if not isinstance(series, pd.Series) or not mapping:
        return series
    return series.map(lambda x: mapping.get(x, x))
