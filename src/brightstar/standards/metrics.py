"""Flexible metric matching/standardization engine.

Capabilities:
- Name normalization (strip punctuation, units, case-insensitive)
- Fuzzy matching via difflib with configurable cutoff
- Alias mapping from JSON file (explicit > fuzzy)
- Resolution function that returns mapping + warnings
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import difflib

from .naming import normalize_metric_name


def _load_aliases_json(path: str | Path | None) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    # ensure str keys/values
    return {str(k): str(v) for k, v in data.items()}


def resolve_configured_metrics(
    config_metrics: Iterable[str],
    available_columns: Iterable[str],
    alias_json: str | Path | None = None,
    fuzzy_cutoff: float = 0.84,
) -> Tuple[Dict[str, str], List[str]]:
    """Resolve scoring metrics to available Phase‑1 metric columns.

    Returns (mapping, warnings): mapping maps available->configured canonical.

    Behavior:
      1) If alias_json defines explicit observed->configured, use it.
      2) Heuristic normalization: exact match on normalized tokens.
      3) Fuzzy match using difflib.get_close_matches with cutoff.
      4) Unresolved configured names produce a warning.
    """
    configured = list(config_metrics)
    available = list(available_columns)
    warnings: List[str] = []
    mapping: Dict[str, str] = {}

    aliases = _load_aliases_json(alias_json)
    # Apply explicit aliases (observed -> configured)
    for obs, canon in aliases.items():
        if obs in available and canon in configured:
            mapping[obs] = canon

    # Normalized exact matches
    norm_to_avail: Dict[str, str] = {}
    for obs in available:
        norm_to_avail.setdefault(normalize_metric_name(obs), obs)
    for canon in configured:
        norm = normalize_metric_name(canon)
        if norm in norm_to_avail:
            obs = norm_to_avail[norm]
            mapping.setdefault(obs, canon)

    # Fuzzy matches for remaining canonicals
    remaining_canons = [c for c in configured if c not in mapping.values()]
    avail_norm_list = [normalize_metric_name(a) for a in available]
    for canon in remaining_canons:
        canon_norm = normalize_metric_name(canon)
        # find best close normalized match
        close = difflib.get_close_matches(canon_norm, avail_norm_list, n=1, cutoff=fuzzy_cutoff)
        if close:
            idx = avail_norm_list.index(close[0])
            obs = available[idx]
            mapping.setdefault(obs, canon)
        else:
            warnings.append(f"Unresolved metric: '{canon}' (no alias/fuzzy match in Phase‑1 columns)")

    return mapping, warnings
