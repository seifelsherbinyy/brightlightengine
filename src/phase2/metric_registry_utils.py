from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

import pandas as pd
import yaml


# ----------------------------- Core structures -----------------------------


@dataclass
class MetricDefinition:
    canonical_name: str
    synonyms: List[str] = field(default_factory=list)
    short_label: str = ""
    description: str = ""
    directionality: str = "higher_is_better"  # or lower_is_better | neutral
    default_transform: str = "robust_zscore"  # zscore | robust_zscore | percentile | minmax | none
    # Families (booleans help downstream grouping; metric_family in config remains canonical)
    is_value_metric: bool = False
    is_volume_metric: bool = False
    is_availability_metric: bool = False
    is_quality_metric: bool = False
    is_inventory_metric: bool = False
    is_price_metric: bool = False
    supports_seasonality_adjustment: bool = False
    category_relevance: List[str] = field(default_factory=list)
    metric_family: str = ""  # value | volume | availability | inventory | price | quality


# ----------------------------- Loader & validation -----------------------------


def _normalize_name(s: Optional[str]) -> str:
    if s is None:
        return ""
    txt = str(s).strip()
    if not txt:
        return ""
    # basic normalization: lowercase, remove spaces/underscores and % sign
    txt = txt.lower().replace("%", "")
    txt = txt.replace(" ", "").replace("_", "")
    return txt


def load_metric_registry(config_path: str) -> Dict[str, MetricDefinition]:
    """Load registry from YAML into a dict keyed by canonical_name.

    Validates uniqueness of canonical names and synonyms across metrics and
    fills safe defaults for missing fields.
    """

    p = Path(config_path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    items = cfg.get("metrics", [])
    registry: Dict[str, MetricDefinition] = {}
    seen_synonyms: Set[str] = set()
    for it in (items or []):
        if not isinstance(it, dict):
            continue
        cname = str(it.get("canonical_name", "")).strip()
        if not cname:
            continue
        if cname in registry:
            # skip duplicates; keep first definition
            continue
        synonyms = list(it.get("synonyms", []) or [])
        # Ensure uniqueness in synonyms across metrics (case/normalized)
        safe_syns: List[str] = []
        for s in synonyms:
            key = _normalize_name(s)
            if not key or key in seen_synonyms:
                continue
            safe_syns.append(str(s))
            seen_synonyms.add(key)

        md = MetricDefinition(
            canonical_name=cname,
            synonyms=safe_syns,
            short_label=str(it.get("short_label", cname)).strip(),
            description=str(it.get("description", "")).strip(),
            directionality=str(it.get("directionality", "higher_is_better")),
            default_transform=str(it.get("default_transform", "robust_zscore")),
            is_value_metric=bool(it.get("is_value_metric", False)),
            is_volume_metric=bool(it.get("is_volume_metric", False)),
            is_availability_metric=bool(it.get("is_availability_metric", False)),
            is_quality_metric=bool(it.get("is_quality_metric", False)),
            is_inventory_metric=bool(it.get("is_inventory_metric", False)),
            is_price_metric=bool(it.get("is_price_metric", False)),
            supports_seasonality_adjustment=bool(it.get("supports_seasonality_adjustment", False)),
            category_relevance=list(it.get("category_relevance", []) or []),
            metric_family=str(it.get("metric_family", "")),
        )
        registry[cname] = md
    return registry


# ----------------------------- Synonym resolution & discovery -----------------------------


def resolve_metric_name(registry: Dict[str, MetricDefinition], raw_column_name: str) -> Tuple[Optional[str], str]:
    """Resolve a raw column to a canonical metric name using registry.

    Returns (canonical_name or None, reason) where reason is one of:
    - exact
    - synonym
    - unregistered_metric
    """

    if not raw_column_name:
        return None, "unregistered_metric"
    norm = _normalize_name(raw_column_name)
    # exact canonical match
    for cname, md in registry.items():
        if _normalize_name(cname) == norm:
            return cname, "exact"
    # synonym match
    for cname, md in registry.items():
        for syn in md.synonyms:
            if _normalize_name(syn) == norm:
                return cname, "synonym"
    return None, "unregistered_metric"


def _is_numeric_series(s: pd.Series) -> bool:
    try:
        if pd.api.types.is_numeric_dtype(s):
            return True
        # try a quick coercion sample
        sample = pd.to_numeric(s.head(100), errors="coerce")
        return sample.notna().any()
    except Exception:
        return False


def discover_metrics_from_phase1(phase1_df: pd.DataFrame, registry: Dict[str, MetricDefinition]) -> Dict[str, str]:
    """For each numeric column in Phase 1 df, attempt to resolve to a canonical name.

    Returns mapping: raw_column_name -> canonical_name or "UNREGISTERED".
    """

    mapping: Dict[str, str] = {}
    for col in phase1_df.columns:
        s = phase1_df[col]
        if not _is_numeric_series(s):
            continue
        cname, reason = resolve_metric_name(registry, col)
        mapping[col] = cname if cname else "UNREGISTERED"
    return mapping


# ----------------------------- Preview export -----------------------------


def build_registry_preview(
    registry: Dict[str, MetricDefinition],
    discovered_mapping: Dict[str, str],
    output_path: str,
) -> None:
    rows: List[Dict[str, Any]] = []
    # Build a reverse index for match status
    norm_to_canonical: Dict[str, Tuple[str, str]] = {}
    for cname, md in registry.items():
        norm_to_canonical[_normalize_name(cname)] = (cname, "exact")
        for syn in md.synonyms:
            norm_to_canonical[_normalize_name(syn)] = (cname, "synonym")

    for raw, canonical in (discovered_mapping or {}).items():
        norm = _normalize_name(raw)
        status = "unregistered"
        match_canonical = None
        if norm in norm_to_canonical:
            match_canonical, status = norm_to_canonical[norm]
        if canonical != "UNREGISTERED" and not match_canonical:
            match_canonical = canonical
            status = "exact"  # conservative default
        md = registry.get(match_canonical or "")
        rows.append({
            "raw_column_name": raw,
            "canonical_metric": match_canonical or "",
            "match_status": status,
            "directionality": (md.directionality if md else ""),
            "default_transform": (md.default_transform if md else ""),
            "metric_family": (md.metric_family if md else ""),
        })

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")


# ----------------------------- Public helper for scoring -----------------------------


def get_active_metric_sets(
    registry: Dict[str, MetricDefinition],
    discovered_mapping: Dict[str, str],
    scoring_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Determine available metrics per dimension and surface missing ones.

    Returns a dict with:
      - metrics_for_profitability/growth/availability/inventory_risk/quality (list of canonical)
      - raw_to_canonical_map (dict)
      - present_canonical (set)
      - missing_required_metrics (list)
      - unregistered_metrics (list of raw names)
    """

    dims = scoring_config.get("dimensions", {})
    dim_metrics = {k: list((dims.get(k, {}).get("metrics", {}) or {}).keys()) for k in (
        "profitability", "growth", "availability", "inventory_risk", "quality"
    )}

    # Present canonical metrics based on discovered mapping
    present: Set[str] = set([v for v in discovered_mapping.values() if v and v != "UNREGISTERED"])
    # Missing required per config
    required: Set[str] = set()
    for lst in dim_metrics.values():
        required.update(lst)
    missing = sorted([m for m in required if m not in present])
    # Build raw->canonical map only for registered
    raw_to_canonical = {raw: can for raw, can in discovered_mapping.items() if can and can != "UNREGISTERED"}
    unregistered = sorted([raw for raw, can in discovered_mapping.items() if can == "UNREGISTERED"])

    def _avail(lst: List[str]) -> List[str]:
        return [m for m in lst if m in present]

    return {
        "metrics_for_profitability": _avail(dim_metrics.get("profitability", [])),
        "metrics_for_growth": _avail(dim_metrics.get("growth", [])),
        "metrics_for_availability": _avail(dim_metrics.get("availability", [])),
        "metrics_for_inventory_risk": _avail(dim_metrics.get("inventory_risk", [])),
        "metrics_for_quality": _avail(dim_metrics.get("quality", [])),
        "raw_to_canonical_map": raw_to_canonical,
        "present_canonical": sorted(present),
        "missing_required_metrics": missing,
        "unregistered_metrics": unregistered,
    }
