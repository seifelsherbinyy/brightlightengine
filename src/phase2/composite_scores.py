from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import yaml


DIMENSION_COLUMNS = [
    "score_profitability",
    "score_growth",
    "score_availability",
    "score_inventory_risk",
    "score_quality",
]


def load_archetype_weights(config_or_path: str | Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Load archetype→dimension weights.

    Accepts either a path to scoring_config.yaml or an already-loaded config dict.
    """

    if isinstance(config_or_path, str):
        with open(config_or_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = config_or_path or {}
    return (cfg.get("archetypes") or {})


def validate_archetype_matrix(weights: Dict[str, Dict[str, float]]) -> Tuple[bool, Dict[str, Any]]:
    """Validate archetype weight matrix.

    Ensures each archetype defines all five dimensions and weights sum approximately to 1.0.
    Returns (ok, details) with optional warnings list in details.
    """
    required = ["profitability", "growth", "availability", "inventory_risk", "quality"]
    ok = True
    warnings: list[str] = []
    for arch, wmap in (weights or {}).items():
        missing = [d for d in required if d not in (wmap or {})]
        if missing:
            ok = False
            warnings.append(f"Archetype '{arch}' missing dimensions: {missing}")
        try:
            total = float(sum(float(wmap.get(k, 0.0)) for k in required))
        except Exception:
            total = 0.0
        if not np.isfinite(total) or abs(total - 1.0) > 1e-6:
            # Not fatal; renormalize on the fly, but record warning
            warnings.append(f"Archetype '{arch}' weights sum to {total:.6f}; will renormalize during scoring")
    return ok, {"warnings": warnings}


def compute_composite_score(row: pd.Series, weights: Dict[str, float]) -> float:
    vals = []
    wts = []
    for dim in DIMENSION_COLUMNS:
        key = dim.replace("score_", "")
        w = float(weights.get(key, 0.0)) if isinstance(weights, dict) else 0.0
        v = row.get(dim, np.nan)
        if pd.notna(v) and w > 0:
            vals.append(float(v))
            wts.append(w)
    if not vals:
        return np.nan
    wsum = float(np.sum(wts))
    if wsum <= 0:
        return float(np.mean(vals))
    norm_w = [w / wsum for w in wts]
    return float(np.dot(vals, norm_w))


def _map_category_to_archetype(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """Map category to archetype using scoring_config mappings.

    Supports matching by 'gl_code' or 'category_name' or the single 'category' string
    present in df (from dim_asin join). Falls back to default archetype if no match.
    """

    arch_cfg = (config or {}).get("category_archetype_mapping", {})
    default_arch = arch_cfg.get("default", "Ambient")
    mapping_items = arch_cfg.get("mappings", [])

    # Build a list of tuples: (match_key, value, archetype, by)
    # where by in {"gl_code", "category_name", "category_contains"}
    rules = []
    for item in mapping_items:
        if "gl_code" in item:
            rules.append(("gl_code", str(item["gl_code"]).upper(), str(item.get("archetype", default_arch))))
        elif "category_name" in item:
            rules.append(("category_name", str(item["category_name"]).strip().title(), str(item.get("archetype", default_arch))))
        elif "category_contains" in item:
            rules.append(("category_contains", str(item["category_contains"]).strip().lower(), str(item.get("archetype", default_arch))))

    def pick(row: pd.Series) -> str:
        gl_code = str(row.get("gl_code", "")).upper()
        category_name = str(row.get("category_name", "")).strip().title()
        category = str(row.get("category", "")).strip()
        category_lower = category.lower()
        for by, val, archetype in rules:
            if by == "gl_code" and gl_code and gl_code == val:
                return archetype
            if by == "category_name" and category_name and category_name == val:
                return archetype
            if by == "category_contains" and category_lower and val in category_lower:
                return archetype
        return default_arch

    return df.apply(pick, axis=1).astype("string")


def map_category_to_archetype(df: pd.DataFrame, config: Dict[str, Any]) -> pd.Series:
    """Public wrapper to map categories to archetype per config."""
    return _map_category_to_archetype(df, config)


def apply_composite_scores(
    df: pd.DataFrame,
    dim_category_df: pd.DataFrame,
    weights_dict: Dict[str, Dict[str, float]],
    scoring_config: Dict[str, Any],
) -> pd.DataFrame:
    """Enrich df with 'archetype' and 'composite_score' based on dimension scores.

    - Attempts to attach 'gl_code' and 'category_name' via provided dim_category_df if not present.
    - Uses scoring_config's category_archetype_mapping to assign 'archetype'.
    - Computes composite score using archetype weights.
    """

    out = df.copy()

    # Attach category dimension details when possible.
    # Expect out to already contain 'category' from dim_asin join; dim_category_df has gl_code/category_name.
    if "gl_code" not in out.columns or "category_name" not in out.columns:
        if not dim_category_df.empty:
            # Attempt join by 'category' matching either 'category_name' or concatenated code+name
            cat = dim_category_df.copy()
            cat["category_key"] = cat["gl_code"].astype("string").fillna("") + "|" + cat["category_name"].astype("string").fillna("")
            out["category_key"] = out["category"].astype("string").fillna("")
            out = out.merge(cat, how="left", left_on="category_key", right_on="category_key")
            out.drop(columns=[c for c in ["category_key"] if c in out.columns], inplace=True)

    # Determine archetype string
    # Need scoring_config for mapping
    archetype_series = _map_category_to_archetype(out, scoring_config)
    out["archetype"] = archetype_series

    # Compute composite using weights for each row's archetype
    def comp(row: pd.Series) -> float:
        arch = str(row.get("archetype", ""))
        weights = weights_dict.get(arch, {})
        # Renormalize weights if needed
        if isinstance(weights, dict):
            total = sum(float(weights.get(k, 0.0)) for k in ["profitability", "growth", "availability", "inventory_risk", "quality"])
            if total > 0:
                weights = {k: float(v) / float(total) for k, v in weights.items()}
        return compute_composite_score(row, weights)

    out["composite_raw"] = out.apply(comp, axis=1)

    # Optional per-dimension contributions json (for transparency)
    try:
        def _contribs(row: pd.Series) -> Dict[str, float]:
            arch = str(row.get("archetype", ""))
            w = weights_dict.get(arch, {}) or {}
            # normalize weights
            denom = sum(float(w.get(k, 0.0)) for k in ["profitability", "growth", "availability", "inventory_risk", "quality"]) or 1.0
            contrib = {}
            for dim_col in DIMENSION_COLUMNS:
                key = dim_col.replace("score_", "")
                sval = row.get(dim_col, np.nan)
                if pd.notna(sval):
                    contrib[key] = float(sval) * float(w.get(key, 0.0)) / float(denom)
            return contrib

        out["composite_inputs_json"] = out.apply(_contribs, axis=1).apply(lambda d: None if not d else d)
    except Exception:
        # best-effort; don't fail pipeline for contributions
        pass

    # Rescale if configured
    comp_cfg = (scoring_config or {}).get("composite", {}) or {}
    min_s = float(comp_cfg.get("min_score", 0))
    max_s = float(comp_cfg.get("max_score", 100))
    rescale = bool(comp_cfg.get("rescale", True))
    if rescale and "composite_raw" in out.columns:
        s = pd.to_numeric(out["composite_raw"], errors="coerce")
        mn = s.min(skipna=True)
        mx = s.max(skipna=True)
        if pd.notna(mn) and pd.notna(mx) and float(mx - mn) > 0:
            out["composite_score"] = (s - mn) / (mx - mn) * (max_s - min_s) + min_s
        else:
            out["composite_score"] = s  # cannot rescale; pass-through
    else:
        out["composite_score"] = out.get("composite_raw")

    return out
