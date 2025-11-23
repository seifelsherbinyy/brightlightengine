from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np
import yaml

from .metric_transforms import (
    winsorize_series,
    compute_growth_metrics,
    normalize_metric,
    apply_all_normalizations,
)
from .utils_time import (
    compute_wow_delta,
    compute_l4w_delta,
    compute_l12w_delta,
    ensure_week_key,
    load_unified_calendar,
    enrich_with_seasonality,
    SEASON_FIELDS_CANONICAL,
)
import logging
from .dimension_signals import (
    compute_profitability_signals,
    compute_growth_signals,
    compute_availability_signals,
    compute_inventory_risk_signals,
    compute_quality_signals,
)
from .dimension_scores import (
    score_profitability,
    score_growth,
    score_availability,
    score_inventory_risk,
    score_quality,
    dimension_coverage_annotations,
)
from .composite_scores import (
    load_archetype_weights,
    apply_composite_scores,
    validate_archetype_matrix,
)
from .readiness import (
    compute_metric_readiness,
    apply_readiness_mask,
    finalize_readiness,
)
from .metric_registry_utils import (
    load_metric_registry,
    discover_metrics_from_phase1,
    build_registry_preview,
    get_active_metric_sets,
)
from .metric_coverage import (
    compute_metric_coverage,
    summarize_metric_coverage,
)


def _load_dimensions(dimensions_config: str) -> Dict[str, pd.DataFrame]:
    with open(dimensions_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    root = Path(cfg.get("default_output_root", "")).expanduser()
    files = cfg.get("files", {})

    def p(name: str) -> Path:
        return (root / files.get(name)).resolve()

    dfs: Dict[str, pd.DataFrame] = {}
    for key in ("dim_asin", "dim_vendor", "dim_category", "dim_subcategory", "dim_brand"):
        path = p(key)
        if path and path.exists():
            try:
                dfs[key] = pd.read_parquet(path)
            except Exception:
                # Fallback try CSV
                try:
                    dfs[key] = pd.read_csv(path)
                except Exception:
                    dfs[key] = pd.DataFrame()
        else:
            dfs[key] = pd.DataFrame()
    return dfs


def _load_scoring_config(scoring_config: str) -> Dict[str, Any]:
    with open(scoring_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    # Normalize structure
    cfg.setdefault("metrics", {})
    # Session 2A metrics grouping and normalization defaults
    m = cfg["metrics"]
    m.setdefault("numeric", [])
    m.setdefault("percent", [])
    m.setdefault("currency", [])
    m.setdefault("normalization_method", "robust_zscore")

    cfg.setdefault("dimensions", {})
    # Ensure dimension keys exist and provide default containers for Session 2A signals
    for d in ("profitability", "growth", "availability", "inventory_risk", "quality"):
        cfg["dimensions"].setdefault(d, {})
        cfg["dimensions"][d].setdefault("metrics", cfg["dimensions"][d].get("metrics", {}))
        cfg["dimensions"][d].setdefault("inputs", [])
        cfg["dimensions"][d].setdefault("weights", [])

    # Transforms and deltas defaults for Session 2A
    cfg.setdefault("transforms", {"winsor_limits": [0.01, 0.99], "enable_minmax": False})
    cfg.setdefault("deltas", {"enable_wow": True, "enable_l4w": True, "enable_l12w": True})
    cfg.setdefault("archetypes", {})
    cfg.setdefault("category_archetype_mapping", {"default": "Ambient", "mappings": []})
    cfg.setdefault("readiness", {"min_history_weeks": 6, "null_pct_max": 0.3, "volatility_max": 1e9, "anomaly_max": 10})
    # Session 2B defaults
    cfg.setdefault("composite", {"min_score": 0, "max_score": 100, "rescale": True})
    cfg.setdefault("readiness", cfg.get("readiness", {}))
    cfg["readiness"].setdefault("final", {"require_history_weeks": 6, "max_null_share": 0.25, "allow_missing_dimensions": False})
    # Phase 2c coverage defaults are not injected unless block exists in YAML (back-compat)
    if "coverage" in cfg and isinstance(cfg.get("coverage"), dict):
        cov = cfg["coverage"] or {}
        cov.setdefault("min_coverage_pct", 0.6)
        cov.setdefault("min_history_weeks", 6)
        cov.setdefault("treat_all_zero_as_inactive", True)
        cov.setdefault("enforce_strict_for_dimensions", ["growth", "availability"])
        cov.setdefault("min_dimensions_for_composite", 2)
        cfg["coverage"] = cov
    return cfg


def _winsorize_all(df: pd.DataFrame, metrics_cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for metric, mcfg in metrics_cfg.items():
        if metric not in out.columns:
            continue
        if not isinstance(mcfg, dict):
            continue
        limits = mcfg.get("winsor_limits", [0.01, 0.99])
        out[metric] = winsorize_series(out[metric], (limits[0], limits[1]))
    return out


def _normalize_all(df: pd.DataFrame, metrics_cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for metric, mcfg in metrics_cfg.items():
        # Skip grouping keys and other non-dict config sections or missing metrics
        if metric not in out.columns:
            continue
        if not isinstance(mcfg, dict):
            continue
        method = mcfg.get("method", "robust_zscore")
        higher_is_better = bool(mcfg.get("higher_is_better", True))
        col_norm = normalize_metric(out, metric, method)
        if not higher_is_better:
            col_norm = -col_norm
        out[f"{metric}__norm"] = col_norm
    return out


def _merge_with_dimensions(df: pd.DataFrame, dims: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = df.copy()
    asin_dim = dims.get("dim_asin", pd.DataFrame())
    if not asin_dim.empty:
        # Ensure expected columns are present
        cols = {c.lower(): c for c in asin_dim.columns}
        asin_key = cols.get("asin_id") or cols.get("asin") or "asin_id"
        join_cols = [c for c in ["vendor_id", "vendor_name", "brand_name", "category", "sub_category"] if c in asin_dim.columns]
        join_df = asin_dim[[asin_key] + join_cols].copy()
        join_df = join_df.rename(columns={asin_key: "asin"})
        out = out.merge(join_df, on="asin", how="left", suffixes=("", "_dim"))

    # Attach category dimension details for archetype mapping if available (gl_code, category_name)
    dim_cat = dims.get("dim_category", pd.DataFrame())
    if not dim_cat.empty and "category" in out.columns:
        cat = dim_cat.copy()
        cat["category_key"] = cat["gl_code"].astype("string").fillna("") + "|" + cat["category_name"].astype("string").fillna("")
        out["category_key"] = out["category"].astype("string").fillna("")
        out = out.merge(cat, how="left", left_on="category_key", right_on="category_key")
        out.drop(columns=[c for c in ["category_key"] if c in out.columns], inplace=True)
    return out


def _write_outputs(
    detailed: pd.DataFrame,
    summary: pd.DataFrame,
    readiness_df: pd.DataFrame,
    metadata: Dict[str, Any],
    output_base: str,
) -> None:
    out_dir = Path(output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    detailed.to_parquet(out_dir / "scoring_detailed.parquet", index=False)
    summary.to_parquet(out_dir / "scoring_summary.parquet", index=False)
    readiness_df.to_parquet(out_dir / "metric_readiness.parquet", index=False)
    with (out_dir / "scoring_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_scoring(
    phase1_path: str,
    dimensions_config: str,
    scoring_config: str,
    output_base: str,
    calendar_config_path: Optional[str] = None,
) -> None:
    # Load Phase 1 validated parquet
    phase1_pq = Path(phase1_path)
    if not phase1_pq.exists():
        raise FileNotFoundError(f"Phase 1 parquet not found: {phase1_pq}")
    df = pd.read_parquet(phase1_pq)

    # Normalize headers to lowercase/strip for safety
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Map identifiers to expected Phase 2 names without altering Phase 1 outputs
    # asin: prefer existing 'asin'; else derive from 'asin_id'
    if "asin" not in df.columns and "asin_id" in df.columns:
        df["asin"] = df["asin_id"].astype("string")

    # vendor_id must be present
    if "vendor_id" not in df.columns:
        raise ValueError("Phase 1 dataset missing required column: vendor_id")

    # Ensure a sortable week integer YYYYWW for growth computations
    if "week" not in df.columns:
        if "canonical_week_id" in df.columns:
            # Convert strings like 2025W36 -> 202536
            s = df["canonical_week_id"].astype("string").fillna("")
            df["week"] = s.str.replace("W", "", regex=False).str.replace("w", "", regex=False)
            # Keep only digits; invalid parse becomes NaN then Int64
            df["week"] = pd.to_numeric(df["week"], errors="coerce").astype("Int64")
        elif "week_start_date" in df.columns:
            # Derive ISO (year, week) from start date
            def _to_week(x: object) -> int | None:
                try:
                    d = pd.to_datetime(x).date()
                    iso_year, iso_week, _ = d.isocalendar()
                    return int(f"{iso_year}{int(iso_week):02d}")
                except Exception:
                    return None
            df["week"] = df["week_start_date"].map(_to_week).astype("Int64")
        else:
            raise ValueError("Phase 1 dataset missing required calendar fields to compute week (canonical_week_id or week_start_date)")

    # Metric Registry (Session 2): discover and normalize metric columns
    registry_cfg_path = str(Path("config") / "metric_registry.yaml")
    registry = load_metric_registry(registry_cfg_path)
    discovered = discover_metrics_from_phase1(df, registry)

    # Create canonical aliases on dataframe for discovered metrics
    for raw, canonical in (discovered or {}).items():
        if canonical and canonical != "UNREGISTERED":
            if canonical not in df.columns and raw in df.columns:
                df[canonical] = pd.to_numeric(df[raw], errors="coerce")

    # Backward-compat: also map common legacy names if still missing
    if "gms" not in df.columns and "gms_value" in df.columns:
        df["gms"] = pd.to_numeric(df["gms_value"], errors="coerce")
    if "units" not in df.columns and "ordered_units" in df.columns:
        df["units"] = pd.to_numeric(df["ordered_units"], errors="coerce")
    if "gv" not in df.columns and "gv_value" in df.columns:
        df["gv"] = pd.to_numeric(df["gv_value"], errors="coerce")

    # Validate mandatory calendar fields per spec
    required_calendar = [
        "canonical_week_id",
        "calendar_year",
        "calendar_week_number",
        "week_start_date",
        "week_end_date",
    ]
    missing_cal = [c for c in required_calendar if c not in df.columns]
    if missing_cal:
        raise ValueError(f"Phase 1 dataset missing required calendar fields: {missing_cal}")

    # Abort scoring if DQ flags indicate unmapped weeks or invalid ids
    if "dq_unmapped_week" in df.columns and bool(pd.Series(df["dq_unmapped_week"]).fillna(False).any()):
        raise ValueError("Aborting scoring: Phase 1 has unmapped weeks (dq_unmapped_week > 0)")
    if "dq_invalid_ids" in df.columns and bool(pd.Series(df["dq_invalid_ids"]).fillna(False).any()):
        raise ValueError("Aborting scoring: Phase 1 has invalid identifiers (dq_invalid_ids > 0)")

    # Validate identifiers needed downstream
    for col in ("asin", "vendor_id", "week"):
        if col not in df.columns:
            raise ValueError(f"Phase 1 dataset missing required column for scoring: {col}")

    # Load dimensions and config
    dims = _load_dimensions(dimensions_config)
    cfg = _load_scoring_config(scoring_config)

    # ------------------
    # Session 2b: Calendar config + enrichment
    # ------------------
    # Load calendar config (optional file)
    cal_defaults = {
        "unified_calendar_path": str(Path("data") / "reference" / "unified_calendar_map.csv"),
        "join_key": "canonical_week_id",
        "fields": {
            "season_label": None,
            "season_type": None,
            "event_code": None,
            "is_peak_week": None,
            "is_event_week": None,
            "quarter_label": None,
        },
        "defaults": {
            "season_label": "UNKNOWN_SEASON",
            "season_type": "BASELINE",
            "event_code": "NONE",
            "is_peak_week": False,
            "is_event_week": False,
            "quarter_label": None,
        },
    }

    calendar_settings: Dict[str, Any] = cal_defaults.copy()
    calendar_cfg_file = Path(calendar_config_path) if calendar_config_path else Path("config") / "phase2_calendar.yaml"
    season_source = "defaults"
    try:
        if calendar_cfg_file and calendar_cfg_file.exists():
            with open(calendar_cfg_file, "r", encoding="utf-8") as f:
                calendar_settings.update(yaml.safe_load(f) or {})
        else:
            logging.getLogger(__name__).warning("Calendar config file not found at %s; using defaults.", str(calendar_cfg_file))
    except Exception:
        logging.getLogger(__name__).warning("Failed to load calendar config; using defaults.")

    # Scoring config switches
    cal_flags = cfg.setdefault("calendar", {
        "use_phase1_enrichment": True,
        "fallback_to_unified_calendar": True,
        "seasonality_flags_enabled": True,
    })

    # Ensure required week key present (critical)
    try:
        ensure_week_key(df, "canonical_week_id")
    except Exception as e:
        # Write minimal metadata with abort_reason then re-raise
        out_dir = Path(output_base)
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "scoring_metadata.json").open("w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "abort_reason": str(e),
            }, f, indent=2)
        raise

    # Determine seasonality source
    season_fields_present = [c for c in SEASON_FIELDS_CANONICAL if c in df.columns]
    if bool(cal_flags.get("use_phase1_enrichment", True)) and season_fields_present:
        # Fill missing season fields with defaults
        defaults = (calendar_settings.get("defaults") or {})
        for c in SEASON_FIELDS_CANONICAL:
            if c not in df.columns:
                df[c] = pd.NA
        for c in SEASON_FIELDS_CANONICAL:
            if c in ("is_peak_week", "is_event_week"):
                df[c] = pd.Series(df[c], dtype="boolean").fillna(bool(defaults.get(c, False)))
            else:
                default_val = defaults.get(c, None)
                df[c] = pd.Series(df[c], dtype="string").fillna(default_val if default_val is not None else pd.NA)
        season_source = "phase1"
    elif bool(cal_flags.get("fallback_to_unified_calendar", True)):
        # Try to load and join unified calendar
        join_key = str(calendar_settings.get("join_key", "canonical_week_id"))
        try:
            fields_cfg = calendar_settings.get("fields")
            field_list = list(fields_cfg.keys()) if isinstance(fields_cfg, dict) else None
            cal_path = str(calendar_settings.get("unified_calendar_path", cal_defaults["unified_calendar_path"]))
            cal_df = load_unified_calendar(cal_path, join_key=join_key, fields=field_list)
            df = enrich_with_seasonality(df, cal_df, join_key=join_key, config_defaults=calendar_settings.get("defaults"))
            season_source = "unified_calendar"
        except Exception as e:
            logging.getLogger(__name__).warning("Unified calendar unavailable (%s); defaulting season fields.", e)
            df = enrich_with_seasonality(df, None, join_key="canonical_week_id", config_defaults=calendar_settings.get("defaults"))
            season_source = "defaults"
    else:
        # No external source; apply defaults
        df = enrich_with_seasonality(df, None, join_key="canonical_week_id", config_defaults=calendar_settings.get("defaults"))
        season_source = "defaults"

    # Determine active metric sets using registry and discovered mapping
    active = get_active_metric_sets(registry, discovered, cfg)

    # Export metric registry preview CSV (Session 2)
    out_dir = Path(output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        build_registry_preview(
            registry,
            discovered,
            str(out_dir / "metric_registry_preview.csv"),
        )
    except Exception:
        pass

    # Merge with dimensions (asin/category, etc.)
    df = _merge_with_dimensions(df, dims)

    # Winsorize configured metrics (overwrites base per existing behavior)
    metrics_cfg: Dict[str, Any] = cfg.get("metrics", {})
    df = _winsorize_all(df, metrics_cfg)

    # Session 2A: Normalization/scaling per config groupings on base metrics
    df = apply_all_normalizations(df, cfg)

    # Session 2A: Delta engine (compute growth before normalizing growth metrics)
    deltas_cfg = cfg.get("deltas", {})
    enable_wow = bool(deltas_cfg.get("enable_wow", True))
    enable_l4w = bool(deltas_cfg.get("enable_l4w", True))
    enable_l12w = bool(deltas_cfg.get("enable_l12w", True))

    # Determine metrics to compute deltas for: at minimum numeric group, plus any referenced by growth inputs
    numeric_metrics: List[str] = list(metrics_cfg.get("numeric", []) or [])
    growth_inputs: List[str] = list(cfg.get("dimensions", {}).get("growth", {}).get("inputs", []) or [])
    # Also include common base metrics if present
    base_candidates = [m for m in ("gms", "units", "gv") if m in df.columns]
    target_delta_metrics: List[str] = []
    for m in numeric_metrics + base_candidates:
        if m not in target_delta_metrics:
            target_delta_metrics.append(m)

    # Compute deltas
    if any((enable_wow, enable_l4w, enable_l12w)):
        for m in target_delta_metrics:
            if m not in df.columns:
                continue
            if enable_wow:
                df = compute_wow_delta(df, m)
            if enable_l4w:
                df = compute_l4w_delta(df, m)
            if enable_l12w:
                df = compute_l12w_delta(df, m)

    # Session 2b: season-aware lightweight flags
    if bool(cal_flags.get("seasonality_flags_enabled", True)):
        # Compute transitions within entity grouping (asin/vendor)
        def _group_and_sort_local(df_in: pd.DataFrame):
            key_asin = "asin_id" if "asin_id" in df_in.columns else "asin"
            keys = [key_asin]
            if "vendor_id" in df_in.columns:
                keys.append("vendor_id")
            order_col = "week_start_date" if "week_start_date" in df_in.columns else ("week" if "week" in df_in.columns else None)
            if order_col is None:
                return df_in.groupby(keys, group_keys=False), None
            return df_in.sort_values(keys + [order_col]).groupby(keys, group_keys=False), order_col

        g, _ = _group_and_sort_local(df)

        def _add_flags(group: pd.DataFrame) -> pd.DataFrame:
            prev_peak = pd.Series(group["is_peak_week"], dtype="boolean").shift(1)
            prev_label = pd.Series(group["season_label"], dtype="string").shift(1)
            cur_peak = pd.Series(group["is_peak_week"], dtype="boolean")
            cur_label = pd.Series(group["season_label"], dtype="string")
            group["is_peak_to_peak_transition"] = (prev_peak.fillna(False) & cur_peak.fillna(False) & (prev_label.fillna("") != cur_label.fillna("")))
            group["is_offpeak_recovery_week"] = (prev_peak.fillna(False) & (~cur_peak.fillna(False)))
            group["is_peak_to_peak_transition"] = group["is_peak_to_peak_transition"].astype("boolean")
            group["is_offpeak_recovery_week"] = group["is_offpeak_recovery_week"].astype("boolean")
            return group

        try:
            df = g.apply(_add_flags)
        except Exception:
            # If grouping fails, create false columns
            df["is_peak_to_peak_transition"] = False
            df["is_offpeak_recovery_week"] = False

    # Now normalize any configured metrics (including growth pct columns produced above)
    df = _normalize_all(df, metrics_cfg)

    # Session 2A: Dimension feature signals
    df = compute_profitability_signals(df, cfg)
    df = compute_growth_signals(df, cfg)
    df = compute_availability_signals(df, cfg)
    df = compute_inventory_risk_signals(df, cfg)
    df = compute_quality_signals(df, cfg)

    # Phase 2c: Metric coverage (optional; enabled when 'coverage' block present)
    coverage_cfg = cfg.get("coverage") if isinstance(cfg.get("coverage"), dict) else None
    coverage_df = pd.DataFrame()
    coverage_summary = pd.DataFrame()
    ok_map: Dict[str, set] = {}
    if coverage_cfg is not None:
        try:
            coverage_df = compute_metric_coverage(df, registry, coverage_cfg)
            # Build asin -> set(metrics ok)
            if not coverage_df.empty:
                ok_rows = coverage_df[coverage_df["metric_ok"] == True]  # noqa: E712
                grouped = ok_rows.groupby("asin")["metric"].apply(lambda s: set(s.astype(str).tolist()))
                ok_map = {k: v for k, v in grouped.to_dict().items()}
                # Inject into config so dimension scoring can adaptively filter
                cfg["__metric_ok_by_asin"] = ok_map
            coverage_summary = summarize_metric_coverage(coverage_df, base_df=df)
        except Exception:
            # Fail-safe: skip coverage if anything goes wrong
            coverage_df = pd.DataFrame()
            coverage_summary = pd.DataFrame()

    # Readiness flags and masking
    readiness_cfg = cfg.get("readiness", {})
    readiness_series = compute_metric_readiness(df, readiness_cfg)
    df_masked = apply_readiness_mask(df, readiness_series)

    # Dimension scores (Phase 2 core) with coverage-aware reweighting when enabled
    df_masked["score_profitability"] = score_profitability(df_masked, cfg)
    df_masked["score_growth"] = score_growth(df_masked, cfg)
    df_masked["score_availability"] = score_availability(df_masked, cfg)
    df_masked["score_inventory_risk"] = score_inventory_risk(df_masked, cfg)
    df_masked["score_quality"] = score_quality(df_masked, cfg)

    # Per-dimension coverage annotations
    if coverage_cfg is not None:
        for dim_name in ["profitability", "growth", "availability", "inventory_risk", "quality"]:
            ann = dimension_coverage_annotations(df_masked, cfg, dim_name)
            for col in ann.columns:
                df_masked[col] = ann[col]

    # Composite scores with archetype mapping (Session 2B)
    weights_dict = load_archetype_weights(scoring_config)
    _ok, arch_val = validate_archetype_matrix(weights_dict)
    df_scored = apply_composite_scores(df_masked, dims.get("dim_category", pd.DataFrame()), weights_dict, cfg)

    # Phase 2c: Composite reliability enforcement
    if coverage_cfg is not None:
        min_dims = int(coverage_cfg.get("min_dimensions_for_composite", 2))
        dim_cols = [
            "score_profitability",
            "score_growth",
            "score_availability",
            "score_inventory_risk",
            "score_quality",
        ]
        present = df_scored[dim_cols].notna().sum(axis=1)
        unreliable = present < min_dims
        df_scored["composite_coverage_flag"] = np.where(unreliable, "UNRELIABLE", "OK")
        # When unreliable, set composite_score to NaN (leave composite_raw as-is for diagnostics)
        if "composite_score" in df_scored.columns:
            df_scored.loc[unreliable, "composite_score"] = np.nan

    # Enforce archetype assignment completeness per spec (abort if nulls)
    if "archetype" not in df_scored.columns or df_scored["archetype"].isna().any():
        raise ValueError("Aborting scoring: category→archetype mapping produced nulls; check scoring_config category_archetype_mapping or dimension joins")

    # Build detailed output subset (preserve calendar fields + seasonality)
    id_cols = [
        c
        for c in [
            "asin",
            "vendor_id",
            "week",
            "canonical_week_id",
            "week_start_date",
            "week_end_date",
            "calendar_year",
            "calendar_week_number",
            # seasonality
            "season_label",
            "season_type",
            "event_code",
            "is_peak_week",
            "is_event_week",
            "quarter_label",
            "brand_name",
            "category",
            "sub_category",
            "gl_code",
            "category_name",
            "archetype",
        ]
        if c in df_scored.columns
    ]
    score_cols = [
        "score_profitability",
        "score_growth",
        "score_availability",
        "score_inventory_risk",
        "score_quality",
        "composite_score",
    ]
    # Include composite coverage flag when present
    extra_score_cols = [c for c in ["composite_coverage_flag"] if c in df_scored.columns]
    # Include normalized metrics for transparency
    norm_cols = [c for c in df_scored.columns if str(c).endswith("__norm")]
    seasonal_flag_cols = [c for c in ["is_peak_to_peak_transition", "is_offpeak_recovery_week"] if c in df_scored.columns]
    # Include per-dimension coverage annotations when present
    cov_ann_cols = [
        c for c in df_scored.columns
        if any(c.endswith(suf) for suf in ("_metric_count", "_metrics_used", "_coverage_flag"))
    ]
    detailed = df_scored[id_cols + norm_cols + score_cols + extra_score_cols + cov_ann_cols + seasonal_flag_cols].copy()

    # Summary at vendor/category level (mean of scores and count of asins)
    agg_dict = {c: "mean" for c in score_cols}
    # Include season dimension at summary level by exposing season_type if available
    group_cols = [c for c in ["vendor_id", "category", "season_type"] if c in df_scored.columns]
    if group_cols and "asin" in df_scored.columns:
        summary = (
            df_scored.groupby(group_cols).agg({**agg_dict, "asin": "nunique"})
            .rename(columns={"asin": "asin_count"})
            .reset_index()
        )
    else:
        summary = pd.DataFrame()

    # Metric readiness export (entity-level)
    readiness_entity = (
        pd.DataFrame({
            "asin": df_scored["asin"],
            "vendor_id": df_scored["vendor_id"],
            "is_ready": readiness_series,
        })
        .drop_duplicates(subset=["asin", "vendor_id"])
        .reset_index(drop=True)
    )

    # Prepare Session 2A artifacts
    id_cols = [
        c
        for c in [
            "asin",
            "vendor_id",
            "week",
            "canonical_week_id",
            "week_start_date",
            "week_end_date",
            "calendar_year",
            "calendar_week_number",
            # seasonality
            "season_label",
            "season_type",
            "event_code",
            "is_peak_week",
            "is_event_week",
            "quarter_label",
        ]
        if c in df.columns
    ]
    norm_cols = [c for c in df.columns if str(c).endswith("__norm")]
    delta_cols = [
        c
        for c in df.columns
        if any(
            c.endswith(suf)
            for suf in (
                "_wow_abs",
                "_wow_pct",
                "_l4w_abs",
                "_l4w_pct",
                "_l12w_abs",
                "_l12w_pct",
                "_l4w_delta_abs",
                "_l4w_delta_pct",
                "_l12w_delta_abs",
                "_l12w_delta_pct",
            )
        )
    ]
    signal_cols = [
        c
        for c in (
            "profitability_signal",
            "growth_signal",
            "availability_signal",
            "inventory_risk_signal",
            "quality_signal",
        )
        if c in df.columns
    ]

    transformed_metrics_df = df[id_cols + norm_cols] if norm_cols else pd.DataFrame()
    delta_signals_df = df[id_cols + delta_cols] if delta_cols else pd.DataFrame()
    # Include season flags with dimension signals export if present
    extra_flag_cols = [c for c in ["is_peak_to_peak_transition", "is_offpeak_recovery_week"] if c in df.columns]
    dimension_signals_df = df[id_cols + signal_cols + extra_flag_cols] if signal_cols or extra_flag_cols else pd.DataFrame()

    # Metadata
    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "phase1_path": str(phase1_pq),
            "dimensions_config": str(Path(dimensions_config).resolve()),
            "scoring_config": str(Path(scoring_config).resolve()),
            "metric_registry_config_path": str(Path(registry_cfg_path).resolve()),
        },
        "rows": {
            "phase1": int(len(df)),
            "detailed": int(len(detailed)),
            "summary_vendors": int(len(summary)) if not summary.empty else 0,
            "entities_ready": int(readiness_entity["is_ready"].sum()),
        },
        "metrics_configured": list(metrics_cfg.keys()),
        "metric_registry": {
            "discovered_metric_count": int(len([1 for v in discovered.values() if v and v != "UNREGISTERED"])) if discovered else 0,
            "registered_metric_count": int(len(registry)),
            "missing_required_metrics": active.get("missing_required_metrics", []) if isinstance(active, dict) else [],
            "unregistered_metrics": active.get("unregistered_metrics", []) if isinstance(active, dict) else [],
            "present_canonical": active.get("present_canonical", []) if isinstance(active, dict) else [],
        },
        "calendar": {
            "season_source": season_source,
            "counts_season_type": df["season_type"].value_counts(dropna=False).astype(int).to_dict() if "season_type" in df.columns else {},
            "counts_event_code": df["event_code"].value_counts(dropna=False).astype(int).to_dict() if "event_code" in df.columns else {},
            "peak_week_count": int(pd.Series(df.get("is_peak_week", False)).fillna(False).sum()),
            "total_weeks": int(len(df)),
        },
        "coverage": {
            "enabled": bool(coverage_cfg is not None),
            "num_metrics_discovered": int(len([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])),
            "num_metrics_unreliable": int(coverage_df.shape[0] - coverage_df[coverage_df.get("metric_ok", False) == True].shape[0]) if not coverage_df.empty else 0,  # noqa: E712
            "num_asins_with_unreliable_composite": int((detailed.get("composite_coverage_flag", "OK") == "UNRELIABLE").sum()) if "composite_coverage_flag" in detailed.columns else 0,
            "artifacts": {
                "coverage_path": str((Path(output_base) / "metric_coverage.parquet").resolve()),
                "coverage_summary_path": str((Path(output_base) / "metric_coverage_summary.parquet").resolve()),
            },
        },
    }

    _write_outputs(detailed, summary, readiness_entity, metadata, output_base)

    # Write Session 2A artifacts
    out_dir = Path(output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not transformed_metrics_df.empty:
        transformed_metrics_df.to_parquet(out_dir / "transformed_metrics.parquet", index=False)
    if not delta_signals_df.empty:
        delta_signals_df.to_parquet(out_dir / "delta_signals.parquet", index=False)
    if not dimension_signals_df.empty:
        dimension_signals_df.to_parquet(out_dir / "dimension_signals.parquet", index=False)

    # Build Session 2A metadata
    try:
        import hashlib

        cfg_text = Path(scoring_config).read_text(encoding="utf-8")
        config_hash = hashlib.sha256(cfg_text.encode("utf-8")).hexdigest()
    except Exception:
        config_hash = None

    engineered_cols = list(set(norm_cols + delta_cols + signal_cols))
    col_min_max = {}
    for c in engineered_cols:
        try:
            s = pd.to_numeric(df[c], errors="coerce")
            col_min_max[c] = {"min": float(s.min(skipna=True)) if s.notna().any() else None,
                              "max": float(s.max(skipna=True)) if s.notna().any() else None}
        except Exception:
            col_min_max[c] = {"min": None, "max": None}

    dq_flags = {col: bool(pd.Series(df[col]).fillna(0).astype(float).gt(0).any()) for col in df.columns if str(col).startswith("dq_")}

    session2a_meta = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "config_hash": config_hash,
        "column_counts": {
            "inputs": len(id_cols),
            "norm": len(norm_cols),
            "deltas": len(delta_cols),
            "signals": len(signal_cols),
        },
        "min_max": col_min_max,
        "dq_propagation": dq_flags,
    }
    with (out_dir / "session2A_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(session2a_meta, f, indent=2)

    # Write Phase 2c coverage artifacts (if computed)
    if coverage_cfg is not None and not coverage_df.empty:
        try:
            coverage_df.to_parquet(out_dir / "metric_coverage.parquet", index=False)
        except Exception:
            pass
    if coverage_cfg is not None and not coverage_summary.empty:
        try:
            coverage_summary.to_parquet(out_dir / "metric_coverage_summary.parquet", index=False)
        except Exception:
            pass

    # ------------------
    # Session 2B: Final readiness and composite artifacts
    # ------------------
    df_final = finalize_readiness(df_scored, cfg.get("readiness", {}))

    # Write composite_scores.parquet (asin-week)
    comp_id_cols = [
        c
        for c in [
            "asin",
            "vendor_id",
            "week",
            "canonical_week_id",
            "week_start_date",
            "week_end_date",
            "calendar_year",
            "calendar_week_number",
            # seasonality
            "season_label",
            "season_type",
            "event_code",
            "is_peak_week",
            "is_event_week",
            "quarter_label",
            "brand_name",
            "category",
            "sub_category",
            "gl_code",
            "category_name",
        ]
        if c in df_final.columns
    ]
    comp_cols = [
        c for c in [
            "profitability_signal",
            "growth_signal",
            "availability_signal",
            "inventory_risk_signal",
            "quality_signal",
            "score_profitability",
            "score_growth",
            "score_availability",
            "score_inventory_risk",
            "score_quality",
            "archetype",
            "composite_raw",
            "composite_score",
            "composite_inputs_json",
            "readiness_flag",
            "readiness_reason",
        ] if c in df_final.columns
    ]
    composite_scores_df = df_final[comp_id_cols + comp_cols].copy()

    # Vendor/category summaries (mean composite_score, asin_count) by vendor/category/week
    def _safe_group(df_in: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
        keys = [k for k in keys if k in df_in.columns]
        if not keys:
            return pd.DataFrame()
        agg = (
            df_in.groupby(keys)
            .agg(composite_score_mean=("composite_score", "mean"), asin_count=("asin", "nunique"))
            .reset_index()
        )
        return agg

    composite_summary_vendor = _safe_group(df_final, ["vendor_id", "week"])  # vendor-week
    # Prefer category from dim join if available
    cat_keys = [k for k in ["category", "gl_code", "category_name", "week"] if k in df_final.columns]
    composite_summary_category = _safe_group(df_final, cat_keys)

    # Write Session 2B artifacts
    if not composite_scores_df.empty:
        composite_scores_df.to_parquet(out_dir / "composite_scores.parquet", index=False)
    if not composite_summary_vendor.empty:
        composite_summary_vendor.to_parquet(out_dir / "composite_summary_vendor.parquet", index=False)
    if not composite_summary_category.empty:
        composite_summary_category.to_parquet(out_dir / "composite_summary_category.parquet", index=False)

    # Readiness final diagnostics export
    readiness_final_cols = comp_id_cols + [c for c in ["readiness_flag", "readiness_reason", "history_weeks", "null_share_overall"] if c in df_final.columns]
    readiness_final = df_final[readiness_final_cols].drop_duplicates() if readiness_final_cols else pd.DataFrame()
    if not readiness_final.empty:
        readiness_final.to_parquet(out_dir / "readiness_final.parquet", index=False)

    # Session 2B metadata
    session2b_meta = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "config_hash": session2a_meta.get("config_hash"),
        "archetype_validation": arch_val,
        "rows": {
            "composite": int(len(composite_scores_df)) if not composite_scores_df.empty else 0,
            "summary_vendor": int(len(composite_summary_vendor)) if not composite_summary_vendor.empty else 0,
            "summary_category": int(len(composite_summary_category)) if not composite_summary_category.empty else 0,
        },
        "warnings": arch_val.get("warnings", []) if isinstance(arch_val, dict) else [],
    }
    with (out_dir / "session2B_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(session2b_meta, f, indent=2)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Brightlight Phase 2 Scoring Engine")
    p.add_argument("--phase1", required=True, help="Path to Phase 1 validated parquet")
    p.add_argument("--dimensions", required=True, help="Path to dimension_paths.yaml")
    p.add_argument("--scoring_config", required=True, help="Path to scoring_config.yaml")
    p.add_argument("--output", required=True, help="Output directory for Brightlight Phase 2 outputs")
    p.add_argument("--calendar_config", required=False, default=None, help="Path to phase2_calendar.yaml (optional)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_scoring(
        phase1_path=args.phase1,
        dimensions_config=args.dimensions,
        scoring_config=args.scoring_config,
        output_base=args.output,
        calendar_config_path=args.calendar_config,
    )


if __name__ == "__main__":
    main()
