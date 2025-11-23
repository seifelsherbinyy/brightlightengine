"""Phase 2 scoring engine entry point."""
from __future__ import annotations

import argparse
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from .ingestion_utils import load_config
import logging
from .scoring_utils import (
    setup_logging,
    # New Phase 2 helpers
    aggregate_metrics_for_scoring,
    compute_metric_normalized_scores,
    compute_dimension_scores,
    compute_composite_score,
    analyze_zero_behavior,
    compute_reliability_flags,
    build_asin_l4w_performance,
    build_vendor_scoring_enriched,
    build_asin_scoring_enriched,
    # Legacy scoring helpers for backward compatibility
    build_metric_map,
    aggregate_metrics,
    ensure_metric_coverage,
    compute_normalised_scores,
    compute_composite_scores,
    assign_performance_category,
    evaluate_improvement_flags,
    add_rankings,
    persist_outputs,
    write_audit,
    write_metadata,
)
from .metric_registry_utils import load_metric_registry
from .logging_utils import log_system_event, log_warning, log_error, record_topic_timing
from .periodic_export_utils import (
    derive_period_column,
    get_recent_periods,
    export_periods_to_tabs,
    export_periods_to_files,
)
from .path_utils import get_phase1_normalized_path
def _prepare_entity_frame(
    normalized_df: pd.DataFrame,
    metric_map: Dict[str, object],
    entity_type: str,
    aggregation: str,
    fallback_normalised: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate and score metrics for the requested entity level (legacy format)."""

    frame = normalized_df.copy()
    frame["entity_type"] = entity_type

    entity_columns: list[str] = ["entity_type", "vendor_code"]
    if "vendor_name" in frame.columns and frame["vendor_name"].notna().any():
        entity_columns.append("vendor_name")

    if entity_type != "vendor":
        entity_columns.append("asin")

    aggregated = aggregate_metrics(frame, entity_columns, aggregation=aggregation)
    if not aggregated.empty:
        aggregated = aggregated[aggregated["metric"].isin(metric_map.keys())]
        aggregated = ensure_metric_coverage(aggregated, entity_columns, metric_map)
        aggregated = compute_normalised_scores(
            aggregated,
            metric_map=metric_map,
            fallback_normalised=fallback_normalised,
        )
    return aggregated, entity_columns


def _apply_scoring(
    scored: pd.DataFrame,
    metric_map: Dict[str, object],
    composite_scale: float,
    thresholds: Dict[str, float],
    improvement_rules: Dict[str, Dict[str, float]],
    entity_columns: Iterable[str],
) -> pd.DataFrame:
    """Compute composite scores, rankings, and qualitative outputs (legacy)."""

    if scored is None or scored.empty:
        return scored
    scored = compute_composite_scores(
        scored,
        metric_map=metric_map,
        composite_scale=composite_scale,
        entity_columns=entity_columns,
    )
    scored = assign_performance_category(scored, thresholds)
    scored = evaluate_improvement_flags(scored, improvement_rules)
    scored = add_rankings(scored, entity_columns)
    return scored


def _prepare_metric_map(config: Dict) -> Dict[str, object]:
    return build_metric_map(config)


def validate_input_schema(df: pd.DataFrame) -> None:
    """
    Validate that input DataFrame has required columns.
    
    Raises:
        ValueError: If required columns are missing
    """
    required_cols = ['entity_id', 'vendor_code', 'Week_Order', 'metric', 'value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(
            f"Input data missing required columns: {missing_cols}. "
            f"Expected columns: {required_cols}"
        )


def _generate_enhanced_outputs(
    config: Dict,
    vendor_scored: pd.DataFrame,
    asin_scored: pd.DataFrame,
    logger
) -> None:
    """
    Generate enhanced scoring outputs: scoring_matrix.parquet and vendor_scoreboard.parquet.
    
    This creates wide-format outputs with per-metric contributions for better explainability.
    """
    from .ingestion_utils import ensure_directory
    
    # Determine output directory
    output_dir = Path(config.get("paths", {}).get("scoring_output_dir", config["paths"]["processed_dir"]))
    ensure_directory(output_dir)
    
    # Build scoring matrix from vendor and ASIN scored data
    # Combine vendor and ASIN data
    all_scored = []
    if not vendor_scored.empty:
        df_v = vendor_scored.copy()
        df_v['entity_id'] = df_v['vendor_code']
        all_scored.append(df_v)
    if not asin_scored.empty:
        df_a = asin_scored.copy()
        df_a['entity_id'] = df_a.get('vendor_code', '') + '_' + df_a.get('asin', '')
        all_scored.append(df_a)
    
    if not all_scored:
        logger.warning("No scored data to generate enhanced outputs")
        return
    
    combined = pd.concat(all_scored, ignore_index=True)
    
    # Build wide format with contributions per metric
    # Get unique entity/week combinations
    if 'Week_Order' not in combined.columns and 'week_label' in combined.columns:
        # Map week labels to order
        combined['Week_Order'] = combined.groupby('week_label').ngroup() + 1
    elif 'Week_Order' not in combined.columns:
        combined['Week_Order'] = 1
    
    # Pivot metrics to wide format
    entity_cols = ['entity_id', 'vendor_code', 'Week_Order']
    
    # Get available entity columns
    if 'vendor_name' in combined.columns:
        entity_cols.append('vendor_name')
    
    # Build pivot with contributions
    pivot_data = []
    for entity_id in combined['entity_id'].unique():
        entity_data = combined[combined['entity_id'] == entity_id]
        
        # Get first row for entity info
        first_row = entity_data.iloc[0]
        row_dict = {
            'entity_id': entity_id,
            'vendor_code': first_row.get('vendor_code', ''),
            'Week_Order': first_row.get('Week_Order', 1)
        }
        
        if 'vendor_name' in entity_data.columns:
            row_dict['vendor_name'] = first_row.get('vendor_name', '')
        
        # Add composite score if available
        if 'Composite_Score' in entity_data.columns:
            row_dict['score_composite'] = first_row['Composite_Score']
            row_dict['score_final'] = first_row['Composite_Score']
        
        # Add per-metric data
        for _, metric_row in entity_data.iterrows():
            metric_name = metric_row.get('metric', '')
            if not metric_name:
                continue
            
            # Add standardized value
            if 'Normalized_Value' in metric_row:
                row_dict[f'std_{metric_name}'] = metric_row['Normalized_Value']
            
            # Add contribution
            if 'Weighted_Score' in metric_row:
                row_dict[f'contrib_{metric_name}'] = metric_row['Weighted_Score']
                
                # Compute percentage contribution
                if 'Composite_Score' in first_row and first_row['Composite_Score'] != 0:
                    pct = (metric_row['Weighted_Score'] / first_row['Composite_Score']) * 100
                    row_dict[f'pct_contrib_{metric_name}'] = pct
        
        # Add placeholder columns for enhanced features
        row_dict['penalty_total'] = 0.0
        row_dict['score_final_conf'] = 1.0
        row_dict['reliability'] = 1.0
        
        pivot_data.append(row_dict)
    
    scoring_matrix = pd.DataFrame(pivot_data)
    
    # Save scoring matrix
    try:
        matrix_path = output_dir / "scoring_matrix.parquet"
        scoring_matrix.to_parquet(matrix_path, index=False)
        logger.info(f"Saved scoring_matrix.parquet with {len(scoring_matrix)} rows")
    except Exception as e:
        logger.warning(f"Failed to save parquet, falling back to CSV: {e}")
        matrix_csv = output_dir / "scoring_matrix.csv"
        scoring_matrix.to_csv(matrix_csv, index=False)
        logger.info(f"Saved scoring_matrix.csv with {len(scoring_matrix)} rows")
    
    # Build vendor scoreboard
    vendor_scoreboard = scoring_matrix.groupby(['vendor_code', 'Week_Order']).agg({
        'score_final': 'mean'
    }).reset_index()
    
    # Add rank within week
    vendor_scoreboard['rank'] = vendor_scoreboard.groupby('Week_Order')['score_final'].rank(
        ascending=False, method='min'
    ).astype(int)
    
    # Save vendor scoreboard
    try:
        scoreboard_path = output_dir / "vendor_scoreboard.parquet"
        vendor_scoreboard.to_parquet(scoreboard_path, index=False)
        logger.info(f"Saved vendor_scoreboard.parquet with {len(vendor_scoreboard)} rows")
    except Exception as e:
        logger.warning(f"Failed to save parquet, falling back to CSV: {e}")
        scoreboard_csv = output_dir / "vendor_scoreboard.csv"
        vendor_scoreboard.to_csv(scoreboard_csv, index=False)
        logger.info(f"Saved vendor_scoreboard.csv with {len(vendor_scoreboard)} rows")


def run_phase2(config_path: str = "config.yaml") -> Dict[str, pd.DataFrame]:
    """Execute the refined Phase 2 scoring pipeline and return key dataframes."""

    config = load_config(config_path)
    logger = setup_logging(config)
    logger.info("Starting Phase 2 scoring with configuration %s", config_path)

    t0 = time.perf_counter()
    # Load normalized Phase 1 output from canonical path
    input_path = get_phase1_normalized_path(config)
    logger.info("Phase 2 reading normalized Phase 1 output from: %s", input_path.resolve())
    if not input_path.exists():
        raise FileNotFoundError(f"Normalized Phase 1 output not found at {input_path}. Run Phase 1 first.")
    norm_df = pd.read_parquet(input_path)
    if norm_df.empty:
        logger.error("Phase 2 loaded an EMPTY Phase 1 file — likely path mismatch.")
        raise RuntimeError("Normalized Phase 1 data missing.")
    if norm_df.empty:
        logger.warning("Normalized dataset is empty. No scoring will be performed.")
        return {"vendor_scorecards": pd.DataFrame(), "asin_scorecards": pd.DataFrame(), "vendor": pd.DataFrame(), "asin": pd.DataFrame(), "audit": pd.DataFrame()}

    # Load metric registry (for direction)
    registry_path = Path(config.get("registry", {}).get("metric_registry", "metadata/metric_registry.csv"))
    metric_registry_df = load_metric_registry(registry_path)

    # 1A) New scoring: Aggregate metrics (window: L4W/ALL)
    v_metrics = aggregate_metrics_for_scoring(norm_df, "vendor", config)
    a_metrics = aggregate_metrics_for_scoring(norm_df, "asin", config)

    # 1B) Zero-semantics and signal analysis (per-entity/metric)
    invalid_map, zero_counters = analyze_zero_behavior(norm_df, config)

    # 2) Normalise metrics and compute dimension + composite scores
    v_scored = compute_metric_normalized_scores(v_metrics, config, metric_registry_df)
    v_scored = compute_dimension_scores(v_scored, config, entity_level="vendor", invalid_metric_map=invalid_map)
    v_scored = compute_composite_score(v_scored, config)

    a_scored = compute_metric_normalized_scores(a_metrics, config, metric_registry_df)
    a_scored = compute_dimension_scores(a_scored, config, entity_level="asin", invalid_metric_map=invalid_map)
    a_scored = compute_composite_score(a_scored, config)

    # 3) Reliability flags
    v_scored = compute_reliability_flags(v_scored, config)
    a_scored = compute_reliability_flags(a_scored, config)

    # 4) Base scorecards
    vendor_scorecards_df = v_scored.copy()
    asin_scorecards_df = a_scored.copy()

    # 5) ASIN L4W performance
    asin_l4w_perf_df = build_asin_l4w_performance(norm_df, config)

    # 6) Enriched tables
    vendor_scoring_enriched_df = build_vendor_scoring_enriched(vendor_scorecards_df, v_metrics, config)
    asin_scoring_enriched_df = build_asin_scoring_enriched(asin_scorecards_df, a_metrics, asin_l4w_perf_df, config)

    # 1B) Legacy scoring path for backward compatibility (per-metric long rows)
    scoring_config = config.get("scoring", {})
    fallback_normalised = float(scoring_config.get("fallback_normalized_value", 0.5))
    aggregation = scoring_config.get("aggregation", "mean")
    composite_scale = float(scoring_config.get("composite_scale", 1000))
    thresholds = scoring_config.get("thresholds", {})
    improvement_rules = scoring_config.get("improvement_rules", {})

    metric_map = _prepare_metric_map(config)
    vendor_scored_legacy, vendor_entity_columns = _prepare_entity_frame(
        norm_df,
        metric_map=metric_map,
        entity_type="vendor",
        aggregation=aggregation,
        fallback_normalised=fallback_normalised,
    )
    vendor_scored_legacy = _apply_scoring(
        vendor_scored_legacy,
        metric_map=metric_map,
        composite_scale=composite_scale,
        thresholds=thresholds,
        improvement_rules=improvement_rules,
        entity_columns=vendor_entity_columns,
    )

    asin_scored_legacy, asin_entity_columns = _prepare_entity_frame(
        norm_df,
        metric_map=metric_map,
        entity_type="asin",
        aggregation=aggregation,
        fallback_normalised=fallback_normalised,
    )
    asin_scored_legacy = _apply_scoring(
        asin_scored_legacy,
        metric_map=metric_map,
        composite_scale=composite_scale,
        thresholds=thresholds,
        improvement_rules=improvement_rules,
        entity_columns=asin_entity_columns,
    )

    # 7) Validation
    def _validate(df: pd.DataFrame, entity_cols: list[str], name: str) -> None:
        if df is None or df.empty:
            return
        if "Composite_Score" not in df.columns or df["Composite_Score"].isna().any():
            logger.warning("Validation: Some %s rows have null Composite_Score", name)
        for col in ["Sales_Score", "Margin_Score", "Availability_Score", "Traffic_Score"]:
            if col not in df.columns:
                logger.warning("Validation: %s missing %s", name, col)
        dupes = df.duplicated(subset=[c for c in entity_cols if c in df.columns], keep=False)
        if dupes.any():
            logger.warning("Validation: %s contains duplicate rows for entity grain", name)

    _validate(vendor_scorecards_df, ["vendor_code", "week_start"], "vendor_scorecards")
    _validate(asin_scorecards_df, ["vendor_code", "asin", "week_start"], "asin_scorecards")

    # 8) Persist outputs to Brightlight scoring folder
    paths = config.get("paths", {})
    scoring_root = Path(paths.get("scoring_root", "data/processed/scoring"))
    scoring_root.mkdir(parents=True, exist_ok=True)

    def _safe_to_parquet_or_csv(df: pd.DataFrame, path: Path):
        try:
            df.to_parquet(path, index=False)
        except Exception as exc:
            logger.warning("Parquet write failed at %s (%s); falling back to CSV", path, exc)
            df.to_csv(path.with_suffix('.csv'), index=False)

    vendor_scorecards_path = scoring_root / "vendor_scorecards.parquet"
    asin_scorecards_path = scoring_root / "asin_scorecards.parquet"
    vendor_enriched_path = scoring_root / "vendor_scoring_enriched.parquet"
    asin_enriched_path = scoring_root / "asin_scoring_enriched.parquet"
    asin_l4w_perf_path = scoring_root / "asin_l4w_performance.parquet"
    metadata_path = scoring_root / "scoring_metadata.json"
    params_path = scoring_root / "scoring_parameters.json"

    _safe_to_parquet_or_csv(vendor_scorecards_df, vendor_scorecards_path)
    _safe_to_parquet_or_csv(asin_scorecards_df, asin_scorecards_path)
    _safe_to_parquet_or_csv(vendor_scoring_enriched_df, vendor_enriched_path)
    _safe_to_parquet_or_csv(asin_scoring_enriched_df, asin_enriched_path)
    _safe_to_parquet_or_csv(asin_l4w_perf_df, asin_l4w_perf_path)

    # 9) Metadata and parameters
    try:
        dims_cfg = config.get("scoring", {}).get("dimensions", {}) or {}
        metrics_used = sorted({m for arr in dims_cfg.values() for m in arr})
        # Parameter ranges per metric from aggregated data
        ranges: Dict[str, Dict[str, float]] = {}
        agg_for_ranges = pd.concat([v_metrics.drop(columns=[c for c in v_metrics.columns if c not in metrics_used], errors='ignore'),
                                    a_metrics.drop(columns=[c for c in a_metrics.columns if c not in metrics_used], errors='ignore')],
                                   ignore_index=True)
        for m in metrics_used:
            if m in agg_for_ranges.columns:
                s = pd.to_numeric(agg_for_ranges[m], errors="coerce")
                ranges[m] = {"min": float(s.min()) if s.notna().any() else None,
                             "max": float(s.max()) if s.notna().any() else None}
        metadata = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics_scored": metrics_used,
            "dimensions": dims_cfg,
            "entity_counts": {
                "vendors": int(vendor_scorecards_df.get("vendor_code", pd.Series()).nunique() if not vendor_scorecards_df.empty else 0),
                "asins": int(asin_scorecards_df.get("asin", pd.Series()).nunique() if not asin_scorecards_df.empty else 0),
            },
        }
        params = {
            "metric_ranges": ranges,
            "dimension_weights": (config.get("scoring", {}).get("weights", {}) or {}).get("dimension", {}),
            "scoring_version": "v2.0",
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write scoring metadata/parameters: %s", exc)

    # 10) Logging summary (zero-semantics + persisted artifact sizes)
    duration = time.perf_counter() - t0

    # Zero-semantics summary
    try:
        logger.info("Zero-semantics summary:")
        logger.info(" - all_zero_series_excluded: %d", int(zero_counters.get("all_zero_series_excluded", 0)))
        logger.info(" - bad_zero_treated_as_missing: %d", int(zero_counters.get("bad_zero_treated_as_missing", 0)))
        logger.info(" - good_zero_kept_as_signal: %d", int(zero_counters.get("good_zero_kept_as_signal", 0)))
        logger.info(" - low_signal_exclusions: %d", int(zero_counters.get("low_signal_exclusions", 0)))
    except Exception:
        pass

    def _rows_from_saved(parquet_path: Path, csv_fallback: Path, in_memory_df: pd.DataFrame) -> int:
        try:
            if parquet_path.exists():
                return int(len(pd.read_parquet(parquet_path)))
        except Exception:
            # Fall back to CSV with same stem if parquet read fails
            pass
        try:
            if csv_fallback.exists():
                return int(sum(1 for _ in csv_fallback.open("r", encoding="utf-8", errors="ignore")) - 1)  # minus header
        except Exception:
            pass
        # Last resort: in-memory length
        try:
            return int(len(in_memory_df)) if in_memory_df is not None else 0
        except Exception:
            return 0

    vendor_rows = _rows_from_saved(
        vendor_scorecards_path,
        vendor_scorecards_path.with_suffix(".csv"),
        vendor_scorecards_df,
    )
    asin_rows = _rows_from_saved(
        asin_scorecards_path,
        asin_scorecards_path.with_suffix(".csv"),
        asin_scorecards_df,
    )
    vendor_enriched_rows = _rows_from_saved(
        vendor_enriched_path,
        vendor_enriched_path.with_suffix(".csv"),
        vendor_scoring_enriched_df,
    )
    asin_enriched_rows = _rows_from_saved(
        asin_enriched_path,
        asin_enriched_path.with_suffix(".csv"),
        asin_scoring_enriched_df,
    )

    logger.info(
        "Phase 2 summary: vendor_scorecards_rows=%d, asin_scorecards_rows=%d, vendor_enriched_rows=%d, asin_enriched_rows=%d, duration=%.2fs",
        vendor_rows,
        asin_rows,
        vendor_enriched_rows,
        asin_enriched_rows,
        duration,
    )
    logger.info("Phase 2 scoring completed successfully")

    # Legacy outputs (files and return keys)
    try:
        # Persist per existing utilities
        vendor_paths = persist_outputs(vendor_scored_legacy, config, "vendor", logger)
        asin_paths = persist_outputs(asin_scored_legacy, config, "asin", logger)
        audit_frames = []
        if not (vendor_scored_legacy is None or vendor_scored_legacy.empty):
            audit_frames.append(vendor_scored_legacy.assign(entity_level="vendor"))
        if not (asin_scored_legacy is None or asin_scored_legacy.empty):
            audit_frames.append(asin_scored_legacy.assign(entity_level="asin"))
        audit_df = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
        audit_path = write_audit(audit_df, config)

        # Minimal metadata for legacy
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata = {
            "timestamp": timestamp,
            "records_vendor": int(len(vendor_scored_legacy)) if vendor_scored_legacy is not None else 0,
            "records_asin": int(len(asin_scored_legacy)) if asin_scored_legacy is not None else 0,
        }
        write_metadata(metadata, config)
    except Exception as exc:
        logger.warning("Legacy scorecard persistence encountered an issue: %s", exc)

    result = {
        "vendor_scorecards": vendor_scorecards_df,
        "asin_scorecards": asin_scorecards_df,
        "vendor_enriched": vendor_scoring_enriched_df,
        "asin_enriched": asin_scoring_enriched_df,
        "asin_l4w_performance": asin_l4w_perf_df,
        # Backward-compatible keys expected by some tests
        "vendor": vendor_scored_legacy,
        "asin": asin_scored_legacy,
        "audit": audit_df if 'audit_df' in locals() else pd.DataFrame(),
    }
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""

    parser = argparse.ArgumentParser(description="Brightstar Phase 2 scoring")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    return parser


def main() -> None:
    """CLI entry point for Phase 2 scoring."""

    parser = build_arg_parser()
    args = parser.parse_args()
    run_phase2(config_path=args.config)


if __name__ == "__main__":
    main()
