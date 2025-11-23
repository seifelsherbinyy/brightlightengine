"""Phase 2 scoring engine entry point."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Dict, Iterable

import pandas as pd

from .ingestion_utils import load_config
from .scoring_utils import (
    add_rankings,
    aggregate_metrics,
    assign_performance_category,
    build_metric_map,
    compute_composite_scores,
    compute_normalised_scores,
    ensure_metric_coverage,
    evaluate_improvement_flags,
    load_normalized_input,
    persist_outputs,
    setup_logging,
    summarize_entities,
    write_audit,
    write_metadata,
)
def _prepare_entity_frame(
    normalized_df: pd.DataFrame,
    metric_map: Dict[str, object],
    entity_type: str,
    aggregation: str,
    fallback_normalised: float,
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate and score metrics for the requested entity level."""

    frame = normalized_df.copy()
    frame["entity_type"] = entity_type

    entity_columns: list[str] = ["entity_type", "vendor_code"]
    if "vendor_name" in frame.columns and frame["vendor_name"].notna().any():
        entity_columns.append("vendor_name")

    if entity_type != "vendor":
        entity_columns.append("asin")

    aggregated = aggregate_metrics(frame, entity_columns, aggregation=aggregation)
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
    """Compute composite scores, rankings, and qualitative outputs."""

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


def run_phase2(config_path: str = "config.yaml") -> Dict[str, pd.DataFrame]:
    """Execute the scoring pipeline and return computed scorecards."""

    config = load_config(config_path)
    logger = setup_logging(config)

    logger.info("Starting Phase 2 scoring with configuration %s", config_path)
    normalized_df = load_normalized_input(config, logger=logger)
    if normalized_df.empty:
        logger.warning("Normalized dataset is empty. No scoring will be performed.")
        return {"vendor": pd.DataFrame(), "asin": pd.DataFrame(), "audit": pd.DataFrame()}

    scoring_config = config.get("scoring", {})
    fallback_normalised = float(scoring_config.get("fallback_normalized_value", 0.5))
    metric_map = _prepare_metric_map(config)
    actual_metric_map = metric_map
    aggregation = scoring_config.get("aggregation", "mean")
    composite_scale = float(scoring_config.get("composite_scale", 1000))
    thresholds = scoring_config.get("thresholds", {})
    improvement_rules = scoring_config.get("improvement_rules", {})

    vendor_scored, vendor_entity_columns = _prepare_entity_frame(
        normalized_df,
        metric_map=actual_metric_map,
        entity_type="vendor",
        aggregation=aggregation,
        fallback_normalised=fallback_normalised,
    )
    vendor_scored = _apply_scoring(
        vendor_scored,
        metric_map=actual_metric_map,
        composite_scale=composite_scale,
        thresholds=thresholds,
        improvement_rules=improvement_rules,
        entity_columns=vendor_entity_columns,
    )

    asin_scored, asin_entity_columns = _prepare_entity_frame(
        normalized_df,
        metric_map=actual_metric_map,
        entity_type="asin",
        aggregation=aggregation,
        fallback_normalised=fallback_normalised,
    )
    asin_scored = _apply_scoring(
        asin_scored,
        metric_map=actual_metric_map,
        composite_scale=composite_scale,
        thresholds=thresholds,
        improvement_rules=improvement_rules,
        entity_columns=asin_entity_columns,
    )

    audit_frames = []
    if not vendor_scored.empty:
        audit_frames.append(vendor_scored.assign(entity_level="vendor"))
    if not asin_scored.empty:
        audit_frames.append(asin_scored.assign(entity_level="asin"))
    audit_df = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()

    vendor_paths = persist_outputs(vendor_scored, config, "vendor", logger)
    asin_paths = persist_outputs(asin_scored, config, "asin", logger)
    audit_path = write_audit(audit_df, config)

    timestamp = datetime.now(timezone.utc).isoformat()
    metadata = {
        "timestamp": timestamp,
        "records_vendor": int(len(vendor_scored)),
        "records_asin": int(len(asin_scored)),
        "metric_weights": {name: config.get("scoring", {}).get("metric_weights", {}).get(name) for name in actual_metric_map},
        "top_vendor": vendor_scored.sort_values("Composite_Score", ascending=False)["vendor_code"].head(1).tolist(),
        "score_range": {
            "min": float(pd.concat([vendor_scored["Composite_Score"], asin_scored["Composite_Score"]]).min())
            if not vendor_scored.empty or not asin_scored.empty
            else None,
            "max": float(pd.concat([vendor_scored["Composite_Score"], asin_scored["Composite_Score"]]).max())
            if not vendor_scored.empty or not asin_scored.empty
            else None,
        },
        "output_paths": {
            "vendor_csv": str(vendor_paths[0]),
            "vendor_parquet": str(vendor_paths[1]) if vendor_paths[1] else None,
            "asin_csv": str(asin_paths[0]),
            "asin_parquet": str(asin_paths[1]) if asin_paths[1] else None,
            "audit_csv": str(audit_path),
        },
    }
    write_metadata(metadata, config)

    top_vendors, bottom_vendors = summarize_entities(vendor_scored, "vendor_code")
    if not top_vendors.empty:
        print("Top 5 vendors:\n", top_vendors.to_string(index=False))
    if not bottom_vendors.empty:
        print("Bottom 5 vendors:\n", bottom_vendors.to_string(index=False))

    logger.info("Phase 2 scoring completed for %s vendors and %s ASINs", metadata["records_vendor"], metadata["records_asin"])
    return {"vendor": vendor_scored, "asin": asin_scored, "audit": audit_df}


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
