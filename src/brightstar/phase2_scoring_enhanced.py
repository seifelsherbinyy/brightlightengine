"""Phase 2 scoring engine with robust transforms, penalties, and explainability.

This module implements a comprehensive scoring pipeline with:
- Configurable metric transforms (winsorize, log1p, logit)
- Robust standardization (percentile rank, robust z-score)
- Penalty system with auditable breakdown
- Per-metric contributions and reliability scores
- Deterministic, reproducible outputs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .common.config_validator import load_and_validate_config
from .ingestion_utils import ensure_directory, load_config
from .scoring.penalties import evaluate_penalties
from .scoring.transforms import (
    apply_direction,
    impute,
    log1p_safe,
    logit_safe,
    parse_percent_strings,
    standardize_percentile,
    standardize_robust_z,
    winsorize,
)
from .scoring.processing import (
    add_grain_columns,
    compute_derived_metrics,
    add_rolling_features,
    compute_signal_quality,
)
from .scoring.weights import compute_dynamic_weights
from .scoring.metric_mapping import build_metric_mapping, apply_metric_mapping
from .standards.naming import normalize_week_label
from .standards.metrics import resolve_configured_metrics


LOGGER_NAME = "brightstar.phase2_scoring"


def setup_logging(config: Dict) -> logging.Logger:
    """Configure logging for the scoring phase."""
    log_dir = ensure_directory(config["paths"]["logs_dir"])
    file_name = config.get("logging", {}).get("scoring_file_name", "phase2_scoring.log")
    log_path = log_dir / file_name

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, config.get("logging", {}).get("scoring_level", "INFO")))
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.debug("Phase 2 logging initialized at %s", log_path)
    return logger


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


def load_phase1_output(config: Dict, logger: logging.Logger) -> pd.DataFrame:
    """
    Load Phase 1 normalized output.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        DataFrame with Phase 1 output
    """
    # Try multiple candidate paths
    candidate_paths = []
    
    # First try the configured path
    ingestion_output_dir = config.get("paths", {}).get("ingestion_output_dir")
    if ingestion_output_dir:
        ingestion_dir = Path(ingestion_output_dir)
        candidate_paths.extend([
            ingestion_dir / "normalized_phase1.parquet",
            ingestion_dir / "normalized_phase1.csv"
        ])
    
    # Try processed_dir
    processed_dir = Path(config["paths"]["processed_dir"])
    candidate_paths.extend([
        processed_dir / "normalized_phase1.parquet",
        processed_dir / "normalized_phase1.csv"
    ])
    
    # Try configured normalized_output
    if config.get("paths", {}).get("normalized_output"):
        candidate_paths.append(Path(config["paths"]["normalized_output"]))
    
    for path in candidate_paths:
        if path.exists():
            logger.info(f"Loading Phase 1 output from {path}")
            if path.suffix.lower() == ".parquet":
                return pd.read_parquet(path)
            else:
                return pd.read_csv(path)
    
    raise FileNotFoundError(
        f"Could not find Phase 1 output. Tried: {[str(p) for p in candidate_paths]}"
    )


def prepare_long_format(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Prepare data in long format with required columns.
    
    Converts Phase 1 output to required format:
    ['entity_id', 'vendor_code', 'Week_Order', 'metric', 'value']
    """
    df = df.copy()
    
    # Create entity_id if missing (combination of vendor_code + asin or just vendor_code)
    if 'entity_id' not in df.columns:
        if 'asin' in df.columns:
            df['entity_id'] = df['vendor_code'].astype(str) + '_' + df['asin'].astype(str)
        else:
            df['entity_id'] = df['vendor_code'].astype(str)
    
    # Ensure Week_Order exists (prefer deterministic order by YYYYWNN label)
    if 'Week_Order' not in df.columns and 'week_label' in df.columns:
        # Determine a stable ordering of weeks by sorted unique labels (YYYYWNN sorts lexicographically)
        weeks = (
            df['week_label'].dropna().astype(str).unique().tolist()
            if 'week_label' in df.columns else []
        )
        weeks_sorted = sorted(weeks)
        week_map = {w: i + 1 for i, w in enumerate(weeks_sorted)}
        df['Week_Order'] = df['week_label'].map(week_map).astype('Int64')
    
    # Parse percent strings
    df['value'] = parse_percent_strings(df['value'])
    
    # Coerce value to numeric
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    
    # Drop rows with missing required fields
    required = ['entity_id', 'vendor_code', 'Week_Order', 'metric', 'value']
    before_count = len(df)
    df = df.dropna(subset=required)
    after_count = len(df)
    
    if before_count > after_count:
        logger.warning(f"Dropped {before_count - after_count} rows with missing required fields")
    
    return df


def process_metric(
    df_metric: pd.DataFrame,
    metric_config: object,
    rolling_weeks: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Process a single metric through the transform pipeline.
    
    Steps:
    1. Winsorize
    2. Apply direction
    3. Apply transform (none/log1p/logit)
    4. Standardize (percentile/robust_z) with rolling window
    5. Impute missing values
    
    Returns:
        DataFrame with std_value and raw_value columns
    """
    df = df_metric.copy()
    metric_name = metric_config.name
    
    logger.info(f"Processing metric: {metric_name}")
    logger.info(f"  Input rows: {len(df)}, NA count: {df['value'].isna().sum()}")
    
    # Keep original value as raw_value
    df['raw_value'] = df['value']
    
    # Step 1: Winsorize
    q_low, q_high = metric_config.clip_quantiles
    df['value'] = winsorize(df['value'], q_low, q_high)
    logger.info(f"  Winsorized at quantiles [{q_low}, {q_high}]")
    logger.info(f"  Range after winsorization: [{df['value'].min():.4f}, {df['value'].max():.4f}]")
    
    # Step 2: Apply direction
    df['value'] = apply_direction(df['value'], metric_config.direction)
    
    # Step 3: Apply transform
    if metric_config.transform == 'log1p':
        df['value'] = log1p_safe(df['value'])
        logger.info(f"  Applied log1p transform")
    elif metric_config.transform == 'logit':
        # For logit, ensure values are in [0, 1]
        if df['value'].min() < 0 or df['value'].max() > 1:
            # Auto-clip to [0, 1] range
            df['value'] = df['value'].clip(lower=0, upper=1)
            logger.warning(f"  Auto-clipped values to [0, 1] for logit transform")
        df['value'] = logit_safe(df['value'])
        logger.info(f"  Applied logit transform")
    else:
        logger.info(f"  No transform applied")
    
    # Step 4: Standardize with rolling window
    # Sort by Week_Order for rolling window
    df = df.sort_values('Week_Order')
    
    # Check if we have enough history for rolling window
    unique_weeks = df['Week_Order'].nunique()
    
    if unique_weeks >= rolling_weeks:
        # Apply rolling standardization
        # For each week, use the past rolling_weeks for standardization
        std_values = []
        
        for week in df['Week_Order'].unique():
            # Get historical window
            window_weeks = sorted(df['Week_Order'].unique())
            week_idx = window_weeks.index(week)
            start_idx = max(0, week_idx - rolling_weeks + 1)
            historical_weeks = window_weeks[start_idx:week_idx + 1]
            
            # Data for this window
            window_df = df[df['Week_Order'].isin(historical_weeks)].copy()
            
            # Standardize within this window
            if metric_config.standardize == 'percentile':
                window_std = standardize_percentile(window_df, 'value')
            else:  # robust_z
                window_std = standardize_robust_z(window_df, 'value')
            
            # Extract std_value for current week
            week_std = window_std[window_df['Week_Order'] == week]
            std_values.append(pd.DataFrame({
                'entity_id': window_df[window_df['Week_Order'] == week]['entity_id'],
                'std_value': week_std.values
            }))
        
        # Combine all standardized values
        std_df = pd.concat(std_values, ignore_index=True)
        df = df.merge(std_df, on='entity_id', how='left')
        logger.info(f"  Applied rolling window standardization ({rolling_weeks} weeks)")
    else:
        # Fallback to cross-sectional standardization per week
        logger.warning(
            f"  Insufficient history ({unique_weeks} weeks < {rolling_weeks}), "
            f"using cross-sectional standardization"
        )
        
        if metric_config.standardize == 'percentile':
            df['std_value'] = standardize_percentile(df, 'value', by_keys=['Week_Order'])
        else:  # robust_z
            df['std_value'] = standardize_robust_z(df, 'value', by_keys=['Week_Order'])
    
    # Step 5: Impute missing values
    df = impute(df, 'std_value', metric_config.missing_policy, group_keys=['Week_Order'])
    logger.info(f"  Applied imputation policy: {metric_config.missing_policy}")
    
    # Add metric name
    df['metric'] = metric_name
    
    return df


def compute_coverage(df_long: pd.DataFrame, rolling_weeks: int) -> pd.DataFrame:
    """
    Compute coverage per metric per entity/week.
    
    Coverage = share of non-NA values in last rolling_weeks windows
    """
    df = df_long.copy()
    
    # Sort by entity and week
    df = df.sort_values(['entity_id', 'metric', 'Week_Order'])
    
    # Compute rolling coverage
    def calc_coverage(group):
        group = group.sort_values('Week_Order')
        # For each row, look back rolling_weeks
        coverage = []
        for idx, row in group.iterrows():
            current_week = row['Week_Order']
            # Get data from current week and previous (rolling_weeks-1) weeks
            window_data = group[
                (group['Week_Order'] <= current_week) &
                (group['Week_Order'] > current_week - rolling_weeks)
            ]
            # Coverage is proportion of non-null values
            non_null_count = window_data['std_value'].notna().sum()
            total_count = len(window_data)
            cov = non_null_count / total_count if total_count > 0 else 0
            coverage.append(cov)
        
        group['coverage'] = coverage
        return group
    
    df = df.groupby(['entity_id', 'metric']).apply(calc_coverage).reset_index(drop=True)
    
    return df


def compute_reliability(df_long: pd.DataFrame, rolling_weeks: int) -> pd.Series:
    """
    Compute reliability score per entity/week.
    
    reliability = clip(min(coverage_weighted, 1.0) * (1 / (1 + variance)), 0, 1)
    
    Where:
    - coverage_weighted: weighted average of coverage across metrics
    - variance: variance of std_values in last rolling_weeks
    """
    df = df_long.copy()
    
    # Group by entity_id and Week_Order
    def calc_reliability(group):
        # Compute weighted coverage
        if 'weight' in group.columns:
            total_weight = group['weight'].sum()
            if total_weight > 0:
                coverage_weighted = (group['coverage'] * group['weight']).sum() / total_weight
            else:
                coverage_weighted = group['coverage'].mean()
        else:
            coverage_weighted = group['coverage'].mean()
        
        # Compute variance of std_values
        variance = group['std_value'].var()
        if pd.isna(variance):
            variance = 0
        
        # Compute reliability
        reliability = min(coverage_weighted, 1.0) * (1 / (1 + variance))
        reliability = np.clip(reliability, 0, 1)
        
        return reliability
    
    reliability = df.groupby(['entity_id', 'Week_Order']).apply(calc_reliability)
    
    return reliability


def build_wide_composite(
    df_long: pd.DataFrame,
    metrics_config: list,
    weights_must_sum_to_1: bool,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Build wide format with composite scores and contributions.
    
    Returns DataFrame with:
    - entity_id, vendor_code, Week_Order
    - std_<metric>, contrib_<metric>, pct_contrib_<metric> for each metric
    - score_composite
    """
    # Pivot to wide format
    wide_dfs = []
    
    for metric_cfg in metrics_config:
        metric_name = metric_cfg.name
        metric_df = df_long[df_long['metric'] == metric_name].copy()
        
        if metric_df.empty:
            logger.warning(f"No data for metric {metric_name}")
            continue
        
        # Compute contribution using effective per-row weight (already merged in df_long_std)
        # Fallback to configured static weight if column missing
        if 'weight' not in metric_df.columns:
            metric_df['weight'] = float(metric_cfg.weight)
        metric_df['contrib'] = metric_df['weight'].fillna(float(metric_cfg.weight)) * metric_df['std_value'].fillna(0)

        # Prepare pivot columns: std, contrib, and audit columns for weights
        pick_cols = ['entity_id', 'vendor_code', 'Week_Order', 'std_value', 'contrib']
        if 'weight' in metric_df.columns:
            pick_cols.append('weight')
        if 'weight_multiplier' in metric_df.columns:
            pick_cols.append('weight_multiplier')
        if 'base_weight' in metric_df.columns:
            pick_cols.append('base_weight')

        pivot_df = metric_df[pick_cols].copy()
        pivot_df = pivot_df.rename(columns={
            'std_value': f'std_{metric_name}',
            'contrib': f'contrib_{metric_name}',
            'weight': f'weight_{metric_name}',
            'weight_multiplier': f'weightmul_{metric_name}',
            'base_weight': f'base_weight_{metric_name}',
        })
        
        wide_dfs.append(pivot_df)
    
    # Merge all metrics
    if not wide_dfs:
        logger.error("No metrics data to process")
        return pd.DataFrame()
    
    wide = wide_dfs[0]
    for df in wide_dfs[1:]:
        wide = wide.merge(df, on=['entity_id', 'vendor_code', 'Week_Order'], how='outer')
    
    # Ensure percent contributions are based on existing contrib columns
    wide['score_composite'] = 0.0
    for metric_cfg in metrics_config:
        contrib_col = f'contrib_{metric_cfg.name}'
        if contrib_col in wide.columns:
            wide['score_composite'] += wide[contrib_col].fillna(0)
    
    # Compute percent contributions
    for metric_cfg in metrics_config:
        metric_name = metric_cfg.name
        contrib_col = f'contrib_{metric_name}'
        pct_contrib_col = f'pct_contrib_{metric_name}'
        
        if contrib_col in wide.columns:
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                wide[pct_contrib_col] = np.where(
                    wide['score_composite'] != 0,
                    wide[contrib_col] / wide['score_composite'] * 100,
                    0
                )
    
    logger.info(f"Built composite scores for {len(wide)} entity/week combinations")
    
    return wide


def apply_penalties_and_caps(
    df_wide: pd.DataFrame,
    df_long: pd.DataFrame,
    penalty_rules: list,
    caps: object,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Apply penalties and score caps.
    
    Returns DataFrame with penalty_total, penalty_breakdown, and score_final columns added.
    """
    # Evaluate penalties on long format
    penalty_total, penalty_breakdown = evaluate_penalties(df_long, penalty_rules)
    
    # Add penalties to long format
    df_long = df_long.copy()
    df_long['penalty_total'] = penalty_total
    df_long['penalty_breakdown'] = penalty_breakdown
    
    # Aggregate penalties to entity/week level
    penalty_agg = df_long.groupby(['entity_id', 'Week_Order']).agg({
        'penalty_total': 'sum',
        'penalty_breakdown': lambda x: json.dumps(
            [item for sublist in x.apply(json.loads) for item in sublist]
        )
    }).reset_index()
    
    # Merge with wide format
    df_wide = df_wide.merge(penalty_agg, on=['entity_id', 'Week_Order'], how='left')
    
    # Fill missing penalties with 0
    df_wide['penalty_total'] = df_wide['penalty_total'].fillna(0)
    df_wide['penalty_breakdown'] = df_wide['penalty_breakdown'].fillna('[]')
    
    # Compute final score with caps
    df_wide['score_final'] = df_wide['score_composite'] + df_wide['penalty_total']
    df_wide['score_final'] = df_wide['score_final'].clip(lower=caps.min_score, upper=caps.max_score)
    
    # Log penalty statistics
    penalty_count = (df_wide['penalty_total'] != 0).sum()
    if penalty_count > 0:
        logger.info(f"Applied {penalty_count} penalties")
        logger.info(f"  Total penalty range: [{df_wide['penalty_total'].min():.1f}, {df_wide['penalty_total'].max():.1f}]")
    
    return df_wide


def create_vendor_scoreboard(df_matrix: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Create vendor scoreboard by aggregating scores per vendor and week.
    
    Returns DataFrame with vendor_code, Week_Order, mean score_final, and rank.
    """
    # Group by vendor_code and Week_Order
    vendor_scores = df_matrix.groupby(['vendor_code', 'Week_Order']).agg({
        'score_final': 'mean',  # Use mean aggregation
        'score_composite': 'mean',
        'penalty_total': 'sum'
    }).reset_index()
    
    # Add rank within each week
    vendor_scores['rank'] = vendor_scores.groupby('Week_Order')['score_final'].rank(
        ascending=False, method='min'
    ).astype(int)
    
    logger.info(f"Created vendor scoreboard with {len(vendor_scores)} vendor/week combinations")
    
    return vendor_scores


def save_outputs(
    df_matrix: pd.DataFrame,
    df_scoreboard: pd.DataFrame,
    config: Dict,
    metadata: Dict,
    logger: logging.Logger
) -> None:
    """
    Save scoring_matrix and vendor_scoreboard to parquet (with CSV fallback).
    """
    output_dir = Path(config.get("paths", {}).get("scoring_output_dir", config["paths"]["processed_dir"]))
    ensure_directory(output_dir)
    
    # Save scoring_matrix
    matrix_parquet = output_dir / "scoring_matrix.parquet"
    matrix_csv = output_dir / "scoring_matrix.csv"
    
    try:
        df_matrix.to_parquet(matrix_parquet, index=False)
        logger.info(f"Saved scoring_matrix.parquet ({len(df_matrix)} rows)")
    except Exception as e:
        logger.error(f"Failed to write parquet: {e}")
        logger.info("Falling back to CSV")
        df_matrix.to_csv(matrix_csv, index=False)
        logger.info(f"Saved scoring_matrix.csv ({len(df_matrix)} rows)")
    
    # Save vendor_scoreboard
    scoreboard_parquet = output_dir / "vendor_scoreboard.parquet"
    scoreboard_csv = output_dir / "vendor_scoreboard.csv"
    
    try:
        df_scoreboard.to_parquet(scoreboard_parquet, index=False)
        logger.info(f"Saved vendor_scoreboard.parquet ({len(df_scoreboard)} rows)")
    except Exception as e:
        logger.error(f"Failed to write parquet: {e}")
        logger.info("Falling back to CSV")
        df_scoreboard.to_csv(scoreboard_csv, index=False)
        logger.info(f"Saved vendor_scoreboard.csv ({len(df_scoreboard)} rows)")
    
    # Save metadata
    metadata_path = output_dir / "scoring_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def run_phase2(config_path: str = "config.yaml") -> Dict[str, pd.DataFrame]:
    """
    Execute Phase 2 scoring pipeline.
    
    Returns:
        Dictionary with 'matrix' and 'scoreboard' DataFrames
    """
    # Load and validate config
    config = load_config(config_path)
    logger = setup_logging(config)
    
    logger.info(f"Starting Phase 2 scoring with config: {config_path}")
    
    # Load and validate scoring config
    try:
        scoring_config = load_and_validate_config(config)
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        raise
    
    # Load Phase 1 output
    df_input = load_phase1_output(config, logger)
    logger.info(f"Loaded {len(df_input)} rows from Phase 1")

    # Opportunistically enrich with derived metrics before further transforms
    try:
        derived_df = compute_derived_metrics(df_input)
        if not derived_df.empty:
            df_input = pd.concat([df_input, derived_df], ignore_index=True)
            logger.info("Added %d derived metric rows (e.g., ASP, CPPU, CP, CM %)", len(derived_df))
        else:
            logger.info("No derived metrics could be computed on this input (skipping)")
    except Exception as ex:
        logger.warning("Skipping derived metrics due to error: %s", ex)

    # Add grain columns for downstream rollups (non-breaking)
    try:
        df_input = add_grain_columns(df_input)
    except Exception as ex:
        logger.warning("Failed to add grain columns: %s", ex)
    
    # Prepare long format
    df_long = prepare_long_format(df_input, logger)
    validate_input_schema(df_long)
    logger.info(f"Prepared {len(df_long)} rows in long format")

    # Normalize week_label to YYYYWNN to avoid wrong label variants
    if 'week_label' in df_long.columns:
        try:
            df_long['week_label'] = normalize_week_label(df_long['week_label'])
        except Exception as ex:
            logger.warning("Failed to normalize week_label: %s", ex)

    # Map observed metric names to configured canonical names (aliases + heuristics/fuzzy)
    try:
        configured_metrics = [m.name for m in scoring_config.metrics]
        alias_json = Path(config.get("paths", {}).get("metric_aliases_json", Path("data") / "reference" / "metric_aliases.json"))
        # Resolve configured → available map; we get mapping available->configured
        available_cols = df_long["metric"].dropna().unique().tolist()
        resolved_map, resolution_warnings = resolve_configured_metrics(configured_metrics, available_cols, alias_json)
        for w in resolution_warnings:
            logger.warning(w)
        if resolved_map:
            before_counts = df_long["metric"].value_counts().to_dict()
            # Build inverse map observed->canonical for series mapping
            obs_to_canon = dict(resolved_map)
            df_long["metric"] = df_long["metric"].map(lambda x: obs_to_canon.get(x, x))
            after_counts = df_long["metric"].value_counts().to_dict()
            logger.info("Resolved %d metrics via aliases/fuzzy; alias file: %s", len(obs_to_canon), alias_json)
            logger.debug("Metric counts before mapping: %s", before_counts)
            logger.debug("Metric counts after mapping: %s", after_counts)
        else:
            logger.info("No metric alias/fuzzy mapping applied (no matches)")
    except Exception as ex:
        logger.warning("Metric alias mapping skipped due to error: %s", ex)

    # Rolling features and signal quality (optional, augments dataset; does not alter scoring pipeline)
    try:
        df_long = add_rolling_features(df_long, window=scoring_config.rolling_weeks)
        df_long = compute_signal_quality(df_long, lookback_weeks=scoring_config.rolling_weeks)
    except Exception as ex:
        logger.warning("Rolling features/signal quality not computed: %s", ex)
    
    # Forward-fill quality snapshots within each (entity, metric) so all weeks have values
    if set(["completeness", "recency_weeks", "trend_quality"]).issubset(df_long.columns):
        df_long[["completeness", "recency_weeks", "trend_quality"]] = (
            df_long.sort_values(["entity_id", "metric", "Week_Order"]) 
                   .groupby(["entity_id", "metric"])[["completeness", "recency_weeks", "trend_quality"]]
                   .ffill()
        )

    # Filter to configured metrics
    df_long = df_long[df_long['metric'].isin(configured_metrics)]
    logger.info(f"Filtered to {len(df_long)} rows for configured metrics: {configured_metrics}")

    # Dynamic weights based on signal quality
    base_weights = {m.name: float(m.weight) for m in scoring_config.metrics}
    try:
        df_weights = compute_dynamic_weights(
            df_quality=df_long,
            base_weights_by_metric=base_weights,
            lookback_weeks=scoring_config.rolling_weeks,
        )
        logger.info("Computed dynamic weights for %d rows", len(df_weights))
    except Exception as ex:
        logger.warning("Dynamic weights not computed (%s). Falling back to base weights only.", ex)
        df_weights = pd.DataFrame(columns=["entity_id", "metric", "Week_Order", "weight_effective"])  # empty

    # Process each metric through transform pipeline
    processed_metrics = []
    for metric_cfg in scoring_config.metrics:
        metric_df = df_long[df_long['metric'] == metric_cfg.name].copy()
        
        if metric_df.empty:
            logger.warning(f"No data found for metric: {metric_cfg.name}")
            continue
        
        processed = process_metric(
            metric_df,
            metric_cfg,
            scoring_config.rolling_weeks,
            logger
        )
        # Merge dynamic weights (fallback to base weight where missing)
        if not df_weights.empty:
            processed = processed.merge(
                df_weights[["entity_id", "metric", "Week_Order", "weight_effective", "weight_multiplier", "base_weight"]],
                on=["entity_id", "metric", "Week_Order"],
                how="left",
            )
        processed['weight_effective'] = processed['weight_effective'].fillna(float(metric_cfg.weight))
        processed['weight_multiplier'] = processed.get('weight_multiplier', pd.Series(index=processed.index, dtype=float)).fillna(1.0)
        processed['base_weight'] = processed.get('base_weight', pd.Series(index=processed.index, dtype=float)).fillna(float(metric_cfg.weight))
        processed['weight'] = processed['weight_effective']
        processed_metrics.append(processed)
    
    if not processed_metrics:
        logger.error("No metrics were processed successfully")
        return {'matrix': pd.DataFrame(), 'scoreboard': pd.DataFrame()}
    
    # Combine all processed metrics
    df_long_std = pd.concat(processed_metrics, ignore_index=True)
    logger.info(f"Processed {len(df_long_std)} metric values")
    
    # Compute coverage
    df_long_std = compute_coverage(df_long_std, scoring_config.rolling_weeks)
    
    # Build wide composite with contributions
    df_wide = build_wide_composite(
        df_long_std,
        scoring_config.metrics,
        scoring_config.weights_must_sum_to_1,
        logger
    )
    
    # Compute reliability
    reliability = compute_reliability(df_long_std, scoring_config.rolling_weeks)
    reliability_df = reliability.reset_index()
    reliability_df.columns = ['entity_id', 'Week_Order', 'reliability']
    
    df_wide = df_wide.merge(reliability_df, on=['entity_id', 'Week_Order'], how='left')
    df_wide['reliability'] = df_wide['reliability'].fillna(0)
    
    # Apply penalties and caps
    df_wide = apply_penalties_and_caps(
        df_wide,
        df_long_std,
        scoring_config.penalties,
        scoring_config.caps,
        logger
    )
    
    # Add score_final_conf (alias for reliability)
    df_wide['score_final_conf'] = df_wide['reliability']
    
    # Create vendor scoreboard
    df_scoreboard = create_vendor_scoreboard(df_wide, logger)
    
    # Prepare metadata
    config_str = json.dumps(config.get('scoring', {}), sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    metadata = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'config_hash': config_hash,
        'input_row_count': len(df_input),
        'output_row_count': len(df_wide),
        'metrics': configured_metrics,
        'rolling_weeks': scoring_config.rolling_weeks,
        'penalty_rules': len(scoring_config.penalties)
    }
    
    # Save outputs
    save_outputs(df_wide, df_scoreboard, config, metadata, logger)
    
    logger.info("Phase 2 scoring completed successfully")
    
    return {
        'matrix': df_wide,        # wide matrix with contrib/weights/composite
        'scoreboard': df_scoreboard,
        'long': df_long_std,      # long standardized with raw/std and Patch-3 fields
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="BrightStar Phase 2 Scoring - Robust metric scoring with penalties and explainability"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    return parser


def main() -> None:
    """CLI entry point for Phase 2 scoring."""
    parser = build_arg_parser()
    args = parser.parse_args()
    
    try:
        results = run_phase2(config_path=args.config)
        
        # Print summary
        print("\n=== Phase 2 Scoring Summary ===")
        print(f"Scoring matrix rows: {len(results['matrix'])}")
        print(f"Vendor scoreboard rows: {len(results['scoreboard'])}")
        
        if not results['scoreboard'].empty:
            print("\nTop 5 vendors (latest week):")
            latest_week = results['scoreboard']['Week_Order'].max()
            top_vendors = results['scoreboard'][
                results['scoreboard']['Week_Order'] == latest_week
            ].nsmallest(5, 'rank')[['vendor_code', 'score_final', 'rank']]
            print(top_vendors.to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
