"""Utility helpers for Phase 2 scoring computations."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .ingestion_utils import ensure_directory


LOGGER_NAME = "brightstar.phase2"


@dataclass
class MetricConfig:
    """Configuration describing how to score a single metric."""

    name: str
    weight: float
    direction: str

    def is_descending(self) -> bool:
        """Return ``True`` when lower values indicate better performance."""

        return self.direction.lower() in {"down", "desc", "lower", "decrease"}


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

    logger.debug("Phase 2 logging initialised at %s", log_path)
    return logger


def load_normalized_input(config: Dict, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Load the normalized dataset produced by Phase 1."""

    scoring_config = config.get("scoring", {})
    candidate_paths: List[Path] = []

    normalized_preference = scoring_config.get("normalized_input")
    if normalized_preference:
        candidate_paths.append(Path(normalized_preference))

    default_normalized = config["paths"].get("normalized_output")
    if default_normalized:
        candidate_paths.append(Path(default_normalized))

    # fall back to CSV representation when parquet is not available
    candidate_paths.append(Path(config["paths"]["processed_dir"]) / "normalized_phase1.csv")

    for path in candidate_paths:
        if path.exists():
            if path.suffix.lower() == ".parquet":
                try:
                    return pd.read_parquet(path)
                except Exception as exc:  # pragma: no cover - pandas engine coverage is environment dependent
                    if logger:
                        logger.warning("Unable to read parquet file %s (%s). Falling back to CSV if available.", path, exc)
                    continue
            if path.suffix.lower() in {".csv", ".txt"}:
                return pd.read_csv(path)

    raise FileNotFoundError(
        "Unable to locate normalized dataset. Ensure Phase 1 output exists before running Phase 2."
    )


def build_metric_map(config: Dict) -> Dict[str, MetricConfig]:
    """Construct the metric configuration mapping from settings."""

    scoring_config = config.get("scoring", {})
    metric_map: Dict[str, MetricConfig] = {}

    configured_metrics = scoring_config.get("metrics") or []
    for entry in configured_metrics:
        name = entry.get("name")
        if not name:
            continue
        weight = float(entry.get("weight", 0.0))
        direction = entry.get("direction", scoring_config.get("metric_directions", {}).get(name, "up"))
        metric_map[str(name)] = MetricConfig(name=str(name), weight=weight, direction=str(direction))

    if not metric_map:
        weights = scoring_config.get("metric_weights", {})
        directions = scoring_config.get("metric_directions", {})
        if not weights:
            raise ValueError("No metric weights configured for scoring.")

        for name, weight in weights.items():
            direction = directions.get(name, "up")
            metric_map[name] = MetricConfig(name=name, weight=float(weight), direction=str(direction))

    return metric_map


def aggregate_metrics(
    df: pd.DataFrame,
    entity_columns: Iterable[str],
    metric_column: str = "metric",
    value_column: str = "value",
    aggregation: str = "mean",
) -> pd.DataFrame:
    """Aggregate metric values per entity according to the configured function."""

    if df.empty:
        return pd.DataFrame(columns=[*entity_columns, metric_column, "Raw_Value"])

    group_cols = list(entity_columns) + [metric_column]
    grouped = df.groupby(group_cols)[value_column]

    if aggregation == "mean":
        values = grouped.mean()
    elif aggregation == "median":
        values = grouped.median()
    elif aggregation == "sum":
        values = grouped.sum()
    else:
        raise ValueError(f"Unsupported aggregation function: {aggregation}")

    aggregated = values.reset_index().rename(columns={value_column: "Raw_Value"})
    return aggregated


def ensure_metric_coverage(
    aggregated: pd.DataFrame,
    entity_columns: Iterable[str],
    metric_map: Dict[str, MetricConfig],
) -> pd.DataFrame:
    """Ensure every entity has an entry for each configured metric."""

    entity_cols = list(entity_columns)
    metrics = list(metric_map.keys())
    if aggregated.empty:
        columns = entity_cols + ["metric", "Raw_Value"]
        return pd.DataFrame(columns=columns)

    unique_entities = aggregated[entity_cols].drop_duplicates()
    entity_cartesian = unique_entities.assign(_key=1).merge(
        pd.DataFrame({"metric": metrics, "_key": 1}), on="_key"
    ).drop(columns="_key")

    merged = entity_cartesian.merge(aggregated, on=entity_cols + ["metric"], how="left")
    merged["Raw_Value"] = merged["Raw_Value"].astype(float)
    return merged


def compute_normalised_scores(
    aggregated: pd.DataFrame,
    metric_map: Dict[str, MetricConfig],
    fallback_normalised: float,
) -> pd.DataFrame:
    """Add normalised and weighted scores for each metric."""

    if aggregated.empty:
        aggregated = aggregated.copy()
        aggregated["Normalized_Value"] = pd.Series(dtype=float)
        aggregated["Weight"] = pd.Series(dtype=float)
        aggregated["Weighted_Score"] = pd.Series(dtype=float)
        return aggregated

    metric_stats = (
        aggregated.groupby("metric")["Raw_Value"].agg(["min", "max"]).rename(columns={"min": "_min", "max": "_max"})
    )

    def normalise(row: pd.Series) -> float:
        config = metric_map.get(row["metric"])
        if config is None:
            return fallback_normalised

        raw_value = row["Raw_Value"]
        if pd.isna(raw_value):
            base = fallback_normalised
        else:
            stats = metric_stats.loc[row["metric"]]
            min_val = stats["_min"]
            max_val = stats["_max"]
            if pd.isna(min_val) or pd.isna(max_val) or max_val == min_val:
                base = 0.5
            else:
                base = (raw_value - min_val) / (max_val - min_val)
                base = max(0.0, min(1.0, base))

        if config.is_descending():
            base = 1.0 - base

        return max(0.0, min(1.0, base))

    aggregated = aggregated.copy()
    aggregated["Normalized_Value"] = aggregated.apply(normalise, axis=1).round(4)
    aggregated["Weight"] = aggregated["metric"].map(lambda m: metric_map.get(m, MetricConfig(m, 0.0, "up")).weight)
    aggregated["Weighted_Score"] = (aggregated["Normalized_Value"] * aggregated["Weight"]).round(4)
    return aggregated


def compute_composite_scores(
    scored: pd.DataFrame,
    metric_map: Dict[str, MetricConfig],
    composite_scale: float,
    entity_columns: Iterable[str],
) -> pd.DataFrame:
    """Aggregate weighted scores into composite entity scores."""

    if scored.empty:
        return scored.assign(Composite_Score=pd.Series(dtype=float))

    total_weight = sum(config.weight for config in metric_map.values())
    if total_weight <= 0:
        raise ValueError("Metric weights must sum to a positive value.")

    entity_cols = list(entity_columns)

    composite = (
        scored.groupby(entity_cols)["Weighted_Score"].sum().reset_index().rename(columns={"Weighted_Score": "_weighted_sum"})
    )
    composite["Composite_Score"] = ((composite["_weighted_sum"] / total_weight) * composite_scale).round(3)
    scored = scored.merge(composite.drop(columns=["_weighted_sum"]), on=entity_cols, how="left")
    return scored


def assign_performance_category(scored: pd.DataFrame, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Assign qualitative categories based on composite score thresholds."""

    if scored.empty or "Composite_Score" not in scored.columns:
        scored["Performance_Category"] = pd.Series(dtype=str)
        return scored

    ordered_thresholds = sorted(((label, value) for label, value in thresholds.items()), key=lambda item: item[1], reverse=True)

    def label_score(score: float) -> str:
        for label, value in ordered_thresholds:
            if score >= value:
                return label.title()
        return ordered_thresholds[-1][0].title() if ordered_thresholds else "Unclassified"

    scored = scored.copy()
    scored["Performance_Category"] = scored["Composite_Score"].apply(label_score)
    return scored


def evaluate_improvement_flags(
    scored: pd.DataFrame,
    improvement_rules: Dict[str, Dict[str, float]],
) -> pd.DataFrame:
    """Apply rule-based improvement flags per metric."""

    if scored.empty:
        scored["Improvement_Flag"] = pd.Series(dtype=str)
        return scored

    default_rule = improvement_rules.get("default", {})

    def apply_rule(row: pd.Series) -> Optional[str]:
        rule = improvement_rules.get(row["metric"], default_rule)
        if not rule:
            return None

        normalized_value = row.get("Normalized_Value")
        raw_value = row.get("Raw_Value")

        if "normalized_lt" in rule and normalized_value < rule["normalized_lt"]:
            return rule.get("label", "Watch")
        if "normalized_gt" in rule and normalized_value > rule["normalized_gt"]:
            return rule.get("label", "Watch")
        if "raw_lt" in rule and raw_value < rule["raw_lt"]:
            return rule.get("label", "Watch")
        if "raw_gt" in rule and raw_value > rule["raw_gt"]:
            return rule.get("label", "Watch")
        return None

    scored = scored.copy()
    scored["Improvement_Flag"] = scored.apply(apply_rule, axis=1)
    return scored


def add_rankings(scored: pd.DataFrame, entity_columns: Iterable[str]) -> pd.DataFrame:
    """Calculate rankings per entity type, metric, and category."""

    if scored.empty:
        scored["Rank_Overall"] = pd.Series(dtype=float)
        scored["Rank_By_Metric"] = pd.Series(dtype=float)
        scored["Rank_By_Category"] = pd.Series(dtype=float)
        return scored

    scored = scored.copy()
    entity_cols = list(entity_columns)
    entity_type_col = entity_cols[0] if entity_cols else "entity_type"

    unique_overall = scored.drop_duplicates(subset=entity_cols + ["Composite_Score"])[entity_cols + ["Composite_Score"]]
    unique_overall["Rank_Overall"] = unique_overall.groupby(entity_type_col)["Composite_Score"].rank(
        ascending=False, method="dense"
    )
    scored = scored.merge(unique_overall[entity_cols + ["Rank_Overall"]], on=entity_cols, how="left")

    metric_unique = scored.drop_duplicates(subset=entity_cols + ["metric"])[entity_cols + ["metric", "Normalized_Value"]]
    metric_unique["Rank_By_Metric"] = metric_unique.groupby([entity_type_col, "metric"])["Normalized_Value"].rank(
        ascending=False, method="dense"
    )
    scored = scored.merge(
        metric_unique[entity_cols + ["metric", "Rank_By_Metric"]],
        on=entity_cols + ["metric"],
        how="left",
    )

    if "Performance_Category" in scored.columns:
        category_unique = scored.drop_duplicates(subset=entity_cols + ["Performance_Category"])[
            entity_cols + ["Performance_Category", "Composite_Score"]
        ]
        category_unique["Rank_By_Category"] = category_unique.groupby([entity_type_col, "Performance_Category"])[
            "Composite_Score"
        ].rank(ascending=False, method="dense")
        scored = scored.merge(
            category_unique[entity_cols + ["Performance_Category", "Rank_By_Category"]],
            on=entity_cols + ["Performance_Category"],
            how="left",
        )
    else:
        scored["Rank_By_Category"] = pd.NA

    improvement_unique = scored.drop_duplicates(subset=entity_cols + ["metric"]).copy()

    def improvement_score(row: pd.Series) -> Optional[float]:
        normalized_value = row.get("Normalized_Value")
        if pd.isna(normalized_value):
            return None
        return float(max(0.0, min(1.0, 1.0 - float(normalized_value))))

    improvement_unique["Improvement_Score"] = improvement_unique.apply(improvement_score, axis=1)
    improvement_unique["Improvement_Rank"] = improvement_unique.groupby([entity_type_col])["Improvement_Score"].rank(
        ascending=False,
        method="dense",
    )

    scored = scored.merge(
        improvement_unique[entity_cols + ["metric", "Improvement_Score", "Improvement_Rank"]],
        on=entity_cols + ["metric"],
        how="left",
    )

    return scored


def persist_outputs(
    scored: pd.DataFrame,
    config: Dict,
    entity_type: str,
    logger: logging.Logger,
) -> Tuple[Path, Optional[Path]]:
    """Write scorecard results for the specified entity type."""

    # Determine output paths robustly to support minimal configs in tests
    paths = config.get("paths", {})
    if entity_type == "vendor":
        csv_path = Path(paths.get("scorecard_vendor_csv", Path(paths.get("processed_dir", "data/processed")) / "scorecards" / "vendor_scorecard.csv"))
        parquet_path = Path(paths.get("scorecard_vendor_parquet", Path(paths.get("processed_dir", "data/processed")) / "scorecards" / "vendor_scorecard.parquet"))
    else:
        csv_path = Path(paths.get("scorecard_asin_csv", Path(paths.get("processed_dir", "data/processed")) / "scorecards" / "asin_scorecard.csv"))
        parquet_path = Path(paths.get("scorecard_asin_parquet", Path(paths.get("processed_dir", "data/processed")) / "scorecards" / "asin_scorecard.parquet"))

    ensure_directory(csv_path.parent)

    scored.to_csv(csv_path, index=False)
    try:
        if parquet_path:
            ensure_directory(parquet_path.parent)
            scored.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Unable to write parquet scorecard %s (%s).", parquet_path, exc)
        parquet_path = None

    return csv_path, parquet_path


def write_audit(audit: pd.DataFrame, config: Dict) -> Path:
    """Persist the consolidated score audit dataframe."""
    paths = config.get("paths", {})
    default_dir = Path(paths.get("processed_dir", "data/processed")) / "scorecards"
    audit_path = Path(paths.get("scorecard_audit_csv", default_dir / "score_audit.csv"))
    ensure_directory(audit_path.parent)
    audit.to_csv(audit_path, index=False)
    return audit_path


def write_metadata(metadata: Dict, config: Dict) -> Path:
    """Persist run metadata for auditability."""

    metadata_path = Path(config["paths"]["scorecard_metadata"])
    ensure_directory(metadata_path.parent)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return metadata_path


def summarize_entities(scored: pd.DataFrame, entity_column: str, top_n: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return top and bottom performers for console summaries."""

    if scored.empty:
        empty = pd.DataFrame(columns=[entity_column, "Composite_Score"])
        return empty, empty

    summary_cols = [entity_column, "Composite_Score", "Performance_Category"]
    sorted_df = (
        scored.drop_duplicates(subset=summary_cols)
        .loc[:, summary_cols]
        .sort_values("Composite_Score", ascending=False)
    )
    top = sorted_df.head(top_n)
    bottom = sorted_df.tail(top_n).iloc[::-1]
    return top, bottom


# ==========================================================
# Phase 5 – ASIN Growth Dashboard helpers
# ==========================================================

def _get_asin_growth_config(config: Dict) -> Dict:
    """Return asin_growth_model config with safe defaults."""
    model = dict(config.get("asin_growth_model", {}) or {})
    thresholds = model.get("thresholds", {}) or {}
    model.setdefault("thresholds", {
        "growth_high": float(thresholds.get("growth_high", 0.20)),
        "growth_low": float(thresholds.get("growth_low", -0.05)),
        "importance_high": float(thresholds.get("importance_high", 0.03)),
        "min_gms_l4w": float(thresholds.get("min_gms_l4w", 1000.0)),
        "min_units_l4w": float(thresholds.get("min_units_l4w", 10)),
    })

    avail = model.get("availability", {}) or {}
    model.setdefault("availability", {
        "weights": {
            "Fill_Rate_Sourceable_L4W": float(avail.get("weights", {}).get("Fill_Rate_Sourceable_L4W", 0.4)),
            "Vendor_Confirmation_Rate_Sourceable_L4W": float(avail.get("weights", {}).get("Vendor_Confirmation_Rate_Sourceable_L4W", 0.3)),
            "SoROOS_L4W": float(avail.get("weights", {}).get("SoROOS_L4W", -0.3)),
        },
        "availability_threshold": float(avail.get("availability_threshold", 0.0)),
    })

    pr = model.get("priority", {}) or {}
    model.setdefault("priority", {
        "growth_weight": float(pr.get("growth_weight", 0.4)),
        "importance_weight": float(pr.get("importance_weight", 0.3)),
        "crap_penalty": float(pr.get("crap_penalty", -0.3)),
        "oos_penalty": float(pr.get("oos_penalty", -0.2)),
        "forecast_gain_weight": float(pr.get("forecast_gain_weight", 0.2)),
        "forecast_risk_penalty": float(pr.get("forecast_risk_penalty", -0.2)),
        "horizon_preference": int(pr.get("horizon_preference", 4)),
    })
    return model


def build_asin_l4w_aggregates(normalized_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Compute last-4-weeks (L4W) and previous-4-weeks aggregates for key ASIN metrics.

    Expected input columns: vendor_code, asin, metric, value, week_start
    Uses canonical metric names already enforced in Phase 1.
    Returns a dataframe with one row per (vendor_code, asin) containing L4W features.
    """
    if normalized_df is None or normalized_df.empty:
        return pd.DataFrame()

    df = normalized_df.copy()
    # Guard required columns
    required = {"vendor_code", "asin", "metric", "value"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logging.getLogger(LOGGER_NAME).warning("ASIN L4W aggregates: missing columns %s", ", ".join(sorted(missing)))
        return pd.DataFrame()

    # Coerce dates
    if "week_start" not in df.columns:
        # Try fallbacks used elsewhere
        for cand in ("Week_Start", "week_label", "Week_Label"):
            if cand in df.columns:
                try:
                    df["week_start"] = pd.to_datetime(df[cand], errors="coerce")
                except Exception:
                    df["week_start"] = pd.NaT
                break
    df["week_start"] = pd.to_datetime(df.get("week_start"), errors="coerce")
    df = df.dropna(subset=["week_start", "vendor_code", "asin"]).copy()
    if df.empty:
        return pd.DataFrame()

    # Identify the 8 most recent distinct weeks
    weeks = sorted(df["week_start"].dropna().unique())
    if not weeks:
        return pd.DataFrame()
    last_weeks = weeks[-8:]
    l4w = set(last_weeks[-4:])
    prev_l4w = set(last_weeks[:max(0, len(last_weeks) - 4)]) - set(last_weeks[-4:])

    # Helper to aggregate by sum/mean per metric set
    def _agg_metric(metric_name: str, how: str = "sum", window: str = "L4W") -> pd.DataFrame:
        mask = df["metric"].astype(str) == metric_name
        sub = df.loc[mask].copy()
        if sub.empty:
            return pd.DataFrame(columns=["vendor_code", "asin", f"{metric_name}_{window}"])
        if window == "L4W":
            sub = sub[sub["week_start"].isin(l4w)]
        elif window == "prev_L4W":
            sub = sub[sub["week_start"].isin(prev_l4w)]
        if sub.empty:
            return pd.DataFrame(columns=["vendor_code", "asin", f"{metric_name}_{window}"])
        grouped = sub.groupby(["vendor_code", "asin"], as_index=False)["value"]
        if how == "mean":
            out = grouped.mean()
        elif how == "last":
            # Latest value in the window
            sub = sub.sort_values(["vendor_code", "asin", "week_start"])  # ascending
            out = sub.groupby(["vendor_code", "asin"], as_index=False).last()[["vendor_code", "asin", "value"]]
        else:
            out = grouped.sum()
        out.rename(columns={"value": f"{metric_name}_{window}"}, inplace=True)
        return out

    # Compute metrics
    parts: List[pd.DataFrame] = []
    # Sales & units
    parts.append(_agg_metric("Product_GMS", how="sum", window="L4W"))
    parts.append(_agg_metric("Product_GMS", how="sum", window="prev_L4W"))
    parts.append(_agg_metric("Net_Ordered_Units", how="sum", window="L4W"))
    parts.append(_agg_metric("Net_Ordered_Units", how="sum", window="prev_L4W"))
    parts.append(_agg_metric("Deal_GMS", how="sum", window="L4W"))
    # Traffic & conversion
    parts.append(_agg_metric("GV", how="sum", window="L4W"))
    parts.append(_agg_metric("GV", how="sum", window="prev_L4W"))
    # Economics
    parts.append(_agg_metric("PPM", how="mean", window="L4W"))
    parts.append(_agg_metric("Net_PPM", how="mean", window="L4W"))
    parts.append(_agg_metric("CPPU", how="mean", window="L4W"))
    parts.append(_agg_metric("CP", how="sum", window="L4W"))
    parts.append(_agg_metric("CM", how="mean", window="L4W"))
    parts.append(_agg_metric("CCOGS_As_A_Percent_Of_Revenue", how="mean", window="L4W"))
    parts.append(_agg_metric("ASP", how="mean", window="L4W"))
    parts.append(_agg_metric("Customer_Returns", how="sum", window="L4W"))
    # Availability (latest within L4W; else mean)
    parts.append(_agg_metric("On_Hand_Inventory_Units_Sellable", how="last", window="L4W"))
    parts.append(_agg_metric("Total_Inventory_Units", how="sum", window="L4W"))
    parts.append(_agg_metric("Fill_Rate_Sourceable", how="mean", window="L4W"))
    parts.append(_agg_metric("Vendor_Confirmation_Rate_Sourceable", how="mean", window="L4W"))
    parts.append(_agg_metric("SoROOS", how="mean", window="L4W"))

    # Merge all parts
    if not parts:
        return pd.DataFrame()
    base = parts[0]
    for p in parts[1:]:
        base = base.merge(p, on=["vendor_code", "asin"], how="outer")

    base.fillna(0, inplace=True)

    # Share of vendor GMS (L4W)
    vendor_gms = base.groupby("vendor_code", as_index=False)["Product_GMS_L4W"].sum().rename(
        columns={"Product_GMS_L4W": "_vendor_gms_l4w"}
    )
    base = base.merge(vendor_gms, on="vendor_code", how="left")
    base["Share_of_vendor_GMS_L4W"] = base.apply(
        lambda r: float(r["Product_GMS_L4W"]) / float(r["_vendor_gms_l4w"]) if float(r.get("_vendor_gms_l4w", 0)) else 0.0,
        axis=1,
    )
    base.drop(columns=["_vendor_gms_l4w"], inplace=True)
    return base


def add_asin_growth_segment(asin_l4w_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Assign 5+1 growth segments based on growth vs importance and low-data rules."""
    if asin_l4w_df is None or asin_l4w_df.empty:
        return pd.DataFrame()
    model = _get_asin_growth_config(config)
    t = model["thresholds"]
    df = asin_l4w_df.copy()
    # Growth and importance
    def _safe_div(num, den):
        try:
            return float(num) / float(den) if float(den) else None
        except Exception:
            return None

    df["Growth_Rate"] = df.apply(lambda r: _safe_div(r.get("Product_GMS_L4W", 0.0) - r.get("Product_GMS_prev_L4W", 0.0), r.get("Product_GMS_prev_L4W", 0.0)), axis=1)
    df["Growth_Rate"] = df["Growth_Rate"].fillna(0.0)
    df["Importance"] = df.get("Share_of_vendor_GMS_L4W", 0.0)

    def _segment(row) -> str:
        # Low data first
        if float(row.get("Product_GMS_L4W", 0.0)) < float(t["min_gms_l4w"]) or float(row.get("Net_Ordered_Units_L4W", 0.0)) < float(t["min_units_l4w"]):
            return "Low_Data"
        gr = float(row.get("Growth_Rate", 0.0))
        imp = float(row.get("Importance", 0.0))
        if gr >= t["growth_high"] and imp >= t["importance_high"]:
            return "Breakout"
        if gr >= t["growth_high"] and imp < t["importance_high"]:
            return "Emerging"
        if gr <= t["growth_low"] and imp >= t["importance_high"]:
            return "Squeezed"
        if gr >= t["growth_low"] and imp >= t["importance_high"]:
            return "Core"
        return "Sunset"

    df["Asin_Growth_Segment"] = df.apply(_segment, axis=1)
    return df


def build_asin_dashboard_df(
    asin_l4w_df: pd.DataFrame,
    forecasts_df: pd.DataFrame | None,
    health_df: pd.DataFrame | None,
    config: Dict,
) -> pd.DataFrame:
    """Create the ASIN dashboard dataframe with derived metrics and optional forecast fields.

    The output schema is stable and nullable when data is missing.
    """
    if asin_l4w_df is None or asin_l4w_df.empty:
        # Provide empty schema
        cols = [
            "vendor_code", "asin", "Title", "Category",
            "Asin_Growth_Segment", "GMS_Delta_Pct", "Share_of_vendor_GMS_L4W",
            "Product_GMS_L4W", "Net_Ordered_Units_L4W", "Units_Delta_Pct", "Promo_Mix",
            "GV_L4W", "GV_Delta_Pct", "Conversion_Rate_L4W", "Conversion_Delta",
            "ASP_L4W", "Gross_Margin", "Net_Margin", "CPPU_L4W", "CP_L4W", "CM_L4W", "CCOGS_Pct",
            "On_Hand_Inventory_Units_Sellable_L4W", "Total_Inventory_Units_L4W",
            "Fill_Rate_Sourceable_L4W", "Vendor_Confirmation_Rate_Sourceable_L4W", "SoROOS_L4W",
            "Availability_Score", "OOS_Risk_Flag", "Return_Rate",
            "Forecast_Potential_Gain", "Forecast_Risk_Flag", "Priority_Score", "Key_Action",
        ]
        return pd.DataFrame(columns=cols)

    model = _get_asin_growth_config(config)
    avail = model["availability"]
    pr = model["priority"]

    df = asin_l4w_df.copy()
    # Sales trends
    def _pct_delta(new, old):
        try:
            return (float(new) - float(old)) / float(old) if float(old) else None
        except Exception:
            return None

    df["GMS_Delta_Pct"] = df.apply(lambda r: _pct_delta(r.get("Product_GMS_L4W", 0), r.get("Product_GMS_prev_L4W", 0)), axis=1).fillna(0.0)
    df["Units_Delta_Pct"] = df.apply(lambda r: _pct_delta(r.get("Net_Ordered_Units_L4W", 0), r.get("Net_Ordered_Units_prev_L4W", 0)), axis=1).fillna(0.0)
    df["Promo_Mix"] = df.apply(lambda r: (float(r.get("Deal_GMS_L4W", 0.0)) / float(r.get("Product_GMS_L4W", 0.0)) if float(r.get("Product_GMS_L4W", 0.0)) else 0.0), axis=1)

    # Traffic & conversion
    df["GV_Delta_Pct"] = df.apply(lambda r: _pct_delta(r.get("GV_L4W", 0), r.get("GV_prev_L4W", 0)), axis=1).fillna(0.0)
    df["Conversion_Rate_L4W"] = df.apply(lambda r: (float(r.get("Net_Ordered_Units_L4W", 0)) / float(r.get("GV_L4W", 0)) if float(r.get("GV_L4W", 0)) else 0.0), axis=1)
    df["Conversion_Rate_prev_L4W"] = df.apply(lambda r: (float(r.get("Net_Ordered_Units_prev_L4W", 0)) / float(r.get("GV_prev_L4W", 0)) if float(r.get("GV_prev_L4W", 0)) else 0.0), axis=1)
    df["Conversion_Delta"] = df["Conversion_Rate_L4W"] - df["Conversion_Rate_prev_L4W"]

    # Economics
    df.rename(columns={
        "PPM_L4W": "Gross_Margin",
        "Net_PPM_L4W": "Net_Margin",
        "CCOGS_As_A_Percent_Of_Revenue_L4W": "CCOGS_Pct",
        "ASP_L4W": "ASP_L4W",
    }, inplace=True)
    # Ensure columns exist
    for col in ["Gross_Margin", "Net_Margin", "CPPU_L4W", "CP_L4W", "CM_L4W", "CCOGS_Pct", "ASP_L4W"]:
        if col not in df.columns:
            df[col] = 0.0
    df["CRaP_Flag"] = (df["Net_Margin"] < 0) | (df["CPPU_L4W"] < 0)

    # Availability composite
    w = avail["weights"]
    # Ensure existence
    for c in ("Fill_Rate_Sourceable_L4W", "Vendor_Confirmation_Rate_Sourceable_L4W", "SoROOS_L4W"):
        if c not in df.columns:
            df[c] = 0.0
    df["Availability_Score"] = (
        df["Fill_Rate_Sourceable_L4W"] * w.get("Fill_Rate_Sourceable_L4W", 0.4)
        + df["Vendor_Confirmation_Rate_Sourceable_L4W"] * w.get("Vendor_Confirmation_Rate_Sourceable_L4W", 0.3)
        + df["SoROOS_L4W"] * w.get("SoROOS_L4W", -0.3)
    )
    df["OOS_Risk_Flag"] = df["Availability_Score"] < float(avail.get("availability_threshold", 0.0))

    # Quality
    if "Customer_Returns_L4W" not in df.columns:
        df["Customer_Returns_L4W"] = 0.0
    df["Return_Rate"] = df.apply(lambda r: (float(r.get("Customer_Returns_L4W", 0)) / float(r.get("Product_GMS_L4W", 0)) if float(r.get("Product_GMS_L4W", 0)) else 0.0), axis=1)

    # Forecast opportunity join (optional)
    fgain = None
    if forecasts_df is not None and isinstance(forecasts_df, pd.DataFrame) and not forecasts_df.empty:
        f = forecasts_df.copy()
        # Standardize columns
        if "potential_gain" not in f.columns and "Forecast_Potential_Gain" in f.columns:
            f.rename(columns={"Forecast_Potential_Gain": "potential_gain"}, inplace=True)
        if "forecast_horizon_weeks" not in f.columns and "Forecast_Horizon_Weeks" in f.columns:
            f.rename(columns={"Forecast_Horizon_Weeks": "forecast_horizon_weeks"}, inplace=True)
        if "Forecast_Risk_Flag" not in f.columns:
            f["Forecast_Risk_Flag"] = pd.NA
        # Keep ASIN level only
        if "entity_type" in f.columns:
            f = f[f["entity_type"].astype(str).str.lower() == "asin"]
        # Prefer configured horizon
        pref_h = int(pr.get("horizon_preference", 4))
        f = f.sort_values(["vendor_code", "asin", "forecast_horizon_weeks", "potential_gain"], ascending=[True, True, True, False])
        # Pick preferred horizon if available, else highest gain
        def _pick(group: pd.DataFrame) -> pd.Series:
            g_pref = group[group["forecast_horizon_weeks"] == pref_h]
            row = g_pref.head(1)
            if row.empty:
                row = group.sort_values("potential_gain", ascending=False).head(1)
            return row.iloc[0]
        fgain = f.groupby(["vendor_code", "asin"], as_index=False).apply(_pick).reset_index(drop=True)
        fgain = fgain[["vendor_code", "asin", "potential_gain", "Forecast_Risk_Flag"]]

    # Evaluation health (optional) — could override Forecast_Risk_Flag via health
    if health_df is not None and isinstance(health_df, pd.DataFrame) and not health_df.empty and fgain is not None:
        # health_df grain: metric/entity_type/horizon/method. Here we may only carry a coarse risk; leaving as-is for now.
        pass  # Keep Phase 4 risk flag for simplicity; health could be surfaced separately.

    if fgain is not None and not fgain.empty:
        df = df.merge(fgain, on=["vendor_code", "asin"], how="left")
        df.rename(columns={"potential_gain": "Forecast_Potential_Gain"}, inplace=True)
    else:
        df["Forecast_Potential_Gain"] = 0.0
        df["Forecast_Risk_Flag"] = pd.NA

    # Priority score (linear composite with safeguards)
    # Normalize forecast gain per-vendor by 90th percentile to keep scale comparable
    def _norm_gain(group: pd.Series) -> pd.Series:
        s = group.fillna(0.0).astype(float)
        if s.empty:
            return s
        q = float(s.quantile(0.9)) if s.quantile(0.9) not in (None, 0) else None
        if not q:
            return s * 0.0
        return s / q
    df["_norm_gain"] = df.groupby("vendor_code")["Forecast_Potential_Gain"].transform(_norm_gain)

    # Booleans to 0/1
    crap = df["CRaP_Flag"].astype(bool).astype(int)
    oos = df["OOS_Risk_Flag"].astype(bool).astype(int)
    risk = df.get("Forecast_Risk_Flag")
    try:
        risk_i = risk.fillna(0).astype(float)
        # Convert to 0/1 riskiness: anything > 2 considered risk (steady/growing/high are >2 per Topic 4 mapping)
        risk_bin = (risk_i <= 2).astype(int)
    except Exception:
        risk_bin = 0

    df["Priority_Score"] = (
        pr.get("growth_weight", 0.4) * df.get("Growth_Rate", 0.0)
        + pr.get("importance_weight", 0.3) * df.get("Importance", 0.0)
        + pr.get("forecast_gain_weight", 0.2) * df.get("_norm_gain", 0.0)
        + pr.get("crap_penalty", -0.3) * crap
        + pr.get("oos_penalty", -0.2) * oos
        + pr.get("forecast_risk_penalty", -0.2) * risk_bin
    )
    df.drop(columns=["_norm_gain"], inplace=True)

    # Identity/metadata placeholders
    for col in ("Title", "Category", "Key_Action"):
        if col not in df.columns:
            df[col] = pd.NA

    # Stable column order per specification layers
    ordered_cols = [
        # Layer 1 – Status & Segment
        "vendor_code", "asin", "Title", "Category", "Asin_Growth_Segment", "GMS_Delta_Pct", "Share_of_vendor_GMS_L4W",
        # Layer 2 – Sales Trends & Volume
        "Product_GMS_L4W", "Net_Ordered_Units_L4W", "Units_Delta_Pct", "Promo_Mix",
        # Layer 3 – Traffic & Conversion
        "GV_L4W", "GV_Delta_Pct", "Conversion_Rate_L4W", "Conversion_Delta",
        # Layer 4 – Economics & Availability
        "ASP_L4W", "Gross_Margin", "Net_Margin", "CPPU_L4W", "CP_L4W", "CM_L4W", "CCOGS_Pct",
        "On_Hand_Inventory_Units_Sellable_L4W", "Total_Inventory_Units_L4W", "Fill_Rate_Sourceable_L4W",
        "Vendor_Confirmation_Rate_Sourceable_L4W", "SoROOS_L4W", "Availability_Score", "OOS_Risk_Flag",
        # Layer 5 – Opportunities & Actions
        "Forecast_Potential_Gain", "Forecast_Risk_Flag", "Priority_Score", "Key_Action",
    ]

    # Ensure all ordered columns exist
    for c in ordered_cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df[ordered_cols].copy()


# ==========================================================
# Phase 5 – Scoring Dashboards (Vendor & ASIN current-period)
# ==========================================================

def _latest_week(normalized_df: pd.DataFrame) -> Optional[pd.Timestamp]:
    if normalized_df is None or normalized_df.empty:
        return None
    if "week_start" not in normalized_df.columns:
        return None
    try:
        return pd.to_datetime(normalized_df["week_start"], errors="coerce").max()
    except Exception:
        return None


def _agg_latest_vendor(normalized_df: pd.DataFrame, week: Optional[pd.Timestamp]) -> pd.DataFrame:
    if normalized_df is None or normalized_df.empty:
        return pd.DataFrame(columns=[
            "vendor_code", "Product_GMS", "Net_PPM", "SoROOS", "Fill_Rate_Sourceable",
            "Vendor_Confirmation_Rate_Sourceable", "On_Hand_Inventory_Units_Sellable", "Total_Inventory_Units", "GV"
        ])
    df = normalized_df.copy()
    if week is not None and "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
        df = df[df["week_start"] == week]
    # helpers
    def _metric(m):
        sub = df[df["metric"].astype(str) == m][["vendor_code", "value"]]
        return sub
    # sums
    sums = [
        ("Product_GMS", _metric("Product_GMS").groupby("vendor_code", as_index=False).sum()),
        ("Total_Inventory_Units", _metric("Total_Inventory_Units").groupby("vendor_code", as_index=False).sum()),
        ("On_Hand_Inventory_Units_Sellable", _metric("On_Hand_Inventory_Units_Sellable").groupby("vendor_code", as_index=False).sum()),
        ("GV", _metric("GV").groupby("vendor_code", as_index=False).sum()),
    ]
    means = [
        ("Net_PPM", _metric("Net_PPM").groupby("vendor_code", as_index=False).mean()),
        ("SoROOS", _metric("SoROOS").groupby("vendor_code", as_index=False).mean()),
        ("Fill_Rate_Sourceable", _metric("Fill_Rate_Sourceable").groupby("vendor_code", as_index=False).mean()),
        ("Vendor_Confirmation_Rate_Sourceable", _metric("Vendor_Confirmation_Rate_Sourceable").groupby("vendor_code", as_index=False).mean()),
    ]
    out = None
    for name, g in sums + means:
        g = g.rename(columns={"value": name})
        out = g if out is None else out.merge(g, on="vendor_code", how="outer")
    return out if out is not None else pd.DataFrame(columns=["vendor_code"]) 


def _agg_latest_asin(normalized_df: pd.DataFrame, week: Optional[pd.Timestamp]) -> pd.DataFrame:
    if normalized_df is None or normalized_df.empty:
        return pd.DataFrame(columns=[
            "vendor_code", "asin", "Product_GMS", "Net_PPM", "SoROOS", "On_Hand_Inventory_Units_Sellable",
            "Total_Inventory_Units", "GV", "item_name_clean", "brand", "category", "sub_category"
        ])
    df = normalized_df.copy()
    if "asin" not in df.columns:
        df["asin"] = pd.NA
    if week is not None and "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
        df = df[df["week_start"] == week]
    # helpers
    def _metric(m):
        sub = df[df["metric"].astype(str) == m][["vendor_code", "asin", "value"]]
        return sub
    sums = [
        ("Product_GMS", _metric("Product_GMS").groupby(["vendor_code", "asin"], as_index=False).sum()),
        ("Total_Inventory_Units", _metric("Total_Inventory_Units").groupby(["vendor_code", "asin"], as_index=False).sum()),
        ("On_Hand_Inventory_Units_Sellable", _metric("On_Hand_Inventory_Units_Sellable").groupby(["vendor_code", "asin"], as_index=False).sum()),
        ("GV", _metric("GV").groupby(["vendor_code", "asin"], as_index=False).sum()),
    ]
    means = [
        ("Net_PPM", _metric("Net_PPM").groupby(["vendor_code", "asin"], as_index=False).mean()),
        ("SoROOS", _metric("SoROOS").groupby(["vendor_code", "asin"], as_index=False).mean()),
    ]
    out = None
    for name, g in sums + means:
        g = g.rename(columns={"value": name})
        out = g if out is None else out.merge(g, on=["vendor_code", "asin"], how="outer")
    # attach product master fields if present in normalized
    pm_cols = [c for c in ["item_name_clean", "brand", "category", "sub_category"] if c in df.columns]
    if pm_cols:
        meta = df.dropna(subset=["asin"]).drop_duplicates(subset=["vendor_code", "asin"], keep="last")[["vendor_code", "asin", *pm_cols]]
        out = out.merge(meta, on=["vendor_code", "asin"], how="left") if out is not None else meta
    return out if out is not None else pd.DataFrame(columns=["vendor_code", "asin"]) 


def prepare_vendor_scoring_dashboard_df(
    config: Dict,
    vendor_scorecards_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare current-period vendor scoring dashboard dataframe.

    Columns include identity, composite and dimension scores, and key KPIs.
    Empty-safe; returns stable but possibly empty schema.
    """
    cols = [
        "vendor_code", "vendor_name", "Composite_Score",
        "Sales_Score", "Margin_Score", "Availability_Score", "Traffic_Score",
        "Product_GMS", "Net_PPM", "SoROOS", "Fill_Rate_Sourceable",
        "Vendor_Confirmation_Rate_Sourceable", "On_Hand_Inventory_Units_Sellable", "Total_Inventory_Units", "GV",
    ]
    if vendor_scorecards_df is None or vendor_scorecards_df.empty:
        return pd.DataFrame(columns=cols)

    latest = _latest_week(normalized_df)
    kpis = _agg_latest_vendor(normalized_df, latest)

    score = vendor_scorecards_df.copy()
    # Prefer a single current-period row per vendor; keep last occurrence
    keep_cols = [c for c in ["vendor_code", "vendor_name", "Composite_Score", "Sales_Score", "Margin_Score", "Availability_Score", "Traffic_Score"] if c in score.columns]
    score = score[keep_cols].drop_duplicates(subset=["vendor_code"], keep="last") if keep_cols else pd.DataFrame(columns=["vendor_code"]) 

    out = score.merge(kpis, on="vendor_code", how="left") if not score.empty else kpis
    # Ensure all expected columns exist
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols]


def prepare_asin_scoring_dashboard_df(
    config: Dict,
    asin_scorecards_df: pd.DataFrame,
    normalized_df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare current-period ASIN scoring dashboard dataframe.

    Includes Product Master attributes and key KPIs; default period is latest week.
    """
    cols = [
        "vendor_code", "vendor_name", "asin", "Item_Name", "Brand", "Category", "Sub_Category",
        "Composite_Score", "Sales_Score", "Margin_Score", "Availability_Score", "Traffic_Score",
        "Product_GMS", "Net_PPM", "GV", "SoROOS",
        "On_Hand_Inventory_Units_Sellable", "Total_Inventory_Units",
    ]
    if asin_scorecards_df is None or asin_scorecards_df.empty:
        return pd.DataFrame(columns=cols)

    latest = _latest_week(normalized_df)
    kpis = _agg_latest_asin(normalized_df, latest)

    score = asin_scorecards_df.copy()
    keep_cols = [c for c in ["vendor_code", "vendor_name", "asin", "Composite_Score", "Sales_Score", "Margin_Score", "Availability_Score", "Traffic_Score"] if c in score.columns]
    score = score[keep_cols].drop_duplicates(subset=["vendor_code", "asin"], keep="last") if keep_cols else pd.DataFrame(columns=["vendor_code", "asin"]) 

    out = score.merge(kpis, on=["vendor_code", "asin"], how="left") if not score.empty else kpis
    # Map item_name_clean and attributes to dashboard-friendly names
    if "item_name_clean" in out.columns:
        out["Item_Name"] = out["item_name_clean"]
    else:
        out["Item_Name"] = pd.NA
    out["Brand"] = out.get("brand", pd.Series(pd.NA, index=out.index))
    out["Category"] = out.get("category", pd.Series(pd.NA, index=out.index))
    out["Sub_Category"] = out.get("sub_category", pd.Series(pd.NA, index=out.index))

    # Ensure all expected columns exist
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    return out[cols]


# ============================================================
# Phase 2 — New scoring helpers (aggregation, normalisation, enrichment)
# ============================================================

def aggregate_metrics_for_scoring(norm_df: pd.DataFrame, entity_level: str, config: Dict) -> pd.DataFrame:
    """Aggregate normalized long table to the desired grain and window.

    entity_level: "vendor" or "asin".
    Uses config["scoring"]["aggregation_window"]. Returns one row per entity/week
    in wide format with canonical metric columns present when available.
    """
    if norm_df is None or norm_df.empty:
        return pd.DataFrame()

    df = norm_df.copy()
    # Ensure minimal columns
    for c in ("vendor_code", "metric", "value"):
        if c not in df.columns:
            return pd.DataFrame()

    # Coerce week_start as the primary temporal key
    if "week_start" not in df.columns:
        df["week_start"] = pd.to_datetime(df.get("week_start_date"), errors="coerce")
    else:
        df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")

    # Filter entity level
    if entity_level == "vendor":
        id_cols = ["vendor_code", "week_start"]
        df = df.copy()
    elif entity_level == "asin":
        if "asin" not in df.columns:
            df["asin"] = pd.NA
        id_cols = ["vendor_code", "asin", "week_start"]
    else:
        raise ValueError("entity_level must be 'vendor' or 'asin'")

    # Window selection
    window = str(config.get("scoring", {}).get("aggregation_window", "L4W")).upper()
    # Determine global last N weeks based on data; simple and robust
    weeks_sorted = sorted([w for w in df["week_start"].dropna().unique()])
    if weeks_sorted:
        if window == "L4W":
            keep_weeks = set(weeks_sorted[-4:])
        else:  # ALL
            keep_weeks = set(weeks_sorted)
        df = df[df["week_start"].isin(keep_weeks)]

    # Pivot metrics → columns; aggregate by mean for rates/levels and sum for flows
    # We keep a simple sum for all then allow downstream normalisation; metrics absent will be NaN.
    grp = df.groupby(id_cols + ["metric"], as_index=False)["value"].mean()
    wide = grp.pivot_table(index=id_cols, columns="metric", values="value", aggfunc="first").reset_index()
    # Flatten column index and ensure simple strings
    wide.columns = [str(c) for c in wide.columns]
    return wide


def compute_metric_normalized_scores(df: pd.DataFrame, config: Dict, metric_registry: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Compute 0–1 normalized scores for each metric used in dimensions.

    - Reads higher/lower-is-better from metric_registry (column 'direction').
    - Winsorizes by 2nd/98th percentiles per metric to reduce outlier influence.
    - Min–max scales to [0,1]; flips when lower-is-better.
    Adds columns: <metric>_norm
    """
    if df is None or df.empty:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    out = df.copy()
    dims = config.get("scoring", {}).get("dimensions", {}) or {}
    metrics_needed = sorted({m for arr in dims.values() for m in arr})

    # Build direction map from registry (fallback: assume up)
    dir_map: Dict[str, str] = {}
    if isinstance(metric_registry, pd.DataFrame) and not metric_registry.empty:
        if "metric_name" in metric_registry.columns and "direction" in metric_registry.columns:
            for _, row in metric_registry.dropna(subset=["metric_name"]).iterrows():
                dir_map[str(row["metric_name"])]= str(row.get("direction") or "up")

    # Helper: per-metric zero-semantics rules
    scoring_section = config.get("scoring", {}) or {}
    metrics_cfg = scoring_section.get("metrics", {}) or {}
    if not isinstance(metrics_cfg, dict):
        # Support alternative key to avoid conflict with list-based enhanced metrics config
        metrics_cfg = scoring_section.get("metric_zero_semantics", {}) or scoring_section.get("metrics_zero", {}) or {}

    def _metric_rules(name: str) -> Dict:
        rules = metrics_cfg.get(name) if isinstance(metrics_cfg, dict) else {}
        # Backward compatible fallbacks
        return {
            "zero_semantics": (rules or {}).get("zero_semantics", "unknown_zero"),
            "treat_zero_as_missing": bool((rules or {}).get("treat_zero_as_missing", False)),
            "min_signal_threshold": float((rules or {}).get("min_signal_threshold", 0.0)) if (rules or {}).get("min_signal_threshold") is not None else 0.0,
            "allow_all_zero_series": bool((rules or {}).get("allow_all_zero_series", True)),
        }

    for m in metrics_needed:
        if m not in out.columns:
            continue
        s = pd.to_numeric(out[m], errors="coerce")

        rules = _metric_rules(m)
        # Apply zero-semantics at normalization stage using a working copy
        s_work = s.copy()
        if rules.get("treat_zero_as_missing", False):
            s_work = s_work.replace(0, pd.NA)
        # winsorize
        lo, hi = s_work.quantile(0.02), s_work.quantile(0.98)
        s_clipped = s_work.clip(lower=lo, upper=hi)
        min_v, max_v = float(pd.to_numeric(s_clipped, errors="coerce").min()), float(pd.to_numeric(s_clipped, errors="coerce").max())
        denom = (max_v - min_v) if (max_v - min_v) != 0 else 1.0
        # Global all-zero handling: if min==max==0 after winsorization
        if pd.isna(min_v) and pd.isna(max_v):
            # All data missing → produce NA, will be masked later
            norm = pd.Series([pd.NA] * len(out))
        elif min_v == 0.0 and max_v == 0.0:
            direction = dir_map.get(m, "up").lower()
            if direction in {"down", "lower", "desc", "decrease"}:
                norm = pd.Series([1.0] * len(out))
            else:
                norm = pd.Series([0.0] * len(out))
        else:
            norm = (s_clipped - min_v) / denom
        # flip if lower-is-better
        direction = dir_map.get(m, "up").lower()
        if direction in {"down", "lower", "desc", "decrease"}:
            norm = 1.0 - norm
        out[f"{m}_norm"] = norm
    return out


def compute_dimension_scores(
    df: pd.DataFrame,
    config: Dict,
    *,
    entity_level: Optional[str] = None,
    invalid_metric_map: Optional[Dict] = None,
) -> pd.DataFrame:
    """Compute dimension scores with NA-safe masked multiplication.

    Rules implemented:
    - Never crash on NA.
    - Use equal weights within each dimension.
    - Metrics with missing normalized values contribute 0 (no weight re-normalization).
    - If all metrics in a dimension are missing, the dimension score is 0.0.
    """
    if df is None or df.empty:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    out = df.copy()
    dims: Dict[str, List[str]] = config.get("scoring", {}).get("dimensions", {}) or {}

    for dim_name, metrics in dims.items():
        # Build the list of expected normalized columns, creating missing ones as NA
        norm_cols = [f"{m}_norm" for m in metrics]
        for c in norm_cols:
            if c not in out.columns:
                out[c] = pd.NA

        if not norm_cols:
            # No metrics configured for this dimension; default to 0
            out[f"{dim_name.capitalize()}_Score"] = 0.0
            continue

        # Apply zero-semantics invalidation masks: set excluded metrics to NaN for affected entities
        if invalid_metric_map and entity_level:
            if entity_level == "vendor":
                key_cols = [col for col in ["vendor_code"] if col in out.columns]
                if key_cols:
                    # Build a Series of keys to look up
                    keys = out[key_cols[0]].astype(str).fillna("")
                    invalid_by_key: Dict[str, set] = invalid_metric_map.get("vendor", {})
                    if isinstance(invalid_by_key, dict) and not out.empty:
                        # For each metric, null out where entity key is in invalid set
                        for m in metrics:
                            inv_set = invalid_by_key.get(m)
                            if isinstance(inv_set, set):
                                mask = keys.isin(inv_set)
                                if f"{m}_norm" in out.columns:
                                    out.loc[mask, f"{m}_norm"] = pd.NA
            elif entity_level == "asin":
                key_cols = [c for c in ["vendor_code", "asin"] if c in out.columns]
                if len(key_cols) == 2:
                    keys = out[key_cols[0]].astype(str).fillna("") + "||" + out[key_cols[1]].astype(str).fillna("")
                    invalid_by_key: Dict[str, set] = invalid_metric_map.get("asin", {})
                    if isinstance(invalid_by_key, dict) and not out.empty:
                        for m in metrics:
                            inv_set = invalid_by_key.get(m)
                            if isinstance(inv_set, set):
                                mask = keys.isin(inv_set)
                                if f"{m}_norm" in out.columns:
                                    out.loc[mask, f"{m}_norm"] = pd.NA

        # Coerce to numeric and apply masked contribution: NaN -> 0 contribution
        vals = out[norm_cols].apply(pd.to_numeric, errors="coerce")
        total = vals.fillna(0.0).sum(axis=1)
        # Equal weights across ALL metrics defined for the dimension (no re-normalization)
        denom = float(len(norm_cols))
        out[f"{dim_name.capitalize()}_Score"] = (total / denom).astype(float)

    return out


def compute_composite_score(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Weighted sum of dimension scores scaled to 0–1000 as Composite_Score.

    NA handling policy:
    - Never crash on NA/NAType.
    - Missing dimension scores contribute 0 (no re-normalization of weights).
    - Always emit a numeric Composite_Score when weights are configured.
    """
    if df is None or df.empty:
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

    out = df.copy()
    weights = (config.get("scoring", {}).get("weights", {}) or {}).get("dimension", {})

    # Collect dimension score columns according to weights
    components: List[Tuple[str, float]] = []
    for dim, w in (weights or {}).items():
        col = f"{dim.capitalize()}_Score"
        components.append((col, float(w)))

    if not components:
        # No dimension weights configured; leave as NA to signal absence of config
        out["Composite_Score"] = pd.NA
        return out

    total = None
    for col, w in components:
        # Safe numeric coercion and masked multiplication
        series = pd.to_numeric(out.get(col, pd.Series([pd.NA] * len(out))), errors="coerce")
        term = series.fillna(0.0) * float(w)
        total = term if total is None else total.add(term, fill_value=0.0)

    # Scale to 0–1000
    out["Composite_Score"] = (total.fillna(0.0) * 1000.0).astype(float)
    return out


def analyze_zero_behavior(norm_df: pd.DataFrame, config: Dict) -> Tuple[Dict[str, Dict[str, set]], Dict[str, int]]:
    """Analyze zero semantics and signal strength per (entity, metric).

    Returns:
      - invalid_map: {"vendor": {metric: set(vendor_code)}, "asin": {metric: set("vendor||asin")}}
      - summary_counts: counters for logging.
    """
    if norm_df is None or norm_df.empty:
        return {"vendor": {}, "asin": {}}, {
            "all_zero_series_excluded": 0,
            "bad_zero_treated_as_missing": 0,
            "good_zero_kept_as_signal": 0,
            "low_signal_exclusions": 0,
        }

    df = norm_df.copy()
    df["week_start"] = pd.to_datetime(df.get("week_start"), errors="coerce")
    df = df.dropna(subset=["week_start"])  

    # Window selection aligned with aggregate_metrics_for_scoring
    window = str(config.get("scoring", {}).get("aggregation_window", "L4W")).upper()
    weeks_sorted = sorted([w for w in df["week_start"].dropna().unique()])
    if weeks_sorted:
        keep_weeks = set(weeks_sorted[-4:]) if window == "L4W" else set(weeks_sorted)
        df = df[df["week_start"].isin(keep_weeks)]

    # Metric rules from config
    scoring_section = config.get("scoring", {}) or {}
    metrics_cfg = scoring_section.get("metrics", {}) or {}
    if not isinstance(metrics_cfg, dict):
        metrics_cfg = scoring_section.get("metric_zero_semantics", {}) or scoring_section.get("metrics_zero", {}) or {}

    def _rules(metric: str) -> Dict:
        rules = metrics_cfg.get(metric, {}) if isinstance(metrics_cfg, dict) else {}
        return {
            "zero_semantics": (rules or {}).get("zero_semantics", "unknown_zero"),
            "treat_zero_as_missing": bool((rules or {}).get("treat_zero_as_missing", False)),
            "min_signal_threshold": float((rules or {}).get("min_signal_threshold", 0.0)) if (rules or {}).get("min_signal_threshold") is not None else 0.0,
            "allow_all_zero_series": bool((rules or {}).get("allow_all_zero_series", True)),
        }

    invalid_vendor: Dict[str, set] = {}
    invalid_asin: Dict[str, set] = {}
    counters = {
        "all_zero_series_excluded": 0,
        "bad_zero_treated_as_missing": 0,
        "good_zero_kept_as_signal": 0,
        "low_signal_exclusions": 0,
    }

    # Prepare data for vendor and asin separately
    def _analyze(group_cols: List[str], store: Dict[str, set]):
        if not set(group_cols).issubset(df.columns):
            return
        g = df.groupby(group_cols + ["metric"], dropna=False)["value"]
        stats = g.agg(zero_count=lambda s: int((pd.to_numeric(s, errors="coerce") == 0).sum()),
                      nonzero_count=lambda s: int((pd.to_numeric(s, errors="coerce") != 0).sum()),
                      sum_abs=lambda s: float(pd.to_numeric(s, errors="coerce").abs().sum()))
        stats = stats.reset_index()
        for _, row in stats.iterrows():
            metric = str(row["metric"]) if pd.notna(row["metric"]) else None
            if not metric:
                continue
            r = _rules(metric)
            all_zero = (row["nonzero_count"] == 0) and (row["zero_count"] > 0)
            low_signal = r.get("min_signal_threshold", 0.0) > 0.0 and float(row["sum_abs"]) < float(r.get("min_signal_threshold", 0.0))

            valid = True
            if all_zero and not r.get("allow_all_zero_series", True):
                valid = False
                counters["all_zero_series_excluded"] += 1
            if low_signal:
                valid = False
                counters["low_signal_exclusions"] += 1

            key = None
            if group_cols == ["vendor_code"]:
                key = str(row["vendor_code"]) if pd.notna(row["vendor_code"]) else None
                target = invalid_vendor
            else:
                key = f"{row['vendor_code']}||{row['asin']}"
                target = invalid_asin
            if not valid and key:
                target.setdefault(metric, set()).add(key)

            # Counters for semantics classification (informational)
            zs = r.get("zero_semantics", "unknown_zero")
            if zs == "bad_zero" and r.get("treat_zero_as_missing", False):
                counters["bad_zero_treated_as_missing"] += 1
            elif zs == "good_zero":
                counters["good_zero_kept_as_signal"] += 1

    _analyze(["vendor_code"], invalid_vendor)
    if {"vendor_code", "asin"}.issubset(df.columns):
        _analyze(["vendor_code", "asin"], invalid_asin)

    invalid_map = {"vendor": invalid_vendor, "asin": invalid_asin}
    return invalid_map, counters


def compute_reliability_flags(scoring_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Assign reliability_flag per entity based on history_length, missing_rate, volatility.

    Expects per-entity/week rows with key columns present. Computes a simple categorical flag.
    """
    if scoring_df is None or scoring_df.empty:
        return scoring_df if isinstance(scoring_df, pd.DataFrame) else pd.DataFrame()
    df = scoring_df.copy()
    # Identify grouping
    if "asin" in df.columns and df["asin"].notna().any():
        group_cols = ["vendor_code", "asin"]
    else:
        group_cols = ["vendor_code"]

    # thresholds
    thr = config.get("scoring", {}).get("phase2_thresholds", {}) or {}
    low_hist = int(thr.get("low_history_weeks", 8))

    # choose core metrics for missing_rate/volatility
    core_metrics = [c for c in ["Product_GMS", "Net_PPM", "SoROOS"] if c in df.columns]
    def _flag(group: pd.DataFrame) -> str:
        history_len = group.get("week_start").nunique() if "week_start" in group.columns else len(group)
        missing_rate = 0.0
        if core_metrics:
            missing_rate = float(group[core_metrics].isna().mean().mean())
        volatility = 0.0
        if "Product_GMS" in group.columns:
            volatility = float(group["Product_GMS"].astype(float).std(ddof=0))
        # Simple rules
        if history_len < low_hist or missing_rate > 0.4:
            return "low"
        if missing_rate > 0.2:
            return "medium"
        return "high"

    flags = df.groupby(group_cols, dropna=False).apply(_flag).reset_index(name="reliability_flag")
    out = df.merge(flags, on=group_cols, how="left")
    return out


def build_asin_l4w_performance(norm_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Compute ASIN L4W performance and deltas for growth inputs."""
    if norm_df is None or norm_df.empty:
        return pd.DataFrame()
    df = norm_df.copy()
    # Coerce date
    df["week_start"] = pd.to_datetime(df.get("week_start"), errors="coerce")
    df = df.dropna(subset=["week_start", "vendor_code"])  # asin may be NA for vendor-level rows
    # Keep ASIN rows only
    if "asin" in df.columns:
        df = df.dropna(subset=["asin"]).copy()
    else:
        return pd.DataFrame()

    # Identify last 8 weeks
    weeks = sorted(df["week_start"].dropna().unique())
    if not weeks:
        return pd.DataFrame()
    last8 = weeks[-8:]
    l4w = set(last8[-4:])
    prev4 = set(last8[:max(0, len(last8) - 4)]) - set(last8[-4:])

    def _metric_window(metric: str, window: str, how: str = "sum") -> pd.DataFrame:
        sub = df[df["metric"].astype(str) == metric].copy()
        sub = sub[sub["week_start"].isin(l4w if window == "L4W" else prev4)]
        if sub.empty:
            return pd.DataFrame(columns=["vendor_code", "asin", f"{metric}_{window}"])
        g = sub.groupby(["vendor_code", "asin"], as_index=False)["value"]
        out = g.sum() if how == "sum" else g.mean()
        out.rename(columns={"value": f"{metric}_{window}"}, inplace=True)
        return out

    parts = [
        _metric_window("Product_GMS", "L4W", how="sum"),
        _metric_window("Product_GMS", "prev_L4W", how="sum"),
        _metric_window("Net_Ordered_Units", "L4W", how="sum"),
        _metric_window("Net_Ordered_Units", "prev_L4W", how="sum"),
        _metric_window("GV", "L4W", how="sum"),
        _metric_window("GV", "prev_L4W", how="sum"),
    ]
    perf = None
    for p in parts:
        perf = p if perf is None else perf.merge(p, on=["vendor_code", "asin"], how="outer")
    if perf is None:
        return pd.DataFrame()

    # Derived fields
    growth_cfg = (config.get("scoring", {}).get("growth", {}) or {})
    eps = float(growth_cfg.get("epsilon_share_for_zero_base", 0.01))
    max_cap = growth_cfg.get("max_growth_pct")
    max_cap = float(max_cap) if max_cap is not None else None

    def _pct_delta(new, old):
        new = pd.to_numeric(new, errors="coerce")
        old = pd.to_numeric(old, errors="coerce")
        # Element-wise safe growth with zero-base handling
        result = pd.Series(index=new.index, dtype="float64")
        prev_zero = (old == 0) | old.isna()
        both_zero = prev_zero & ((new == 0) | new.isna())
        breakout = prev_zero & (pd.to_numeric(new, errors="coerce") > 0)
        normal = ~(prev_zero)
        result[both_zero] = 0.0
        result[breakout] = (new[breakout] / eps) - 1.0
        result[normal] = (new[normal] - old[normal]) / old[normal]
        if max_cap is not None:
            result = result.clip(lower=-abs(max_cap), upper=abs(max_cap))
        return result

    perf["GMS_Delta_Pct"] = _pct_delta(perf.get("Product_GMS_L4W"), perf.get("Product_GMS_prev_L4W"))
    perf["Units_Delta_Pct"] = _pct_delta(perf.get("Net_Ordered_Units_L4W"), perf.get("Net_Ordered_Units_prev_L4W"))
    perf["Conversion_Rate_L4W"] = (perf.get("Net_Ordered_Units_L4W") / perf.get("GV_L4W")).replace([pd.NA, float("inf")], pd.NA)
    perf["Conversion_Rate_prev_L4W"] = (perf.get("Net_Ordered_Units_prev_L4W") / perf.get("GV_prev_L4W")).replace([pd.NA, float("inf")], pd.NA)
    perf["Conversion_Delta"] = (perf["Conversion_Rate_L4W"] - perf["Conversion_Rate_prev_L4W"]) if "Conversion_Rate_prev_L4W" in perf.columns else pd.NA

    # Importance share by vendor L4W GMS
    if {"vendor_code", "Product_GMS_L4W"}.issubset(perf.columns):
        totals = perf.groupby("vendor_code", as_index=False)["Product_GMS_L4W"].sum().rename(columns={"Product_GMS_L4W": "_vendor_gms"})
        perf = perf.merge(totals, on="vendor_code", how="left")
        perf["importance_share_l4w"] = (perf["Product_GMS_L4W"] / perf["_vendor_gms"]).replace([pd.NA, float("inf")], pd.NA)
        perf.drop(columns=["_vendor_gms"], inplace=True)
    return perf


def build_vendor_scoring_enriched(vendor_scores_df: pd.DataFrame, aggregated_vendor_metrics: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Merge vendor composite+dimension scores with key KPIs and reliability flag."""
    if vendor_scores_df is None:
        return pd.DataFrame()
    df = vendor_scores_df.copy()
    if aggregated_vendor_metrics is not None and not aggregated_vendor_metrics.empty:
        # Merge on vendor_code & week_start when present
        join_cols = [c for c in ["vendor_code", "week_start"] if c in aggregated_vendor_metrics.columns and c in df.columns]
        if not join_cols:
            join_cols = ["vendor_code"]
        df = df.merge(aggregated_vendor_metrics, on=join_cols, how="left")
    # Ensure reliability_flag present
    if "reliability_flag" not in df.columns:
        df["reliability_flag"] = pd.NA
    return df


def build_asin_scoring_enriched(asin_scores_df: pd.DataFrame, aggregated_asin_metrics: pd.DataFrame, asin_l4w_perf_df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Merge ASIN scores with attributes, KPIs, growth inputs, and flags."""
    if asin_scores_df is None:
        return pd.DataFrame()
    df = asin_scores_df.copy()
    # Merge aggregates
    if aggregated_asin_metrics is not None and not aggregated_asin_metrics.empty:
        join_cols = [c for c in ["vendor_code", "asin", "week_start"] if c in aggregated_asin_metrics.columns and c in df.columns]
        if not join_cols:
            join_cols = [c for c in ["vendor_code", "asin"] if c in df.columns]
        df = df.merge(aggregated_asin_metrics, on=join_cols, how="left")
    # Merge L4W perf (on vendor_code, asin)
    if asin_l4w_perf_df is not None and not asin_l4w_perf_df.empty:
        join_cols2 = [c for c in ["vendor_code", "asin"] if c in df.columns and c in asin_l4w_perf_df.columns]
        if join_cols2:
            df = df.merge(asin_l4w_perf_df, on=join_cols2, how="left")
    # Flags
    thr = config.get("scoring", {}).get("phase2_thresholds", {}) or {}
    crap_threshold = float(thr.get("crap_ppm", 0.0))
    avail_threshold = float(thr.get("availability_risk", 0.7))
    # CRaP when Net_PPM < crap_ppm or CPPU < 0
    df["CRaP_Flag"] = False
    if "Net_PPM" in df.columns:
        df.loc[pd.to_numeric(df["Net_PPM"], errors="coerce") < crap_threshold, "CRaP_Flag"] = True
    if "CPPU" in df.columns:
        df.loc[pd.to_numeric(df["CPPU"], errors="coerce") < 0, "CRaP_Flag"] = True
    # Availability risk
    if "Availability_Score" in df.columns:
        df["Availability_Risk_Flag"] = pd.to_numeric(df["Availability_Score"], errors="coerce") < avail_threshold
    else:
        df["Availability_Risk_Flag"] = pd.NA
    # Ensure Item_Name
    if "item_name_clean" in df.columns and "Item_Name" not in df.columns:
        df["Item_Name"] = df["item_name_clean"]
    return df
