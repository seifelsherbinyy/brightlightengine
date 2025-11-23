"""Utility helpers for Phase 3 commentary generation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

from .ingestion_utils import ensure_directory, load_config
from .path_utils import get_phase1_normalized_path


LOGGER_NAME = "brightstar.phase3"


class SafeDict(dict):
    """Dictionary returning the placeholder when keys are missing."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - defensive
        return "{" + key + "}"


@dataclass
class ChangeResult:
    """Stores change information used for commentary rendering."""

    entity_type: str
    vendor_code: str
    asin: Optional[str]
    metric: str
    current_week: Optional[str]
    previous_week: Optional[str]
    current_value: Optional[float]
    previous_value: Optional[float]
    change_value: Optional[float]
    change_pct: Optional[float]
    classification: str


CHANGE_DIRECTION_MAP = {
    "strong_increase": "up strongly",
    "moderate_increase": "up",
    "flat": "flat",
    "moderate_decline": "down",
    "strong_decline": "down sharply",
}

COMMENTARY_COLUMNS = [
    "entity_type",
    "vendor_code",
    "asin",
    "week_label",
    "metric",
    "Commentary_Type",
    "Commentary_Text",
    "Change_Class",
    "Change_Pct",
    "Current_Value",
    "Previous_Value",
]


def setup_logging(config: Dict) -> logging.Logger:
    """Configure logging for the commentary phase."""

    log_dir = ensure_directory(config["paths"]["logs_dir"])
    commentary_config = config.get("commentary", {})
    file_name = commentary_config.get("logging_file_name", "phase3_commentary.log")
    log_path = log_dir / file_name

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, commentary_config.get("logging_level", "INFO")))
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

    logger.debug("Phase 3 logging initialised at %s", log_path)
    return logger


def load_normalized_dataset(config: Dict, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """Load the normalized dataset used for trend analysis."""

    commentary_config = config.get("commentary", {})
    # 1) Canonical path from Phase 1 helper
    try:
        canonical = get_phase1_normalized_path(config)
        if canonical.exists():
            return pd.read_parquet(canonical)
    except Exception as exc:
        if logger:
            logger.debug("Canonical normalized parquet not available yet (%s). Trying fallbacks.", exc)

    # 2) Legacy/config-driven fallbacks
    candidate_paths: List[Path] = []
    pref = commentary_config.get("normalized_input")
    if pref:
        candidate_paths.append(Path(pref))

    default = config["paths"].get("normalized_output")
    if default:
        candidate_paths.append(Path(default))

    candidate_paths.append(Path(config["paths"]["processed_dir"]) / "normalized_phase1.csv")

    for path in candidate_paths:
        if path.exists():
            if path.suffix.lower() == ".parquet":
                try:
                    return pd.read_parquet(path)
                except Exception as exc:  # pragma: no cover - engine availability
                    if logger:
                        logger.warning("Unable to read parquet file %s (%s). Falling back if possible.", path, exc)
                    continue
            if path.suffix.lower() in {".csv", ".txt"}:
                return pd.read_csv(path)

    raise FileNotFoundError(
        "Unable to locate normalized dataset. Ensure Phases 1 and 2 outputs exist before running Phase 3."
    )


def _load_scorecard(path: Path, fallback_csv: Path, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    if path and path.is_file():
        if path.suffix.lower() == ".parquet":
            try:
                return pd.read_parquet(path)
            except Exception as exc:  # pragma: no cover - optional engine
                if logger:
                    logger.warning("Unable to read parquet scorecard %s (%s). Trying CSV fallback.", path, exc)
        else:
            return pd.read_csv(path)

    if fallback_csv and fallback_csv.is_file():
        return pd.read_csv(fallback_csv)

    raise FileNotFoundError(f"Scorecard file not found at {path} or {fallback_csv}.")


def load_scorecards(config: Dict, logger: Optional[logging.Logger] = None) -> Dict[str, pd.DataFrame]:
    """Load vendor and ASIN scorecards for commentary generation."""

    commentary_config = config.get("commentary", {})
    vendor_pref = Path(commentary_config.get("scorecard_vendor", config["paths"].get("scorecard_vendor_parquet", "")))
    asin_pref = Path(commentary_config.get("scorecard_asin", config["paths"].get("scorecard_asin_parquet", "")))

    vendor_csv = Path(config["paths"].get("scorecard_vendor_csv", ""))
    asin_csv = Path(config["paths"].get("scorecard_asin_csv", ""))

    scorecards = {}
    for label, preferred, fallback in (
        ("vendor", vendor_pref, vendor_csv),
        ("asin", asin_pref, asin_csv),
    ):
        try:
            scorecards[label] = _load_scorecard(preferred, fallback, logger=logger)
        except FileNotFoundError as exc:
            if logger:
                logger.warning("Scorecard missing for %s level (%s). Using empty frame.", label, exc)
            scorecards[label] = pd.DataFrame()
    return scorecards


def _aggregate_weekly_values(
    df: pd.DataFrame,
    entity_columns: Iterable[str],
    aggregation: str,
) -> pd.DataFrame:
    if df.empty:
        cols = list(entity_columns) + ["metric", "week_label", "week_start", "value"]
        return pd.DataFrame(columns=cols)

    working = df.copy()
    working["week_start"] = pd.to_datetime(working["week_start"], errors="coerce")
    working = working.dropna(subset=["week_start"])

    group_cols = list(entity_columns) + ["metric", "week_label", "week_start"]
    if aggregation == "mean":
        aggregated = working.groupby(group_cols, as_index=False)["value"].mean()
    elif aggregation == "sum":
        aggregated = working.groupby(group_cols, as_index=False)["value"].sum()
    elif aggregation == "median":
        aggregated = working.groupby(group_cols, as_index=False)["value"].median()
    else:
        raise ValueError(f"Unsupported aggregation function: {aggregation}")

    return aggregated


def _classify_change(change_pct: Optional[float], change_bands: Dict[str, float]) -> str:
    if change_pct is None:
        return "insufficient_history"

    strong_inc = float(change_bands.get("strong_increase", 0.1))
    moderate_inc = float(change_bands.get("moderate_increase", 0.04))
    flat_threshold = abs(float(change_bands.get("flat", 0.01)))
    moderate_dec = float(change_bands.get("moderate_decline", -0.04))
    strong_dec = float(change_bands.get("strong_decline", -0.1))

    if change_pct >= strong_inc:
        return "strong_increase"
    if change_pct >= moderate_inc:
        return "moderate_increase"
    if change_pct <= strong_dec:
        return "strong_decline"
    if change_pct <= moderate_dec:
        return "moderate_decline"
    if abs(change_pct) <= flat_threshold:
        return "flat"
    return "moderate_increase" if change_pct > 0 else "moderate_decline"


def _format_percent(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:+.1%}"


def _format_number(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.2f}"


def _change_records(
    aggregated: pd.DataFrame,
    entity_columns: List[str],
    change_bands: Dict[str, float],
) -> List[ChangeResult]:
    records: List[ChangeResult] = []
    if aggregated.empty:
        return records

    grouped = aggregated.sort_values("week_start").groupby(entity_columns + ["metric"], sort=False)

    for keys, group in grouped:
        group = group.sort_values("week_start")
        current = group.iloc[-1]
        previous = group.iloc[-2] if len(group) > 1 else None

        current_value = float(current["value"]) if not pd.isna(current["value"]) else None
        previous_value = float(previous["value"]) if previous is not None and not pd.isna(previous["value"]) else None
        change_value = None
        change_pct = None
        if previous is not None and previous_value is not None and current_value is not None:
            change_value = current_value - previous_value
            if previous_value != 0:
                change_pct = change_value / abs(previous_value)

        classification = _classify_change(change_pct, change_bands)

        key_dict = dict(zip(entity_columns, keys if isinstance(keys, tuple) else (keys,)))
        records.append(
            ChangeResult(
                entity_type=key_dict.get("entity_type", "vendor"),
                vendor_code=key_dict.get("vendor_code"),
                asin=key_dict.get("asin"),
                metric=current["metric"],
                current_week=current["week_label"],
                previous_week=previous["week_label"] if previous is not None else None,
                current_value=current_value,
                previous_value=previous_value,
                change_value=change_value,
                change_pct=change_pct,
                classification=classification,
            )
        )

    return records


def _resolve_threshold(thresholds: Dict[str, float], metric: str) -> float:
    if isinstance(thresholds, dict):
        if metric in thresholds:
            return float(thresholds[metric])
        if "default" in thresholds:
            return float(thresholds["default"])
    return float(thresholds) if thresholds else 0.0


def _select_template(templates: Dict[str, str], key: str, fallback: str) -> str:
    if key in templates:
        return templates[key]
    if fallback in templates:
        return templates[fallback]
    return next(iter(templates.values()))


def _direction_label(classification: str) -> str:
    return CHANGE_DIRECTION_MAP.get(classification, "flat")


def _score_lookup(scorecard: pd.DataFrame, entity_type: str) -> pd.DataFrame:
    if scorecard.empty:
        return scorecard

    frame = scorecard.copy()
    if "entity_type" not in frame.columns:
        frame["entity_type"] = entity_type

    if entity_type == "vendor" and "asin" not in frame.columns:
        frame["asin"] = None

    key_cols = ["entity_type", "vendor_code"]
    if entity_type == "asin":
        key_cols.append("asin")

    key_cols.append("metric")
    frame = frame.set_index(key_cols)
    return frame


def _render_commentary(
    changes: List[ChangeResult],
    score_lookup: pd.DataFrame,
    commentary_config: Dict,
) -> pd.DataFrame:
    templates = commentary_config.get("templates", {})
    performance_templates = templates.get("performance_summary", {})
    opportunity_templates = templates.get("opportunity", {})
    risk_templates = templates.get("risk", {})

    opp_thresholds = commentary_config.get("opportunity_thresholds", {})
    risk_thresholds = commentary_config.get("risk_thresholds", {})

    records: List[Dict[str, object]] = []
    for change in changes:
        key = (change.entity_type, change.vendor_code)
        if change.entity_type == "asin":
            key = key + (change.asin,)
        key = key + (change.metric,)

        lookup_row = None
        if not score_lookup.empty and key in score_lookup.index:
            lookup_row = score_lookup.loc[key]

        normalized_value = None
        if lookup_row is not None and "Normalized_Value" in lookup_row:
            normalized_value = float(lookup_row["Normalized_Value"])

        perf_template = _select_template(
            performance_templates,
            change.classification,
            fallback="insufficient_history",
        )

        direction = _direction_label(change.classification)
        context = SafeDict(
            {
                "entity": change.vendor_code if change.entity_type == "vendor" else f"{change.vendor_code}-{change.asin}",
                "metric": change.metric,
                "change_pct": _format_percent(change.change_pct),
                "change_value": _format_number(change.change_value),
                "direction": direction,
                "current_value": _format_number(change.current_value),
                "previous_value": _format_number(change.previous_value),
            }
        )

        records.append(
            {
                "entity_type": change.entity_type,
                "vendor_code": change.vendor_code,
                "asin": change.asin,
                "week_label": change.current_week,
                "metric": change.metric,
                "Commentary_Type": "performance_summary",
                "Commentary_Text": perf_template.format_map(context),
                "Change_Class": change.classification,
                "Change_Pct": change.change_pct,
                "Current_Value": change.current_value,
                "Previous_Value": change.previous_value,
            }
        )

        if normalized_value is not None:
            opportunity_threshold = _resolve_threshold(opp_thresholds, change.metric)
            risk_threshold = _resolve_threshold(risk_thresholds, change.metric)
        else:
            opportunity_threshold = _resolve_threshold(opp_thresholds, "default")
            risk_threshold = _resolve_threshold(risk_thresholds, "default")

        opportunity_key = "neutral"
        if normalized_value is not None:
            if normalized_value < opportunity_threshold:
                opportunity_key = "below_threshold"
            elif change.classification in {"moderate_increase", "strong_increase"}:
                opportunity_key = "improving"

        opportunity_template = _select_template(
            opportunity_templates,
            opportunity_key,
            fallback="neutral",
        )

        records.append(
            {
                "entity_type": change.entity_type,
                "vendor_code": change.vendor_code,
                "asin": change.asin,
                "week_label": change.current_week,
                "metric": change.metric,
                "Commentary_Type": "opportunity",
                "Commentary_Text": opportunity_template.format_map(context),
                "Change_Class": change.classification,
                "Change_Pct": change.change_pct,
                "Current_Value": change.current_value,
                "Previous_Value": change.previous_value,
            }
        )

        risk_key = "neutral"
        if change.classification == "strong_decline":
            risk_key = "strong_decline"
        elif change.classification == "moderate_decline":
            risk_key = "moderate_decline"
        elif normalized_value is not None and normalized_value < risk_threshold:
            risk_key = "below_threshold"

        risk_template = _select_template(risk_templates, risk_key, fallback="neutral")
        records.append(
            {
                "entity_type": change.entity_type,
                "vendor_code": change.vendor_code,
                "asin": change.asin,
                "week_label": change.current_week,
                "metric": change.metric,
                "Commentary_Type": "risk",
                "Commentary_Text": risk_template.format_map(context),
                "Change_Class": change.classification,
                "Change_Pct": change.change_pct,
                "Current_Value": change.current_value,
                "Previous_Value": change.previous_value,
            }
        )

    return pd.DataFrame.from_records(records)


def generate_commentary(config_path: str) -> pd.DataFrame:
    """Main orchestration function for generating commentary."""

    config = load_config(config_path)
    logger = setup_logging(config)

    logger.info("Starting Phase 3 commentary with configuration %s", config_path)
    normalized_df = load_normalized_dataset(config, logger=logger)
    if normalized_df.empty:
        logger.warning("Normalized dataset is empty; no commentary produced.")
        return pd.DataFrame()

    scorecards = load_scorecards(config, logger=logger)
    commentary_config = config.get("commentary", {})
    scoring_config = config.get("scoring", {})
    aggregation = scoring_config.get("aggregation", "mean")
    change_bands = commentary_config.get("change_bands", {})

    vendor_df = normalized_df.copy()
    vendor_df["entity_type"] = "vendor"
    vendor_weekly = _aggregate_weekly_values(vendor_df, ["entity_type", "vendor_code"], aggregation)
    asin_df = normalized_df.copy()
    asin_df["entity_type"] = "asin"
    asin_weekly = _aggregate_weekly_values(asin_df, ["entity_type", "vendor_code", "asin"], aggregation)

    changes: List[ChangeResult] = []
    changes.extend(_change_records(vendor_weekly, ["entity_type", "vendor_code"], change_bands))
    changes.extend(_change_records(asin_weekly, ["entity_type", "vendor_code", "asin"], change_bands))

    score_lookup_vendor = _score_lookup(scorecards.get("vendor", pd.DataFrame()), "vendor")
    score_lookup_asin = _score_lookup(scorecards.get("asin", pd.DataFrame()), "asin")

    vendor_records = _render_commentary(
        changes=[c for c in changes if c.entity_type == "vendor"],
        score_lookup=score_lookup_vendor,
        commentary_config=commentary_config,
    )
    asin_records = _render_commentary(
        changes=[c for c in changes if c.entity_type == "asin"],
        score_lookup=score_lookup_asin,
        commentary_config=commentary_config,
    )

    combined_frames = [df for df in (vendor_records, asin_records) if not df.empty]
    if combined_frames:
        commentary_df = pd.concat(combined_frames, ignore_index=True)
    else:
        commentary_df = pd.DataFrame(columns=COMMENTARY_COLUMNS)

    commentary_df = commentary_df.reindex(columns=COMMENTARY_COLUMNS)

    output_dir = ensure_directory(Path(config["paths"]["processed_dir"]) / "commentary")
    csv_path = output_dir / "commentary.csv"
    parquet_path = output_dir / "commentary.parquet"

    commentary_df.to_csv(csv_path, index=False)
    try:
        commentary_df.to_parquet(parquet_path, index=False)
    except Exception as exc:  # pragma: no cover - parquet optional dependency
        logger.warning("Unable to write parquet commentary file: %s", exc)
        parquet_path = None

    logger.info("Phase 3 commentary completed with %s records", len(commentary_df))
    print(f"Generated {len(commentary_df)} commentary records. Output: {csv_path}")

    return commentary_df

