"""Phase 1 ingestion and normalization entry point."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from .ingestion_utils import (
    MANDATORY_OUTPUT_DIR,
    apply_master_lookup,
    deduplicate_records,
    detect_metric_column,
    detect_structure,
    detect_value_column,
    detect_week_column,
    detect_identifier_column,
    ensure_directory,
    finalize_weeks,
    get_timestamped_output_dir,
    load_calendar_map,
    load_master_lookup,
    load_config,
    melt_wide_dataframe,
    normalize_headers,
    normalize_long_dataframe,
    read_input_file,
    setup_logging,
    summarize_phase,
    validate_and_standardize_asins,
    validate_extension,
    write_output,
    write_state,
    write_phase1_reports,
    load_product_master,
    attach_product_master,
    apply_value_flags,
    ensure_export_contract,
    validate_phase1_output,
    load_masterfile_frame,
    extract_masterfile_item_names,
)
from .logging_utils import log_system_event
from .path_utils import get_phase1_normalized_path, cleanup_tmp_files
from .metric_registry_utils import match_to_canonical, metric_variants


def resolve_id_columns(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, List[str]]:
    """Determine identifier columns using configured aliases or positional hints."""

    ingestion_config = config.get("ingestion", {})
    explicit = ingestion_config.get("id_columns", [])
    if explicit and all(col in df.columns for col in explicit):
        return df, explicit

    aliases = ingestion_config.get("column_aliases", {})
    asin_aliases: Iterable[str] = aliases.get("asin", ["asin"])
    vendor_aliases: Iterable[str] = aliases.get("vendor", ["vendor", "vendor_code"])

    vendor_fallback = ingestion_config.get("vendor_column")
    asin_fallback = ingestion_config.get("asin_column")

    working_df, vendor_column = detect_identifier_column(
        df,
        vendor_aliases,
        fallback_index=vendor_fallback,
        default_name="Vendor",
    )

    working_df, asin_column = detect_identifier_column(
        working_df,
        asin_aliases,
        fallback_index=asin_fallback,
        default_name="ASIN",
    )

    if asin_column is None:
        raise ValueError("Unable to detect ASIN column. Provide aliases in configuration or ensure header includes ASIN.")

    if vendor_column is None:
        working_df = working_df.copy()
        vendor_column = "Vendor"
        working_df[vendor_column] = pd.NA

    return working_df, [vendor_column, asin_column]


def ingest_file(path: Path, config: dict, lookup, master_lookup: pd.DataFrame | None = None) -> pd.DataFrame:
    """Normalize a single raw export file."""

    # Support a wide set of default extensions if not specified in config
    allowed = config["ingestion"].get(
        "allowed_extensions",
        [".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls"],
    )
    validate_extension(path, allowed)
    df_raw = read_input_file(path)
    df_raw = normalize_headers(df_raw)
    df_raw, id_columns = resolve_id_columns(df_raw, config)

    structure = detect_structure(df_raw, id_columns)
    value_round = config["ingestion"].get("value_round")

    if structure == "wide" and not config["ingestion"].get("force_long_format", False):
        normalized = melt_wide_dataframe(
            df_raw,
            id_columns=id_columns,
            lookup=lookup,
            value_round=value_round,
        )
    else:
        week_column = detect_week_column(df_raw)
        metric_column = detect_metric_column(df_raw)
        value_column = detect_value_column(df_raw)
        if not week_column or not metric_column or not value_column:
            raise ValueError(
                f"Unable to detect columns for long format in {path}. Found week={week_column}, metric={metric_column}, value={value_column}"
            )
        normalized = normalize_long_dataframe(
            df_raw,
            lookup=lookup,
            week_column=week_column,
            metric_column=metric_column,
            value_column=value_column,
            id_columns=id_columns,
            value_round=value_round,
        )

    normalized["source_file"] = path.name
    if master_lookup is not None:
        normalized = apply_master_lookup(normalized, master_lookup)
    return normalized


def gather_input_files(config: dict, explicit: Optional[Iterable[Path]] = None) -> List[Path]:
    """Collect all files that should be ingested."""

    if explicit:
        return [Path(p) for p in explicit]
    raw_dir = Path(config["paths"]["raw_dir"])
    ensure_directory(raw_dir)
    allowed = {ext.lower() for ext in config["ingestion"].get("allowed_extensions", [".csv", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls"])}
    files = [
        path
        for path in raw_dir.iterdir()
        if path.suffix.lower() in allowed
    ]
    # Apply deterministic source file priority if configured
    priorities: List[str] = config.get("ingestion", {}).get("source_priority", []) or []
    def prio_key(p: Path) -> tuple[int, str]:
        name = p.name.lower()
        # Higher score means higher priority, so that it will be placed later in the list
        score = 0  # default lowest priority
        for i, token in enumerate(priorities):
            if token and token.lower() in name:
                score = len(priorities) - i
                break
        return (score, name)
    return sorted(files, key=prio_key)


def run_phase1(config_path: str = "config.yaml", input_files: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """Execute the ingestion pipeline and return the normalized dataframe."""

    config = load_config(config_path)
    # Create timestamped output directory for this run and route all outputs there
    output_dir = get_timestamped_output_dir()
    logger = setup_logging(config, output_dir=output_dir)
    # Safety: clean up stale tmp files in canonical normalized directory
    try:
        cleanup_tmp_files(config)
    except Exception:
        pass
    try:
        log_system_event(logger, "Phase 1 ingestion started.")
    except Exception:
        pass
    lookup = load_calendar_map(config["paths"]["calendar_map"])
    # If a unified calendar was emitted to the base directory during load, copy it into this run folder
    try:
        unified_src = MANDATORY_OUTPUT_DIR / "unified_calendar_map.csv"
        if unified_src.exists():
            shutil.copyfile(unified_src, output_dir / "unified_calendar_map.csv")
        # Also copy the updated reference calendar_map.csv used by this run
        ref_calendar = Path(config["paths"]["calendar_map"])  # after load it should be updated in-place
        if ref_calendar.exists():
            shutil.copyfile(ref_calendar, output_dir / "calendar_map.csv")
    except Exception:
        pass
    master_lookup = load_master_lookup(
        config["paths"].get("masterfile"),
        config.get("masterfile"),
    )

    file_paths = gather_input_files(config, [Path(p) for p in input_files] if input_files else None)
    if not file_paths:
        logger.warning("No input files found for Phase 1 ingestion.")
        return pd.DataFrame(columns=[
            "vendor_code",
            "vendor_name",
            "asin",
            "metric",
            "week_label",
            "week_raw",
            "week_start",
            "week_end",
            "value",
            "source_file",
        ])

    frames = []
    for path in file_paths:
        logger.info("Ingesting %s", path)
        normalized = ingest_file(path, config, lookup, master_lookup)
        frames.append(normalized)

    combined = pd.concat(frames, ignore_index=True)
    # Ensure ASINs are standardized and invalids are exported for debugging
    combined = validate_and_standardize_asins(combined, output_dir=output_dir)
    combined = deduplicate_records(combined, logger)
    # Eliminate null/blank week labels; log unresolved cases
    combined = finalize_weeks(combined, lookup, logger, issues_output=output_dir / "week_parsing_issues.csv")

    # Canonicalize metric names using predefined variant map (pure local logic)
    try:
        if not combined.empty and "metric" in combined.columns:
            before = combined["metric"].astype(str)
            mapped = before.apply(lambda x: match_to_canonical(x, metric_variants) or x)
            changes = int((before != mapped).sum())
            combined["metric"] = mapped
            if changes:
                logger.info("Canonicalized %d metric values based on variant map.", changes)
    except Exception as _exc:
        # Degrade safely: never interrupt ingestion due to canonicalization
        logger.warning("Metric canonicalization skipped due to an issue; proceeding with raw metric names.")
    # Attach Product Master attributes (item_name_clean, brand, category, etc.) for ASIN rows
    try:
        pm_df = load_product_master(config)
        combined = attach_product_master(combined, pm_df)
    except Exception as _exc:
        logger.warning("Product Master enrichment skipped due to an issue; proceeding without item attributes.")

    # Phase 1 patch: item_name fallback from Masterfile when product_master missing
    try:
        mf_frame = load_masterfile_frame(config)
        names_df = extract_masterfile_item_names(mf_frame)
        if not names_df.empty and "asin" in combined.columns:
            # Left-join on ASIN only, then fill item_name_clean where null for ASIN rows
            combined = combined.merge(names_df, on="asin", how="left")
            if "item_name_clean" not in combined.columns:
                combined["item_name_clean"] = pd.NA
            asin_mask = combined["asin"].notna() & (combined["asin"].astype(str).str.len() > 0)
            # Prefer existing product_master value; fill remaining with masterfile name
            combined.loc[asin_mask, "item_name_clean"] = (
                combined.loc[asin_mask, "item_name_clean"].where(
                    combined.loc[asin_mask, "item_name_clean"].notna(),
                    other=combined.loc[asin_mask, "item_name_master"]
                )
            )
            # Optional: drop helper column to keep schema clean
            combined.drop(columns=[c for c in ["item_name_master"] if c in combined.columns], inplace=True)
    except Exception:
        # Non-fatal; keep pipeline moving
        pass

    # Apply value flags and enforce export contract
    try:
        combined = apply_value_flags(combined, config)
    except Exception as _exc:
        logger.warning("Value casting/flagging skipped due to an issue; proceeding with raw values.")

    try:
        combined = ensure_export_contract(combined)
    except Exception as _exc:
        logger.warning("Export contract enforcement skipped; proceeding with current columns.")
    combined.sort_values(
        ["vendor_code", "vendor_name", "asin", "metric", "week_start", "week_label"],
        inplace=True,
    )

    # Best-effort validation prior to export (non-fatal)
    try:
        validate_phase1_output(combined, logger)
    except Exception:
        pass

    output_path = write_output(combined, config, logger, output_dir=output_dir)
    logger.info("Normalized dataset written to %s", output_path)

    # Canonical normalized location under output_root with atomic replace
    try:
        canonical_path = get_phase1_normalized_path(config)
        tmp_path = canonical_path.with_suffix(".tmp.parquet")
        combined.to_parquet(tmp_path, index=False)
        tmp_path.replace(canonical_path)
        logger.info("Phase 1 wrote normalized output to canonical path: %s", canonical_path.resolve())
    except Exception as _exc:
        logger.warning("Failed to persist canonical normalized parquet; downstream phases may not find it.")

    # Write validation and summary artifacts
    report_paths = write_phase1_reports(combined, logger, output_dir=output_dir)
    for key, rpath in report_paths.items():
        logger.info("Wrote %s to %s", key, rpath)

    metadata = {
        "records": int(len(combined)),
        "vendors": combined["vendor_code"].nunique(),
        "asins": combined["asin"].nunique(),
        "weeks": combined["week_label"].nunique(),
        "last_week": combined.sort_values("week_start", na_position="last")["week_label"].iloc[-1]
        if not combined.empty
        else None,
        "output_path": str(output_path),
        "output_dir": str(output_dir),
    }
    write_state(metadata, config, output_dir=output_dir)

    template = config.get("summary", {}).get(
        "template", "Processed {total_rows} rows across {unique_weeks} weeks. Last week: {last_week}."
    )
    summary = summarize_phase(combined, template)
    if config.get("summary", {}).get("enabled", True):
        print(summary)
    logger.info(summary)

    # Ingestion diagnostics summary file in logs
    try:
        logs_dir = ensure_directory(config["paths"]["logs_dir"])
        run_id = output_dir.name
        diag_path = logs_dir / f"ingestion_{run_id}.txt"
        metrics_detected = int(combined["metric"].nunique()) if not combined.empty and "metric" in combined.columns else 0
        unmapped_weeks = int(combined["week_start_date"].isna().sum()) if "week_start_date" in combined.columns else 0
        new_metrics_msg = "N/A"
        try:
            # heuristic: metrics not in canonical list length vs unique
            from .metric_registry_utils import canonical_metrics
            new_count = len(set(map(str, combined.get("metric", pd.Series(dtype=str)).dropna().unique())) - set(canonical_metrics))
            new_metrics_msg = str(new_count)
        except Exception:
            pass
        lines = [
            f"Files processed: {len(file_paths)}",
            f"Rows processed: {len(combined)}",
            f"Metrics detected: {metrics_detected}",
            f"New metrics detected (approx): {new_metrics_msg}",
            f"Unmapped week labels: {unmapped_weeks}",
            f"Output: {output_path}",
        ]
        diag_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote ingestion diagnostics to %s", diag_path)
    except Exception:
        pass
    try:
        log_system_event(logger, "Phase 1 ingestion completed.")
    except Exception:
        pass
    return combined


def build_arg_parser() -> argparse.ArgumentParser:
    """Create a CLI argument parser for the phase runner."""

    parser = argparse.ArgumentParser(description="Brightstar Phase 1 ingestion")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration file")
    parser.add_argument("inputs", nargs="*", help="Explicit input files to ingest")
    return parser


def main() -> None:
    """CLI entry point."""

    parser = build_arg_parser()
    args = parser.parse_args()
    run_phase1(config_path=args.config, input_files=args.inputs or None)


if __name__ == "__main__":
    main()
