"""Phase 4 forecasting engine entry point."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Dict

import pandas as pd

from .ingestion_utils import load_config
from .forecasting_utils import (
    build_context,
    generate_forecasts,
    load_normalized_dataset,
    persist_forecast_outputs,
    setup_logging,
    summarise_potential,
    write_metadata,
)


def _entity_column(entity_type: str) -> str:
    return "vendor_code" if entity_type == "vendor" else "asin"


def run_phase4(config_path: str = "config.yaml") -> Dict[str, pd.DataFrame]:
    """Execute the forecasting pipeline and return generated forecasts."""

    config = load_config(config_path)
    logger = setup_logging(config)

    logger.info("Starting Phase 4 forecasting with configuration %s", config_path)
    normalized_df = load_normalized_dataset(config, logger=logger)
    if normalized_df.empty:
        logger.warning("Normalized dataset is empty. No forecasts will be produced.")
        empty = pd.DataFrame()
        return {"vendor": empty, "asin": empty}

    context = build_context(config)

    vendor_forecasts = generate_forecasts(normalized_df, context, "vendor", logger=logger)
    asin_forecasts = generate_forecasts(normalized_df, context, "asin", logger=logger)

    vendor_paths = persist_forecast_outputs(vendor_forecasts, config, "vendor", logger=logger)
    asin_paths = persist_forecast_outputs(asin_forecasts, config, "asin", logger=logger)

    forecasting_config = config.get("forecasting", {})
    metadata_fields = forecasting_config.get("metadata_fields", {})

    timestamp = datetime.now(timezone.utc).isoformat()
    metadata: Dict[str, object] = {
        "timestamp": timestamp,
        "horizons_weeks": list(context.horizons),
        "records_vendor": int(len(vendor_forecasts)),
        "records_asin": int(len(asin_forecasts)),
        "metrics": list(context.metrics),
        "output_paths": {
            "vendor_csv": str(vendor_paths[0]),
            "vendor_parquet": str(vendor_paths[1]) if vendor_paths[1] else None,
            "asin_csv": str(asin_paths[0]),
            "asin_parquet": str(asin_paths[1]) if asin_paths[1] else None,
        },
    }

    if metadata_fields.get("include_history", True):
        metadata["history_summary"] = {
            "min_weeks_vendor": int(vendor_forecasts["history_weeks"].min())
            if not vendor_forecasts.empty
            else None,
            "max_weeks_vendor": int(vendor_forecasts["history_weeks"].max())
            if not vendor_forecasts.empty
            else None,
            "min_weeks_asin": int(asin_forecasts["history_weeks"].min()) if not asin_forecasts.empty else None,
            "max_weeks_asin": int(asin_forecasts["history_weeks"].max()) if not asin_forecasts.empty else None,
        }

    if metadata_fields.get("include_methods", True):
        methods = pd.concat([vendor_forecasts["methods"], asin_forecasts["methods"]], ignore_index=True)
        metadata["methods_used"] = sorted({method for entry in methods.dropna() for method in entry.split(",") if method})

    metadata_path = write_metadata(metadata, config)

    logger.info(
        "Forecasting completed. Vendor records: %s | ASIN records: %s | Metadata: %s",
        metadata["records_vendor"],
        metadata["records_asin"],
        metadata_path,
    )

    vendor_top, vendor_bottom = summarise_potential(vendor_forecasts, _entity_column("vendor"))
    asin_top, asin_bottom = summarise_potential(asin_forecasts, _entity_column("asin"))

    if not vendor_top.empty:
        print("Top vendor potential gains:\n", vendor_top.to_string(index=False))
    if not vendor_bottom.empty:
        print("Vendors needing attention:\n", vendor_bottom.to_string(index=False))
    if not asin_top.empty:
        print("Top ASIN potential gains:\n", asin_top.to_string(index=False))
    if not asin_bottom.empty:
        print("ASINs needing attention:\n", asin_bottom.to_string(index=False))

    return {"vendor": vendor_forecasts, "asin": asin_forecasts}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brightstar Phase 4 forecasting")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_phase4(config_path=args.config)


if __name__ == "__main__":  # pragma: no cover - module is executed as a script
    main()
