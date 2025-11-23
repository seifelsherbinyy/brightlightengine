"""Phase 5 dashboard generation entry point."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

from .dashboard_utils import create_dashboard, setup_logging
from .logging_utils import log_system_event
from .ingestion_utils import load_config


def run_phase5(config_path: str = "config.yaml") -> Tuple[Path, Dict[str, object]]:
    """Execute the dashboard generation workflow."""

    config = load_config(config_path)
    logger = setup_logging(config)
    logger.info("Starting Phase 5 dashboard build with configuration %s", config_path)
    try:
        log_system_event(logger, "Dashboard build started.")
    except Exception:
        pass
    output_path, summary = create_dashboard(config, logger)
    logger.info("Dashboard build complete: %s", output_path)
    try:
        log_system_event(logger, "Dashboard build completed.")
    except Exception:
        pass
    return output_path, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brightstar Phase 5 dashboard generator")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_phase5(config_path=args.config)


if __name__ == "__main__":  # pragma: no cover - CLI execution guard
    main()

