"""Phase 3 commentary engine entry point."""
from __future__ import annotations

import argparse

import pandas as pd

from .commentary_utils import generate_commentary


def run_phase3(config_path: str = "config.yaml") -> pd.DataFrame:
    """Execute the commentary generation workflow."""

    return generate_commentary(config_path)


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for Phase 3."""

    parser = argparse.ArgumentParser(description="Brightstar Phase 3 commentary engine")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    return parser


def main() -> None:
    """CLI entry point for executing the commentary engine."""

    parser = build_arg_parser()
    args = parser.parse_args()
    run_phase3(config_path=args.config)


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
