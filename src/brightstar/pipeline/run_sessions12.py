from __future__ import annotations

import argparse
from pathlib import Path

from ..logging_utils import log_system_event


def run_sessions_1_2(
    input_path: str | None = None,
    phase1_output: str = "data/processed/phase1",
    phase2_output: str = "data/processed/scoring",
    schema_path: str = "config/schema_phase1.yaml",
    column_map_path: str | None = "config/phase1_column_map.yaml",
    calendar_config: str | None = "config/phase1_calendar.yaml",
    dimensions_config: str = "config/dimension_paths.yaml",
    scoring_config: str = "config/scoring_config.yaml",
) -> dict:
    """Run Session 1 (Phase 1 ingestion/validation) then Session 2 (scoring engine).

    Returns a dict with key paths produced.
    """

    # Late imports to keep module import light (use absolute imports to avoid package-relative issues)
    from phase1.loader import load_phase1
    from phase2.scoring_engine import run_scoring

    # Resolve defaults
    if input_path is None:
        # Prefer the data/raw directory if present; otherwise, raise a friendly error
        raw_dir = Path("data") / "raw"
        if raw_dir.exists():
            input_path = str(raw_dir)
        else:
            raise FileNotFoundError("No input_path provided and data/raw does not exist. Please specify --input.")

    # Ensure output directories exist
    p1_out = Path(phase1_output)
    p1_out.mkdir(parents=True, exist_ok=True)
    p2_out = Path(phase2_output)
    p2_out.mkdir(parents=True, exist_ok=True)

    # Run Session 1
    log_system_event(None, f"Session 1: Loading Phase 1 from {input_path}")
    load_phase1(
        input_path=str(input_path),
        schema_path=str(schema_path),
        column_map_path=str(column_map_path) if column_map_path else None,
        calendar_config=str(calendar_config) if calendar_config else None,
        output_base=str(p1_out),
    )

    # Locate Phase 1 validated parquet
    phase1_parquet = p1_out / "phase1_validated.parquet"
    if not phase1_parquet.exists():
        # Fallback: search common subfolders
        candidates = list(p1_out.rglob("phase1_validated.parquet"))
        if candidates:
            phase1_parquet = candidates[0]
        else:
            raise FileNotFoundError(f"Expected Phase 1 output not found at {phase1_parquet}")

    # Run Session 2 scoring engine (with Session 2A features inside)
    log_system_event(None, f"Session 2: Running scoring_engine on {phase1_parquet}")
    run_scoring(
        phase1_path=str(phase1_parquet),
        dimensions_config=str(dimensions_config),
        scoring_config=str(scoring_config),
        output_base=str(p2_out),
    )

    return {
        "phase1_parquet": str(phase1_parquet.resolve()),
        "phase2_output": str(p2_out.resolve()),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Sessions 1 and 2 (Phase 1 + Phase 2)")
    p.add_argument("--input", help="Input file or directory for Phase 1", default=None)
    p.add_argument("--phase1_output", help="Output directory for Phase 1 artifacts", default="data/processed/phase1")
    p.add_argument("--phase2_output", help="Output directory for Phase 2 artifacts", default="data/processed/scoring")
    p.add_argument("--schema", help="Path to Phase 1 schema YAML", default="config/schema_phase1.yaml")
    p.add_argument("--column_map", help="Path to Phase 1 column map YAML", default="config/phase1_column_map.yaml")
    p.add_argument("--calendar", help="Path to Phase 1 calendar config YAML", default="config/phase1_calendar.yaml")
    p.add_argument("--dimensions", help="Path to dimension_paths.yaml for Phase 2", default="config/dimension_paths.yaml")
    p.add_argument("--scoring_config", help="Path to scoring_config.yaml for Phase 2", default="config/scoring_config.yaml")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_sessions_1_2(
        input_path=args.input,
        phase1_output=args.phase1_output,
        phase2_output=args.phase2_output,
        schema_path=args.schema,
        column_map_path=args.column_map,
        calendar_config=args.calendar,
        dimensions_config=args.dimensions,
        scoring_config=args.scoring_config,
    )
