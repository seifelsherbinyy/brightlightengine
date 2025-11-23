from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any


def _exists(p: Path) -> bool:
    try:
        return p.exists()
    except Exception:
        return False


def _collect_status(output_dir: Path) -> Dict[str, Any]:
    """Scan Phase 2 output directory for expected artifacts and build a status map."""
    files = {
        "scoring_detailed": output_dir / "scoring_detailed.parquet",
        "scoring_summary": output_dir / "scoring_summary.parquet",
        "metric_readiness": output_dir / "metric_readiness.parquet",
        "scoring_metadata": output_dir / "scoring_metadata.json",
        # Session 2A artifacts
        "transformed_metrics": output_dir / "transformed_metrics.parquet",
        "delta_signals": output_dir / "delta_signals.parquet",
        "dimension_signals": output_dir / "dimension_signals.parquet",
        "session2A_metadata": output_dir / "session2A_metadata.json",
        # Legacy/enhanced outputs present in some runs
        "asin_scoring_enriched": output_dir / "asin_scoring_enriched.parquet",
        "vendor_scoring_enriched": output_dir / "vendor_scoring_enriched.parquet",
        "asin_scorecards": output_dir / "asin_scorecards.parquet",
        "vendor_scorecards": output_dir / "vendor_scorecards.parquet",
        "asin_l4w_performance": output_dir / "asin_l4w_performance.parquet",
        "scoring_parameters": output_dir / "scoring_parameters.json",
    }
    status: Dict[str, Any] = {}
    for k, p in files.items():
        status[k] = {
            "path": str(p.resolve()),
            "exists": _exists(p),
            "size_bytes": (p.stat().st_size if _exists(p) else None),
        }
    return status


def dry_run_sessions_1_2(
    input_path: str | None = None,
    phase1_output: str = "data/processed/phase1",
    phase2_output: str = "data/processed/scoring",
    schema_path: str = "config/schema_phase1.yaml",
    column_map_path: str | None = "config/phase1_column_map.yaml",
    calendar_config: str | None = "config/phase1_calendar.yaml",
    dimensions_config: str = "config/dimension_paths.yaml",
    scoring_config: str = "config/scoring_config.yaml",
) -> Dict[str, Any]:
    """Attempt to run Sessions 1 and 2; on failure, still report current artifact statuses.

    Returns a dict with run result and artifact statuses.
    """

    p1_out = Path(phase1_output)
    p2_out = Path(phase2_output)
    p1_out.mkdir(parents=True, exist_ok=True)
    p2_out.mkdir(parents=True, exist_ok=True)

    ran_ok = False
    error: str | None = None
    phase1_parquet_path: str | None = None
    # Preferred path: if Phase 1 parquet already exists, run only Phase 2 to avoid import issues
    p1_parquet = p1_out / "phase1_validated.parquet"
    if p1_parquet.exists():
        try:
            # Late import to avoid package import issues
            from phase2.scoring_engine import run_scoring
            run_scoring(
                phase1_path=str(p1_parquet),
                dimensions_config=str(dimensions_config),
                scoring_config=str(scoring_config),
                output_base=str(p2_out),
            )
            ran_ok = True
            phase1_parquet_path = str(p1_parquet.resolve())
        except Exception as ex:
            error = str(ex)
    else:
        # Fallback: attempt full Sessions 1+2 run
        try:
            from .run_sessions12 import run_sessions_1_2
            res = run_sessions_1_2(
                input_path=input_path,
                phase1_output=str(p1_out),
                phase2_output=str(p2_out),
                schema_path=schema_path,
                column_map_path=column_map_path,
                calendar_config=calendar_config,
                dimensions_config=dimensions_config,
                scoring_config=scoring_config,
            )
            ran_ok = True
            phase1_parquet_path = res.get("phase1_parquet")
        except Exception as ex:
            error = str(ex)

    # Collect statuses regardless of run result
    status = _collect_status(p2_out)

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase1_parquet": phase1_parquet_path or str((p1_out / "phase1_validated.parquet").resolve()),
        "phase2_output": str(p2_out.resolve()),
        "ran_successfully": ran_ok,
        "error": error,
        "artifacts": status,
    }

    # Write a summary file under runs/
    runs_dir = Path("runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = runs_dir / f"dryrun_sessions12_{stamp}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dry-run Sessions 1 and 2 and report artifact statuses")
    p.add_argument("--input", help="Input file or directory for Phase 1", default=None)
    p.add_argument("--phase1_output", help="Output directory for Phase 1 artifacts", default="data/processed/phase1")
    p.add_argument("--phase2_output", help="Output directory for Phase 2 artifacts", default="data/processed/scoring")
    p.add_argument("--schema", help="Path to Phase 1 schema YAML", default="config/schema_phase1.yaml")
    p.add_argument("--column_map", help="Path to Phase 1 column map YAML", default="config/phase1_column_map.yaml")
    p.add_argument("--calendar", help="Path to Phase 1 calendar config YAML", default="config/phase1_calendar.yaml")
    p.add_argument("--dimensions", help="Path to dimension_paths.yaml for Phase 2", default="config/dimension_paths.yaml")
    p.add_argument("--scoring_config", help="Path to scoring_config.yaml for Phase 2", default="config/scoring_config.yaml")
    p.add_argument("--json", help="Print JSON to stdout", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    res = dry_run_sessions_1_2(
        input_path=args.input,
        phase1_output=args.phase1_output,
        phase2_output=args.phase2_output,
        schema_path=args.schema,
        column_map_path=args.column_map,
        calendar_config=args.calendar,
        dimensions_config=args.dimensions,
        scoring_config=args.scoring_config,
    )
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        # Human-readable summary
        print("Brightstar Sessions 1+2 Dry-Run Summary:\n")
        print(f"  Phase 1 parquet: {res['phase1_parquet']}")
        print(f"  Phase 2 output:  {res['phase2_output']}")
        print(f"  Ran successfully: {res['ran_successfully']}")
        if res.get("error"):
            print(f"  Error: {res['error']}")
        print("\nArtifacts:")
        for k, info in res["artifacts"].items():
            status = "OK" if info["exists"] else "MISSING"
            size = info["size_bytes"] if info["size_bytes"] is not None else "-"
            print(f"  - {k:24s} {status:8s} {info['path']} (size={size})")


if __name__ == "__main__":
    main()
