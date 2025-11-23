"""Full pipeline orchestrator for Phases 1 → 4.

Responsibilities:
- Optionally run Phase 1 (ingestion)
- Run Phase 2 enhanced scoring (Patches 1–3)
- Pre-Phase-4 sanity check (Patch 4a)
- Build timestamped Phase-4 run directory and write machine + Excel outputs
- Persist metadata.json with run details

This module is designed for programmatic use and simple CLI.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ..ingestion_utils import load_config
from ..phase1_ingestion import run_phase1
from ..phase2_scoring_enhanced import run_phase2, build_wide_composite
from ..phase3_commentary import run_phase3
from ..common.config_validator import load_and_validate_config
from ..output.phase4_outputs import (
    build_phase4_run_dir,
    collect_git_commit_hash,
    write_machine_outputs,
    write_excel_outputs,
    write_metadata_json,
    schema_checksum,
)
from ..utils.pipeline_phase_sanity_check import run_sanity_check
from ..output.vendor_routing import write_vendor_partitioned_outputs
from ..standards.schemas import validate_phase_output
from ..standards.run_clean import mark_success, mark_failed


def _dummy_logger():
    class _L:
        def info(self, *args, **kwargs):
            return None
        def warning(self, *args, **kwargs):
            return None
        def error(self, *args, **kwargs):
            return None
    return _L()


def run_all_phases(
    config_path: str = "config.yaml",
    skip_phase1: bool = False,
    skip_sanity: bool = False,
    skip_phase3: bool = False,
) -> Dict[str, object]:
    config = load_config(config_path)
    config["_config_path"] = config_path  # hint for downstream calls

    # Phase 1
    if not skip_phase1:
        run_phase1(config_path=config_path)

    # Phase 2 (enhanced)
    t0 = datetime.now(timezone.utc)
    p2 = run_phase2(config_path=config_path)
    t2 = datetime.now(timezone.utc)
    # Prefer long standardized frame for sanity and potential rebuilds
    df_long: pd.DataFrame = p2.get("long", pd.DataFrame())
    df_wide_from_p2: pd.DataFrame = p2.get("matrix", pd.DataFrame())

    # Backward compatibility: some callers return only 'matrix' which may be long-shaped
    if df_long.empty and not df_wide_from_p2.empty:
        # Heuristics: if it has 'metric' and 'std_value' it is likely long
        if {"entity_id", "metric", "std_value", "Week_Order"}.issubset(df_wide_from_p2.columns):
            df_long = df_wide_from_p2.copy()
            df_wide_from_p2 = pd.DataFrame()
    df_vendor: pd.DataFrame = p2.get("scoreboard", pd.DataFrame())

    if df_long.empty:
        raise RuntimeError("Phase 2 returned empty long matrix; cannot proceed to Phase 4 outputs.")

    # Phase 3 must run unless explicitly skipped
    t2_end = t2
    if not skip_phase3:
        p3_df = run_phase3(config_path=config_path)
        t3 = datetime.now(timezone.utc)
        t2_end = t3
    
    # Prepare scoring config (needed for sanity and wide build)
    scoring_cfg = load_and_validate_config(config)

    # Sanity check (gate)
    if not skip_sanity:
        ready = run_sanity_check(config, df_long=df_long, metrics_cfg=scoring_cfg.metrics)
        if not ready.ok:
            raise RuntimeError(f"Sanity check failed: {ready.errors}")

    # Build wide composite (for contributions and composite score), unless Phase 2 already produced it
    # Only accept p2['matrix'] as wide if it contains wide audit columns
    is_p2_wide = False
    if not df_wide_from_p2.empty:
        cols = list(df_wide_from_p2.columns)
        is_p2_wide = any(c.startswith("std_") for c in cols) or ("score_composite" in cols)
    if is_p2_wide:
        df_wide = df_wide_from_p2
    else:
        df_wide = build_wide_composite(df_long, scoring_cfg.metrics, scoring_cfg.weights_must_sum_to_1, logger=_dummy_logger())

    # Construct asin-level scores as latest composite per entity_id
    if not df_wide.empty:
        latest = df_wide.sort_values(["entity_id", "Week_Order"]).groupby("entity_id", as_index=False).tail(1)
        df_asin_scores = latest[[c for c in latest.columns if c in {"entity_id", "vendor_code", "Week_Order", "score_composite"}]].copy()
    else:
        df_asin_scores = pd.DataFrame(columns=["entity_id", "vendor_code", "Week_Order", "score_composite"])

    # Signal quality snapshot (if present)
    sig_cols = [c for c in ["entity_id", "metric", "Week_Order", "completeness", "recency_weeks", "variance", "trend_quality"] if c in df_long.columns]
    df_signal = df_long[sig_cols].copy() if sig_cols else pd.DataFrame()

    # Phase 4 run directories
    run_dirs = build_phase4_run_dir(Path("outputs") / "phase4")

    # Validate wide schema (basic) before writing
    try:
        validate_phase_output(df_wide, phase="phase4_wide")
    except Exception as ex:
        # Continue; write_machine_outputs also validates and will raise hard if needed
        pass

    # Machine-readable outputs
    machine_paths = write_machine_outputs(
        run_dirs,
        long_df=df_long,
        wide_df=df_wide,
        asin_scores=df_asin_scores,
        vendor_scores=df_vendor,
        signal_quality_df=df_signal,
    )

    # Excel outputs
    taxonomy_df = pd.DataFrame()  # optional; could be loaded from taxonomy engine if available
    excel_path = None
    try:
        excel_path = write_excel_outputs(
            run_dirs,
            vendor_scores=df_vendor,
            asin_long=df_wide if not df_wide.empty else df_long,
            taxonomy_df=taxonomy_df,
            anomalies_df=None,
        )
    except Exception as ex:
        # Do not abort the run; record in metadata later
        excel_error = str(ex)
        excel_path = None
    else:
        excel_error = None

    # Metadata
    t_end = datetime.now(timezone.utc)
    timestamp = t_end.isoformat()
    git_hash = collect_git_commit_hash(Path(__file__).resolve().parents[3])
    metrics_processed = list({m.name for m in scoring_cfg.metrics})
    # Dynamic weight settings observed (bounds) and config-derived context
    mul_series = df_long["weight_multiplier"] if "weight_multiplier" in df_long.columns else pd.Series(dtype=float)
    dyn_weights_meta = {
        "lookback_weeks": int(scoring_cfg.rolling_weeks),
        "min_multiplier_observed": float(mul_series.min()) if not mul_series.empty else None,
        "max_multiplier_observed": float(mul_series.max()) if not mul_series.empty else None,
    }
    trend_meta = {
        "window_weeks": int(scoring_cfg.rolling_weeks),
        "slope_method": "ols_rolling",
    }
    outlier_meta = {
        "method": "winsorize",
        "per_metric_clip_quantiles": {m.name: list(m.clip_quantiles) for m in scoring_cfg.metrics},
    }
    # Orchestrator args
    args_meta = {
        "skip_phase1": bool(skip_phase1),
        "skip_phase3": bool(skip_phase3),
        "skip_sanity": bool(skip_sanity),
        "config_path": str(config_path),
    }
    # Phase execution times
    exec_meta = {
        "phase2_started": t0.isoformat(),
        "phase2_finished": t2.isoformat(),
        "phase3_finished": t2_end.isoformat() if not skip_phase3 else None,
        "phase4_finished": t_end.isoformat(),
        "durations_sec": {
            "phase2": (t2 - t0).total_seconds(),
            "phase3": (t2_end - t2).total_seconds() if not skip_phase3 else 0.0,
            "phase4": (t_end - t2_end).total_seconds(),
        },
    }
    meta = {
        "timestamp": timestamp,
        "git_commit": git_hash,
        "metrics_processed": metrics_processed,
        "entity_counts": {
            "vendors": int(df_vendor["vendor_code"].nunique()) if not df_vendor.empty else 0,
            "asins": int(df_long["entity_id"].nunique()) if "entity_id" in df_long.columns else 0,
        },
        "metric_counts": {
            "rows_long": int(len(df_long)),
            "rows_wide": int(len(df_wide)),
        },
        "outputs": {k: str(v) for k, v in machine_paths.items()} | {"excel": str(excel_path) if excel_path else None},
        "config_snapshot": scoring_cfg.model_dump() if hasattr(scoring_cfg, "model_dump") else {},
        "dynamic_weight_settings": dyn_weights_meta,
        "trend_config": trend_meta,
        "outlier_detection": outlier_meta,
        "orchestrator_args": args_meta,
        "execution_times": exec_meta,
        "schema_checksums": {
            "metrics_long": schema_checksum(df_long),
            "metrics_wide": schema_checksum(df_wide),
            "scores_vendor": schema_checksum(df_vendor) if not df_vendor.empty else None,
            "scores_asin": schema_checksum(df_asin_scores) if not df_asin_scores.empty else None,
        },
        "excel_error": excel_error,
    }
    write_metadata_json(run_dirs["run"] / "metadata.json", meta)

    # External SEIF vendor routing
    run_tag = run_dirs["run"].name
    external_base = Path(config.get("paths", {}).get("external_output_dir", r"C:\\Users\\selsherb\\Documents\\AVS-E\\CL\\Seif Vendors"))
    try:
        write_vendor_partitioned_outputs(
            metrics_long=df_long,
            metrics_wide=df_wide,
            scores_asin=df_asin_scores,
            scores_vendor=df_vendor,
            run_timestamp=run_tag,
            external_base=external_base,
            excel_path=Path(excel_path) if excel_path else None,
            run_metadata=meta,
        )
    except Exception as ex:
        # Record failure but do not abort
        meta["vendor_routing_error"] = str(ex)
        write_metadata_json(run_dirs["run"] / "metadata.json", meta)

    # Mark run status manifest
    if excel_error or meta.get("vendor_routing_error"):
        mark_failed(run_dirs["run"], errors=[e for e in [excel_error, meta.get("vendor_routing_error")] if e])
    else:
        mark_success(run_dirs["run"]) 

    return {"paths": {"run": str(run_dirs["run"])}, "metadata": meta}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run Brightstar Phases 1–4 and emit outputs")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument("--skip-phase1", action="store_true", help="Reuse existing Phase 1 outputs")
    p.add_argument("--skip-sanity", action="store_true", help="Skip pre-execution sanity check")
    p.add_argument("--skip-phase3", action="store_true", help="Skip Phase 3 scoring/commentary (not recommended)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_all_phases(
        config_path=args.config,
        skip_phase1=args.skip_phase1,
        skip_sanity=args.skip_sanity,
        skip_phase3=args.skip_phase3,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
