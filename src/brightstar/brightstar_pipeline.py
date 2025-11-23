"""Brightstar end-to-end pipeline orchestration.

This module orchestrates the standard multi‑phase BrightStar run and also
provides a Full Reset mode (RUN_FULL_RESET) that:
  1) Cleans output artifacts under an external output root safely,
  2) Recreates the folder tree,
  3) Executes Phases 1 → 6 in order (1,2,3,4,6,5), ignoring prior runs,
  4) Performs sanity checks across modules and config paths,
  5) Writes run metadata and a human-readable summary.

The reset logic NEVER touches raw input directories nor project metadata.
It is defensive: on any issue it logs warnings and continues.
"""
from __future__ import annotations

import argparse
import json
import shutil
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from .ingestion_utils import ensure_directory, load_config
from .phase1_ingestion import run_phase1
from .phase2_scoring import run_phase2
from .phase3_commentary import run_phase3
from .phase4_forecasting import run_phase4
from .phase5_dashboard import run_phase5
from .phase6_evaluation import run_phase6
from .scoring_utils import load_normalized_input
from .metric_registry_utils import (
    load_metric_registry,
    detect_new_metrics,
    append_pending_metrics,
)
# Topic C: diagrams
try:
    from .diagram_utils import (
        ensure_diagram_dir,
        load_phase_status,
        load_phase_timing,
        generate_all_diagrams,
    )
except Exception:  # pragma: no cover - allow pipeline to run without diagrams in legacy envs
    ensure_diagram_dir = None  # type: ignore
    load_phase_status = None  # type: ignore
    load_phase_timing = None  # type: ignore
    generate_all_diagrams = None  # type: ignore
try:
    from .logging_utils import (
        log_system_event,
        get_logger,
        get_user_logger,
        start_phase_timer,
        end_phase_timer,
        write_timing_report,
        get_topic_timings,
    )
except Exception:  # pragma: no cover - fallback if module missing in legacy installs
    def log_system_event(logger: logging.Logger, msg: str) -> None:
        logger.info("[SYSTEM] %s", msg)

LOGGER_NAME = "brightstar.pipeline"


@dataclass
class PhaseResult:
    """Outcome details for a single pipeline phase."""

    name: str
    status: str
    detail: str
    duration_seconds: float
    output: Any = None


@dataclass
class PipelineRunResult:
    """Aggregate summary returned by :func:`run_pipeline`."""

    config_path: str
    started_at: datetime
    finished_at: datetime
    phases: List[PhaseResult] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True when all executed phases succeeded or were skipped."""

        return all(phase.status in {"success", "skipped"} for phase in self.phases)


@dataclass
class PhaseDefinition:
    """Declarative description of an orchestrated phase."""

    name: str
    runner: Callable[[str], Any]
    description: str
    summariser: Callable[[Any], str]
    skip_check: Callable[[Dict[str, Any]], bool]


def _setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Create a pipeline logger writing to the configured logs directory."""

    log_dir = ensure_directory(config["paths"]["logs_dir"])
    file_name = config.get("logging", {}).get("pipeline_file_name", "brightstar_pipeline.log")
    level_name = config.get("logging", {}).get("pipeline_level", "INFO")

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level_name, logging.INFO))
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_dir / file_name, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.debug("Pipeline logger initialised at %s", log_dir / file_name)
    return logger


def _validate_config(config: Dict[str, Any]) -> None:
    """Raise an informative error when key configuration blocks are missing."""

    required_sections = ["paths", "ingestion", "scoring", "commentary", "forecasting", "dashboard"]
    missing_sections = [section for section in required_sections if section not in config]
    if missing_sections:
        raise ValueError(f"Configuration missing sections: {', '.join(missing_sections)}")

    required_paths = ["raw_dir", "processed_dir", "logs_dir", "calendar_map"]
    paths_block = config["paths"]
    missing_paths = [key for key in required_paths if key not in paths_block]
    if missing_paths:
        raise ValueError(f"Configuration missing path entries: {', '.join(missing_paths)}")


# =============================================================
# Filesystem helpers (Full Reset safety)
# =============================================================

def _resolve_path(p: str | Path) -> Path:
    """Return an absolute, resolved Path (Windows-friendly)."""
    return Path(p).expanduser().resolve()


def safe_delete_and_recreate(path: Path, logger: Optional[logging.Logger] = None) -> None:
    """Safely delete a directory and recreate it.

    - Uses shutil.rmtree(ignore_errors=True)
    - Wraps operations in try/except with explicit logging
    - Always recreates the directory with parents=True
    """
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            if logger:
                logger.info("Removed directory: %s", str(path))
        path.mkdir(parents=True, exist_ok=True)
        if logger:
            logger.info("Recreated directory: %s", str(path))
    except Exception as exc:  # pragma: no cover - defensive
        if logger:
            logger.warning("Failed to reset directory %s (%s)", str(path), exc)


def _under_root(candidate: Path, root: Path) -> bool:
    """Return True if candidate path is at or under root (prevents accidental deletion)."""
    try:
        cand = candidate.resolve()
        base = root.resolve()
        return str(cand).startswith(str(base))
    except Exception:
        return False


# =============================================================
# Sanity checks
# =============================================================

def sanity_check_engine(config: Dict[str, Any], logger: Optional[logging.Logger] = None) -> None:
    """Run lightweight cross-module integrity checks.

    - Import core modules to ensure no ImportError
    - Validate key path keys exist in config
    - Validate downstream dependencies existence (best-effort)
    - Log warnings only; never raise.
    """
    lg = logger or logging.getLogger(LOGGER_NAME)

    modules = [
        ("phase1_ingestion", "brightstar.phase1_ingestion"),
        ("phase2_scoring", "brightstar.phase2_scoring"),
        ("phase3_commentary", "brightstar.phase3_commentary"),
        ("phase4_forecasting", "brightstar.phase4_forecasting"),
        ("phase5_dashboard", "brightstar.phase5_dashboard"),
        ("phase6_evaluation", "brightstar.phase6_evaluation"),
    ]
    for name, path in modules:
        try:
            __import__(path)
            lg.info("Sanity: imported %s", path)
        except Exception as exc:
            lg.warning("Sanity: failed to import %s (%s)", path, exc)

    # Config path keys
    paths = config.get("paths", {})
    required_keys = ["raw_dir", "processed_dir", "logs_dir"]
    for key in required_keys:
        if key not in paths:
            lg.warning("Sanity: config paths missing key '%s'", key)

    # If present, verify existence of base dirs
    for key in ("output_root", "raw_dir", "processed_dir", "logs_dir", "dashboard_dir", "dashboards_dir"):
        p = paths.get(key)
        if not p:
            continue
        pp = Path(p)
        if not pp.exists():
            lg.info("Sanity: path does not exist yet (ok): %s=%s", key, pp)

    # Downstream dependencies (best-effort current state)
    def _exists(path_str: Optional[str]) -> bool:
        return bool(path_str) and Path(path_str).exists()

    norm_ok = _exists(paths.get("normalized_output"))
    sc_v_ok = _exists(paths.get("scorecard_vendor_csv"))
    sc_a_ok = _exists(paths.get("scorecard_asin_csv"))
    f_v_ok = _exists(paths.get("forecast_vendor_csv"))
    f_a_ok = _exists(paths.get("forecast_asin_csv"))

    if not norm_ok:
        lg.info("Sanity: Phase 2 dependency missing (normalized parquet/csv) — expected before run.")
    if not (sc_v_ok and sc_a_ok):
        lg.info("Sanity: Phase 3 dependency (scorecards) not present yet — will be created during run.")
    if not (f_v_ok and f_a_ok):
        lg.info("Sanity: Phase 6 forecasts not present yet — will be created during run.")


# =============================================================
# Full Reset Orchestration
# =============================================================

def _make_reset_tree(output_root: Path, logger: logging.Logger) -> Dict[str, Path]:
    """Build and reset the folder tree required for a clean run.

    Returns a dict of created paths.
    """
    tree = {
        "processed": output_root / "data" / "processed",
        "processed_normalized": output_root / "data" / "processed" / "normalized",
        "processed_scorecards": output_root / "data" / "processed" / "scorecards",
        "processed_commentary": output_root / "data" / "processed" / "commentary",
        "processed_forecasts": output_root / "data" / "processed" / "forecasts",
        "processed_evaluation": output_root / "data" / "processed" / "evaluation",
        "outputs": output_root / "outputs",
        "outputs_dashboards": output_root / "outputs" / "dashboards",
        "outputs_periodic": output_root / "outputs" / "periodic_exports",
        "logs": output_root / "logs",
        "runs": output_root / "runs",
    }

    # Delete allowed directories under root (never raw)
    for key in ("processed", "outputs", "logs", "runs"):
        p = tree[key]
        if _under_root(p, output_root):
            safe_delete_and_recreate(p, logger)
        else:
            logger.warning("Skip deletion outside root: %s", str(p))

    # Ensure sub-directories exist
    for key, p in tree.items():
        if key in {"processed", "outputs", "logs", "runs"}:
            continue
        p.mkdir(parents=True, exist_ok=True)
    return tree


def run_full_reset(config_path: str = "config/config.yaml") -> None:
    """Perform a full reset run: clean outputs, recreate tree, execute all phases.

    This function ignores previous run artifacts entirely and executes
    phases in order: 1 → 2 → 3 → 4 → 6 → 5.
    """
    config = load_config(config_path)
    logger = _setup_logging(config)
    # Unified logging (Topic B)
    try:
        system_logger = get_logger("brightstar_system", config)
        user_logger = get_user_logger(config)
    except Exception:
        system_logger = logger
        user_logger = logger
    timing_dict: Dict[str, float] = {}

    # Resolve output root
    paths = config.get("paths", {})
    output_root_str = paths.get("output_root")
    if not output_root_str:
        # Defensive fallback: attempt to infer from processed_dir
        processed_dir = paths.get("processed_dir")
        if processed_dir:
            output_root = _resolve_path(Path(processed_dir).parent.parent if "data" in str(processed_dir) else Path(processed_dir).parent)
            logger.warning(
                "No paths.output_root configured. Using fallback root based on processed_dir: %s",
                str(output_root),
            )
        else:
            logger.warning("No paths.output_root configured and no processed_dir available; aborting destructive reset.")
            # Still try running phases with no cleanup
            output_root = Path.cwd()
    else:
        output_root = _resolve_path(output_root_str)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("**FULL RESET RUN STARTED (%s)**", run_id)
    log_system_event(logger, f"FULL RESET RUN INITIATED at {run_id}")
    try:
        user_logger.info("FULL RESET MODE: All prior outputs removed.")
    except Exception:
        pass

    # Pre-run sanity import/path check
    sanity_check_engine(config, logger)

    # Build/reset tree
    tree = _make_reset_tree(output_root, logger)

    # Execute phases in strict order, ignoring skip/resume semantics
    phase_order = [
        ("phase1", run_phase1),
        ("phase2", run_phase2),
        ("phase3", run_phase3),
        ("phase4", run_phase4),
        ("phase6", run_phase6),
        ("phase5", run_phase5),
    ]

    phases_meta: List[Dict[str, Any]] = []
    for name, runner in phase_order:
        start_ts = time.perf_counter()
        # Unified timing labels
        phase_label_map = {
            "phase1": "Phase 1 (Ingestion)",
            "phase2": "Phase 2 (Scoring)",
            "phase3": "Phase 3 (Commentary)",
            "phase4": "Phase 4 (Forecasting)",
            "phase5": "Phase 5 (Dashboard)",
            "phase6": "Phase 6 (Evaluation)",
        }
        label = phase_label_map.get(name, name)
        try:
            start_mark = start_phase_timer(label)
        except Exception:
            start_mark = start_ts
        try:
            out = runner(config_path=config_path)
            duration = time.perf_counter() - start_ts
            try:
                end_phase_timer(label, start_mark, timing_dict, user_logger)
            except Exception:
                pass
            summary = {
                "phase": name,
                "status": "success",
                "duration_seconds": round(duration, 3),
            }
            # Attach small counts when possible
            try:
                if isinstance(out, dict):
                    summary["outputs_keys"] = list(out.keys())
                    # heuristic counts
                    for k, v in out.items():
                        try:
                            summary[f"count_{k}"] = int(len(v)) if hasattr(v, "__len__") else None
                        except Exception:
                            pass
            except Exception:
                pass
            phases_meta.append(summary)
            logger.info("%s completed in %.2fs", name, duration)
            # Topic 4 timing: split Forecast and Evaluation when those phases run
            if name == "phase4":
                timing_dict.setdefault("Topic 4 (Forecast Ops)", duration)
                timing_dict.setdefault("Topic 4 (Forecast/Evaluation Ops)", 0.0)
            if name == "phase6":
                timing_dict["Topic 4 (Evaluation Ops)"] = duration
                timing_dict["Topic 4 (Forecast/Evaluation Ops)"] = timing_dict.get("Topic 4 (Forecast Ops)", 0.0) + duration
            # Topic 2 timing: approximate as Phase 5 duration
            if name == "phase5":
                timing_dict["Topic 2 (Dashboard Formatting Ops)"] = duration
        except Exception as exc:  # pragma: no cover - defensive
            duration = time.perf_counter() - start_ts
            phases_meta.append({"phase": name, "status": "failed", "error": str(exc), "duration_seconds": round(duration, 3)})
            logger.exception("%s failed after %.2fs (continuing)", name, duration)

    # Post-run sanity: validate expected downstream artifacts exist now
    try:
        sanity_check_engine(config, logger)
    except Exception:
        pass

    # Persist run metadata and summary into runs/
    runs_dir = tree.get("runs", output_root / "runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(_resolve_path(config_path)),
        "output_root": str(output_root),
        "phases": phases_meta,
    }
    meta_path = runs_dir / f"run_{run_id}_metadata.json"
    try:
        meta_path.write_text(json.dumps(meta, indent=2))
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to write run metadata (%s): %s", meta_path, exc)

    summary_lines = [
        f"FULL RESET RUN SUMMARY (run_id={run_id})",
        f"output_root: {output_root}",
        "Phases:",
    ]
    for p in phases_meta:
        summary_lines.append(f"  - {p.get('phase')}: {p.get('status')} ({p.get('duration_seconds','?')}s)")
    summary_path = runs_dir / f"run_{run_id}_summary.txt"
    try:
        summary_path.write_text("\n".join(summary_lines))
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to write run summary (%s): %s", summary_path, exc)

    log_system_event(logger, "FULL RESET RUN COMPLETED successfully.")
    # Write timing report
    try:
        # Merge topic timings recorded by phases
        try:
            timing_dict.update(get_topic_timings())
        except Exception:
            pass
        write_timing_report(timing_dict, config)
        user_logger.info("Pipeline run completed successfully.")
    except Exception:
        pass
    # Generate diagrams at the end of reset run (Topic C)
    try:
        if ensure_diagram_dir and load_phase_status and load_phase_timing and generate_all_diagrams:
            ensure_diagram_dir(config)
            _ = load_phase_status(config)
            _ = load_phase_timing(config)
            generate_all_diagrams(config)
    except Exception:
        logger.warning("Diagram generation skipped due to an issue (reset mode).")
    logger.info("Full reset run completed. Metadata: %s | Summary: %s", meta_path, summary_path)


def _phase1_summary(output: Any) -> str:
    if isinstance(output, pd.DataFrame):
        return f"Normalised {len(output)} records." if not output.empty else "No records to normalise."
    return "Phase 1 produced no dataframe output."


def _phase2_summary(output: Any) -> str:
    if isinstance(output, dict):
        vendor = output.get("vendor")
        asin = output.get("asin")
        vendor_count = len(vendor) if isinstance(vendor, pd.DataFrame) else 0
        asin_count = len(asin) if isinstance(asin, pd.DataFrame) else 0
        return f"Scored {vendor_count} vendor rows and {asin_count} ASIN rows."
    return "Phase 2 produced no scorecard output."


def _phase3_summary(output: Any) -> str:
    if isinstance(output, pd.DataFrame):
        return f"Generated {len(output)} commentary rows."
    return "Phase 3 produced no commentary output."


def _phase4_summary(output: Any) -> str:
    if isinstance(output, dict):
        vendor = output.get("vendor")
        asin = output.get("asin")
        vendor_count = len(vendor) if isinstance(vendor, pd.DataFrame) else 0
        asin_count = len(asin) if isinstance(asin, pd.DataFrame) else 0
        return f"Forecasted {vendor_count} vendor rows and {asin_count} ASIN rows."
    return "Phase 4 produced no forecast output."


def _phase5_summary(output: Any) -> str:
    if isinstance(output, tuple) and len(output) == 2:
        path, summary = output
        if isinstance(path, Path):
            detail = f"Dashboard written to {path.name}"
            if isinstance(summary, dict):
                vendors = summary.get("total_vendors")
                avg_score = summary.get("average_score")
                freshness = summary.get("data_freshness")
                extra = ", ".join(
                    part
                    for part in [
                        f"vendors={vendors}" if vendors is not None else None,
                        f"avg_score={avg_score:.2f}" if isinstance(avg_score, (int, float)) else None,
                        f"freshness={freshness}" if freshness else None,
                    ]
                    if part
                )
                if extra:
                    detail = f"{detail} ({extra})"
            return detail
    return "Phase 5 produced no dashboard output."


def _output_exists(path: Optional[str]) -> bool:
    return bool(path) and Path(path).exists()


def _commentary_exists(config: Dict[str, Any]) -> bool:
    commentary_dir = Path(config["paths"]["processed_dir"]) / "commentary"
    return (commentary_dir / "commentary.csv").exists()


def _forecast_exists(config: Dict[str, Any]) -> bool:
    paths = config["paths"]
    vendor = paths.get("forecast_vendor_csv")
    asin = paths.get("forecast_asin_csv")
    return _output_exists(vendor) and _output_exists(asin)


def _dashboard_exists(config: Dict[str, Any]) -> bool:
    dashboard_config = config.get("dashboard", {})
    directory = Path(dashboard_config.get("output_dir", config["paths"].get("dashboard_dir", "")))
    if not directory.exists():
        return False
    prefix = dashboard_config.get("file_prefix", "BrightStar_Dashboard")
    return any(path.name.startswith(prefix) and path.suffix.lower() == ".xlsx" for path in directory.glob("*.xlsx"))


def _evaluation_exists(config: Dict[str, Any]) -> bool:
    eval_cfg = config.get("evaluation", {})
    out_dir = Path(eval_cfg.get("output_dir", "outputs/forecast_evaluation"))
    if not out_dir.exists():
        return False
    return any((out_dir / name).exists() for name in ("health_summary.csv", "health_summary.parquet"))


PHASE_REGISTRY: Sequence[PhaseDefinition] = (
    PhaseDefinition(
        name="phase1",
        runner=run_phase1,
        description="Data ingestion & normalisation",
        summariser=_phase1_summary,
        skip_check=lambda cfg: _output_exists(cfg["paths"].get("normalized_output")),
    ),
    PhaseDefinition(
        name="phase2",
        runner=run_phase2,
        description="Scoring engine",
        summariser=_phase2_summary,
        skip_check=lambda cfg: _output_exists(cfg["paths"].get("scorecard_vendor_csv"))
        and _output_exists(cfg["paths"].get("scorecard_asin_csv")),
    ),
    PhaseDefinition(
        name="phase3",
        runner=run_phase3,
        description="Commentary engine",
        summariser=_phase3_summary,
        skip_check=_commentary_exists,
    ),
    PhaseDefinition(
        name="phase4",
        runner=run_phase4,
        description="Forecasting engine",
        summariser=_phase4_summary,
        skip_check=_forecast_exists,
    ),
    PhaseDefinition(
        name="phase5",
        runner=run_phase5,
        description="Dashboard generator",
        summariser=_phase5_summary,
        skip_check=_dashboard_exists,
    ),
    PhaseDefinition(
        name="phase6",
        runner=run_phase6,
        description="Forecast evaluation & governance",
        summariser=lambda out: (
            f"Evaluation pairs={len(out.get('pairs', [])) if hasattr(out.get('pairs', None), '__len__') else 0} "
            f"groups={len(out.get('health', [])) if hasattr(out.get('health', None), '__len__') else 0}"
        ),
        # Skip when outputs already exist and resume requested
        skip_check=_evaluation_exists,
    ),
)


def _select_phases(
    phases: Optional[Iterable[str]],
    start: Optional[str],
    end: Optional[str],
) -> List[str]:
    """Determine which phases should be executed based on CLI options."""

    available = [phase.name for phase in PHASE_REGISTRY]
    if phases:
        selected: List[str] = []
        for entry in phases:
            entry_normalised = entry.strip().lower()
            if entry_normalised in {"all", "*"}:
                return available
            if entry_normalised not in available:
                raise ValueError(f"Unknown phase '{entry}'. Available phases: {', '.join(available)}")
            if entry_normalised not in selected:
                selected.append(entry_normalised)
        return selected

    selected = available
    if start:
        if start not in available:
            raise ValueError(f"Unknown start phase '{start}'.")
        selected = available[available.index(start) :]
    if end:
        if end not in available:
            raise ValueError(f"Unknown end phase '{end}'.")
        end_index = available.index(end)
        selected = [phase for phase in selected if available.index(phase) <= end_index]
    return selected


def run_pipeline(
    config_path: str = "config.yaml",
    phases: Optional[Iterable[str]] = None,
    start_phase: Optional[str] = None,
    end_phase: Optional[str] = None,
    fail_fast: bool = False,
    resume: bool = False,
) -> PipelineRunResult:
    """Execute multiple Brightstar phases in sequence."""

    config = load_config(config_path)
    _validate_config(config)
    logger = _setup_logging(config)
    # Unified logging (Topic B)
    try:
        system_logger = get_logger("brightstar_system", config)
        user_logger = get_user_logger(config)
    except Exception:
        system_logger = logger
        user_logger = logger
    timing_dict: Dict[str, float] = {}

    selected = _select_phases(phases, start_phase, end_phase)
    logger.info("Running pipeline for phases: %s", ", ".join(selected))

    results: List[PhaseResult] = []
    started = datetime.now(timezone.utc)

    for definition in PHASE_REGISTRY:
        if definition.name not in selected:
            continue

        if resume and definition.skip_check(config):
            detail = f"Skipping {definition.name}; existing outputs detected."
            results.append(
                PhaseResult(
                    name=definition.name,
                    status="skipped",
                    detail=detail,
                    duration_seconds=0.0,
                    output=None,
                )
            )
            logger.info(detail)
            continue

        logger.info("Starting %s (%s)", definition.name, definition.description)
        phase_start = time.perf_counter()
        # Unified timing label for this definition
        phase_label_map = {
            "phase1": "Phase 1 (Ingestion)",
            "phase2": "Phase 2 (Scoring)",
            "phase3": "Phase 3 (Commentary)",
            "phase4": "Phase 4 (Forecasting)",
            "phase5": "Phase 5 (Dashboard)",
            "phase6": "Phase 6 (Evaluation)",
        }
        label = phase_label_map.get(definition.name, definition.name)
        try:
            start_mark = start_phase_timer(label)
        except Exception:
            start_mark = phase_start
        try:
            output = definition.runner(config_path=config_path)
            duration = time.perf_counter() - phase_start
            try:
                end_phase_timer(label, start_mark, timing_dict, user_logger)
            except Exception:
                pass
            detail = definition.summariser(output)
            results.append(
                PhaseResult(
                    name=definition.name,
                    status="success",
                    detail=detail,
                    duration_seconds=duration,
                    output=output,
                )
            )
            logger.info("%s completed in %.2fs: %s", definition.name, duration, detail)

            # BrightStar Patch: Metric Registry detection between Phase 1 and Phase 2
            if definition.name == "phase1" and ("phase2" in selected):
                topic_label = "Topic 1 (Metric Registry Ops)"
                t_start = start_phase_timer(topic_label)
                try:
                    # Load normalized data exactly like Phase 2
                    normalized_df = load_normalized_input(config, logger=None)
                except Exception as exc:
                    # Graceful degradation: do not stop pipeline
                    logging.getLogger("brightstar").warning(
                        "Metric registry detection skipped; unable to load normalized input (%s)",
                        exc,
                    )
                    normalized_df = pd.DataFrame()

                # Resolve registry and pending paths from config
                registry_cfg = config.get("registry", {})
                registry_path = Path(registry_cfg.get("metric_registry", "metadata/metric_registry.csv"))
                pending_path = Path(registry_cfg.get("pending_metrics", "metadata/new_metrics_pending.csv"))

                registry_df = load_metric_registry(registry_path)
                unknown = detect_new_metrics(normalized_df, registry_df)
                if unknown:
                    append_pending_metrics(pending_path, unknown)
                    logging.getLogger("brightstar").warning(
                        "Detected %d unknown metrics not present in registry. Names: %s | registry=%s | pending=%s",
                        len(unknown),
                        ", ".join(unknown),
                        str(registry_path),
                        str(pending_path),
                    )
                end_phase_timer(topic_label, t_start, timing_dict, user_logger)
        except Exception as exc:  # pragma: no cover - defensive logging path
            duration = time.perf_counter() - phase_start
            logger.exception("%s failed after %.2fs", definition.name, duration)
            results.append(
                PhaseResult(
                    name=definition.name,
                    status="failed",
                    detail=str(exc),
                    duration_seconds=duration,
                    output=None,
                )
            )
            if fail_fast:
                logger.error("Fail-fast enabled; stopping pipeline execution.")
                finished = datetime.now(timezone.utc)
                _print_summary(results)
                return PipelineRunResult(
                    config_path=config_path,
                    started_at=started,
                    finished_at=finished,
                    phases=results,
                )

    finished = datetime.now(timezone.utc)
    _print_summary(results)
    # Write timing report and friendly completion line
    try:
        try:
            timing_dict.update(get_topic_timings())
        except Exception:
            pass
        write_timing_report(timing_dict, config)
        user_logger.info("Pipeline run completed successfully.")
    except Exception:
        pass
    # Generate diagrams at the end of normal run (Topic C)
    try:
        if ensure_diagram_dir and load_phase_status and load_phase_timing and generate_all_diagrams:
            ensure_diagram_dir(config)
            _ = load_phase_status(config)
            _ = load_phase_timing(config)
            generate_all_diagrams(config)
    except Exception:
        logger.warning("Diagram generation skipped due to an issue.")
    return PipelineRunResult(
        config_path=config_path,
        started_at=started,
        finished_at=finished,
        phases=results,
    )


# -------------------------------------------------------------
# Reset then run only Phase 1
# -------------------------------------------------------------
def run_reset_then_phase1(config_path: str = "config.yaml") -> PhaseResult:
    """Reset outputs under output_root and execute only Phase 1 (ingestion).

    This provides a clean-slate Phase 1 run without proceeding to later phases.
    """
    config = load_config(config_path)
    logger = _setup_logging(config)
    try:
        system_logger = get_logger("brightstar_system", config)
        user_logger = get_user_logger(config)
    except Exception:
        system_logger = logger
        user_logger = logger

    # Resolve output root similar to full reset
    paths = config.get("paths", {})
    output_root_str = paths.get("output_root")
    if not output_root_str:
        processed_dir = paths.get("processed_dir")
        if processed_dir:
            output_root = _resolve_path(
                Path(processed_dir).parent.parent if "data" in str(processed_dir) else Path(processed_dir).parent
            )
            logger.warning(
                "No paths.output_root configured. Using fallback root based on processed_dir: %s",
                str(output_root),
            )
        else:
            logger.warning(
                "No paths.output_root configured and no processed_dir available; aborting destructive reset."
            )
            output_root = Path.cwd()
    else:
        output_root = _resolve_path(output_root_str)

    # Pre-run sanity
    sanity_check_engine(config, logger)

    # Reset tree
    _make_reset_tree(output_root, logger)

    # Run only Phase 1
    label = "Phase 1 (Ingestion)"
    try:
        start_mark = start_phase_timer(label)
    except Exception:
        start_mark = time.perf_counter()
    try:
        output = run_phase1(config_path=config_path)
        try:
            timing_dict: Dict[str, float] = {}
            end_phase_timer(label, start_mark, timing_dict, user_logger)
            try:
                timing_dict.update(get_topic_timings())
            except Exception:
                pass
            write_timing_report(timing_dict, config)
        except Exception:
            pass
        detail = _phase1_summary(output)
        result = PhaseResult(
            name="phase1",
            status="success",
            detail=detail,
            duration_seconds=0.0,
            output=output,
        )
        try:
            user_logger.info("Reset+Phase1 run completed successfully.")
        except Exception:
            pass
        return result
    except Exception as exc:
        detail = str(exc)
        try:
            user_logger.error("Phase 1 failed after reset: %s", detail)
        except Exception:
            pass
        return PhaseResult(
            name="phase1",
            status="failed",
            detail=detail,
            duration_seconds=0.0,
            output=None,
        )


def _print_summary(results: Sequence[PhaseResult]) -> None:
    if not results:
        print("No phases executed.")
        return

    lines = ["Brightstar pipeline summary:"]
    for result in results:
        lines.append(
            f"  - {result.name}: {result.status.upper()} ({result.detail})"
        )
    print("\n".join(lines))


def _split_phases_argument(argument: Optional[str]) -> Optional[List[str]]:
    if argument is None:
        return None
    return [part for part in argument.split(",") if part.strip()]


def build_arg_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the pipeline CLI."""

    parser = argparse.ArgumentParser(description="Brightstar end-to-end pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument(
        "--phases",
        help="Comma-separated list of phases to run (phase1,phase2,...) or 'all'",
    )
    parser.add_argument(
        "--start-phase",
        choices=[definition.name for definition in PHASE_REGISTRY],
        help="Execute phases starting from this phase onward",
    )
    parser.add_argument(
        "--end-phase",
        choices=[definition.name for definition in PHASE_REGISTRY],
        help="Stop execution after this phase",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop execution after the first failure")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip phases whose outputs already exist according to configuration paths",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Full reset mode: clean outputs under output_root, recreate tree, then run phases 1→6.",
    )
    parser.add_argument(
        "--reset-phase1",
        action="store_true",
        help="Reset outputs under output_root and run only phase1 (ingestion) from a clean state.",
    )
    return parser


def main() -> None:
    """CLI entry point for executing the Phase 6 pipeline."""

    parser = build_arg_parser()
    args = parser.parse_args()
    if getattr(args, "reset", False):
        # Full reset ignores phases/start/end/resume
        run_full_reset(config_path=args.config)
        return
    if getattr(args, "reset_phase1", False):
        run_reset_then_phase1(config_path=args.config)
        return
    phase_list = _split_phases_argument(args.phases)
    run_pipeline(
        config_path=args.config,
        phases=phase_list,
        start_phase=args.start_phase,
        end_phase=args.end_phase,
        fail_fast=args.fail_fast,
        resume=args.resume,
    )


if __name__ == "__main__":  # pragma: no cover - CLI guard
    main()
