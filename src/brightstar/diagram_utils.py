"""Diagram generation utilities for BrightStar (Topic C).

This module produces three offline diagrams using Graphviz (python-graphviz):
  1) Pipeline flow diagram (Phases 1→2→3→4→6→5) with status coloring and durations.
  2) Neural map diagram (conceptual data flow and optional branches).
  3) Run-status diagram (six colored nodes summarizing per-phase health).

Outputs are written to the configured diagrams directory under Brightlight,
typically: paths.diagrams_dir (defaults to "diagrams" when not configured).

All functions degrade gracefully: if Graphviz is unavailable or logs are
missing, they will log a warning and skip diagram creation rather than raising.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import logging

from .ingestion_utils import ensure_directory


LOGGER = logging.getLogger("brightstar.diagram")


def _resolve_logs_and_diagrams(config: dict) -> tuple[Path, Path]:
    paths = config.get("paths", {}) if isinstance(config, dict) else {}
    logs_dir = Path(paths.get("logs_dir", "logs")).expanduser().resolve()
    diagrams_dir = Path(paths.get("diagrams_dir", "diagrams")).expanduser().resolve()
    return logs_dir, diagrams_dir


def ensure_diagram_dir(config: dict) -> Path:
    """Ensure Brightlight/diagrams directory exists and return its path."""
    _, diagrams_dir = _resolve_logs_and_diagrams(config)
    ensure_directory(diagrams_dir)
    LOGGER.info("Diagrams directory ready: %s", diagrams_dir)
    return diagrams_dir


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def load_phase_status(config: dict) -> Dict[str, str]:
    """Infer per-phase status from logs.

    Heuristics (best-effort, non-fatal):
      - Prefer logs/system.log; fallback to pipeline log file if present.
      - success if a "completed" line for the phase is present.
      - failure if a line mentioning the phase also contains "failed" or "ERROR".
      - warning if a line mentioning the phase contains "WARNING".
      - if timing exists but no explicit markers, assume success.
    """
    logs_dir, _ = _resolve_logs_and_diagrams(config)
    system_log = logs_dir / "system.log"
    pipeline_log = logs_dir / (config.get("logging", {}).get("pipeline_file_name", "brightstar_pipeline.log"))
    timing = load_phase_timing(config)

    text = _read_text(system_log) or _read_text(pipeline_log)
    text_lower = text.lower()

    phase_labels = {
        "Phase 1": ["phase1", "phase 1", "ingestion"],
        "Phase 2": ["phase2", "phase 2", "scoring"],
        "Phase 3": ["phase3", "phase 3", "commentary"],
        "Phase 4": ["phase4", "phase 4", "forecast"],
        "Phase 5": ["phase5", "phase 5", "dashboard"],
        "Phase 6": ["phase6", "phase 6", "evaluation"],
    }

    results: Dict[str, str] = {}
    for phase_key, tokens in phase_labels.items():
        status: Optional[str] = None
        for token in tokens:
            if token in text_lower:
                # failure dominates
                if "failed" in text_lower or "error" in text_lower:
                    status = "failure"
                    break
                if "warning" in text_lower:
                    status = "warning"
                if "completed" in text_lower:
                    status = status or "success"
        if not status:
            # If we have timing for the phase, treat as success
            if phase_key in timing:
                status = "success"
            else:
                status = "warning"  # conservative default
        results[phase_key] = status

    return results


def load_phase_timing(config: dict) -> Dict[str, float]:
    """Parse logs/timing.log to extract phase durations in seconds.

    Expected lines like: "Phase 1 (Ingestion): 4.92 seconds".
    Returns a dict: {"Phase 1": 4.92, ...}
    """
    logs_dir, _ = _resolve_logs_and_diagrams(config)
    timing_log = logs_dir / "timing.log"
    text = _read_text(timing_log)
    out: Dict[str, float] = {}
    if not text:
        return out
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if not line.lower().startswith("phase "):
            continue
        # e.g., Phase 1 (Ingestion): 3.21 seconds
        try:
            left, right = line.split(":", 1)
            phase_num = left.split()[1]
            secs_str = right.strip().split()[0]
            secs = float(secs_str)
            out[f"Phase {phase_num}"] = secs
        except Exception:
            continue
    return out


def _status_color(status: str) -> str:
    s = (status or "").lower()
    if s.startswith("fail") or s == "failure":
        return "#F44336"  # red
    if s.startswith("warn") or s == "warning":
        return "#FFC107"  # amber
    return "#4CAF50"  # green (success/default)


def _safe_render(graph, out_path: Path) -> None:
    try:
        graph.render(filename=out_path.stem, directory=str(out_path.parent), format="png", cleanup=True)
        LOGGER.info("Diagram written: %s", out_path)
    except Exception as exc:  # pragma: no cover - environment specific graphviz availability
        LOGGER.warning("Diagram rendering skipped (%s): %s", out_path, exc)


def generate_pipeline_flow_diagram(status_dict: Dict[str, str], timing_dict: Dict[str, float], config: dict) -> Optional[Path]:
    """Generate a left-to-right pipeline flow with colored nodes and durations."""
    try:
        from graphviz import Digraph
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Graphviz unavailable; skipping pipeline flow diagram (%s)", exc)
        return None

    _, diagrams_dir = _resolve_logs_and_diagrams(config)
    ensure_directory(diagrams_dir)
    out_path = Path(diagrams_dir) / "brightstar_pipeline_flow.png"

    g = Digraph("BrightStarPipeline", format="png")
    g.attr(rankdir="LR", concentrate="true", splines="spline")

    phases = [
        ("Phase 1", "Ingestion"),
        ("Phase 2", "Scoring"),
        ("Phase 3", "Commentary"),
        ("Phase 4", "Forecasting"),
        ("Phase 6", "Evaluation"),
        ("Phase 5", "Dashboard"),
    ]

    for key, label in phases:
        status = status_dict.get(key, "success")
        dur = timing_dict.get(key, None)
        dur_str = f"\n{dur:.2f}s" if isinstance(dur, (int, float)) else ""
        g.node(key, f"{key}\n{label}{dur_str}", style="filled", fillcolor=_status_color(status), shape="rect")

    edges = [("Phase 1", "Phase 2"), ("Phase 2", "Phase 3"), ("Phase 2", "Phase 5"), ("Phase 2", "Phase 4"), ("Phase 4", "Phase 6"), ("Phase 6", "Phase 5")]
    for a, b in edges:
        g.edge(a, b)

    _safe_render(g, out_path)
    return out_path


def generate_neural_map_diagram(config: dict) -> Optional[Path]:
    """Generate a conceptual neural-map style diagram of BrightStar."""
    try:
        from graphviz import Digraph
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Graphviz unavailable; skipping neural map diagram (%s)", exc)
        return None

    _, diagrams_dir = _resolve_logs_and_diagrams(config)
    ensure_directory(diagrams_dir)
    out_path = Path(diagrams_dir) / "brightstar_neural_map.png"

    g = Digraph("BrightStarNeuralMap", format="png")
    g.attr(rankdir="LR")

    # Nodes
    g.node("Raw", "Raw Inputs", shape="cylinder")
    g.node("P1", "Phase 1\nIngestion", shape="box")
    g.node("Reg", "Metric Registry\n(Topic 1)", shape="component")
    g.node("P2", "Phase 2\nScoring", shape="box")
    g.node("Snap", "Periodic Snapshots\n(Topic 3)", shape="folder")
    g.node("Com", "Phase 3\nCommentary", shape="box")
    g.node("P4", "Phase 4\nForecasting", shape="box")
    g.node("P6", "Phase 6\nEvaluation", shape="box")
    g.node("P5", "Phase 5\nDashboard", shape="box")

    # Edges (conceptual neural connections)
    g.edge("Raw", "P1")
    g.edge("P1", "Reg")
    g.edge("Reg", "P2")
    g.edge("P2", "Com")
    g.edge("P2", "Snap", style="dashed")
    g.edge("P2", "P5", style="dotted")
    g.edge("P2", "P4")
    g.edge("P4", "P6")
    g.edge("P6", "P5")

    _safe_render(g, out_path)
    return out_path


def generate_run_status_diagram(status_dict: Dict[str, str], config: dict) -> Optional[Path]:
    """Generate a simplified run-status diagram with colored phase nodes."""
    try:
        from graphviz import Digraph
    except Exception as exc:  # pragma: no cover
        LOGGER.warning("Graphviz unavailable; skipping run status diagram (%s)", exc)
        return None

    _, diagrams_dir = _resolve_logs_and_diagrams(config)
    ensure_directory(diagrams_dir)
    out_path = Path(diagrams_dir) / "brightstar_run_status.png"

    g = Digraph("BrightStarRunStatus", format="png")
    g.attr(rankdir="LR")
    with g.subgraph(name="cluster_row1") as c1:
        for key in ("Phase 1", "Phase 2", "Phase 3"):
            c1.node(key, key, style="filled", fillcolor=_status_color(status_dict.get(key, "success")))
    with g.subgraph(name="cluster_row2") as c2:
        for key in ("Phase 4", "Phase 6", "Phase 5"):
            c2.node(key, key, style="filled", fillcolor=_status_color(status_dict.get(key, "success")))
    # progression hints
    g.edges([("Phase 1", "Phase 2"), ("Phase 2", "Phase 3"), ("Phase 3", "Phase 4"), ("Phase 4", "Phase 6"), ("Phase 6", "Phase 5")])

    _safe_render(g, out_path)
    return out_path


def generate_all_diagrams(config: dict) -> None:
    """Wrapper to generate all diagrams using current logs/timing."""
    ensure_diagram_dir(config)
    status = load_phase_status(config)
    timing = load_phase_timing(config)
    try:
        generate_pipeline_flow_diagram(status, timing, config)
        generate_neural_map_diagram(config)
        generate_run_status_diagram(status, config)
    except Exception as exc:  # pragma: no cover - diagrams are best-effort
        LOGGER.warning("Diagram generation encountered an issue and was partially skipped: %s", exc)
