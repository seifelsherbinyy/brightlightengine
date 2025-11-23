"""Unified logging utilities for BrightStar (Topic B).

This module centralizes logging setup and timing helpers so all phases can emit:
  - system-readable logs (system.log)
  - user-readable logs (user_readable.log)
  - timing breakdowns (timing.log)

Design constraints:
  - No imports of phase modules to avoid circular dependencies.
  - Windows-safe filesystem interactions using ``Path.resolve()``.
  - Graceful degradation: if file handlers fail, keep console logging.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict


SYSTEM_FMT = "[%(asctime)s] [%(levelname)s] %(message)s"
HUMAN_FMT = "%(message)s"


def _ensure_logs_dir(config: dict) -> Path:
    paths = (config or {}).get("paths", {})
    logs_dir = Path(paths.get("logs_dir", "logs")).expanduser().resolve()
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def _safe_add_file_handler(logger: logging.Logger, path: Path, fmt: str, level: int) -> None:
    try:
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    except Exception as exc:  # pragma: no cover - defensive on FS issues
        # Fall back silently to console-only; also emit to console
        logger.warning("[WARNING] Failed to attach file handler %s (%s)", str(path), exc)


def get_logger(name: str, config: dict) -> logging.Logger:
    """Return a system logger with console + file handlers.

    - File: system.log (machine-friendly format)
    - Console: same format
    - Level: INFO by default
    """
    logs_dir = _ensure_logs_dir(config)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Reset handlers to avoid duplication across repeated initializations
    logger.handlers = []

    # Console handler
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(SYSTEM_FMT))
    logger.addHandler(sh)

    # File handler
    _safe_add_file_handler(logger, logs_dir / "system.log", SYSTEM_FMT, logging.INFO)
    return logger


def get_user_logger(config: dict) -> logging.Logger:
    """Return a user-friendly logger that writes to user_readable.log and console.

    The format intentionally avoids stack traces unless DEBUG level is set by caller.
    """
    logs_dir = _ensure_logs_dir(config)
    logger = logging.getLogger("brightstar.user")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # Console handler (human-readable)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter(HUMAN_FMT))
    logger.addHandler(sh)

    # File handler
    _safe_add_file_handler(logger, logs_dir / "user_readable.log", HUMAN_FMT, logging.INFO)
    return logger


def start_phase_timer(phase_name: str) -> float:
    """Start a timer for a given phase/topic and return the perf counter."""
    return time.perf_counter()


def end_phase_timer(phase_name: str, start_time: float, timing_dict: Dict[str, float], user_logger: logging.Logger) -> None:
    """End timer, record to ``timing_dict`` and log to user logger."""
    try:
        elapsed = time.perf_counter() - float(start_time)
    except Exception:
        elapsed = 0.0
    timing_dict[phase_name] = float(elapsed)
    try:
        user_logger.info(f"Phase {phase_name} completed in {elapsed:.2f} seconds")
    except Exception:  # pragma: no cover - logging should never break pipeline
        pass


def write_timing_report(timing_dict: Dict[str, float], config: dict) -> Path | None:
    """Write a timing breakdown to logs/timing.log.

    The report lists phases and any topic timers captured by the caller.
    Returns the path to the written report, or None on error.
    """
    logs_dir = _ensure_logs_dir(config)
    out_path = logs_dir / "timing.log"
    try:
        lines = [
            "---- BRIGHTSTAR PIPELINE TIMING REPORT ----",
        ]
        # Phases 1–6 (if present)
        phase_keys = [
            "Phase 1 (Ingestion)",
            "Phase 2 (Scoring)",
            "Phase 3 (Commentary)",
            "Phase 4 (Forecasting)",
            "Phase 5 (Dashboard)",
            "Phase 6 (Evaluation)",
        ]
        total = 0.0
        for key in phase_keys:
            if key in timing_dict:
                val = float(timing_dict.get(key, 0.0))
                total += val
                lines.append(f"{key}: {val:.2f} seconds")
        lines.append(f"Total Duration: {total:.2f} seconds")

        # Enhancement Topics (optional)
        topic_keys = [
            "Topic 1 (Metric Registry Ops)",
            "Topic 2 (Dashboard Formatting Ops)",
            "Topic 3 (Periodic Export Ops)",
            "Topic 4 (Forecast/Evaluation Ops)",
            "Topic 4 (Forecast Ops)",
            "Topic 4 (Evaluation Ops)",
        ]
        for key in topic_keys:
            if key in timing_dict:
                lines.append(f"{key}: {float(timing_dict[key]):.2f} seconds")

        out_path.write_text("\n".join(lines), encoding="utf-8")
        return out_path
    except Exception as exc:  # pragma: no cover - defensive on FS issues
        # As a fallback, emit to console
        logging.getLogger("brightstar").warning("[WARNING] Failed to write timing report (%s): %s", str(out_path), exc)
        return None


def log_system_event(logger: logging.Logger, message: str):
    logger.info("[SYSTEM] %s", message)


def log_warning(logger: logging.Logger, message: str):
    logger.warning("[WARNING] %s", message)


def log_error(logger: logging.Logger, message: str):
    logger.error("[ERROR] %s", message)


# --- Simple in-memory topic timing recorder (for cross-phase reporting) ---
_TOPIC_TIMINGS: Dict[str, float] = {}


def record_topic_timing(topic_name: str, seconds: float) -> None:
    """Record timing for an enhancement topic in a module-level dict.

    This allows phases to report their own sub-operations; the pipeline can then
    merge these timings into the final report using :func:`get_topic_timings`.
    """
    try:
        val = float(seconds)
    except Exception:
        val = 0.0
    _TOPIC_TIMINGS[topic_name] = _TOPIC_TIMINGS.get(topic_name, 0.0) + val


def get_topic_timings() -> Dict[str, float]:
    """Return a shallow copy of recorded topic timings."""
    return dict(_TOPIC_TIMINGS)
