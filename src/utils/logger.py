from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


_LOG_PATH: Optional[Path] = None


def set_log_path(path: str | os.PathLike[str]) -> None:
    """Set the default JSON log file path used by append_log and helpers.

    The log format is JSON lines (one object per line) for easy appendability.
    """

    global _LOG_PATH
    _LOG_PATH = Path(path)
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _resolve_log_path() -> Optional[Path]:
    global _LOG_PATH
    if _LOG_PATH is not None:
        return _LOG_PATH
    # Environment override for ad-hoc usage
    env = os.environ.get("BRIGHTSTAR_LOG_PATH")
    if env:
        p = Path(env)
        p.parent.mkdir(parents=True, exist_ok=True)
        _set_if_none(p)
        return p
    return None


def _set_if_none(p: Path) -> None:
    global _LOG_PATH
    if _LOG_PATH is None:
        _LOG_PATH = p


def append_log(phase: str, message: str, severity: str = "INFO", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Append a JSON object to the configured log file.

    - Non-blocking best-effort: failures are swallowed after constructing the entry.
    - Returns the entry dict for further processing or testing.
    """

    entry: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "phase": phase,
        "severity": severity.upper(),
        "message": message,
    }
    if extra:
        entry.update({"extra": extra})

    path = _resolve_log_path()
    if path is not None:
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            # Best-effort: ignore logging errors
            pass
    return entry


def log_cleanup_activity(moved_files: Iterable[str], archive_path: str) -> Dict[str, Any]:
    """Log a summary of cleanup archival activity.

    Returns the entry dict.
    """

    moved = list(moved_files or [])
    return append_log(
        phase="phase1",
        message="Cleanup archival completed",
        severity="INFO",
        extra={
            "archive_path": archive_path,
            "files_archived": len(moved),
            "sample": moved[:25],
        },
    )


def log_validation_summary(check_results: Dict[str, Any]) -> Dict[str, Any]:
    """Log a connectivity/validation summary for Phase 1 modules and configs."""

    return append_log(
        phase="phase1",
        message="Connectivity validation completed",
        severity=("ERROR" if not check_results.get("all_passed", False) else "INFO"),
        extra=check_results,
    )
