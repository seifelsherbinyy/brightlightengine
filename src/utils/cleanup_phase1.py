from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

from .logger import set_log_path, append_log, log_cleanup_activity, log_validation_summary


OBSOLETE_PATTERNS = [
    r".*_old\.py$",
    r".*_backup\.parquet$",
    r".*\.(tmp|bak|old)$",
    r"^temp_.*\.csv$",
    r".*legacy.*\.(yaml|yml|json|csv|parquet)$",
]


def _is_obsolete(name: str) -> bool:
    for rx in OBSOLETE_PATTERNS:
        if re.search(rx, name, flags=re.IGNORECASE):
            return True
    return False


def _collect_candidates(root: Path) -> List[Path]:
    files: List[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file() and _is_obsolete(p.name):
            files.append(p)
    return files


def _should_retain(p: Path, project_root: Path) -> bool:
    # Retain active schema YAMLs under config
    try:
        rel = p.resolve().relative_to(project_root.resolve())
    except Exception:
        rel = p.name
    rel_str = str(rel).replace("/", "\\")
    if rel_str.lower().startswith("config\\"):
        # Keep active schema/config files by default
        return True
    # Do not move anything under data/reference per constraints
    if rel_str.lower().startswith("data\\reference\\"):
        return True
    # Retain metric_registry_utils.py
    if rel_str.lower().endswith("src\\brightstar\\metric_registry_utils.py"):
        return True
    # Retain validated outputs under data/processed/phase1
    if "data\\processed\\phase1" in rel_str.lower():
        return True
    # Retain active src/phase1 modules except temp
    if rel_str.lower().startswith("src\\phase1\\") and "\\temp\\" not in rel_str.lower():
        return True
    return False


def _move_to_archive(files: List[Path], archive_dir: Path, project_root: Path) -> Tuple[int, List[str]]:
    moved: List[str] = []
    for f in files:
        if _should_retain(f, project_root):
            continue
        # Preserve relative path segments under archive
        try:
            rel = f.resolve().relative_to(project_root.resolve())
        except Exception:
            rel = Path(f.name)
        dest = archive_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(f), str(dest))
            moved.append(str(rel))
        except Exception:
            # Non-blocking; continue
            pass
    return len(moved), moved


@dataclass
class ConnectivityResult:
    imports_ok: bool
    config_ok: bool
    artifacts_ok: bool
    details: Dict[str, Any]


def _check_imports() -> Tuple[bool, List[str]]:
    modules = [
        "src.phase1.loader",
        "src.phase1.dq_checks",
        "src.phase1.validation_report",
        "src.phase1.calendar_mapper",
        "src.phase1.id_normalizer",
    ]
    errors: List[str] = []
    ok = True
    for m in modules:
        try:
            __import__(m)
        except Exception as e:
            ok = False
            errors.append(f"import_failed:{m}:{type(e).__name__}:{e}")
    return ok, errors


def _check_configs(project_root: Path) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    needed = [
        project_root / "config" / "dimension_paths.yaml",
        project_root / "config" / "phase1_calendar.yaml",
    ]
    for p in needed:
        if not p.exists():
            missing.append(str(p))
    return len(missing) == 0, missing


def _check_artifacts(log_dir: Path) -> Tuple[bool, List[str]]:
    run_log = log_dir / "phase1_run_log.json"
    if not run_log.exists():
        return False, [f"missing_run_log:{run_log}"]
    try:
        data = json.loads(run_log.read_text(encoding="utf-8"))
        paths = list((data.get("outputs") or {}).values())
        missing = [p for p in paths if not Path(p).exists()]
        return len(missing) == 0, missing
    except Exception as e:
        return False, [f"run_log_parse_error:{type(e).__name__}:{e}"]


def run_cleanup(project_root: str, archive_dir: str, log_dir: str) -> Dict[str, Any]:
    project = Path(project_root)
    archive_root = Path(archive_dir)
    log_root = Path(log_dir)

    # Prepare logger file path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    archive_path = archive_root / timestamp
    archive_path.mkdir(parents=True, exist_ok=True)
    cleanup_log_path = log_root / "phase1_cleanup_log.json"
    cleanup_log_path.parent.mkdir(parents=True, exist_ok=True)
    set_log_path(str(cleanup_log_path))

    # Identify candidates in data root and phase1\temp
    data_root = project / "data"
    temp_root = project / "src" / "phase1" / "temp"
    candidates = _collect_candidates(data_root)
    candidates += _collect_candidates(temp_root)

    count, moved = _move_to_archive(candidates, archive_path, project)

    # Connectivity validation
    imports_ok, import_errors = _check_imports()
    config_ok, missing_configs = _check_configs(project)
    artifacts_ok, artifacts_missing = _check_artifacts(log_root)

    checks = {
        "imports_ok": imports_ok,
        "config_ok": config_ok,
        "artifacts_ok": artifacts_ok,
        "import_errors": import_errors,
        "missing_configs": missing_configs,
        "missing_artifacts": artifacts_missing,
    }
    checks["all_passed"] = all([imports_ok, config_ok, artifacts_ok])

    # Logging
    log_cleanup_activity(moved, str(archive_path.resolve()))
    log_validation_summary(checks)
    append_log(
        phase="phase1",
        message="Cleanup completed",
        severity="INFO",
        extra={
            "user": getpass.getuser(),
            "timestamp": datetime.now().isoformat(),
            "files_archived": count,
        },
    )

    summary = {
        "archived_count": count,
        "archive_path": str(archive_path.resolve()),
        "checks": checks,
        "log_file": str(cleanup_log_path.resolve()),
        "user": getpass.getuser(),
        "timestamp": datetime.now().isoformat(),
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Brightlight Phase 1 cleanup & connectivity validator")
    p.add_argument("--project_root", required=True, help="Path to project root (brightstar)")
    p.add_argument("--archive_dir", required=True, help="Directory where archive timestamp folders will be created")
    p.add_argument("--log_dir", required=True, help="Phase 1 output directory where cleanup log will be written")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    summary = run_cleanup(args.project_root, args.archive_dir, args.log_dir)
    # Print compact JSON to stdout
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
