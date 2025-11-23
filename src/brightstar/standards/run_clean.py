"""Run cleanliness utilities: manifests and safe metadata writing."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


def write_run_manifest(run_dir: Path, status: str, errors: Optional[List[str]] = None, extra: Optional[Dict[str, object]] = None) -> Path:
    run_dir = Path(run_dir)
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "errors": list(errors or []),
    }
    if extra:
        manifest.update(extra)
    out = run_dir / "run_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out


def mark_success(run_dir: Path, extra: Optional[Dict[str, object]] = None) -> Path:
    return write_run_manifest(run_dir, "SUCCESS", [], extra)


def mark_failed(run_dir: Path, errors: Optional[List[str]] = None, extra: Optional[Dict[str, object]] = None) -> Path:
    return write_run_manifest(run_dir, "FAILED", errors or [], extra)
