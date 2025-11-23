from __future__ import annotations

from pathlib import Path


def get_phase1_normalized_path(config: dict) -> Path:
    """Return the canonical absolute path for the Phase 1 normalized output.

    Uses paths.output_root as the base and ensures the parent directory exists.
    """

    root = Path(config["paths"]["output_root"]) if "output_root" in config.get("paths", {}) else Path.cwd()
    out_path = root / "data" / "processed" / "normalized" / "normalized_phase1.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def cleanup_tmp_files(config: dict) -> None:
    """Remove stale temporary parquet files from the normalized output folder."""

    root = Path(config["paths"]["output_root"]) if "output_root" in config.get("paths", {}) else Path.cwd()
    norm_dir = root / "data" / "processed" / "normalized"
    if not norm_dir.exists():
        return
    for tmp in norm_dir.glob("*.tmp.parquet"):
        try:
            tmp.unlink()
        except Exception:
            # Best-effort cleanup; ignore failures
            pass
