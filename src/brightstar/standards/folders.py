"""Standard folder builders for internal and external outputs."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict


DEFAULT_EXTERNAL_BASE = Path(r"C:\Users\selsherb\Documents\AVS-E\CL\Seif Vendors")


def build_internal_run_dirs(base_dir: Path | str = "outputs/phase4") -> Dict[str, Path]:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = base / ts
    excel_dir = run_dir / "excel"
    machine_dir = run_dir / "machine"
    excel_dir.mkdir(parents=True, exist_ok=True)
    machine_dir.mkdir(parents=True, exist_ok=True)
    return {"run": run_dir, "excel": excel_dir, "machine": machine_dir}


def build_external_vendor_dir(external_base: Path | str, vendor_code: str, run_tag: str) -> Dict[str, Path]:
    base = Path(external_base)
    vendor_root = base / vendor_code / run_tag
    asin_dir = vendor_root / "asin"
    vendor_dir = vendor_root / "vendor"
    machine_dir = vendor_root / "machine"
    excel_dir = vendor_root / "excel"
    metadata_dir = vendor_root / "metadata"
    for p in (asin_dir, vendor_dir, machine_dir, excel_dir, metadata_dir):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "root": vendor_root,
        "asin": asin_dir,
        "vendor": vendor_dir,
        "machine": machine_dir,
        "excel": excel_dir,
        "metadata": metadata_dir,
    }
