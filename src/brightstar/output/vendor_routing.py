"""SEIF vendor output routing.

Creates a vendor-partitioned directory structure under the external base path
and writes per-vendor/per-ASIN outputs plus metadata and an updated history index.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .phase4_outputs import write_metadata_json
from ..output.phase4_outputs import _extract_run_tag  # reuse helper


EXTERNAL_BASE_DEFAULT = Path(r"C:\Users\selsherb\Documents\AVS-E\CL\Seif Vendors")


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_history_index(base: Path, run_tag: str, vendors: List[str]) -> Path:
    idx = base / "history_index.json"
    history = []
    if idx.exists():
        try:
            history = json.loads(idx.read_text(encoding="utf-8"))
        except Exception:
            history = []
    history.append({"run": run_tag, "vendors": vendors})
    idx.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return idx


def write_vendor_partitioned_outputs(
    metrics_long: pd.DataFrame,
    metrics_wide: pd.DataFrame,
    scores_asin: pd.DataFrame,
    scores_vendor: pd.DataFrame,
    run_timestamp: str,
    external_base: Path | str = EXTERNAL_BASE_DEFAULT,
    excel_path: Optional[Path] = None,
    run_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, Path]]:
    """Write vendor-partitioned outputs into the SEIF directory structure.

    Returns a mapping of vendor_code -> dict of written paths (root, asin, vendor, machine, excel, metadata).
    """
    base = Path(external_base)
    base.mkdir(parents=True, exist_ok=True)

    if scores_vendor is None or scores_vendor.empty:
        # Fall back to any vendor codes in frames
        vendors = []
        for df in (metrics_wide, scores_asin, metrics_long):
            if df is not None and not df.empty and "vendor_code" in df.columns:
                vendors.extend(df["vendor_code"].dropna().unique().tolist())
        vendors = sorted(list({v for v in vendors}))
    else:
        vendors = sorted(scores_vendor["vendor_code"].dropna().unique().tolist())

    out: Dict[str, Dict[str, Path]] = {}
    for v in vendors:
        vroot = base / v / run_timestamp
        asin_dir = _ensure_dir(vroot / "asin")
        vendor_dir = _ensure_dir(vroot / "vendor")
        machine_dir = _ensure_dir(vroot / "machine")
        excel_dir = _ensure_dir(vroot / "excel")
        meta_dir = _ensure_dir(vroot / "metadata")

        # Filter and write
        if metrics_wide is not None and not metrics_wide.empty:
            mwv = metrics_wide[metrics_wide.get("vendor_code") == v] if "vendor_code" in metrics_wide.columns else metrics_wide
            if not mwv.empty:
                mwv.to_parquet(machine_dir / "metrics_wide.parquet", index=False)
        if scores_asin is not None and not scores_asin.empty:
            sav = scores_asin[scores_asin.get("vendor_code") == v] if "vendor_code" in scores_asin.columns else scores_asin
            if not sav.empty:
                sav.to_parquet(asin_dir / "scores_asin.parquet", index=False)
        if scores_vendor is not None and not scores_vendor.empty:
            svv = scores_vendor[scores_vendor.get("vendor_code") == v] if "vendor_code" in scores_vendor.columns else scores_vendor
            if not svv.empty:
                svv.to_parquet(vendor_dir / "scores_vendor.parquet", index=False)
        if excel_path and Path(excel_path).exists():
            # Copy workbook per vendor for convenience (could be single global file otherwise)
            import shutil
            shutil.copy2(excel_path, excel_dir / Path(excel_path).name)

        # Metadata
        meta = dict(run_metadata or {})
        meta["vendor_code"] = v
        write_metadata_json(meta_dir / "metadata.json", meta)

        out[v] = {
            "root": vroot,
            "asin": asin_dir,
            "vendor": vendor_dir,
            "machine": machine_dir,
            "excel": excel_dir,
            "metadata": meta_dir,
        }

    # Update cross-vendor history index
    _write_history_index(base, run_timestamp, vendors)
    return out
