from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import yaml

from .commentary_utils import (
    detect_metric_movements,
    detect_dimension_shifts,
    detect_inventory_risks,
    detect_buyability_failures,
    detect_category_outliers,
)
from .filters import filter_high_impact_events, dedupe_related_signals, prioritize_by_materiality
from .formatters import compose_wbr_callout


REQUIRED_FIELDS = [
    "asin_id",
    "vendor_id",
    "canonical_week_id",
    "composite_score",
]


@dataclass
class Inputs:
    detailed: pd.DataFrame
    summary: pd.DataFrame
    readiness: pd.DataFrame
    metadata: Dict[str, Any]
    dim_asin: pd.DataFrame
    dim_vendor: pd.DataFrame


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_inputs(phase2_path: str, dimensions_path: str) -> Tuple[Inputs, Dict[str, str], List[str]]:
    warnings: List[str] = []
    p = Path(phase2_path)
    if not p.exists():
        raise FileNotFoundError(f"Phase 2 directory not found: {p}")

    detailed_path = p / "scoring_detailed.parquet"
    summary_path = p / "scoring_summary.parquet"
    readiness_path = p / "metric_readiness.parquet"
    meta_path = p / "scoring_metadata.json"

    if not detailed_path.exists() or not summary_path.exists() or not readiness_path.exists() or not meta_path.exists():
        missing = [str(x) for x in [detailed_path, summary_path, readiness_path, meta_path] if not x.exists()]
        raise FileNotFoundError(f"Missing Phase 2 artifacts: {missing}")

    detailed = pd.read_parquet(detailed_path)
    summary = pd.read_parquet(summary_path)
    readiness = pd.read_parquet(readiness_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # Validate required fields presence
    cols = set([str(c) for c in detailed.columns])
    for col in REQUIRED_FIELDS:
        if col not in cols:
            warnings.append(f"Missing required field in detailed: {col}")
    # Dimension scores and WoW deltas
    expected_dims = ["profit", "growth", "availability", "inventory_risk", "quality"]
    for d in expected_dims:
        if f"{d}_score" not in cols:
            warnings.append(f"Missing dimension score: {d}_score")
    for m in ["gms", "units", "gv", "instock_pct", "fo_pct", "returns_rate", "asp"]:
        colname = f"{m}_wow_pct"
        if colname not in cols:
            warnings.append(f"Missing WoW delta: {colname}")
    # Readiness flags presence
    rcols = set([str(c) for c in readiness.columns])
    if not any(c in rcols for c in ("readiness_flag", "is_ready", "completeness")):
        warnings.append("Readiness flags/diagnostics missing expected columns")

    # Load dimensions lookups
    dims_cfg = yaml.safe_load(Path(dimensions_path).read_text(encoding="utf-8"))
    def _p(name: str) -> Path:
        v = dims_cfg.get(name)
        return Path(v) if v else Path("nonexistent")

    dim_asin_path = _p("dim_asin")
    dim_vendor_path = _p("dim_vendor")
    dim_asin = pd.read_parquet(dim_asin_path) if dim_asin_path.exists() else pd.DataFrame()
    dim_vendor = pd.read_parquet(dim_vendor_path) if dim_vendor_path.exists() else pd.DataFrame()

    # Join if keys present
    out = detailed.copy()
    if not dim_asin.empty and "asin_id" in out.columns and "asin_id" in dim_asin.columns:
        out = out.merge(dim_asin, how="left", on="asin_id")
    if not dim_vendor.empty and "vendor_id" in out.columns and "vendor_id" in dim_vendor.columns:
        out = out.merge(dim_vendor, how="left", on="vendor_id")
    detailed = out

    input_hashes = {
        "scoring_detailed": _hash_file(detailed_path),
        "scoring_summary": _hash_file(summary_path),
        "metric_readiness": _hash_file(readiness_path),
        "scoring_metadata": _hash_file(meta_path),
        "dim_asin": _hash_file(dim_asin_path) if dim_asin_path.exists() else None,
        "dim_vendor": _hash_file(dim_vendor_path) if dim_vendor_path.exists() else None,
    }

    return Inputs(detailed, summary, readiness, metadata, dim_asin, dim_vendor), input_hashes, warnings


def _select_signals(df: pd.DataFrame, readiness: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    # Detect events per spec, concatenate and return unified event frame
    parts: List[pd.DataFrame] = []
    parts.append(detect_metric_movements(df, config))
    parts.append(detect_dimension_shifts(df, config))
    parts.append(detect_inventory_risks(df))
    parts.append(detect_buyability_failures(df))
    parts.append(detect_category_outliers(df))

    # Readiness warnings from readiness df
    readiness_cols = {c.lower(): c for c in readiness.columns}
    if readiness is not None and not readiness.empty:
        # flag low completeness
        comp_col = None
        for cand in ["completeness", "metric_completeness"]:
            if cand in readiness_cols:
                comp_col = readiness_cols[cand]
                break
        if comp_col is not None:
            low = readiness[readiness[comp_col] < float(config.get("readiness", {}).get("min_completeness", 0.85))]
            if not low.empty:
                rr = low[[c for c in low.columns if c in ("vendor_id", "asin_id", "canonical_week_id")]].copy()
                rr["signal_type"] = "readiness_low_completeness"
                rr["severity"] = 0.2
                rr["metric"] = "readiness"
                parts.append(rr)

    events = pd.concat([p for p in parts if p is not None and not p.empty], ignore_index=True) if parts else pd.DataFrame()
    # Standardize required columns
    if events.empty:
        return events
    needed = ["vendor_id", "asin_id", "canonical_week_id", "signal_type", "severity"]
    for c in needed:
        if c not in events.columns:
            events[c] = None
    return events


def _generate_callouts(events: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame(columns=[
            "vendor_id", "asin_id", "canonical_week_id", "signal_type", "severity",
            "setup_text", "driver_text", "next_steps_text",
        ])
    # Apply filtering pipeline
    filt = filter_high_impact_events(events, config)
    deduped = dedupe_related_signals(filt)
    prioritized = prioritize_by_materiality(deduped, config)

    # Compose WBR callouts
    prioritized = prioritized.copy()
    texts = prioritized.apply(lambda r: compose_wbr_callout(r.to_dict(), config), axis=1)
    prioritized[["setup_text", "driver_text", "next_steps_text"]] = pd.DataFrame(texts.tolist(), index=prioritized.index)

    # Deterministic ordering
    order_cols = ["vendor_id", "severity", "signal_type", "canonical_week_id", "asin_id"]
    for col in order_cols:
        if col not in prioritized.columns:
            prioritized[col] = None
    prioritized.sort_values(by=["vendor_id", "severity", "canonical_week_id", "signal_type", "asin_id"], ascending=[True, False, False, True, True], inplace=True)
    return prioritized


def _write_outputs(callouts: pd.DataFrame, metadata_in: Dict[str, Any], config: Dict[str, Any], input_hashes: Dict[str, str], warnings: List[str], output_base: str, commentary_config_path: str) -> Tuple[Path, Path, Path]:
    out_dir = Path(output_base)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Metadata
    cfg_text = Path(commentary_config_path).read_text(encoding="utf-8")
    cfg_hash = hashlib.sha256(cfg_text.encode("utf-8")).hexdigest()
    ts = datetime.utcnow().isoformat()
    applied_thresholds = config.get("thresholds", {})
    md = {
        "timestamp": ts,
        "config_hash": cfg_hash,
        "input_hashes": input_hashes,
        "applied_thresholds": applied_thresholds,
        "phase2_metadata": metadata_in,
        "detected_events": int(len(callouts)),
        "final_callouts": int(len(callouts)),
        "warnings": warnings,
    }

    # Outputs
    pq_path = out_dir / "phase3_commentary.parquet"
    jsonl_path = out_dir / "phase3_commentary.json"
    meta_path = out_dir / "phase3_metadata.json"

    cols = [
        "vendor_id", "asin_id", "canonical_week_id", "signal_type", "severity",
        "setup_text", "driver_text", "next_steps_text",
    ]
    to_write = callouts[cols] if not callouts.empty else pd.DataFrame(columns=cols)
    to_write.to_parquet(pq_path, index=False)
    # Write json lines
    with jsonl_path.open("w", encoding="utf-8") as f:
        for _, row in to_write.iterrows():
            f.write(json.dumps({k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}, ensure_ascii=False) + "\n")
    meta_path.write_text(json.dumps(md, indent=2), encoding="utf-8")

    return pq_path, jsonl_path, meta_path


def run_commentary(phase2_path: str, dimensions_path: str, commentary_config_path: str, output_base: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cfg = yaml.safe_load(Path(commentary_config_path).read_text(encoding="utf-8"))
    inputs, input_hashes, warnings = _load_inputs(phase2_path, dimensions_path)
    events = _select_signals(inputs.detailed, inputs.readiness, cfg)
    callouts = _generate_callouts(events, cfg)
    pq_path, jsonl_path, meta_path = _write_outputs(
        callouts, inputs.metadata, cfg, input_hashes, warnings, output_base, commentary_config_path
    )
    return callouts, {
        "warnings": warnings,
        "outputs": {
            "parquet": str(pq_path.resolve()),
            "json": str(jsonl_path.resolve()),
            "metadata": str(meta_path.resolve()),
        },
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Brightlight Phase 3 Commentary Engine")
    p.add_argument("--phase2", required=True, help="Path to Phase 2 output directory")
    p.add_argument("--dimensions", required=True, help="Path to dimension_paths.yaml")
    p.add_argument("--commentary_config", required=True, help="Path to commentary_config.yaml")
    p.add_argument("--output", required=True, help="Output directory for Phase 3 artifacts")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    df, extra = run_commentary(args.phase2, args.dimensions, args.commentary_config, args.output)
    # Validation run printout: row count, top 3 callouts, missing field warnings
    top3 = []
    if df is not None and not df.empty:
        disp = df[["vendor_id", "asin_id", "signal_type", "setup_text"]].head(3)
        top3 = [f"{r.vendor_id}|{r.asin_id}|{r.signal_type}: {r.setup_text}" for r in disp.itertuples(index=False)]
    print(json.dumps({
        "row_count": 0 if df is None else int(len(df)),
        "top3": top3,
        "warnings": extra.get("warnings", []),
        "outputs": extra.get("outputs", {}),
    }))


if __name__ == "__main__":
    main()
