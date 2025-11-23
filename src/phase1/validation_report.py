from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
from .dq_checks import aggregate_severity_counts, build_attention_rows


def build_report(df: pd.DataFrame, schema: Dict[str, Any] | None = None) -> Dict[str, Any]:
    total = int(len(df))
    valid = int(df.get("is_valid_row", pd.Series([False] * total)).sum())
    invalid = total - valid
    by_vendor = (
        df.groupby("vendor_id")["is_valid_row"].agg(["count", "sum"]).reset_index()
        if "vendor_id" in df.columns and "is_valid_row" in df.columns
        else pd.DataFrame(columns=["vendor_id", "count", "sum"])
    )
    report: Dict[str, Any] = {
        "summary": {
            "total_rows": total,
            "valid_rows": valid,
            "invalid_rows": invalid,
        },
    }
    if schema is not None:
        report["schema"] = {
            "required_columns": schema.get("required_columns", []),
            "dtypes": schema.get("dtypes", {}),
        }
    if not by_vendor.empty:
        report["by_vendor"] = [
            {
                "vendor_id": str(row["vendor_id"]),
                "rows": int(row["count"]),
                "valid_rows": int(row["sum"]),
            }
            for _, row in by_vendor.iterrows()
        ]
    # Count of each dq flag if present
    dq_cols = [c for c in df.columns if str(c).startswith("dq_")]
    if dq_cols:
        report["dq_flags"] = {
            c: int(df[c].sum()) for c in dq_cols if df[c].dtype == bool or pd.api.types.is_bool_dtype(df[c])
        }
        # Add severity bucket counts
        report["dq_severity_counts"] = aggregate_severity_counts(df)
        # Add distribution heatmap (JSON-friendly): by vendor_id when available
        if "vendor_id" in df.columns and "dq_severity" in df.columns:
            dist = (
                df.assign(dq_severity=df["dq_severity"].astype("string").fillna("").str.title())
                  .groupby(["vendor_id", "dq_severity"]).size().reset_index(name="count")
            )
            # pivot to wide format per vendor for readability
            if not dist.empty:
                severities = ["Critical", "High", "Medium", "Low", ""]
                pivot = {str(v): {s: 0 for s in severities} for v in dist["vendor_id"].astype("string").unique()}
                for _, r in dist.iterrows():
                    v = str(r["vendor_id"])
                    s = str(r["dq_severity"])
                    pivot.setdefault(v, {})
                    pivot[v][s] = int(r["count"])
                # Convert to list of rows
                report["dq_severity_distribution"] = [
                    {
                        "vendor_id": v,
                        "Critical": pivot[v].get("Critical", 0),
                        "High": pivot[v].get("High", 0),
                        "Medium": pivot[v].get("Medium", 0),
                        "Low": pivot[v].get("Low", 0),
                    }
                    for v in sorted(pivot.keys())
                ]
        # Category/Archetype health (Session 3)
        cat_health = {}
        for key, col in [
            ("unknown_categories", "dq_missing_category"),
            ("inferred_categories", "dq_category_inferred"),
            ("category_conflicts", "dq_category_conflict"),
            ("archetype_unassigned", "dq_archetype_unassigned"),
        ]:
            if col in df.columns:
                cat_health[key] = int(df[col].fillna(False).astype(bool).sum())
        # Also include simple counts of canonical category/archetype presence if available
        if "category_canonical" in df.columns:
            cat_health["with_category_canonical"] = int(df["category_canonical"].astype("string").fillna("").str.strip().ne("").sum())
        if "archetype_gcc" in df.columns:
            cat_health["with_archetype_gcc"] = int(df["archetype_gcc"].astype("string").fillna("").str.strip().ne("").sum())
        if cat_health:
            report["category_archetype_health"] = cat_health
    return report


def write_report(report: Dict[str, Any], output_path: str):
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def write_attention_csv(df: pd.DataFrame, output_path: str) -> None:
    """Emit CSV of rows requiring attention (High/Medium) with compact columns.

    Filename convention: phase1_attention_required.csv
    """

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    attn = build_attention_rows(df)
    if not attn.empty:
        attn.to_csv(out, index=False, encoding="utf-8-sig")
    else:
        # Write header-only file for consistency
        attn.to_csv(out, index=False, encoding="utf-8-sig")
