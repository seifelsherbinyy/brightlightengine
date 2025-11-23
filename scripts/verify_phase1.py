"""Verification script for Phase 1 normalized outputs.

Loads the final dataset from the mandatory output directory and runs
consistency checks on schema and values.
"""
from __future__ import annotations

from pathlib import Path
import json
import sys
import pandas as pd

MANDATORY_OUTPUT_DIR = Path(r"C:\Users\selsherb\Documents\AVS-E\CL\Seif Vendors")


def load_normalized() -> pd.DataFrame:
    pq = MANDATORY_OUTPUT_DIR / "normalized_phase1.parquet"
    csv = MANDATORY_OUTPUT_DIR / "normalized_phase1.csv"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception:
            pass
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError("normalized_phase1 not found in mandatory output directory")


def check_schema(df: pd.DataFrame) -> dict:
    required = [
        "vendor_code",
        "vendor_name",
        "asin",
        "metric",
        "week_label",
        "week_raw",
        "week_start",
        "week_end",
        "value",
        "source_file",
    ]
    missing = [c for c in required if c not in df.columns]
    bad_week = df[~df["week_label"].astype(str).str.match(r"^\d{4}W\d{2}$", na=False)] if "week_label" in df.columns else pd.DataFrame()
    return {
        "missing_columns": missing,
        "bad_week_labels": int(len(bad_week)) if not bad_week.empty else 0,
        "row_count": int(len(df)),
        "unique_weeks": int(df["week_label"].nunique()) if "week_label" in df.columns else 0,
        "unique_asins": int(df["asin"].nunique()) if "asin" in df.columns else 0,
        "unique_vendors": int(df["vendor_code"].nunique()) if "vendor_code" in df.columns else 0,
    }


def main() -> int:
    try:
        df = load_normalized()
    except Exception as exc:
        print(json.dumps({"status": "error", "message": str(exc)}, indent=2))
        return 1

    report = check_schema(df)
    ok = (not report["missing_columns"]) and report["bad_week_labels"] == 0 and report["row_count"] > 0
    status = {"status": "ok" if ok else "warn", **report}
    print(json.dumps(status, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
