"""
metric_registry_update.py

Small helper to sync the metric registry file with the expanded
canonical metric set defined in metric_registry_utils.py.

Usage (from project root):

    python metric_registry_update.py --registry config/metric_registry.csv

If the file exists, it will be updated in-place to include any missing
canonical metrics with descriptions. If it does not exist, a new file
will be created with all canonical metrics.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from metric_registry_utils import (
    load_metric_registry,
    bootstrap_predefined_metrics,
    REGISTRY_COLUMNS,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync Brightlight metric registry with canonical metrics.")
    parser.add_argument(
        "--registry",
        type=str,
        required=True,
        help="Path to the metric registry file (CSV or Parquet).",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)

    # Load existing registry (or empty structured DF)
    df = load_metric_registry(registry_path)

    # Bootstrap / extend with canonical metrics + descriptions
    updated = bootstrap_predefined_metrics(df)

    # Decide format based on extension
    suffix = registry_path.suffix.lower()
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".parquet":
        updated.to_parquet(registry_path, index=False)
    else:
        # default to CSV
        updated.to_csv(registry_path, index=False)

    print(f"[metric-registry-update] Updated registry written to: {registry_path}")
    print(f"[metric-registry-update] Rows: {len(updated)}, Columns: {len(updated.columns)}")
    missing_cols = [c for c in REGISTRY_COLUMNS if c not in updated.columns]
    if missing_cols:
        print(f"[metric-registry-update] WARNING: missing expected columns: {missing_cols}")


if __name__ == "__main__":
    main()
