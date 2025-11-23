import logging
from pathlib import Path

import pandas as pd

from brightstar.metric_registry_utils import (
    load_metric_registry,
    detect_new_metrics,
    append_pending_metrics,
    normalize_metric_key,
    match_to_canonical,
    bootstrap_predefined_metrics,
    REGISTRY_COLUMNS,
    metric_variants,
)


def test_load_registry_when_missing(tmp_path: Path):
    path = tmp_path / "metadata" / "metric_registry.csv"
    df = load_metric_registry(path)
    # Should not raise and must have expected columns
    assert list(df.columns) == REGISTRY_COLUMNS
    assert df.empty


def test_detect_new_metrics():
    normalized_df = pd.DataFrame({
        "metric": ["A", "B", "A", None],
        "value": [1, 2, 3, 4],
        "vendor_code": ["V1", "V1", "V2", "V2"],
        "asin": ["X", "Y", "Z", "Z"],
    })
    registry_df = pd.DataFrame({"metric_name": ["A"]})
    unknown = detect_new_metrics(normalized_df, registry_df)
    assert unknown == ["B"]


def test_normalize_metric_key_variants():
    # Hyphens, symbols, spaces
    assert normalize_metric_key("CCOGS As A % Of Revenue") == "ccogs_as_a_of_revenue"
    assert normalize_metric_key("On-Hand Inventory Units (Sellable)") == "on_hand_inventory_units_sellable"
    assert normalize_metric_key("CP($)") == "cp"


def test_match_to_canonical_success_and_none():
    assert match_to_canonical("Contra COGS % of revenue", metric_variants) == "CCOGS_As_A_Percent_Of_Revenue"
    assert match_to_canonical("Glance Views", metric_variants) == "GV"
    # Unknown metric should return None
    assert match_to_canonical("MadeUp Metric", metric_variants) is None


def test_bootstrap_predefined_metrics_idempotent():
    # Start from empty registry with only columns
    base = pd.DataFrame(columns=REGISTRY_COLUMNS)
    boot = bootstrap_predefined_metrics(base)
    # Running bootstrap again should not increase row count
    boot2 = bootstrap_predefined_metrics(boot)
    assert len(boot2) == len(boot)
    # Descriptive columns exist
    assert "description" in boot2.columns
    assert "keywords" in boot2.columns


def test_append_pending_metrics_appends(tmp_path: Path):
    pending = tmp_path / "metadata" / "new_metrics_pending.csv"
    # First append
    append_pending_metrics(pending, ["A", "B"])
    df1 = pd.read_csv(pending)
    assert df1.shape[0] == 2
    assert set(df1["metric_name"]) == {"A", "B"}
    # Second append should not overwrite previous contents
    append_pending_metrics(pending, ["C"])
    df2 = pd.read_csv(pending)
    assert df2.shape[0] == 3
    assert set(df2["metric_name"]) == {"A", "B", "C"}


def test_scoring_fallback_when_registry_empty(tmp_path: Path, monkeypatch):
    # Create minimal config
    processed_dir = tmp_path / "data" / "processed"
    logs_dir = tmp_path / "logs"
    processed_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)

    # Normalized CSV (fallback reader in scoring_utils will pick this up)
    norm_csv = processed_dir / "normalized_phase1.csv"
    pd.DataFrame(
        {
            "vendor_code": ["V1", "V1"],
            "asin": ["A1", "A2"],
            "metric": ["Foo", "Bar"],
            "value": [10.0, 5.0],
        }
    ).to_csv(norm_csv, index=False)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
paths:
  processed_dir: {}
  logs_dir: {}
  normalized_output: {}
logging:
  scoring_level: INFO
  scoring_file_name: phase2_scoring.log
registry:
  metric_registry: {}
  pending_metrics: {}
scoring:
  aggregation: mean
  fallback_normalized_value: 0.5
  composite_scale: 1000
  metrics:
    - name: Foo
      weight: 1.0
      direction: higher_is_better
        """.format(
            processed_dir.as_posix(),
            logs_dir.as_posix(),
            norm_csv.as_posix(),
            # Point registry to non-existent file to force empty registry
            (tmp_path / "metadata" / "metric_registry.csv").as_posix(),
            (tmp_path / "metadata" / "new_metrics_pending.csv").as_posix(),
        )
    )

    # Import here to avoid global config interference
    from brightstar.phase2_scoring import run_phase2

    # Capture root brightstar logger warnings
    logger = logging.getLogger("brightstar")
    logger.setLevel(logging.WARNING)
    records = []

    class ListHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = ListHandler()
    logger.addHandler(handler)
    try:
        outputs = run_phase2(config_path=str(config_path))
    finally:
        logger.removeHandler(handler)

    # Verify fallback warning was emitted
    assert any("Metric registry empty. Falling back" in r.getMessage() for r in records)

    # Verify that scoring produced output only for config-defined metric 'Foo'
    vendor_df = outputs.get("vendor")
    asin_df = outputs.get("asin")
    assert isinstance(vendor_df, pd.DataFrame)
    assert isinstance(asin_df, pd.DataFrame)
    if not vendor_df.empty:
        assert set(vendor_df["metric"]) <= {"Foo"}
    if not asin_df.empty:
        assert set(asin_df["metric"]) <= {"Foo"}
