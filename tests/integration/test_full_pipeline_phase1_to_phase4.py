from pathlib import Path

import pandas as pd
import pytest


def test_full_pipeline_creates_phase4_outputs(tmp_path, monkeypatch):
    # Prepare a minimal config file
    config_path = tmp_path / "config.yaml"
    processed_dir = tmp_path / "data" / "processed"
    logs_dir = tmp_path / "logs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""
paths:
  processed_dir: {processed_dir}
  logs_dir: {logs_dir}
scoring:
  rolling_weeks: 4
  weights_must_sum_to_1: true
  metrics:
    - name: Units
      direction: higher_is_better
      weight: 1.0
      clip_quantiles: [0.01, 0.99]
      transform: none
      standardize: percentile
      missing_policy: group_median
        """.strip()
    )

    # Craft a tiny Phase 2 return payload to bypass heavy upstream dependencies
    long_df = pd.DataFrame(
        {
            "entity_id": ["V1_A1", "V1_A1"],
            "vendor_code": ["V1", "V1"],
            "Week_Order": [1, 2],
            "metric": ["Units", "Units"],
            "value": [10.0, 12.0],
            "raw_value": [10.0, 12.0],
            "std_value": [0.1, 0.2],
            # signal quality + weights
            "volatility": [0.0, 0.1],
            "trend_slope": [0.0, 0.1],
            "trend_strength": [0.0, 1.0],
            "completeness": [1.0, 1.0],
            "recency_weeks": [0.0, 0.0],
            "variance": [0.0, 0.01],
            "trend_quality": [0.0, 1.0],
            "weight_effective": [1.0, 1.0],
            "weight_multiplier": [1.0, 1.0],
            "base_weight": [1.0, 1.0],
        }
    )
    vendor_df = pd.DataFrame({"vendor_code": ["V1"], "vendor_name": ["Vendor"], "Composite_Score": [0.1]})

    # Monkeypatch Phase 2 runner to return our tiny frames
    import brightstar.pipeline.run_all_phases as pipeline

    def fake_run_phase2(config_path: str = "config.yaml"):
        return {"matrix": long_df.copy(), "scoreboard": vendor_df.copy()}

    monkeypatch.setattr("brightstar.pipeline.run_all_phases.run_phase2", fake_run_phase2)

    # Execute the orchestrator with sanity check skipped (we test writers here)
    result = pipeline.run_all_phases(config_path=str(config_path), skip_phase1=True, skip_sanity=True)

    run_dir = Path(result["paths"]["run"])
    assert run_dir.exists()
    # Machine outputs present
    machine = run_dir / "machine"
    assert (machine / "metrics_long.parquet").exists()
    assert (machine / "metrics_wide.parquet").exists()
    assert (machine / "scores_asin.parquet").exists()
    assert (machine / "scores_vendor.parquet").exists()
    # Excel present
    assert (run_dir / "excel" / "diagnostics.xlsx").exists()
    # Metadata present
    meta = run_dir / "metadata.json"
    assert meta.exists()
