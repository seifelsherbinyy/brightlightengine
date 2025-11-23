from pathlib import Path

import pandas as pd

from brightstar.output.phase4_outputs import (
    build_phase4_run_dir,
    write_machine_outputs,
    write_excel_outputs,
)


def make_long_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["V1_A1", "V1_A1"],
            "vendor_code": ["V1", "V1"],
            "Week_Order": [1, 2],
            "metric": ["Units", "Units"],
            "value": [10.0, 12.0],
            "std_value": [0.1, 0.2],
            "raw_value": [10.0, 12.0],
            "volatility": [0.0, 0.1],
            "trend_slope": [0.0, 0.1],
            "trend_strength": [0.0, 1.0],
            "completeness": [1.0, 1.0],
            "recency_weeks": [0.0, 0.0],
            "variance": [0.0, 0.01],
            "trend_quality": [0.0, 1.0],
            "weight_effective": [0.5, 0.5],
            "weight_multiplier": [1.0, 1.0],
            "base_weight": [0.5, 0.5],
        }
    )


def make_wide_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": ["V1_A1"],
            "vendor_code": ["V1"],
            "Week_Order": [2],
            "std_Units": [0.2],
            "contrib_Units": [0.1],
            "weight_Units": [0.5],
            "weightmul_Units": [1.0],
            "base_weight_Units": [0.5],
            "score_composite": [0.1],
        }
    )


def test_phase4_machine_and_excel_outputs(tmp_path: Path):
    run_dirs = build_phase4_run_dir(tmp_path / "phase4")
    long_df = make_long_df()
    wide_df = make_wide_df()
    asin_scores = pd.DataFrame({
        "entity_id": ["V1_A1"],
        "vendor_code": ["V1"],
        "Week_Order": [2],
        "score_composite": [0.1],
    })
    vendor_scores = pd.DataFrame({
        "vendor_code": ["V1"],
        "vendor_name": ["Vendor One"],
        "Composite_Score": [0.1],
    })

    paths = write_machine_outputs(
        run_dirs,
        long_df=long_df,
        wide_df=wide_df,
        asin_scores=asin_scores,
        vendor_scores=vendor_scores,
        signal_quality_df=long_df[[
            "entity_id", "metric", "Week_Order", "completeness", "recency_weeks", "variance", "trend_quality"
        ]],
    )
    # Ensure files exist
    for key in ["metrics_long", "metrics_wide", "scores_asin", "scores_vendor", "signal_quality"]:
        assert (paths[key]).exists()

    # Excel: pass wide-like frame to ensure Patch-3 audit fields present in sheet
    xlsx = write_excel_outputs(run_dirs, vendor_scores=vendor_scores, asin_long=wide_df)
    assert xlsx.exists()
