import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd


sys.path.append(str(Path(__file__).resolve().parents[1]))

from brightstar.phase2_scoring import run_phase2


def build_config(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    logs_dir = tmp_path / "logs"
    scorecard_dir = processed_dir / "scorecards"
    logs_dir.mkdir(parents=True, exist_ok=True)
    scorecard_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = processed_dir / "normalized_phase1.csv"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        dedent(
            f"""
            paths:
              raw_dir: {data_dir / 'raw'}
              processed_dir: {processed_dir}
              logs_dir: {logs_dir}
              calendar_map: {data_dir / 'reference' / 'calendar_map.csv'}
              normalized_output: {normalized_path}
              ingestion_state: {processed_dir / 'state.json'}
              scorecards_dir: {scorecard_dir}
              scorecard_vendor_csv: {scorecard_dir / 'vendor.csv'}
              scorecard_vendor_parquet: {scorecard_dir / 'vendor.parquet'}
              scorecard_asin_csv: {scorecard_dir / 'asin.csv'}
              scorecard_asin_parquet: {scorecard_dir / 'asin.parquet'}
              scorecard_audit_csv: {scorecard_dir / 'audit.csv'}
              scorecard_metadata: {scorecard_dir / 'metadata.json'}
            logging:
              scoring_level: INFO
              scoring_file_name: phase2.log
            ingestion:
              allowed_extensions:
                - .csv
            scoring:
              normalized_input: {normalized_path}
              aggregation: mean
              fallback_normalized_value: 0.5
              composite_scale: 1000
              thresholds:
                excellent: 800
                good: 600
                fair: 400
                poor: 0
              metric_weights:
                GMS($): 0.5
                FillRate(%): 0.3
                SoROOS(%): 0.2
              metric_directions:
                GMS($): up
                FillRate(%): up
                SoROOS(%): down
            """
        ).strip()
        + "\n"
    )
    return config_path, normalized_path


def create_normalized_dataset(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {"vendor_code": "V1", "vendor_name": "Vendor One", "asin": "A1", "metric": "GMS($)", "value": 100},
            {"vendor_code": "V1", "vendor_name": "Vendor One", "asin": "A1", "metric": "FillRate(%)", "value": 0.9},
            {"vendor_code": "V1", "vendor_name": "Vendor One", "asin": "A1", "metric": "SoROOS(%)", "value": 0.1},
            {"vendor_code": "V2", "vendor_name": "Vendor Two", "asin": "A2", "metric": "GMS($)", "value": 50},
            {"vendor_code": "V2", "vendor_name": "Vendor Two", "asin": "A2", "metric": "FillRate(%)", "value": 0.6},
            {"vendor_code": "V2", "vendor_name": "Vendor Two", "asin": "A2", "metric": "SoROOS(%)", "value": 0.3},
            {"vendor_code": "V3", "vendor_name": "Vendor Three", "asin": "A3", "metric": "GMS($)", "value": 70},
            {"vendor_code": "V3", "vendor_name": "Vendor Three", "asin": "A3", "metric": "FillRate(%)", "value": None},
            {"vendor_code": "V3", "vendor_name": "Vendor Three", "asin": "A3", "metric": "SoROOS(%)", "value": 0.2},
        ]
    )
    df["week_label"] = "2025W37"
    df.to_csv(path, index=False)
    return df


def test_scoring_generates_ranked_scorecards(tmp_path):
    config_path, normalized_path = build_config(tmp_path)
    create_normalized_dataset(normalized_path)

    results = run_phase2(config_path=str(config_path))

    vendor_df = results["vendor"]
    asin_df = results["asin"]

    assert not vendor_df.empty
    assert not asin_df.empty
    assert "vendor_name" in vendor_df.columns
    expected_names = {"Vendor One", "Vendor Two", "Vendor Three"}
    assert set(vendor_df.drop_duplicates(subset=["vendor_code"])["vendor_name"]) == expected_names

    composite_scores = (
        vendor_df.drop_duplicates(subset=["vendor_code", "Composite_Score"])
        .set_index("vendor_code")["Composite_Score"]
    )
    assert composite_scores.loc["V1"] > composite_scores.loc["V3"] > composite_scores.loc["V2"]

    v1_fill_rate = vendor_df[(vendor_df["vendor_code"] == "V1") & (vendor_df["metric"] == "FillRate(%)")].iloc[0]
    assert v1_fill_rate["Normalized_Value"] == 1.0
    assert v1_fill_rate["Weighted_Score"] == 0.3

    v3_fill_rate = vendor_df[(vendor_df["vendor_code"] == "V3") & (vendor_df["metric"] == "FillRate(%)")].iloc[0]
    assert v3_fill_rate["Normalized_Value"] == 0.5

    ranks = vendor_df.drop_duplicates(subset=["vendor_code", "Rank_Overall"]).set_index("vendor_code")
    assert ranks.loc["V1", "Rank_Overall"] == 1
    assert ranks.loc["V2", "Rank_Overall"] == ranks["Rank_Overall"].max()

    assert "Improvement_Rank" in vendor_df.columns
    improvement = vendor_df[vendor_df["metric"] == "GMS($)"]
    assert improvement["Improvement_Rank"].notna().all()

    gms_ranks = (
        vendor_df[vendor_df["metric"] == "GMS($)"]
        .set_index("vendor_code")["Rank_By_Metric"]
    )
    assert gms_ranks.loc["V1"] == 1
    assert gms_ranks.loc["V2"] > gms_ranks.loc["V3"]

    metadata_path = Path(config_path.parent / "data" / "processed" / "scorecards" / "metadata.json")
    assert metadata_path.exists()


def test_scorecard_outputs_created(tmp_path):
    config_path, normalized_path = build_config(tmp_path)
    create_normalized_dataset(normalized_path)

    run_phase2(config_path=str(config_path))

    scorecard_dir = Path(config_path.parent / "data" / "processed" / "scorecards")
    assert (scorecard_dir / "vendor.csv").exists()
    assert (scorecard_dir / "asin.csv").exists()
    assert (scorecard_dir / "audit.csv").exists()
