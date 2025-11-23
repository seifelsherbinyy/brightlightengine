import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd


sys.path.append(str(Path(__file__).resolve().parents[1]))

from brightstar.phase3_commentary import run_phase3


def _build_config(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    processed_dir = data_dir / "processed"
    scorecard_dir = processed_dir / "scorecards"
    logs_dir = tmp_path / "logs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    scorecard_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    normalized_path = processed_dir / "normalized.csv"
    vendor_scorecard = scorecard_dir / "vendor.csv"
    asin_scorecard = scorecard_dir / "asin.csv"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        dedent(
            f"""
            paths:
              processed_dir: {processed_dir}
              logs_dir: {logs_dir}
              scorecard_vendor_csv: {vendor_scorecard}
              scorecard_asin_csv: {asin_scorecard}
            scoring:
              aggregation: mean
            commentary:
              normalized_input: {normalized_path}
              scorecard_vendor: {vendor_scorecard}
              scorecard_asin: {asin_scorecard}
              logging_file_name: phase3.log
              change_bands:
                strong_increase: 0.1
                moderate_increase: 0.05
                flat: 0.01
                moderate_decline: -0.05
                strong_decline: -0.1
              opportunity_thresholds:
                default: 0.65
              risk_thresholds:
                default: 0.45
              templates:
                performance_summary:
                  strong_increase: "{{entity}} surged in {{metric}} with {{change_pct}} week over week."
                  moderate_increase: "{{entity}} improved {{metric}} {{change_pct}}."
                  flat: "{{entity}} held {{metric}} steady."
                  moderate_decline: "{{entity}} softened in {{metric}} ({{change_pct}})."
                  strong_decline: "{{entity}} declined sharply in {{metric}} ({{change_pct}})."
                  insufficient_history: "{{entity}} lacks history for {{metric}}; monitor upcoming weeks."
                opportunity:
                  below_threshold: "{{metric}} is lagging for {{entity}} at {{current_value}}."
                  improving: "{{entity}} is recovering in {{metric}}; maintain momentum."
                  neutral: "No major opportunity flagged for {{entity}} in {{metric}}."
                risk:
                  strong_decline: "Risk alert: {{metric}} falling {{change_pct}} for {{entity}}."
                  moderate_decline: "Monitor {{metric}}; {{entity}} dipped {{change_pct}}."
                  below_threshold: "{{entity}} remains weak in {{metric}} ({{current_value}})."
                  neutral: "No elevated risk detected for {{entity}} in {{metric}}."
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path, normalized_path, vendor_scorecard, asin_scorecard


def _create_inputs(normalized_path: Path, vendor_scorecard: Path, asin_scorecard: Path) -> None:
    normalized = pd.DataFrame(
        [
            {"entity_type": "vendor", "vendor_code": "V1", "asin": "A1", "metric": "GMS($)", "week_label": "2025W01", "week_start": "2025-01-01", "value": 100},
            {"entity_type": "vendor", "vendor_code": "V1", "asin": "A1", "metric": "GMS($)", "week_label": "2025W02", "week_start": "2025-01-08", "value": 120},
            {"entity_type": "vendor", "vendor_code": "V1", "asin": "A1", "metric": "FillRate(%)", "week_label": "2025W01", "week_start": "2025-01-01", "value": 0.8},
            {"entity_type": "vendor", "vendor_code": "V1", "asin": "A1", "metric": "FillRate(%)", "week_label": "2025W02", "week_start": "2025-01-08", "value": 0.7},
            {"entity_type": "vendor", "vendor_code": "V2", "asin": "A2", "metric": "GMS($)", "week_label": "2025W01", "week_start": "2025-01-01", "value": 90},
            {"entity_type": "vendor", "vendor_code": "V2", "asin": "A2", "metric": "GMS($)", "week_label": "2025W02", "week_start": "2025-01-08", "value": 90},
            {"entity_type": "vendor", "vendor_code": "V3", "asin": "A3", "metric": "GMS($)", "week_label": "2025W02", "week_start": "2025-01-08", "value": 150},
        ]
    )
    normalized.drop(columns=["entity_type"], inplace=True)
    normalized.to_csv(normalized_path, index=False)

    vendor_df = pd.DataFrame(
        [
            {"entity_type": "vendor", "vendor_code": "V1", "metric": "GMS($)", "Normalized_Value": 0.72},
            {"entity_type": "vendor", "vendor_code": "V1", "metric": "FillRate(%)", "Normalized_Value": 0.55},
            {"entity_type": "vendor", "vendor_code": "V2", "metric": "GMS($)", "Normalized_Value": 0.4},
            {"entity_type": "vendor", "vendor_code": "V3", "metric": "GMS($)", "Normalized_Value": 0.65},
        ]
    )
    vendor_df.to_csv(vendor_scorecard, index=False)

    asin_df = pd.DataFrame(
        [
            {"entity_type": "asin", "vendor_code": "V1", "asin": "A1", "metric": "GMS($)", "Normalized_Value": 0.75},
            {"entity_type": "asin", "vendor_code": "V2", "asin": "A2", "metric": "GMS($)", "Normalized_Value": 0.38},
        ]
    )
    asin_df.to_csv(asin_scorecard, index=False)


def test_commentary_generation_with_trend_detection(tmp_path):
    config_path, normalized_path, vendor_scorecard, asin_scorecard = _build_config(tmp_path)
    _create_inputs(normalized_path, vendor_scorecard, asin_scorecard)

    commentary = run_phase3(config_path=str(config_path))

    assert not commentary.empty
    v1_gms_perf = commentary[
        (commentary["entity_type"] == "vendor")
        & (commentary["vendor_code"] == "V1")
        & (commentary["metric"] == "GMS($)")
        & (commentary["Commentary_Type"] == "performance_summary")
    ].iloc[0]
    assert v1_gms_perf["Change_Class"] == "strong_increase"
    assert "+20.0%" in v1_gms_perf["Commentary_Text"]

    v1_gms_opp = commentary[
        (commentary["entity_type"] == "vendor")
        & (commentary["vendor_code"] == "V1")
        & (commentary["metric"] == "GMS($)")
        & (commentary["Commentary_Type"] == "opportunity")
    ].iloc[0]
    assert "recovering" in v1_gms_opp["Commentary_Text"]

    v1_fill_risk = commentary[
        (commentary["entity_type"] == "vendor")
        & (commentary["vendor_code"] == "V1")
        & (commentary["metric"] == "FillRate(%)")
        & (commentary["Commentary_Type"] == "risk")
    ].iloc[0]
    assert "Risk alert" in v1_fill_risk["Commentary_Text"]

    asin_records = commentary[commentary["entity_type"] == "asin"]
    assert not asin_records.empty

    v3_history = commentary[
        (commentary["vendor_code"] == "V3")
        & (commentary["metric"] == "GMS($)")
        & (commentary["Commentary_Type"] == "performance_summary")
    ].iloc[0]
    assert v3_history["Change_Class"] == "insufficient_history"
    assert "lacks history" in v3_history["Commentary_Text"]


def test_commentary_outputs_written(tmp_path):
    config_path, normalized_path, vendor_scorecard, asin_scorecard = _build_config(tmp_path)
    _create_inputs(normalized_path, vendor_scorecard, asin_scorecard)

    run_phase3(config_path=str(config_path))

    commentary_dir = config_path.parent / "data" / "processed" / "commentary"
    assert (commentary_dir / "commentary.csv").exists()
