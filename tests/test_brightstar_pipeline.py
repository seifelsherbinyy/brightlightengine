"""Integration tests for the Brightstar end-to-end pipeline."""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from brightstar.brightstar_pipeline import run_pipeline


def _build_config(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    scorecard_dir = processed_dir / "scorecards"
    forecast_dir = processed_dir / "forecasts"
    commentary_dir = processed_dir / "commentary"
    dashboard_dir = processed_dir / "dashboard"
    logs_dir = tmp_path / "logs"
    reference_dir = data_dir / "reference"

    for directory in [raw_dir, processed_dir, scorecard_dir, forecast_dir, commentary_dir, dashboard_dir, logs_dir, reference_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    calendar_map = reference_dir / "calendar_map.csv"
    calendar_map.write_text(
        "raw_week_label,week_label,week_start,week_end\n"
        "09/01/2025-09/07/2025,2025W36,2025-09-01,2025-09-07\n"
        "09/08/2025-09/14/2025,2025W37,2025-09-08,2025-09-14\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        dedent(
            f"""
            paths:
              raw_dir: {raw_dir}
              processed_dir: {processed_dir}
              logs_dir: {logs_dir}
              calendar_map: {calendar_map}
              normalized_output: {processed_dir / 'normalized_phase1.csv'}
              ingestion_state: {processed_dir / 'ingestion_state.json'}
              scorecards_dir: {scorecard_dir}
              scorecard_vendor_csv: {scorecard_dir / 'vendor_scorecard.csv'}
              scorecard_vendor_parquet: {scorecard_dir / 'vendor_scorecard.parquet'}
              scorecard_asin_csv: {scorecard_dir / 'asin_scorecard.csv'}
              scorecard_asin_parquet: {scorecard_dir / 'asin_scorecard.parquet'}
              scorecard_audit_csv: {scorecard_dir / 'score_audit.csv'}
              scorecard_metadata: {scorecard_dir / 'run_metadata.json'}
              forecast_vendor_csv: {forecast_dir / 'vendor_forecasts.csv'}
              forecast_asin_csv: {forecast_dir / 'asin_forecasts.csv'}
              forecast_metadata: {forecast_dir / 'run_metadata.json'}
              dashboard_dir: {dashboard_dir}
            logging:
              level: INFO
              file_name: phase1.log
              scoring_level: INFO
              scoring_file_name: phase2.log
              forecasting_level: INFO
              forecasting_file_name: phase4.log
              pipeline_level: INFO
              pipeline_file_name: pipeline.log
            ingestion:
              vendor_column: 0
              asin_column: 1
              id_columns:
                - Vendor
                - ASIN
              value_round: 2
              allowed_extensions:
                - .csv
              output_format: csv
            scoring:
              normalized_input: {processed_dir / 'normalized_phase1.csv'}
              metric_weights:
                GMS($): 0.6
                FillRate(%): 0.4
              metric_directions:
                GMS($): up
                FillRate(%): up
              thresholds:
                excellent: 850
                good: 700
                fair: 550
                poor: 0
              composite_scale: 1000
              fallback_normalized_value: 0.5
            commentary:
              normalized_input: {processed_dir / 'normalized_phase1.csv'}
              scorecard_vendor: {scorecard_dir / 'vendor_scorecard.csv'}
              scorecard_asin: {scorecard_dir / 'asin_scorecard.csv'}
              logging_level: INFO
              logging_file_name: phase3.log
              change_bands:
                strong_increase: 0.1
                moderate_increase: 0.05
                flat: 0.01
                moderate_decline: -0.05
                strong_decline: -0.1
              templates:
                performance_summary:
                  strong_increase: "{{entity}} surged in {{metric}} ({{change_pct}})"
                  moderate_increase: "{{entity}} improved {{metric}} ({{change_pct}})"
                  flat: "{{entity}} held steady on {{metric}}."
                  moderate_decline: "{{entity}} softened in {{metric}} ({{change_pct}})."
                  strong_decline: "{{entity}} declined sharply in {{metric}} ({{change_pct}})."
                  insufficient_history: "{{entity}} requires more history for {{metric}}."
                opportunity:
                  below_threshold: "{{entity}} has opportunity in {{metric}} ({{current_value}})."
                  improving: "{{entity}} is improving {{metric}}."
                  neutral: "No opportunity flag for {{metric}}."
                risk:
                  strong_decline: "Risk: {{metric}} falling for {{entity}}."
                  moderate_decline: "Monitor {{metric}} for {{entity}}."
                  below_threshold: "{{entity}} below threshold in {{metric}}."
                  neutral: "No risk this week."
            forecasting:
              normalized_input: {processed_dir / 'normalized_phase1.csv'}
              output_dir: {forecast_dir}
              horizon_weeks: 2
              min_history_weeks: 2
              frequency: W-MON
              metrics:
                - GMS($)
                - FillRate(%)
              ensemble_weights:
                fallback: 1.0
              metric_directions:
                GMS($): up
                FillRate(%): up
              uncertainty_buffer_pct: 0.1
              metadata_fields:
                include_history: true
                include_methods: true
            dashboard:
              output_dir: {dashboard_dir}
              file_prefix: BrightstarTest
              sheet_names:
                summary: Summary
                vendor_scorecard: Vendor Scorecards
                asin_scorecard: ASIN Scorecards
                commentary: Commentary
                forecasts: Forecasts
              sheet_order:
                - Summary
                - Vendor Scorecards
                - ASIN Scorecards
                - Commentary
                - Forecasts
              formatting:
                header_fill: "1F4E78"
                header_font_color: "FFFFFF"
                body_font: Calibri
                precision: 2
                column_widths:
                  default: 18
                  overrides:
                    Commentary_Text: 40
                score_color_scale:
                  start_color: "F8696B"
                  mid_color: "FFEB84"
                  end_color: "63BE7B"
                  mid_value: 700
                  end_value: 950
                commentary_fills:
                  performance_summary: "C6EFCE"
                  opportunity: "FFEB9C"
                  risk: "FFC7CE"
                  default: "FFFFFF"
            summary:
              enabled: false
            pipeline:
              default_phases:
                - phase1
                - phase2
                - phase3
                - phase4
                - phase5
              allow_resume: true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path, raw_dir, processed_dir


def _create_raw_input(raw_dir: Path) -> None:
    raw_data = (
        "Vendor,ASIN,GMS($)(09/01/2025-09/07/2025),GMS($)(09/08/2025-09/14/2025),FillRate(%)(09/01/2025-09/07/2025),FillRate(%)(09/08/2025-09/14/2025)\n"
        "V1,A1,100,120,0.9,0.95\n"
        "V2,A2,80,75,0.8,0.78\n"
    )
    (raw_dir / "sample.csv").write_text(raw_data, encoding="utf-8")


def test_pipeline_runs_all_phases(tmp_path):
    config_path, raw_dir, processed_dir = _build_config(tmp_path)
    _create_raw_input(raw_dir)

    result = run_pipeline(config_path=str(config_path))

    assert result.success
    statuses = {phase.name: phase.status for phase in result.phases}
    for phase_name in ["phase1", "phase2", "phase3", "phase4", "phase5"]:
        assert statuses.get(phase_name) == "success"

    assert (processed_dir / "normalized_phase1.csv").exists()
    assert (processed_dir / "scorecards" / "vendor_scorecard.csv").exists()
    assert (processed_dir / "commentary" / "commentary.csv").exists()
    assert (processed_dir / "forecasts" / "vendor_forecasts.csv").exists()
    dashboard_files = list((processed_dir / "dashboard").glob("*.xlsx"))
    assert dashboard_files
