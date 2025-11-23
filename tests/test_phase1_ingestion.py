import json
import json
import sys
from pathlib import Path
from textwrap import dedent

import pandas as pd
import yaml


sys.path.append(str(Path(__file__).resolve().parents[1]))

from brightstar.phase1_ingestion import run_phase1


def build_config(tmp_path: Path) -> Path:
    paths = tmp_path / "data"
    raw_dir = paths / "raw"
    processed_dir = paths / "processed"
    logs_dir = tmp_path / "logs"
    reference_dir = paths / "reference"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    calendar_map = reference_dir / "calendar_map.csv"
    calendar_map.write_text(
        "raw_week_label,week_label,week_start,week_end\n"
        "09/01/2025-09/07/2025,2025W36,2025-09-01,2025-09-07\n"
        "09/08/2025-09/14/2025,2025W37,2025-09-08,2025-09-14\n"
    )

    master_path = reference_dir / "Masterfile.xlsx"
    pd.DataFrame(
        [
            {"ASIN": "A1", "Vendor Code": "V_MASTER", "Vendor Name": "Vendor One"},
        ]
    ).to_excel(master_path, index=False, sheet_name="Master")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        dedent(
            f"""
            paths:
              raw_dir: {raw_dir}
              processed_dir: {processed_dir}
              logs_dir: {logs_dir}
              calendar_map: {calendar_map}
              normalized_output: {processed_dir}/normalized.parquet
              ingestion_state: {processed_dir}/state.json
              masterfile: {master_path}
            logging:
              level: INFO
              file_name: phase1.log
            ingestion:
              vendor_column: 0
              asin_column: 1
              id_columns:
                - Vendor
                - ASIN
              value_round: 2
              deduplicate: true
              allowed_extensions:
                - .csv
              output_format: csv
            masterfile:
              sheet_name: Master
              columns:
                asin: ASIN
                vendor_code: Vendor Code
                vendor_name: Vendor Name
            summary:
              enabled: true
              template: "Processed {{total_rows}} rows across {{unique_weeks}} weeks. Last week: {{last_week}}."
            """
        ).strip()
        + "\n"
    )
    return config_path


def create_sample_raw(raw_path: Path) -> Path:
    content = (
        "Vendor,ASIN,GMS($) 09/01/2025-09/07/2025,GMS($) 09/08/2025-09/14/2025,Fill Rate (%) W36 2026,Revenue(09/22/2026-09/28/2026)\n"
        "V1,A1,100,110,0.95,120\n"
        "V1,A1,100,110,0.95,120\n"
    )
    file_path = raw_path / "sample.csv"
    file_path.write_text(content)
    return file_path


def test_run_phase1_normalizes_weeks(tmp_path, capsys):
    config_path = build_config(tmp_path)
    config_text = yaml.safe_load(config_path.read_text())
    raw_dir = Path(config_text["paths"]["raw_dir"])
    create_sample_raw(raw_dir)

    df = run_phase1(config_path=str(config_path))

    captured = capsys.readouterr().out
    assert "Processed" in captured
    assert not df.empty
    assert "vendor_name" in df.columns
    assert set(df["vendor_name"].dropna()) == {"Vendor One"}
    assert set(df["week_label"]) == {"2025W36", "2025W37", "2026W36", "2026W39"}
    fill_rate_row = df[df["metric"] == "Fill Rate (%)"].iloc[0]
    assert fill_rate_row["week_label"] == "2026W36"
    assert fill_rate_row["week_start"] == "2026-08-31"
    range_row = df[df["metric"] == "Revenue"].iloc[0]
    assert range_row["week_label"] == "2026W39"
    assert range_row["week_start"] == "2026-09-21"
    assert (tmp_path / "logs" / "phase1.log").exists()
    state_path = tmp_path / "data" / "processed" / "state.json"
    assert state_path.exists()
    state = json.loads(state_path.read_text())
    assert state["records"] == 4
    assert state["last_week"] == "2026W39"


def test_deduplication(tmp_path):
    config_path = build_config(tmp_path)
    config_text = yaml.safe_load(config_path.read_text())
    raw_dir = Path(config_text["paths"]["raw_dir"])
    create_sample_raw(raw_dir)

    df = run_phase1(config_path=str(config_path))

    assert len(df) == 4
    assert df.drop_duplicates(subset=["week_label"]).shape[0] == 4
