# Testing Guide

Brightstar relies on pytest for automated verification. The repository includes targeted tests per phase in the `tests/` directory. Phase 1 focuses on validating ingestion, week normalization, and deduplication. Phase 2 adds unit coverage for the scoring engine, with later phases extending the surface to commentary, forecasting, dashboard generation, and pipeline orchestration.

## Running the Test Suite

```bash
pytest
```

## Phase 1 Test Coverage

- **Week normalization** – ensures date-range headers map to canonical fiscal weeks defined in `calendar_map.csv`.
- **Deduplication** – verifies that repeated vendor/ASIN/week combinations collapse into a single record.

## Phase 2 Test Coverage

- **Weighted scoring** – checks that configured weights and metric directions produce the expected composite ordering.
- **Graceful null handling** – ensures missing metrics default to neutral (0.5 normalized) instead of failing the run.
- **Ranking metadata** – validates `Rank_Overall`, `Rank_By_Metric`, and output file creation for the vendor/ASIN scorecards.

## Phase 3 Test Coverage

- **Template rendering** – ensures commentary templates fill correctly for each change band and commentary type.
- **Missing history** – verifies neutral messaging when week-over-week comparisons are not possible.
- **Output persistence** – checks that commentary CSV/Parquet artifacts are written to the processed directory.

## Phase 4 Test Coverage

- **Forecast table schema** – validates vendor and ASIN forecasts contain the expected metadata columns.
- **Potential gains** – ensures directional potential-gain logic respects metric direction.
- **Metadata export** – confirms JSON metadata captures methods used and history depth.

## Phase 5 Test Coverage

- **Workbook structure** – asserts all configured sheets are created with expected columns.
- **Styling hooks** – verifies conditional formatting helpers run without error.
- **Summary sheet metrics** – checks run statistics reflect source data.

## Phase 6 Test Coverage

- **End-to-end orchestration** – runs the pipeline across Phases 1–5 using synthetic sample data and asserts every output artifact exists.
- **Resume/selection controls** – covered indirectly by unit tests verifying the pipeline’s selection logic.

Future contributions should continue mirroring phase names for new tests so the suite remains organized and discoverable.
