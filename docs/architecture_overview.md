# Brightstar Architecture Overview

Brightstar is a modular analytics engine designed to ingest Amazon Vendor exports, transform them into a canonical schema, and power downstream scoring, commentary, forecasting, and dashboard delivery. Each phase of the project builds on the previous one so that data flows seamlessly from ingestion to insights.

## Phase 1 – Data Ingestion & Normalization

Phase 1 reads raw CSV/XLSX exports, detects whether the file is delivered in a wide metric-per-column layout or a long metric/value layout, and converts every dataset into a unified long-form table.

Key capabilities introduced in this phase:

- Calendar-aware week normalization using `data/reference/calendar_map.csv` and the `WeekLookup` helper.
- Configurable column detection based on `config.yaml` so that vendor and ASIN identifiers are resilient to header changes.
- Deduplication logic that protects against double-counted weeks when multiple files are appended.
- Structured logging plus persisted ingestion state, enabling incremental runs in Phase 6.

Later phases will extend this canonical dataset with scores, commentary, forecasts, and dashboard outputs.

## Phase 2 – Scoring Engine

Phase 2 consumes the normalized long-form dataset and produces weighted performance scorecards for both vendors and ASINs. The
engine is fully configuration-driven through `config.yaml`, enabling analytics teams to tweak metric weights, threshold bands,
and improvement rules without code changes.

Highlights of the scoring architecture:

- Metric definitions (weight + direction) are assembled into a reusable `MetricConfig` map that downstream commentary and
  forecasting phases can re-use for consistent semantics.
- Normalization leverages min–max scaling with direction-aware inversion so metrics where “lower is better” (e.g., SoROOS) score
  correctly alongside “higher is better” measures such as GMS or Fill Rate.
- Composite scores scale to a 0–1000 range with 1/1000 precision to provide fine-grained ranking granularity across the vendor
  portfolio.
- An audit dataframe captures `Raw_Value`, `Normalized_Value`, `Weight`, `Weighted_Score`, qualitative performance categories,
  and rule-based improvement flags for each metric and entity. This audit surface enables Phase 3 commentary templating and Phase
  4 forecasting diagnostics.
- Outputs persist to CSV/Parquet plus a JSON metadata file summarizing the run (record counts, score ranges, top performers),
  establishing the foundation for later dashboard assembly and pipeline orchestration.

## Phase 3 – Commentary Engine

Phase 3 layers a narrative generation engine on top of the normalized dataset and scorecards. It evaluates week-over-week
movement for each vendor and ASIN, classifies the trend direction using configurable change bands, and renders multi-topic
commentary (performance summaries, opportunities, risks) with templates stored in `config.yaml`.

Key responsibilities of the commentary phase:

- Load vendor/ASIN scorecards alongside the normalized week-level metrics so commentary reflects both absolute values and
  normalised performance scores.
- Aggregate the latest two weeks for every entity/metric combination to detect change magnitude, handling missing history with
  neutral messaging.
- Apply template-driven rendering with safe placeholder substitution so analytics teams can tune tone, terminology, and
  thresholds without code changes.
- Persist structured commentary tables (CSV/Parquet) under `data/processed/commentary/` for later consumption by the dashboard
  generator and alerting workflows.

## Phase 4 – Forecasting Engine

Phase 4 adds an ensemble forecasting layer that projects each configured metric for vendors and ASINs. The engine blends
Prophet, LSTM (TensorFlow), and a resilient baseline forecaster so production runs remain stable even when optional dependencies
are unavailable. Forecast configuration resides entirely in `config.yaml`, enabling data teams to adjust horizons, ensemble
weights, and history expectations without code changes.

Highlights of the forecasting architecture:

- Normalized history is aggregated per entity/metric to create clean weekly time series with canonical week start dates.
- Each series flows through Prophet/LSTM components when available; a deterministic baseline supplies predictions and
  uncertainty bounds when minimum history thresholds are not met.
- The ensemble combiner supports configurable weighting and emits potential gain calculations that factor in whether “higher”
  or “lower” values are desirable for the metric (re-using direction metadata from scoring when needed).
- Outputs persist to CSV/Parquet under `data/processed/forecasts/` along with JSON metadata capturing run statistics, history
  coverage, and the modelling methods applied. These artifacts feed Phase 5 dashboards and Phase 6 orchestration/alerts.

## Phase 5 – Dashboard Generator

Phase 5 assembles a presentation-ready Excel dashboard that consolidates normalized data, weighted scorecards, commentary, and
forecasts. Configuration continues to drive workbook structure so analytics teams can change sheet order, styles, and
thresholds without modifying Python modules.

Key responsibilities of the dashboard phase:

- Load processed tables from Phases 1–4 and enrich vendor/ASIN scorecards with linked commentary narratives and forecast
  potential gains for quick executive review.
- Produce a `Summary` sheet that captures run timestamp, vendor counts, composite score statistics, top forecast opportunities,
  and overall data freshness to make pipeline health visible.
- Create dedicated sheets for vendor scorecards, ASIN scorecards, commentary, and forecasts with configurable column widths,
  precision, and colour schemes defined in `config.yaml`.
- Apply conditional formatting (colour scales for composite scores, fills for commentary types) so hot spots and risks are easy
  to scan before Phase 6 automation.
- Persist timestamped workbooks to `data/processed/dashboard/`, ready to be shared or embedded in reporting workflows.

## Phase 6 – Pipeline Orchestration & Deployment

Phase 6 introduces the `brightstar_pipeline` orchestrator that unifies every phase behind a single CLI and logging surface.
This layer enables reproducible, idempotent production runs while keeping the modular architecture intact for development.

Key responsibilities of the pipeline phase:

- Validate `config.yaml` before any work begins, ensuring critical sections/paths exist and directories are created.
- Execute Phases 1–5 sequentially with detailed timing metrics, summary strings, and consolidated console output so operators
  can verify outcomes rapidly.
- Support granular execution controls (`--phases`, `--start-phase`, `--end-phase`) plus a `--resume` switch that skips phases
  when their configured outputs already exist—ideal for incremental reruns.
- Provide centralised logging to `logs/brightstar_pipeline.log`, complementing the existing per-phase log files for deep
  troubleshooting.
- Integrate with CI/CD and deployment scripts (`scripts/run_pipeline.sh`, the `brightstar-pipeline` console entry point) so the
  entire analytics suite can be scheduled, packaged, and distributed confidently.
