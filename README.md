
### Calendar Mapping Engine

Purpose: Robustly parse vendor‑supplied week labels into a canonical week id (`YYYYWNN`), enrich rows with unified calendar attributes, and set `dq_unmapped_week` when mapping fails.

Supported raw formats (parsed in this order):
- Direct canonical: `^YYYYWNN$` (e.g., `2025W36`)
- Hyphenated: `YYYY-WNN` (e.g., `2025-W36` → `2025W36`)
- With leading W and spaced year: `WNN YYYY` (e.g., `W36 2025` → `2025W36`)
- Date range: `DD/MM/YYYY-DD/MM/YYYY` or `YYYY-MM-DD–YYYY-MM-DD` (en dash or hyphen). The start date determines the canonical ISO week.
- Single date: When no week column exists, a single date column such as `week_start` (configurable) can be mapped to the unified week that contains that date.

Fallback lookup:
- If a raw label cannot be parsed, the engine attempts a bridge lookup using `data/reference/calendar_map.csv` by joining on `raw_week_label` to obtain `canonical_week_id`.

Unified calendar enrichment (from `data/reference/unified_calendar_map.csv`):
- Adds: `calendar_year`, `calendar_week_number`, `week_start_date`, `week_end_date`, `month_label`, `quarter_label` (and `month_number` when available).

DQ flag behavior:
- `dq_unmapped_week = True` when `canonical_week_id` is blank or the unified calendar join did not succeed (e.g., no `week_start_date`). Otherwise `False`.

Configuration (`config/phase1_calendar.yaml`):
- Backward‑compatible flat keys are supported, but the preferred structure is nested:

```
calendar:
  raw_week_column: "week"           # optional; auto‑detected if missing
  raw_start_column: "week_start"     # optional
  raw_end_column: "week_end"         # optional
  unified_calendar_path: "data/reference/unified_calendar_map.csv"
  raw_calendar_map_path: "data/reference/calendar_map.csv"
  date_formats:
    - "%d/%m/%Y"
    - "%Y-%m-%d"
```

Integration in Phase 1 Loader:
- The loader constructs a `CalendarConfig` from the YAML (nested or flat keys) and calls the calendar mapper before metric normalization. Calendar fields appear in all Phase 1 outputs.

---

### Brightlight Phase 2 Scoring Engine

Purpose: Compute robust, archetype-aware performance scores from Phase 1 validated data. The engine ingests the calendar- and dimension-enriched Phase 1 parquet, applies metric transforms (winsorization, robust scaling, normalization), computes growth deltas, aggregates dimension-level scores, applies category→archetype weights, and gates results with a readiness layer before writing Phase 2 scoring artifacts.

Key capabilities:
- Input validation and gating:
  - Verifies presence of `asin_id/asin`, `vendor_id`, and unified calendar fields: `canonical_week_id`, `calendar_year`, `calendar_week_number`, `week_start_date`, `week_end_date`.
  - Aborts if `dq_unmapped_week > 0` or `dq_invalid_ids > 0`, or any required calendar fields are missing.
  - Derives sortable `week` (int `YYYYWW`) from `canonical_week_id` when not present.
- Metric transforms (config-driven via `config/scoring_config.yaml`):
  - Winsorization with quantile `limits` per metric (e.g., `[0.01, 0.99]`).
  - Normalization methods: `zscore`, `robust_zscore` (median/MAD), `percentile`, and optional `minmax` (only if enabled).
  - Direction handling with `higher_is_better`; metrics where lower is better are inverted post-normalization.
- Growth deltas (per-metric, group by ASIN and order by week):
  - Week-over-week absolute and percent deltas: `gms_wow_abs`, `gms_wow_pct`, etc.
  - L4W vs prior L4W window deltas: `*_l4w_delta_abs`, `*_l4w_delta_pct`.
  - L12W vs prior L12W window deltas: `*_l12w_delta_abs`, `*_l12w_delta_pct`.
- Dimension scores (weighted aggregates):
  - Profitability: `cp`, `cppu`, `net_ppm`, `contribution_margin`.
  - Growth: growth signals from `gms/units/gv` deltas.
  - Availability: `instock_pct`, `soroos_pct`, `fo_pct` (direction-aware).
  - Inventory risk: `onhand_units`, `aging_days`, `stockout_count` (lower is better).
  - Quality: `returns_rate`, `defect_rate` (lower is better).
- Archetype composite:
  - Category→archetype mapping via `category_archetype_mapping` in `scoring_config.yaml` (with a default archetype).
  - Archetype-specific weights across dimension scores using `archetypes` matrix.
  - Composite score computed row-wise as Σ(dim_score × weight).
- Readiness layer:
  - Per-entity (ASIN×Vendor) diagnostics: history length, null share, volatility (MAD), anomaly counts (robust z).
  - `readiness` thresholds in config: `min_history_weeks`, `null_pct_max`, `volatility_max`, `anomaly_max`.
  - Rows deemed not ready are masked for `__norm` metrics prior to scoring; `readiness_flag` included in readiness export.

Configuration (`config/scoring_config.yaml`):
- `metrics`: winsorization limits, normalization method, and `higher_is_better` per metric.
- `dimensions`: metric weights for dimension scores.
- `archetypes`: weightings of dimensions per archetype (Fresh/Ambient/Beauty/Baby/Health).
- `category_archetype_mapping`: rules to map categories to archetypes (by `gl_code`, `category_name`, or substring).
- `readiness`: thresholds that govern masking and inclusion.

Run command:

```
python -m src.phase2.scoring_engine ^
  --phase1 "C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase1\phase1_validated.parquet" ^
  --dimensions config/dimension_paths.yaml ^
  --scoring_config config/scoring_config.yaml ^
  --output "C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase2" ^
  --calendar_config config/phase2_calendar.yaml  
```
(Note: `--calendar_config` is optional. If omitted or the file is missing, the engine uses neutral defaults.)

Outputs (written to the Brightlight Phase 2 directory):
- `scoring_detailed.parquet` — ASIN×week rows with ids, calendar fields, seasonality columns, normalized metrics (`__norm`), season-aware flags, dimension scores, archetype, and composite score.
- `scoring_summary.parquet` — vendor/category level aggregates (mean dimension/composite scores and ASIN counts).
- `metric_readiness.parquet` — per-entity readiness flags/diagnostics.
- `scoring_metadata.json` — run parameters, config summary, counts, and timestamps.

Notes:
- Metric names and canonical columns are preserved; no new metrics or dimensions are introduced beyond configuration.
- The engine is fully driven by `config/scoring_config.yaml` and the Brightlight dimension paths in `config/dimension_paths.yaml`.

---

### Phase 2c – Metric Coverage & Score Reliability

Purpose:
- Evaluate the reliability of metric signals at the ASIN level based on weekly coverage and activity.
- Adapt dimension scoring to only use metrics that meet coverage thresholds, renormalizing weights over available metrics.
- Enforce composite score reliability by requiring a minimum number of valid dimensions.

How it works:
1) Metric coverage computation (optional, enabled when `coverage:` exists in `config/scoring_config.yaml`):
   - For every ASIN × metric we compute:
     - `weeks_total`, `weeks_non_null`, `weeks_non_zero`
     - `coverage_pct = weeks_non_null / weeks_total`
     - `signal_pct = weeks_non_zero / weeks_total`
     - Flags: `metric_sparse`, `metric_short_history`, `metric_all_zero`, and `metric_ok`.
   - Thresholds are controlled by `coverage:` in `scoring_config.yaml`:
     - `min_coverage_pct` (default 0.6)
     - `min_history_weeks` (default 6)
     - `treat_all_zero_as_inactive` (default true)
     - `min_dimensions_for_composite` (default 2)

2) Adaptive dimension scoring:
   - For each dimension, metrics with `metric_ok = true` are used; others are excluded.
   - Weights are renormalized over the subset of available metrics for that row.
   - If no metrics are valid for a dimension, the dimension score is set to `NaN`.
   - Additional annotations are written per row:
     - `<dim>_metric_count` (configured count), `<dim>_metrics_used` (list), `<dim>_coverage_flag` in {`OK`, `PARTIAL`, `NO_VALID_METRICS`}.

3) Composite score reliability:
   - The composite score only combines dimensions with non-null scores.
   - If the number of usable dimensions is less than `min_dimensions_for_composite`, the engine sets:
     - `composite_score = NaN`
     - `composite_coverage_flag = "UNRELIABLE"`

Artifacts:
- `metric_coverage.parquet` — per ASIN × metric coverage and flags.
- `metric_coverage_summary.parquet` — grouped aggregates by vendor/category.
- `scoring_detailed.parquet` — now includes `composite_coverage_flag` when coverage is enabled.
- `scoring_metadata.json` — extended with a `coverage` section: enabled flag, counts, and paths to artifacts.

Configuration (append to `config/scoring_config.yaml` or use defaults already provided):

```
coverage:
  min_coverage_pct: 0.6
  min_history_weeks: 6
  treat_all_zero_as_inactive: true
  enforce_strict_for_dimensions: ["growth", "availability"]
  min_dimensions_for_composite: 2
```

Backward compatibility:
- If the `coverage:` block is omitted, all new behavior is disabled. Scores and outputs remain identical to earlier versions (no extra columns are required by consumers).

How to run & validate:
1. Run Phase 1 and Phase 2 as usual.
2. Confirm new artifacts in the Phase 2 output folder:
   - `metric_coverage.parquet`, `metric_coverage_summary.parquet`, `scoring_metadata.json` contains `coverage` section.
3. Inspect `scoring_detailed.parquet`:
   - Verify dimension coverage annotations and `composite_coverage_flag` appear when `coverage:` is present.
4. Ensure that when removing or commenting out the `coverage:` block in `config/scoring_config.yaml`, these additions do not appear and scoring remains unchanged.

#### Calendar & Seasonality (Session 2b)

What’s new in Phase 2:
- Seasonality enrichment for each week with the following columns now present in detailed outputs and propagated to other artifacts where calendar fields are present:
  - `season_label`, `season_type`, `event_code`, `is_peak_week`, `is_event_week`, `quarter_label`.
- Lightweight season-aware flags (do not affect numeric deltas or scores):
  - `is_peak_to_peak_transition`: previous week was a peak and current week is also peak but `season_label` changed (e.g., BF → CM).
  - `is_offpeak_recovery_week`: previous week was `is_peak_week == True` and current week is `False`.

How enrichment is sourced (precedence):
1. Phase 1 season fields (if already present and `calendar.use_phase1_enrichment: true`).
2. Unified calendar join (if enabled via `calendar.fallback_to_unified_calendar: true` and `config/phase2_calendar.yaml` points to a valid CSV at `unified_calendar_path`).
3. Neutral defaults (when neither of the above is available).

Configuration:
- `config/scoring_config.yaml` now includes a `calendar` section with switches:
  - `use_phase1_enrichment` (default true)
  - `fallback_to_unified_calendar` (default true)
  - `seasonality_flags_enabled` (default true)
- `config/phase2_calendar.yaml` (optional) controls the unified calendar path, join key, field names, and defaults:
  - `unified_calendar_path`: default `data/reference/unified_calendar_map.csv` (CSV with at least `canonical_week_id` and season fields)
  - `join_key`: default `canonical_week_id`
  - `fields`: expected field names
  - `defaults`: safe fallback values when fields are missing or null

Metadata (`scoring_metadata.json`) additions under `calendar`:
- `season_source`: one of `phase1`, `unified_calendar`, or `defaults`.
- `counts_season_type`: frequency map of `season_type`.
- `counts_event_code`: frequency map of `event_code`.
- `peak_week_count` and `total_weeks`.

Safety and compatibility:
- If `phase2_calendar.yaml` is missing or unreadable, or the unified calendar CSV is missing/malformed, the engine falls back to defaults and continues, marking `season_source = "defaults"` in metadata.
- All new columns are nullable; numeric delta and score formulas are unchanged.

---

### Metric Registry (Session 2)

Purpose: Centralize metric definitions and synonyms, auto-discover metric columns from Phase 1 outputs, and ensure Phase 2 operates on canonical metric names while remaining backward compatible.

What it does:
- Loads canonical metric definitions and synonyms from `config/metric_registry.yaml`.
- Scans Phase 1 dataset to discover numeric columns and resolve them to canonical metrics using case-insensitive, normalized matching (spaces/underscores/% removed).
- Creates canonical alias columns in-memory so downstream transforms run on canonical names, regardless of raw column header variations.
- Exports a human-readable preview `metric_registry_preview.csv` into the Phase 2 output directory showing raw→canonical mapping, match status (exact/synonym/unregistered), and key metadata (directionality, default_transform, family).
- Surfaces metrics not present in the registry in `scoring_metadata.json` under `metric_registry.unregistered_metrics` for later registry extension. These are not used in scoring in this session.

Files and paths:
- Registry config: `config/metric_registry.yaml`
- Preview CSV (generated): `<phase2_output>/metric_registry_preview.csv`

Scoring metadata additions (in `<phase2_output>/scoring_metadata.json`):
- inputs.metric_registry_config_path
- metric_registry.discovered_metric_count
- metric_registry.registered_metric_count
- metric_registry.missing_required_metrics
- metric_registry.unregistered_metrics
- metric_registry.present_canonical

Backward compatibility:
- No Phase 1 schemas were changed.
- Existing Phase 2 formulas and dimension logic are unchanged; only metric selection/mapping was adapted.
- Legacy mappings (e.g., `gms_value`→`gms`, `ordered_units`→`units`, `gv_value`→`gv`) are preserved as a safety net.

---

### Brightlight Phase 3 Commentary Engine (AVS-style)

Purpose: Convert deterministic Phase 2 scoring outputs into high-impact AVS WBR callouts. The engine ingests Phase 2 artifacts, detects meaningful events (metric movements, dimension shifts, inventory and buyability risks, readiness warnings), filters and prioritizes them, and formats prescriptive callouts.

Inputs (read from the Phase 2 output directory provided via --phase2):
- scoring_detailed.parquet
- scoring_summary.parquet
- metric_readiness.parquet
- scoring_metadata.json

Additional joins:
- dim_asin and dim_vendor from `config/dimension_paths.yaml`

Configuration:
- `config/commentary_config.yaml` controls thresholds, minimum severity, and vendor caps.

Detection coverage:
- Composite and metric movements (WoW) including GMS, Units, GV, Instock, FO%, Returns, ASP
- Dimension shifts: profitability, growth, availability, inventory_risk, quality
- Inventory risk flags: HCPO_present, LCPO_only, Aging, Overstock, OOS_risk
- Buyability failures: listing_blocked, suppressed_from_website, no_retail_offer, no_inventory_offer
- Readiness warnings: low completeness, missing weeks, null-heavy metrics (as available)

Filtering and prioritization:
- Keep only high-impact events per config thresholds
- Deduplicate correlated signals per ASIN×Vendor×Week×Signal
- Cap 3–7 callouts per vendor deterministically

Formatting (AVS WBR standard):
- Setup block with precise metric/percent deltas and context
- Driver block explaining root cause using metric/dimension signals
- Next steps block with 2–3 prescriptive, deadline-oriented actions

Outputs (written to the Phase 3 output directory provided via --output):
- phase3_commentary.parquet — final callout rows with text blocks
- phase3_commentary.json — JSON lines with the same columns
- phase3_metadata.json — run metadata including timestamp, config and input hashes, counts, and thresholds

Required columns in outputs:
- vendor_id, asin_id (nullable), canonical_week_id, signal_type, severity, setup_text, driver_text, next_steps_text

Run command:

```
python -m src.phase3.commentary_engine ^
  --phase2 "C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase2" ^
  --dimensions config/dimension_paths.yaml ^
  --commentary_config config/commentary_config.yaml ^
  --output "C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase3"
```

CLI returns a compact JSON with:
- commentary row count
- top 3 callouts (vendor|asin|signal: setup_text)
- missing field warnings (if any)
- outputs: absolute file paths to phase3_commentary.parquet, phase3_commentary.json, and phase3_metadata.json
 
---

### Phase 1e Cleanup & Logger Integration

Purpose: After Phase 1e integration, provide a deterministic maintenance utility to archive obsolete files and validate connectivity between Phase 1 modules and configs. A unified JSON logger captures cleanup activity and validation results.

What it does:
- Automatic file cleanup (safe archival, no deletions):
  - Scans these locations for unused/legacy artifacts:
    - C:\Users\selsherb\PycharmProjects\brightstar\data\
    - C:\Users\selsherb\PycharmProjects\brightstar\src\phase1\temp\
  - Moves matched files into an archive folder: C:\Users\selsherb\PycharmProjects\brightstar\archive\YYYYMMDD_HHMM\
  - Targets typical obsolete patterns only: *_old.py, *_backup.parquet, *.tmp|.bak|.old, temp_*.csv, *legacy*.(yaml|yml|json|csv|parquet)
  - Retains and never archives:
    - Active schema/config files under config\
    - All files under data\reference\
    - metric_registry_utils.py
    - Validated Phase 1 outputs under data\processed\phase1\
    - Active src\phase1\ modules (excluding src\phase1\temp\)

- Connectivity validation:
  - Attempts to import Phase 1 modules (loader, dq_checks, validation_report, calendar_mapper, id_normalizer).
  - Verifies config presence: config\dimension_paths.yaml and config\phase1_calendar.yaml under the project root.
  - Reads phase1_run_log.json from your Phase 1 output directory and checks that all referenced artifacts exist.

- Unified JSON logger (src/utils/logger.py):
  - append_log(phase, message, severity="INFO") — JSONL append, non-blocking.
  - log_cleanup_activity(moved_files, archive_path) — summarizes archival.
  - log_validation_summary(check_results) — summarizes connectivity checks.
  - Cleanup writes to: C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase1\phase1_cleanup_log.json

How to run (Windows PowerShell):

```
python -m src.utils.cleanup_phase1 ^
  --project_root "C:\Users\selsherb\PycharmProjects\brightstar" ^
  --archive_dir "C:\Users\selsherb\PycharmProjects\brightstar\archive" ^
  --log_dir "C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase1"
```

Expected outputs:
- Archive folder created: C:\Users\selsherb\PycharmProjects\brightstar\archive\YYYYMMDD_HHMM\ containing archived files, preserving relative paths.
- Log report JSONL: C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase1\phase1_cleanup_log.json with entries for archival and validation.
- Console JSON summary: archived_count, archive_path, checks (passed/failed and details), log_file, timestamp, user.

Notes and constraints:
- The utility never deletes files; it moves only items matching explicit obsolete patterns.
- Files under data\reference\ and active modules/configs are always retained.
- The logger is append-only JSON lines and ignores IO errors to avoid blocking pipelines.
- No changes to Phase 2+ modules or outputs.

---

### Phase 1e Enrichment (Masterfile + Contra + Unified Calendar)

Objective: Upgrade Phase 1 ingestion to integrate Masterfile sheets (Info, Info2, Buyability, Inventory), Contra COGS Agreements, and unified calendar maps into a single enriched, auditable universe with DQ severities and escalation artifacts.

What’s included in this patch:
- Masterfile integration
  - Info: canonical ASIN attributes (item_name, brand_name, category, sub_category, short item name when available).
  - Info2: canonical vendor lookup.
  - Buyability: `is_buyable` joined on ASIN; if missing, `dq_buyability_missing` is set and `phase1_missing_buyability.json` is emitted.
  - Inventory: weekly aggregation and risk flags supported when provided; audit summary written to `phase1_inventory_audit.json`.
- Contra COGS agreements
  - Dimension builder support to parse agreements and explode included ASINs to a bridge; loader emits `phase1_contra_audit.json` summary and default neutral fields when inputs are absent.
- Unified calendar
  - Loader uses the unified calendar map to enrich with `canonical_week_id`, `calendar_year`, `calendar_week_number`, `week_start_date`, `week_end_date`, and any additional labels present in the source (month/quarter/season/holiday/is_peak_week when provided). Coverage summary is written to `phase1_week_completeness.parquet`.
- Data Quality (DQ) flags and severities
  - New flags: `dq_invalid_vendor_code`, `dq_misplaced_id`, `dq_buyability_missing`, `dq_unbuyable`, `dq_inventory_risk`, `dq_contra_mapping_broken`, `dq_unknown_brand`, `dq_unknown_category`, `dq_week_coverage_low`.
  - Severity mapping: Critical / High / Medium / Low. Critical issues are logged to `phase1_fatal_errors.log` for escalation; High/Medium rows are exported to `phase1_attention_required.csv`.
- Metric normalization and compatibility
  - Continues to use the existing metric registry utilities and schema enforcement; no breaking changes to Phase 2 interfaces.

Outputs added/confirmed under your Phase 1 output directory:
- phase1_validated.parquet
- phase1_validated_view.csv
- phase1_id_issues.csv (misplaced ids quarantine)
- phase1_unbuyable.csv
- phase1_inventory_audit.json
- phase1_contra_audit.json
- phase1_week_completeness.parquet
- phase1_validation_report.json (now includes severity counts)
- phase1_attention_required.csv
- phase1_fatal_errors.log (only when Critical flags detected)
- phase1_missing_buyability.json (only when buyability sheet/context is missing)

Run (Windows PowerShell):

```
python -m src.phase1.loader ^
  --input "C:\Users\selsherb\PycharmProjects\brightstar\data\raw\input_phase1.csv" ^
  --schema config/schema_phase1.yaml ^
  --column_map config/phase1_column_map.yaml ^
  --calendar_config config/phase1_calendar.yaml ^
  --output "C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase1"
```

Validation checklist:
- Row count > 0, and phase1_validated.parquet exists.
- DQ artifacts present (id issues, attention CSV, week completeness, audits).
- No Critical errors logged unless expected; if logged, review `phase1_fatal_errors.log`.
- Calendar joins successful: canonical fields and labels populated.
- phase1_validated includes enriched fields: `item_name`/`short_item_name` (if present), `brand`/`brand_name`, `category`, `sub_category`, `is_buyable`, inventory risk flags, and contra fields (`active_agreement_count`, `contra_active_flag`, `days_to_next_expiry`, `funding_type`).

### Phase 1 Schema-Enforced Loader

Purpose: normalize input rows, enforce the ingestion schema, standardize identifiers, apply DQ checks, and persist validated outputs.

Command:

```
python -m src.phase1.loader --input <path-to-input.csv-or.parquet> --schema config/schema_phase1.yaml --output "C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase1"
```

Behavior:
- Detects CSV or Parquet and loads with dtype="string" (when applicable).
- Normalizes column names to lowercase + stripped.
- Validates row count > 0.
- Loads schema YAML and validates required columns; enforces dtypes; coerces numeric fields.
- Normalizes identifiers:
  - `asin`: uppercase alphanumeric
  - `vendor_id`: uppercase
  - `week`: converts to integer `YYYYWW`
  - `category`: Title Case
- Applies DQ checks and appends flags:
  - `dq_missing`, `dq_invalid_ids`, `dq_numeric_issues`, `dq_out_of_range`, `is_valid_row`
- Writes outputs:
  - `C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase1\phase1_validated.parquet`
  - `C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase1\phase1_validation_report.json`

Schema expectations (config/schema_phase1.yaml):
- `required_columns`: list of required column names
- `dtypes`: mapping of column -> logical dtype (`string`, `int`, `float`, `bool`)
- `numeric_fields`: list of numeric columns to coerce with `to_numeric`
- `id_fields`: list of identifier columns (for downstream awareness)
- `valid_ranges`: optional per-metric ranges (not enforced beyond percent heuristics in this phase)

### Neural Mapping Awareness of Dimensions

The generated dimensions (vendor, ASIN, brand, category, subcategory) are intended to be discoverable by downstream mapping and learning components. No changes were made to Phases 2–6, but these dimensions can be joined in future enhancements for enriched modeling and routing.

### Brightlight Directory Structure

All generated artifacts for these modules are written under the Brightlight root:

```
C:\Users\selsherb\Documents\AVS-E\CL\Brightlight
├── dimensions
│   ├── dim_vendor.parquet
│   ├── dim_asin.parquet
│   ├── dim_brand.parquet
│   ├── dim_category.parquet
│   ├── dim_subcategory.parquet
│   └── dimension_builder_log.json
└── phase1
    ├── phase1_validated.parquet
    └── phase1_validation_report.json
```

These folders are autocreated if missing.

## 📁 Repository Structure

```
brightstar/
├── README.md  
├── requirements.txt  
├── config.yaml  
├── src/  
│   ├── phase1_ingestion.py  
│   ├── phase2_scoring.py  
│   └── …  
├── tests/  
├── .github/  
│   └── workflows/  
└── docs/  
```

## 🌟 Brightlight Phase 2 Scoring Engine

Phase 2 computes normalized metric values, rolls them up into 5 dimension scores, applies an archetype weight model (by category), and produces composite scores per ASIN/week and vendor-level summaries. It also computes a readiness layer to mask unreliable metrics.

What it does:
- Loads Phase 1 validated fact table and joins to dimensions (ASIN, vendor, category, etc.)
- Applies metric winsorization and normalization (zscore, robust_zscore, percentile, optional minmax)
- Computes growth deltas: WoW, L4W vs previous L4W, L12W vs previous L12W
- Computes 5 dimension scores as weighted means of normalized metrics:
  - Profitability (CP, CPPU, Net PPM, Contribution Margin)
  - Growth (GMS/Units/GV growth signals)
  - Availability/Service (Instock, SoROOS, FO%)
  - Inventory Risk (On‑hand, Aging, Stockout frequency)
  - Quality/Returns (Returns/defect signals)
- Applies archetype weights (Fresh, Ambient, Beauty, Baby, Health) to build a composite score per entity
- Computes readiness flags (history length, null%, volatility, anomalies) and masks metrics when not ready
- Writes detailed, summary, readiness, and metadata outputs to your Brightlight Phase 2 folder

Configuration:
- `config/scoring_config.yaml` controls:
  - Metric registry: winsorization limits, normalization method, higher‑is‑better flag
  - Dimension score composition: which normalized metrics and their weights
  - Archetype weights per dimension and category→archetype mapping rules
  - Readiness thresholds (min history weeks, max null%, volatility, anomaly limits)

Run command (Validation Run):

```
python -m src.phase2.scoring_engine ^
    --phase1 "C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase1\phase1_validated.parquet" ^
    --dimensions config/dimension_paths.yaml ^
    --scoring_config config/scoring_config.yaml ^
    --output "C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase2"
```

Outputs are written to:

```
C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase2\
    scoring_detailed.parquet       # per ASIN/week: normalized metrics, 5 dimension scores, composite score, archetype
    scoring_summary.parquet        # vendor-level aggregates of scores + ASIN count
    metric_readiness.parquet       # readiness flags per ASIN/vendor
    scoring_metadata.json          # run metadata (inputs, row counts, configured metrics)
```

Notes:
- Only metrics defined in `config/scoring_config.yaml` are used in scoring; the engine gracefully skips metrics that are missing in the Phase 1 dataset.
- Category→archetype mapping can be configured by category name contains, or extended to use GL codes; defaults to Ambient when unmatched.
- The output directory is autocreated.

---

### Phase 1f: Severity-Driven DQ and Row Validity Engine

Objective:
- Replace the previous "any DQ flag invalidates a row" behavior with a severity-aware system.
- Only CRITICAL-severity flags mark rows invalid; HIGH/MEDIUM/LOW remain visible but do not block downstream scoring or commentary.

How it works:
- Each DQ flag (columns starting with `dq_`) is mapped to a severity bucket using `config/dq_severity.yaml`.
- The loader computes a `dq_severity` column per row as the maximum severity across all active DQ flags.
- `is_valid_row` is set to `False` only when `dq_severity == 'CRITICAL'`; otherwise `True`.

Default severity mapping:
- Provided in `config/dq_severity.yaml` (you can extend/override):
  - CRITICAL: `dq_invalid_asin`, `dq_invalid_vendor_code`, `dq_misplaced_id`, `dq_week_mapping_failed`
  - HIGH: `dq_unbuyable`, `dq_unmapped_week`, `dq_inventory_risk`
  - MEDIUM: `dq_missing_item_name`, `dq_missing_brand`, `dq_missing_category`, `dq_buyability_missing`, `dq_contra_mapping_broken`
  - LOW: `dq_unknown_brand`, `dq_unknown_category`, `dq_archetype_unknown`, percent/out-of-range, numeric cast issues

Outputs (Phase 1):
- `phase1_validated.parquet` now includes `dq_severity` and `is_valid_row`.
- `phase1_validated_view.csv` includes severity-aware flags and remains human-friendly.
- `phase1_severity_audit.csv` contains `asin_id`, `vendor_id`, calendar key(s), all `dq_*` columns, `dq_severity`, and `is_valid_row` for audit.
- `phase1_validation_report.json` adds:
  - `dq_severity_counts`: counts per severity bucket
  - `dq_severity_distribution`: JSON-friendly heatmap by vendor

Backwards compatibility:
- All existing DQ flag column names are preserved.
- Phase 2 remains unchanged and will respect `is_valid_row` for gating/masking as configured. Registry loading, calendar fields, and archetype fields are unaffected.

How to run & validate:
1. Run Phase 1 as usual (loader CLI unchanged). Example:
   ```
   python -m src.phase1.loader ^
     --input "data/raw/sample_export.csv" ^
     --schema config/schema_phase1.yaml ^
     --column_map config/phase1_column_map.yaml ^
     --calendar_config config/phase1_calendar.yaml ^
     --output "C:\\Users\\selsherb\\Documents\\AVS-E\\CL\\Brightlight\\phase1"
   ```
2. Inspect `phase1_validation_report.json` for `dq_severity_counts` and ensure CRITICAL rows are a small subset.
3. Review `phase1_severity_audit.csv` to verify which flags triggered severity classification.
4. Proceed to Phase 2; outputs and load logic remain stable.

## 🛠️ Getting Started

### Prerequisites

* Python 3.12 (recommended)
* `requirements.txt` installed:

  ```bash
  pip install -r requirements.txt
  ```

### Setup

1. Clone the repo:

   ```bash
   git clone https://github.com/yourusername/brightstar.git
   cd brightstar
   ```
2. Update `config.yaml` to reflect your folder paths, column numbers, identifier columns.
3. Place your weekly export file into the `raw_data_dir` defined in config.
4. Run Phase 1 ingestion:

   ```bash
   python -m brightstar.phase1_ingestion --config config.yaml
   ```

   The script prints a phase summary (rows processed, number of weeks, last week ingested) and writes ALL Phase 1 outputs
   exclusively to the mandatory directory:

   ```
   C:\\Users\\selsherb\\Documents\\AVS-E\\CL\\Seif Vendors
   ```

   Outputs include:
   - normalized_phase1.parquet (primary)
   - normalized_phase1.csv (fallback)
   - normalized_phase1.xlsx (human-friendly view)
   - ingestion_state.json
   - unified_calendar_map.csv (if calendar mapping had inconsistencies)
   - schema_snapshot.json, schema_validation.json
   - ingestion_audit.json
   - vendor_summary.csv, asin_summary.csv, week_summary.csv
   - validation_rows.csv (problematic rows, if any)

   Note: Phase 1 logs are also written to the same directory (phase1_ingestion.log).

   A verification helper script is provided to validate the final dataset:

   ```bash
   python scripts/verify_phase1.py
   ```
   This prints a JSON summary of schema completeness, row count, and uniqueness counts.

5. Run Phase 2 scoring to generate vendor/ASIN scorecards:

   ```bash
   python -m brightstar.phase2_scoring --config config.yaml
   ```

   This reads the normalized dataset, applies the configurable metric weights/directions, and writes CSV/Parquet scorecards plus
   an audit trail under `data/processed/scorecards/`.

   **Phase 2 Enhanced Features:**

   Phase 2 now includes robust transforms, penalties, and explainability:

   - **Configurable Transforms**: Apply winsorization, log1p, or logit transforms per metric
   - **Robust Standardization**: Use percentile rank or robust z-scores with rolling windows
   - **Penalty System**: Apply rule-based penalties with auditable JSON breakdown
   - **Contributions**: Per-metric contributions to composite scores with percentage breakdowns
   - **Reliability Scores**: Confidence metrics based on coverage and variance

   **Outputs**:
   - `data/processed/scorecards/`: Traditional long-format scorecards (vendor, ASIN, audit)
   - `data/processed/scoring_matrix.parquet`: Wide-format matrix with per-metric contributions
   - `data/processed/vendor_scoreboard.parquet`: Vendor rankings with composite scores

   **Configuration**: Update `config.yaml` under the `scoring:` section with:
   ```yaml
   scoring:
     rolling_weeks: 8                    # Rolling window for standardization
     weights_must_sum_to_1: true         # Enforce weight normalization
     metrics:
       - name: GMS($)
         direction: higher_is_better     # or lower_is_better
         weight: 0.40
         clip_quantiles: [0.01, 0.99]   # Winsorization bounds
         transform: log1p                # none|log1p|logit
         standardize: robust_z           # percentile|robust_z
         missing_policy: group_median    # drop|group_median|zero
     penalties:
       - name: low-registration-rate
         when: "(metric == 'RegRate%') & (std_value < -1.0)"
         apply: -30
     caps:
       min_score: -200
       max_score: 1000
   ```

6. Run Phase 3 commentary to produce automated narratives:

   ```bash
   python -m brightstar.phase3_commentary --config config.yaml
   ```

   The commentary engine compares week-over-week movements, applies the config-driven templates, and writes CSV/Parquet outputs
   under `data/processed/commentary/` ready for dashboard integration.

7. Run Phase 4 forecasting to project key metrics and potential gains:

   ```bash
   python -m brightstar.phase4_forecasting --config config.yaml
   ```

   The forecasting engine loads normalized history, applies the ensemble (Prophet, LSTM, baseline) models per vendor/ASIN, and
   saves CSV/Parquet outputs under `data/processed/forecasts/` alongside run metadata summarizing methods and history coverage.

8. Run Phase 5 dashboard generation to consolidate insights into Excel:

   ```bash
   python -m brightstar.phase5_dashboard --config config.yaml
   ```

   The dashboard generator loads normalized data, scorecards, commentary, and forecasts, enriches vendor scorecards with linked
   insights, and emits a timestamped workbook under `data/processed/dashboard/` with styled sheets and run summaries.

9. Automate everything with the Phase 6 pipeline (optional when developing, recommended in production):

   ```bash
   python -m brightstar.brightstar_pipeline --config config.yaml
   # or install the package locally and use the console script
   # brightstar-pipeline --config config.yaml --resume
   ```

   The orchestrator validates configuration, executes Phases 1–5 in sequence, supports resumable runs (`--resume`) that skip
   phases with existing outputs, and prints a consolidated run summary.

   A convenience shell wrapper is also available:

   ```bash
   scripts/run_pipeline.sh path/to/config.yaml --resume
   ```

### Tests

Run unit and integration tests to check ingestion, scoring, commentary, forecasting, dashboard, and pipeline orchestration logic:

```bash
pytest -v
```

CI and local developers both execute the full pipeline integration test to ensure end-to-end reproducibility using the bundled
sample data and configuration templates.

## 📊 Usage

After ingestion, following phases calculate scores, run forecasts, and output dashboard workbook. Example:

```bash
python -m brightstar.phase2_scoring --config config.yaml
python -m brightstar.phase3_commentary --config config.yaml
python -m brightstar.phase4_forecasting --config config.yaml
python -m brightstar.phase5_dashboard --config config.yaml
```

Outputs are placed in the `processed_dir` as defined in config. Scorecards are stored under `data/processed/scorecards/`,
commentary narratives under `data/processed/commentary/`, forecast projections under `data/processed/forecasts/`, and Excel
dashboards under `data/processed/dashboard/`.

## 🧰 Phase 6 – Full Pipeline Automation

The `brightstar_pipeline` module coordinates every phase with logging, error handling, and optional resume semantics. Use it
whenever you need deterministic reproducibility or production scheduling:

```bash
python -m brightstar.brightstar_pipeline --config config.yaml --fail-fast
```

Key flags:

* `--phases phase1,phase3` – run a custom subset.
* `--start-phase phase3 --end-phase phase5` – execute a contiguous slice.
* `--resume` – skip phases that already produced outputs according to `config.yaml` paths.

When installed via `setup.py`, the `brightstar-pipeline` console script provides the same interface without referencing the
module path explicitly.

## 📦 Packaging & Deployment

* `setup.py` enables editable installs (`pip install -e .`) and exposes the `brightstar-pipeline` console entry point.
* `scripts/run_pipeline.sh` offers a portable shell wrapper for CI/CD or cron scheduling.
* `.github/workflows/ci-python.yml` now runs unit tests and an end-to-end pipeline smoke test on every push/PR to guarantee the
  shipped configuration and sample data remain reproducible.

## 📈 Forecasting & Scoring Engine

* Scoring: Fine granularity per vendor, per metric, scaling to 1/1000 precision.
* Forecasting: Combines time-series models (Prophet, LSTM) in an ensemble with configurable weights for better prediction of future metric values (e.g., GMS growth, SoROOS improvement).
* Commentary: Automated text templates populated with calculated results (Week-over-Week growth, vendor spotlight, potential points to gain).

## ⚙️ Configuration

The `config.yaml` allows controlling:

* Column number or name for each metric (supports updates easily)
* Week detection mode (date-range pattern, fiscal week map)
* Score thresholds, icon sets (supports 5-icon system, not just 3)
* Forecast horizon (3–4 weeks)
* Output workbook sheet names and layout

## 💡 Future Improvements

* Add richer icon-sets for conditional formatting (5 icons instead of 3)
* Expand dashboard to columns A :P for extended metrics
* Add user prompt interface to dynamically map columns if layout changes
* Publish web-based interactive dashboard (in addition to Excel)
* Enable scheduling of weekly runs (cron or workflow)


## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) (if created) for guidelines.


## Phase 1 — Data Quality Enhancements (ID hygiene, Buyability, Completeness)

The Phase 1 ingestion has been extended to improve upstream data reliability without aborting the pipeline. New Boolean flags are added to `phase1_validated.parquet` and several sidecar artifacts are produced under your Brightlight output root.

New DQ flags added to Phase 1 output:
- `dq_misplaced_id`: The row appears to have swapped identifiers, e.g., `asin_id` looks like a vendor id or `vendor_id` looks like an ASIN (10 uppercase alphanumerics).
- `dq_unbuyable`: The associated ASIN is not buyable per Masterfile Buyability sheet; these rows are excluded from downstream scoring via `is_valid_row=False`.
- `dq_missing_item_name`: No item name available after applying Masterfile-first and raw fallback logic.
- `dq_category_mismatch`: Category/subcategory in Masterfile conflicts with the raw-provided values (when both present).
- `dq_week_coverage_low`: Vendor (and vendor×category when available) has <60% week coverage between its observed min/max weeks over the unified calendar.

Related existing flags continue to apply (e.g., `dq_missing`, `dq_invalid_ids`, `dq_numeric_cast_fail`, `dq_unmapped_week`, etc.). The composite `is_valid_row` is derived as `not any(dq_*)`.

New Phase 1 artifacts (written to `${paths.output_root}/phase1/`):
- `phase1_id_issues.csv`: All rows flagged with `dq_misplaced_id` for quick triage.
- `phase1_unbuyable.csv`: All rows flagged with `dq_unbuyable`.
- `phase1_week_completeness.parquet`: One row per vendor (and vendor×category if available) summarizing `min_week`, `max_week`, `expected_weeks`, `actual_weeks`, `coverage_ratio`, and `dq_week_coverage_low`.

Dimension Builder addition (written to `${paths.output_root}/dimensions/`):
- `dim_buyability.parquet`: Two columns (`asin_id`, `is_buyable`) sourced from the Masterfile `Buyability` sheet and also merged into `dim_asin`.

Item/category fallback and mismatch rules (applied in Phase 1 loader):
- `item_name`: Masterfile value takes precedence. If empty, falls back to the raw item name column. If still empty, `dq_missing_item_name=True`.
- `category` / `sub_category`: Masterfile values take precedence. If empty, falls back to raw. When both present and non-empty but unequal, `dq_category_mismatch=True`.

Run instructions (Phase 1 only):
```
python -m src.phase1.loader \
  --input "<your csv file>" \
  --schema config/schema_phase1.yaml \
  --column_map config/phase1_column_map.yaml \
  --calendar_config config/phase1_calendar.yaml \
  --output "C:\\Users\\<you>\\Documents\\AVS-E\\CL\\Brightlight\\phase1"
```

Ensure the dimension builder has been executed so that `dim_vendor.parquet`, `dim_asin.parquet`, and the new `dim_buyability.parquet` exist at `${paths.output_root}/dimensions/` or as configured in `config/dimension_paths.yaml`.


## Masterfile Auto‑Repair (Brand / Category / Item Name)

To increase robustness against incomplete or inconsistent Masterfile inputs, the Dimension Builder and Phase 1 Loader include deterministic auto‑repair logic. This logic is governed (logged with counts and samples) and remains fully backward‑compatible with downstream phases.

What is repaired:
- Brand repair (Dimension Builder + Phase 1):
  - If `brand_name` is empty, infer from `item_name` using deterministic rules:
    - Tokenize on spaces/dashes.
    - If a token matches a configured alias (case‑insensitive), use the canonical brand.
    - Otherwise prefer ALL‑CAPS tokens (2–15 chars); fallback to the first token (title‑cased).
  - If inference fails, assign `UNKNOWN_BRAND`.
  - DQ flags (Phase 1): `dq_inferred_brand`, `dq_missing_brand`.

- Category / Subcategory repair (Dimension Builder + Phase 1):
  - If `category` is empty but `sub_category` present, derive `category` using a mapping table.
  - If both are empty, assign `UNKNOWN_CATEGORY` (to `category`).
  - Validate hierarchy when both present; `dq_category_mismatch` remains set where Masterfile conflicts with raw values; additional Phase 1 flags `dq_inferred_category`, `dq_missing_category` reflect the repair outcomes.

- Item name fallback (Phase 1):
  - Prefer dimension `item_name`, else raw item_name (captured pre‑alias), else `UNKNOWN_ITEM`.
  - `dq_missing_item_name` is set when `UNKNOWN_ITEM` is used.

Governing configs (optional; patch works without them):
- `config/masterfile_brand_aliases.yaml` — brand aliases. Example:
  ```yaml
  P&G: Procter & Gamble
  PG: Procter & Gamble
  APPLE: Apple
  ```
- `config/masterfile_category_map.yaml` — category mapping and hierarchy, e.g.:
  ```yaml
  subcategory_to_category:
    "Wireless Earbuds": "Electronics"
    "Floor Cleaners": "Home Care"
  valid_hierarchy:
    - { category: "Electronics", sub_category: "Wireless Earbuds" }
    - { category: "Home Care", sub_category: "Floor Cleaners" }
  ```

Builder/Loader telemetry and artifacts:
- Dimension Builder augments `dimension_builder_log.json` with:
  - `repairs.brand` (inferred/unknown counts; sample ASINs)
  - `repairs.category` (inferred/unknown/mismatch counts; sample ASINs)
  - `config.hash` over the optional mapping files.
- Phase 1 writes `phase1_masterfile_repairs.json` under your Phase 1 output directory with:
  - `timestamp`, `input`, `config_hash`
  - `counts` for inferred/missing brand/category and missing item_name
  - `samples` (bounded lists) showing before/after for brand/category/item by `asin_id`

Downstream compatibility:
- No Phase 2+ code is modified.
- `dim_asin.parquet` keeps its schema; repaired values populate `brand_name`, `category`, and `sub_category`.
- Phase 1 outputs (`phase1_validated.parquet`, `phase1_validated_view.csv`) include repaired `brand`/`brand_name`, `category`, `sub_category`, and `item_name`, plus new DQ flags.

How to run (validation):
1) Build Dimensions:
```
python -m src.dimensions.build_dimensions ^
  --input data/reference/Masterfile.xlsx ^
  --output "C:\\Users\\<you>\\Documents\\AVS-E\\CL\\Brightlight\\dimensions"
```
2) Phase 1 ingestion (after building dimensions):
```
python -m src.phase1.loader ^
  --input "<your csv file>" ^
  --schema config/schema_phase1.yaml ^
  --column_map config/phase1_column_map.yaml ^
  --calendar_config config/phase1_calendar.yaml ^
  --output "C:\\Users\\<you>\\Documents\\AVS-E\\CL\\Brightlight\\phase1"
```

## 🧮 Phase 2B — Composite Score Engine, Archetypes, and Readiness Finalizer

Session 2B completes Phase 2 by producing a final ASIN‑level composite score using archetype‑specific weights over dimension scores, along with final readiness gating and vendor/category summaries.

What it does:
- Loads Session 2A dimension signals and Phase 2 dimension scores:
  - profitability_signal, growth_signal, availability_signal, inventory_risk_signal, quality_signal
  - score_profitability, score_growth, score_availability, score_inventory_risk, score_quality
- Maps each ASIN’s category to an archetype per config.category_archetype_mapping.
- Applies archetype‑specific weights from config.archetypes to compute a composite score.
- Optionally rescales scores to [min_score, max_score] per global distribution.
- Finalizes readiness using config.readiness.final thresholds and masks composite_score for non‑ready entities.
- Writes artifacts for downstream Phases (3–4).

Configuration (config/scoring_config.yaml):
- archetypes: per‑archetype weights across the five dimensions. Weights are renormalized if they don’t sum exactly to 1.0.
- category_archetype_mapping: rules to map categories to archetypes (by gl_code, category_name, or substring contains).
- composite:
  - min_score: 0
  - max_score: 100
  - rescale: true
- readiness.final:
  - require_history_weeks: 6
  - max_null_share: 0.25
  - allow_missing_dimensions: false

Outputs (data/processed/scoring):
- composite_scores.parquet — asin/week level with archetype, dimension signals/scores, composite_raw, composite_score, readiness fields.
- composite_summary_vendor.parquet — vendor/week aggregates (mean composite_score, ASIN count).
- composite_summary_category.parquet — category/week aggregates (mean composite_score, ASIN count).
- readiness_final.parquet — readiness diagnostics by ASIN/vendor/week where applicable.
- session2B_metadata.json — hash of config, archetype validation warnings, row counts.

Run (Windows PowerShell):
```
python -m src.phase2.scoring_engine ^
  --phase1 "C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase1\phase1_validated.parquet" ^
  --dimensions config/dimension_paths.yaml ^
  --scoring_config config/scoring_config.yaml ^
  --output "C:\Users\selsherb\Documents\AVS-E\CL\Brightlight\phase2"
```

Validation checklist:
- composite_score non‑null for ready entities; null where readiness_flag is False.
- archetype assigned for all rows; no null archetypes.
- dimension contributions consistent with config weights and signs.
- Calendar fields preserved across artifacts; no dtype drift.
