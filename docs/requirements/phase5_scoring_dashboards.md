### Phase 5 Scoring Dashboards — Requirements

#### Purpose
- Provide two analysis-ready sheets in the weekly Excel workbook:
  - Vendor_Scoring: vendor-level composite and dimension scores with key KPIs.
  - ASIN_Scoring: ASIN-level composite/dimension scores, KPIs, and Product Master attributes.
- Remain fully offline and backward compatible. No heavy logic inside Excel formatting; all calculations performed in DataFrames.

#### Data Inputs
- Phase 2 scorecards (paths from `config.yaml`):
  - `paths.scorecard_vendor_parquet|csv`
  - `paths.scorecard_asin_parquet|csv`
- Normalized Phase 1 data (`paths.normalized_output`) for KPIs and latest-week selection.
- Product Master (project-root relative): `paths.product_master_path` (CSV) with columns:
  - `vendor_code, vendor_name, asin, item_name_raw, item_name_clean, brand, category, sub_category, pack_size, pack_unit, notes`

#### Outputs
- Excel workbook (Phase 5) gains two sheets:
  - `Vendor_Scoring`
  - `ASIN_Scoring`
  - Sheets are filter-ready (Excel Tables), with frozen panes and conditional formatting.

#### Column Layouts (ordered)
- Vendor_Scoring
  - Identity: `vendor_code, vendor_name`
  - Scores: `Composite_Score, Sales_Score, Margin_Score, Availability_Score, Traffic_Score`
  - Sales/Traffic: `Product_GMS, GV`
  - Margin/Economics: `Net_PPM`
  - Availability: `SoROOS, Fill_Rate_Sourceable, Vendor_Confirmation_Rate_Sourceable, On_Hand_Inventory_Units_Sellable, Total_Inventory_Units`

- ASIN_Scoring
  - Identity: `vendor_code, vendor_name, asin, Item_Name, Brand, Category, Sub_Category`
  - Scores: `Composite_Score, Sales_Score, Margin_Score, Availability_Score, Traffic_Score`
  - Sales/Traffic: `Product_GMS, GV`
  - Margin/Economics: `Net_PPM`
  - Availability: `SoROOS, On_Hand_Inventory_Units_Sellable, Total_Inventory_Units`

Note: `Item_Name` maps to `item_name_clean` from Product Master; `item_name_raw` remains only in the Product Master CSV.

#### Formatting Rules (config-driven)
- Configure under `dashboard.scoring` in `config.yaml`:
  - `vendor_sheet_name`, `asin_sheet_name` (defaults: `Vendor_Scoring`, `ASIN_Scoring`)
  - `composite_color_scale`, `dimension_color_scale`, `margin_color_scale` color triplets
  - `availability_icon_set` (defaults to apply to `SoROOS` with traffic lights)
- Header bands (dashboard.formatting or safe defaults):
  - Identity section: light gray header
  - Scores section: blue header + greenish color scales for scores
  - Margin section: orange header + color scale for `Net_PPM`
  - Availability section: purple header + icon set for `SoROOS`
- Freeze panes after identity columns; apply autofilter and convert ranges to Excel Tables for filterability.

#### Dependencies and Config Sections
- Paths: `paths.normalized_output`, `paths.scorecard_vendor_*`, `paths.scorecard_asin_*`, `paths.product_master_path`, `paths.logs_dir`, `dashboard.output_dir`.
- Dashboard: `dashboard.scoring` for sheet names and formatting; base `dashboard.formatting` for fonts, borders, column widths, and general color scales.

#### Behavior and Safeguards
- Weekly grain assumed; “current period” is the latest `week_start` in normalized data.
- All computations are performed in Python DataFrames (no Excel formulas).
- Empty-safe: if inputs are missing/empty, sheets are created with headers only.
- Backward compatible: if `dashboard.scoring` or Product Master is absent, defaults apply and enrichment is skipped gracefully.
