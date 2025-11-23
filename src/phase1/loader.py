from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd

from .schema_validator import (
    load_schema,
    validate_required_columns,
    enforce_dtypes,
    resolve_column_aliases,
)
from .id_normalizer import (
    normalize_asin,
    normalize_vendor_id,
    normalize_week,
    normalize_category,
    detect_misplaced_ids,
    quarantine_misplaced_ids,
)
from .dq_checks import apply_dq_flags
from .validation_report import build_report, write_report, write_attention_csv
from .calendar_mapper import (
    load_unified_calendar,
    load_raw_calendar_map,
    derive_canonical_week_id,
    CalendarConfig,
)
from .metric_normalizer import normalize_metrics
import yaml
from ..dimensions.build_dimensions import infer_brand_from_item_name  # reuse deterministic inference
from .id_normalizer import normalize_brand, normalize_item_name


def _read_input(input_path: str) -> pd.DataFrame:
    p = Path(input_path)
    if not p.exists():
        raise FileNotFoundError(f"Input not found: {p}")
    ext = p.suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(p, dtype="string")
    elif ext in (".parquet", ".pq"):
        df = pd.read_parquet(p)
        # Ensure string dtype normalization
        for c in df.columns:
            if pd.api.types.is_object_dtype(df[c]):
                df[c] = df[c].astype("string")
    else:
        # Fallback try CSV
        df = pd.read_csv(p, dtype="string")
    # Normalize headers: lowercase + strip
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _normalize_identifiers(df: pd.DataFrame) -> pd.DataFrame:
    res = df.copy()
    # Accept asin or asin_id, produce asin_id canonical
    if "asin_id" in res.columns:
        res["asin_id"] = res["asin_id"].map(normalize_asin)
    if "asin" in res.columns and "asin_id" not in res.columns:
        res["asin_id"] = res["asin"].map(normalize_asin)
    elif "asin" in res.columns:
        # keep asin for backward compatibility, normalized
        res["asin"] = res["asin"].map(normalize_asin)
    if "vendor_id" in res.columns:
        res["vendor_id"] = res["vendor_id"].map(normalize_vendor_id)
    if "week" in res.columns:
        res["week"] = res["week"].map(normalize_week)
    if "category" in res.columns:
        res["category"] = res["category"].map(normalize_category)
    return res


def load_phase1(
    input_path: str,
    schema_path: str,
    column_map_path: Optional[str] = None,
    calendar_config: Optional[str] = None,
    output_base: str = ".",
) -> pd.DataFrame:
    """Phase 1 Overhauled Loader.

    Steps:
      - Read input (csv/parquet) with dtype='string' and normalized headers
      - Resolve column aliases against column_map_path
      - Validate schema and enforce dtypes
      - Normalize identifiers
      - Calendar mapping to canonical_week_id and enrich with calendar dims
      - Normalize metrics per metric config (loaded from calendar_config neighbor metrics file if available)
      - Join Brightlight dimensions on asin_id
      - Data Quality flags
      - Write parquet, curated CSV view, validation report, and run log
    """

    # Backward compatibility shim for old signature (input, schema, output_base)
    # If third positional argument looks like an output directory (not a YAML),
    # treat it as output_base and use default column_map if available.
    if column_map_path and not str(column_map_path).lower().endswith((".yaml", ".yml")) and output_base == ".":
        # Shift arguments
        output_base = column_map_path
        # try default column map
        default_colmap = Path("config") / "phase1_column_map.yaml"
        column_map_path = str(default_colmap) if default_colmap.exists() else None

    # Load
    df = _read_input(input_path)
    # Preserve potential raw attributes for fallback logic
    raw_item_name = df.get("item_name").copy() if "item_name" in df.columns else None
    raw_category = df.get("category").copy() if "category" in df.columns else None
    raw_subcategory = df.get("sub_category").copy() if "sub_category" in df.columns else None

    if df.shape[0] == 0:
        raise ValueError("Input file has no rows")

    # Column alias resolution
    if column_map_path and Path(column_map_path).exists():
        df = resolve_column_aliases(df, column_map_path)

    # Load schema
    schema = load_schema(schema_path)

    # Validate columns and enforce types
    validate_required_columns(df, schema)
    df = enforce_dtypes(df, schema)

    # Normalize IDs
    df = _normalize_identifiers(df)
    # ID hygiene: detect misplaced IDs early
    df = detect_misplaced_ids(df)

    # Calendar mapping
    cal_cfg: Dict[str, Any] = {}
    if calendar_config and Path(calendar_config).exists():
        with open(calendar_config, "r", encoding="utf-8") as f:
            cal_cfg = yaml.safe_load(f) or {}
    # Build CalendarConfig (supports nested and flat keys)
    cal = CalendarConfig.from_mapping(cal_cfg)
    unified_calendar = load_unified_calendar(cal.unified_calendar_path)
    raw_calendar_map = load_raw_calendar_map(cal.raw_calendar_map_path or "")
    df = derive_canonical_week_id(df, cal, unified_calendar, raw_calendar_map)

    # Metric normalization
    # Load metric config path: default beside calendar_config (phase1_metrics.yaml) or config/phase1_metrics.yaml
    metric_cfg_path = None
    if cal_cfg.get("metric_config"):
        metric_cfg_path = cal_cfg.get("metric_config")
    else:
        # try sibling file or default location
        sibling = Path(calendar_config).with_name("phase1_metrics.yaml") if calendar_config else None
        if sibling and sibling.exists():
            metric_cfg_path = str(sibling)
        else:
            default_metrics = Path("config") / "phase1_metrics.yaml"
            if default_metrics.exists():
                metric_cfg_path = str(default_metrics)
    metric_cfg: Dict[str, Any] = {}
    if metric_cfg_path and Path(metric_cfg_path).exists():
        with open(metric_cfg_path, "r", encoding="utf-8") as f:
            metric_cfg = yaml.safe_load(f) or {}
    df = normalize_metrics(df, metric_cfg)

    # Join Dimensions
    dim_paths_cfg = Path("config") / "dimension_paths.yaml"
    vendor_name_col = "vendor_name"
    if dim_paths_cfg.exists():
        with open(dim_paths_cfg, "r", encoding="utf-8") as f:
            dp = yaml.safe_load(f) or {}
        root = Path(dp.get("default_output_root", "C:\\Users\\selsherb\\Documents\\AVS-E\\CL\\Brightlight\\dimensions"))
        files = dp.get("files", {})
        def _load_dim(name: str) -> pd.DataFrame:
            fn = files.get(name)
            if not fn:
                return pd.DataFrame()
            p = root / fn
            if not p.exists():
                return pd.DataFrame()
            return pd.read_parquet(p)

        dim_asin = _load_dim("dim_asin")
        dim_vendor = _load_dim("dim_vendor")
        dim_category = _load_dim("dim_category")
        dim_category_intel = _load_dim("dim_category_intelligence")
        dim_buyability = _load_dim("dim_buyability")
        # Optional contra dimensions
        dim_contra_agreement = _load_dim("dim_contra_agreement")
        bridge_contra_asin = _load_dim("bridge_contra_asin")

        if not dim_asin.empty and "asin_id" in df.columns:
            # Normalize header cases
            dim_asin.columns = [str(c).strip().lower() for c in dim_asin.columns]
            # expected columns: asin_id, vendor_id, vendor_name, item_name, brand_name, category, sub_category
            join_cols = [c for c in ["vendor_id", "vendor_name", "item_name", "brand_name", "category", "sub_category", "is_buyable"] if c in dim_asin.columns]
            df = df.merge(dim_asin[["asin_id", *join_cols]].drop_duplicates("asin_id"), on="asin_id", how="left")
            # rename brand_name -> brand for curated output
            if "brand_name" in df.columns and "brand" not in df.columns:
                df = df.rename(columns={"brand_name": "brand"})

        # Fallback: join standalone buyability if not present from dim_asin
        if "is_buyable" not in df.columns and not dim_buyability.empty and "asin_id" in df.columns:
            dim_buyability.columns = [str(c).strip().lower() for c in dim_buyability.columns]
            if {"asin_id", "is_buyable"}.issubset(dim_buyability.columns):
                df = df.merge(dim_buyability[["asin_id", "is_buyable"]].drop_duplicates("asin_id"), on="asin_id", how="left")

        if not dim_vendor.empty and "vendor_id" in df.columns:
            dim_vendor.columns = [str(c).strip().lower() for c in dim_vendor.columns]
            if "vendor_name" in dim_vendor.columns:
                df = df.merge(dim_vendor[["vendor_id", "vendor_name"]].drop_duplicates("vendor_id"), on="vendor_id", how="left", suffixes=("", "_v"))
                # prefer vendor_name from asin dim when available; otherwise use vendor dim
                if "vendor_name_v" in df.columns:
                    df["vendor_name"] = df.get("vendor_name").fillna(df["vendor_name_v"]) if "vendor_name" in df.columns else df["vendor_name_v"]
                    df = df.drop(columns=["vendor_name_v"])

        # Join Category Intelligence on asin_id
        if not dim_category_intel.empty and "asin_id" in df.columns:
            dim_category_intel.columns = [str(c).strip().lower() for c in dim_category_intel.columns]
            keep_cols = [
                "asin_id",
                "category_final",
                "category_from_codes",
                "category_from_names",
                "category_confidence_score",
                "archetype_industry",
                "archetype_gcc",
                "dq_category_conflict",
                "dq_category_inferred",
                "dq_missing_category",
                "dq_archetype_unassigned",
            ]
            keep_cols = [c for c in keep_cols if c in dim_category_intel.columns]
            df = df.merge(dim_category_intel[keep_cols].drop_duplicates("asin_id"), on="asin_id", how="left")
            # Expose canonical category for downstream
            if "category_final" in df.columns:
                df["category_canonical"] = df["category_final"].astype("string")

        # Priority/fallback for item_name and category/subcategory, plus mismatch flags
        # Preserve original raw strings for comparison
        if raw_item_name is not None:
            df["_raw_item_name"] = raw_item_name.astype("string")
        if raw_category is not None:
            df["_raw_category"] = raw_category.astype("string")
        if raw_subcategory is not None:
            df["_raw_sub_category"] = raw_subcategory.astype("string")

        # Item name priority: masterfile (dim_asin) then raw; flag missing
        if "item_name" in df.columns or "_raw_item_name" in df.columns:
            dim_item = df.get("item_name").astype("string") if "item_name" in df.columns else pd.Series([pd.NA]*len(df), dtype="string")
            raw_item = df.get("_raw_item_name").astype("string") if "_raw_item_name" in df.columns else pd.Series([pd.NA]*len(df), dtype="string")
            final_item = dim_item.fillna("").str.strip()
            need_fallback = final_item.eq("")
            final_item = final_item.where(~need_fallback, raw_item.fillna("").str.strip())
            # If still missing, set UNKNOWN_ITEM
            final_item2 = final_item.where(final_item.ne(""), pd.NA)
            unknown_item = final_item2.isna()
            df["item_name"] = final_item2.where(~unknown_item, "UNKNOWN_ITEM")
            # Normalize item name text
            df["item_name"] = df["item_name"].map(normalize_item_name).astype("string")
            df["dq_missing_item_name"] = df["item_name"].isna() | df["item_name"].astype("string").str.strip().eq("")
            # If we explicitly filled UNKNOWN_ITEM, mark as missing
            df.loc[df["item_name"].astype("string").str.upper().eq("UNKNOWN_ITEM"), "dq_missing_item_name"] = True

        # Category/subcategory priority and mismatch
        if "category" in df.columns or "_raw_category" in df.columns:
            dim_cat = df.get("category").astype("string") if "category" in df.columns else pd.Series([pd.NA]*len(df), dtype="string")
            raw_cat = df.get("_raw_category").astype("string") if "_raw_category" in df.columns else pd.Series([pd.NA]*len(df), dtype="string")
            final_cat = dim_cat.fillna("").str.strip()
            need_fallback_cat = final_cat.eq("")
            final_cat = final_cat.where(~need_fallback_cat, raw_cat.fillna("").str.strip())
            df["category"] = final_cat.where(final_cat.ne(""), pd.NA)
            # mismatch when both present and non-empty and different (case-insensitive)
            both_present = dim_cat.fillna("").str.strip().ne("") & raw_cat.fillna("").str.strip().ne("")
            df["dq_category_mismatch"] = (dim_cat.fillna("").str.strip().str.lower() != raw_cat.fillna("").str.strip().str.lower()) & both_present

        if "sub_category" in df.columns or "_raw_sub_category" in df.columns:
            dim_sub = df.get("sub_category").astype("string") if "sub_category" in df.columns else pd.Series([pd.NA]*len(df), dtype="string")
            raw_sub = df.get("_raw_sub_category").astype("string") if "_raw_sub_category" in df.columns else pd.Series([pd.NA]*len(df), dtype="string")
            final_sub = dim_sub.fillna("").str.strip()
            need_fallback_sub = final_sub.eq("")
            final_sub = final_sub.where(~need_fallback_sub, raw_sub.fillna("").str.strip())
            df["sub_category"] = final_sub.where(final_sub.ne(""), pd.NA)

        # Category repair mapping as fallback when still missing
        try:
            cat_map_path = Path("config") / "masterfile_category_map.yaml"
            cat_cfg = {}
            if cat_map_path.exists():
                with open(cat_map_path, "r", encoding="utf-8") as f:
                    import yaml as _yaml
                    cat_cfg = _yaml.safe_load(f) or {}
            sub_to_cat = (cat_cfg or {}).get("subcategory_to_category", {})
            if "category" in df.columns:
                cat_s = df["category"].astype("string")
            else:
                cat_s = pd.Series([pd.NA]*len(df), index=df.index, dtype="string")
            # choose subcategory source: prefer current sub_category else raw
            if "sub_category" in df.columns:
                sub_s = df["sub_category"].astype("string")
            elif "_raw_sub_category" in df.columns:
                sub_s = df["_raw_sub_category"].astype("string")
            else:
                sub_s = pd.Series([pd.NA]*len(df), index=df.index, dtype="string")

            cat_empty_mask = cat_s.isna() | cat_s.str.strip().eq("")
            sub_present_mask = sub_s.notna() & sub_s.str.strip().ne("")
            def _map_cat(v: str | None) -> str | None:
                if v is None or pd.isna(v):
                    return None
                key = str(v).strip()
                return sub_to_cat.get(key, None)
            mapped = sub_s.map(_map_cat)
            apply_map_mask = cat_empty_mask & sub_present_mask & mapped.notna()
            if apply_map_mask.any():
                df.loc[apply_map_mask, "category"] = mapped[apply_map_mask]
            # After mapping, fill UNKNOWN_CATEGORY if still empty
            cat_s2 = df["category"].astype("string") if "category" in df.columns else pd.Series([pd.NA]*len(df), index=df.index, dtype="string")
            still_empty_cat = cat_s2.isna() | cat_s2.str.strip().eq("")
            if still_empty_cat.any():
                df.loc[still_empty_cat, "category"] = "UNKNOWN_CATEGORY"
            # Set DQ flags
            df["dq_inferred_category"] = apply_map_mask.fillna(False) if isinstance(apply_map_mask, pd.Series) else False
            df["dq_missing_category"] = df["category"].astype("string").str.upper().eq("UNKNOWN_CATEGORY") if "category" in df.columns else False
        except Exception:
            # Safe fallback when config not present or mapping invalid
            pass

        # Brand repair/inference: prefer dimension brand, then infer from item_name, else UNKNOWN_BRAND
        # Ensure we work with a 'brand' column for downstream/view
        brand_series = None
        if "brand" in df.columns:
            brand_series = df["brand"].astype("string")
        elif "brand_name" in df.columns:
            brand_series = df["brand_name"].astype("string")
        if brand_series is not None:
            before_brand = brand_series.copy()
            empty_brand = brand_series.isna() | brand_series.str.strip().eq("")
            # Infer from item_name where empty
            inferred = df.get("item_name", pd.Series([pd.NA]*len(df), dtype="string")).map(lambda x: infer_brand_from_item_name(x))
            apply_infer = empty_brand & inferred.notna()
            brand_series = brand_series.where(~apply_infer, inferred.astype("string"))
            # Still empty -> UNKNOWN_BRAND
            still_empty = brand_series.isna() | brand_series.astype("string").str.strip().eq("")
            brand_series = brand_series.where(~still_empty, "UNKNOWN_BRAND")
            # Normalize brand text
            brand_series = brand_series.map(normalize_brand).astype("string")
            # Assign back to df in both brand and brand_name as applicable
            if "brand" in df.columns:
                df["brand"] = brand_series
            if "brand_name" in df.columns:
                df["brand_name"] = brand_series
            else:
                # ensure brand_name exists for compatibility if brand present only
                df["brand_name"] = brand_series
            # DQ flags for brand
            df["dq_inferred_brand"] = apply_infer.fillna(False)
            df["dq_missing_brand"] = brand_series.astype("string").str.upper().eq("UNKNOWN_BRAND")

        # Unknown flags based on join results
        asin_missing = df["asin_id"].isna() | df["asin_id"].astype(str).str.strip().eq("") if "asin_id" in df.columns else True
        # If asin present but no enrichment from dim_asin, consider unknown
        enrich_cols = [c for c in ["item_name", "brand", "vendor_id"] if c in df.columns]
        if enrich_cols:
            no_enrichment = df[enrich_cols].isna().all(axis=1)
            df["dq_unknown_asin"] = asin_missing | (~asin_missing & no_enrichment)
        else:
            df["dq_unknown_asin"] = asin_missing

        # Vendor unknown if no vendor_name after joins
        df["dq_unknown_vendor"] = df.get("vendor_name").isna() if "vendor_name" in df.columns else True
        # Category unknown if category missing
        df["dq_unknown_category"] = df.get("category").isna() if "category" in df.columns else True

        # Buyability gating: flag unbuyable rows
        if "is_buyable" in df.columns:
            df["dq_unbuyable"] = df["is_buyable"].astype("boolean").fillna(False).eq(False)

        # Contra overlay (safe defaults when inputs missing)
        if not dim_contra_agreement.empty and not bridge_contra_asin.empty and {"asin_id", "canonical_week_id"}.issubset(df.columns):
            # Normalize columns
            dim_contra_agreement.columns = [str(c).strip().lower() for c in dim_contra_agreement.columns]
            bridge_contra_asin.columns = [str(c).strip().lower() for c in bridge_contra_asin.columns]
            if {"agreement_id", "asin_id"}.issubset(bridge_contra_asin.columns) and "agreement_id" in dim_contra_agreement.columns:
                # Active by week: agreement is active if canonical_week_id date range between start/end
                # First, map week_id to its start_date via unified calendar already joined on df
                wk = df[["canonical_week_id", "week_start_date"]].drop_duplicates("canonical_week_id")
                wk["week_start_date"] = pd.to_datetime(wk["week_start_date"], errors="coerce")
                ag = dim_contra_agreement.copy()
                ag["start_date"] = pd.to_datetime(ag.get("start_date"), errors="coerce")
                ag["end_date"] = pd.to_datetime(ag.get("end_date"), errors="coerce")
                # Build asin->agreements join
                br = bridge_contra_asin[["agreement_id", "asin_id"]].dropna()
                # Expand df rows with agreements by ASIN
                df_ag = df[["asin_id", "canonical_week_id"]].copy()
                df_ag = df_ag.merge(br, on="asin_id", how="left")
                df_ag = df_ag.merge(wk, on="canonical_week_id", how="left")
                df_ag = df_ag.merge(ag, on="agreement_id", how="left")
                # Determine active flag per row
                def _active(row: pd.Series) -> bool:
                    ws = row.get("week_start_date")
                    sd = row.get("start_date")
                    ed = row.get("end_date")
                    try:
                        if pd.isna(ws):
                            return False
                        if sd is not None and not pd.isna(sd) and ws < sd:
                            return False
                        if ed is not None and not pd.isna(ed) and ws > ed:
                            return False
                        return True
                    except Exception:
                        return False
                df_ag["_active"] = df_ag.apply(_active, axis=1)
                # Aggregate per ASIN x week
                grp = (
                    df_ag.groupby(["asin_id", "canonical_week_id"]).agg(
                        active_agreement_count=("_active", lambda s: int(s.fillna(False).sum())),
                        days_to_expiry=("end_date", lambda s: int((s.max() - s.min()).days) if s.notna().any() else None),
                        has_promo_allowance=("funding_type", lambda s: bool(s.astype("string").str.contains("promo", case=False, na=False).any())),
                        has_volume_incentive=("funding_type", lambda s: bool(s.astype("string").str.contains("volume|tier", case=False, na=False).any())),
                    ).reset_index()
                )
                df = df.merge(grp, on=["asin_id", "canonical_week_id"], how="left")
                # contra active boolean
                if "active_agreement_count" in df.columns:
                    df["contra_active_flag"] = df["active_agreement_count"].fillna(0).astype(int) > 0
                else:
                    df["contra_active_flag"] = False
                # dq for mapping broken: asin appears in bridge but no agreement details or count null
                if "dq_contra_mapping_broken" not in df.columns:
                    try:
                        present_map = df[["asin_id"]].merge(br.drop_duplicates("asin_id"), on="asin_id", how="left")["agreement_id"].notna()
                        df["dq_contra_mapping_broken"] = present_map & df.get("active_agreement_count", pd.Series([0]*len(df))).isna()
                    except Exception:
                        df["dq_contra_mapping_broken"] = False
            else:
                # Safe defaults if structure unexpected
                for c, default in [("active_agreement_count", 0), ("contra_active_flag", False), ("days_to_expiry", pd.NA), ("has_promo_allowance", False), ("has_volume_incentive", False), ("dq_contra_mapping_broken", False)]:
                    if c not in df.columns:
                        df[c] = default
        else:
            # Ensure columns exist with safe defaults
            for c, default in [("active_agreement_count", 0), ("contra_active_flag", False), ("days_to_expiry", pd.NA), ("has_promo_allowance", False), ("has_volume_incentive", False), ("dq_contra_mapping_broken", False)]:
                if c not in df.columns:
                    df[c] = default

    # Quarantine misplaced IDs (write file, mark invalid via dq flag handled later)
    out_dir = Path(output_base)
    df = quarantine_misplaced_ids(df, str(out_dir))

    # Weekly completeness summary per vendor_id (and vendor_id x category when category present)
    # Build a calendar ordinal mapping from unified_calendar for expected week counts
    if "canonical_week_id" in df.columns and not df["canonical_week_id"].isna().all():
        cal = unified_calendar.copy()
        if not cal.empty and "canonical_week_id" in cal.columns:
            cal = cal.dropna(subset=["canonical_week_id"]).drop_duplicates("canonical_week_id")
            cal = cal.sort_values(by=["year", "week"], na_position="last") if set(["year", "week"]).issubset(cal.columns) else cal.sort_values("canonical_week_id")
            cal["_ord"] = range(1, len(cal) + 1)
            ord_map = dict(zip(cal["canonical_week_id"].astype("string"), cal["_ord"]))

            group_keys: List[str] = ["vendor_id"] if "vendor_id" in df.columns else []
            if "category" in df.columns:
                group_keys.append("category")
            if group_keys:
                grp = df.dropna(subset=["canonical_week_id"]).copy()
                grp["_wk"] = grp["canonical_week_id"].astype("string")
                agg = (
                    grp.groupby(group_keys)
                    .agg(
                        min_week=("_wk", "min"),
                        max_week=("_wk", "max"),
                        actual_weeks=("_wk", pd.Series.nunique),
                    )
                    .reset_index()
                )
                # expected weeks via ordinal difference
                def _exp(row: pd.Series) -> int:
                    mi = ord_map.get(str(row["min_week"]))
                    ma = ord_map.get(str(row["max_week"]))
                    if mi is None or ma is None or ma < mi:
                        return row["actual_weeks"]
                    return int(ma - mi + 1)

                agg["expected_weeks"] = agg.apply(_exp, axis=1).astype(int)
                agg["coverage_ratio"] = (agg["actual_weeks"] / agg["expected_weeks"]).astype(float).clip(upper=1.0)
                agg["dq_week_coverage_low"] = agg["coverage_ratio"] < 0.60

                # Write summary parquet
                wc_path = out_dir / "phase1_week_completeness.parquet"
                wc_path.parent.mkdir(parents=True, exist_ok=True)
                agg.to_parquet(wc_path, index=False)

                # Propagate group flag back to rows
                cols_join = group_keys + ["dq_week_coverage_low"]
                df = df.merge(agg[cols_join], on=group_keys, how="left")
                df["dq_week_coverage_low"] = df["dq_week_coverage_low"].fillna(False)

    # Export unbuyable rows if present
    if "dq_unbuyable" in df.columns:
        unb = df[df["dq_unbuyable"] == True]  # noqa: E712
        if not unb.empty:
            (out_dir / "phase1_unbuyable.csv").parent.mkdir(parents=True, exist_ok=True)
            unb.to_csv(out_dir / "phase1_unbuyable.csv", index=False, encoding="utf-8-sig")

    # Build and write masterfile repair audit log
    try:
        import json
        from hashlib import sha256
        from datetime import datetime
        # Counts
        counts = {
            "inferred_brand": int(df.get("dq_inferred_brand", pd.Series(False, index=df.index)).sum()) if "dq_inferred_brand" in df.columns else 0,
            "missing_brand": int(df.get("dq_missing_brand", pd.Series(False, index=df.index)).sum()) if "dq_missing_brand" in df.columns else 0,
            "inferred_category": 0,  # filled below if possible
            "missing_category": 0,
            "missing_item_name": int(df.get("dq_missing_item_name", pd.Series(False, index=df.index)).sum()) if "dq_missing_item_name" in df.columns else 0,
        }
        # Attempt to reconstruct inferred/missing category from available columns
        # We treat inferred category as those rows where category not null but _raw_category is null/empty
        if "category" in df.columns:
            rawc = df.get("_raw_category").astype("string") if "_raw_category" in df.columns else pd.Series([pd.NA]*len(df), index=df.index, dtype="string")
            cat_s = df["category"].astype("string")
            inferred_cat_mask = cat_s.str.strip().ne("") & (rawc.isna() | rawc.str.strip().eq(""))
            missing_cat_mask = cat_s.isna() | cat_s.str.strip().eq("") | cat_s.str.upper().eq("UNKNOWN_CATEGORY")
            counts["inferred_category"] = int(inferred_cat_mask.sum())
            counts["missing_category"] = int(missing_cat_mask.sum())

        # Samples (before/after) for brand/category/item
        samples = {"brand": [], "category": [], "item_name": []}
        if "asin_id" in df.columns:
            # brand samples
            if "brand" in df.columns:
                sub = df.loc[:, [c for c in ["asin_id", "brand", "brand_name"] if c in df.columns]].head(500)
                samples["brand"] = sub.rename(columns={"brand": "brand_after"}).to_dict(orient="records")
            # category samples
            if "category" in df.columns:
                sub = df.loc[:, [c for c in ["asin_id", "category", "_raw_category", "sub_category"] if c in df.columns]].head(500)
                samples["category"] = sub.rename(columns={"category": "category_after"}).to_dict(orient="records")
            # item samples
            if "item_name" in df.columns:
                sub = df.loc[:, [c for c in ["asin_id", "item_name", "_raw_item_name"] if c in df.columns]].head(500)
                samples["item_name"] = sub.rename(columns={"item_name": "item_after"}).to_dict(orient="records")

        # Config hash over optional alias/map files
        cfg_paths = [Path("config") / "masterfile_brand_aliases.yaml", Path("config") / "masterfile_category_map.yaml"]
        h = sha256()
        for p in cfg_paths:
            try:
                if p.exists():
                    h.update(p.name.encode("utf-8"))
                    h.update(p.read_bytes())
            except Exception:
                continue
        audit = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "input": str(Path(input_path).resolve()),
            "counts": counts,
            "samples": samples,
            "config_hash": h.hexdigest(),
        }
        with open(out_dir / "phase1_masterfile_repairs.json", "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2, ensure_ascii=False)
    except Exception:
        # best-effort logging
        pass

    # Ensure presence of optional enrichment fields per Phase 1e when dims missing
    # Buyability: if missing, set dq_buyability_missing and emit placeholder audit
    out_dir = Path(output_base)
    if "is_buyable" not in df.columns:
        df["dq_buyability_missing"] = True
        try:
            (out_dir / "phase1_missing_buyability.json").write_text("{\n  \"message\": \"Buyability sheet missing; defaults applied\"\n}\n", encoding="utf-8")
        except Exception:
            pass
    # Inventory risk placeholders if not present
    for c in ["inventory_risk_lcpo_only", "inventory_risk_overstock", "inventory_risk_oos", "inventory_risk_aging"]:
        if c not in df.columns:
            df[c] = False
    # Inventory presence detection flags per Phase 1f
    if {"inventory_risk_lcpo_only", "inventory_risk_overstock", "inventory_risk_oos", "inventory_risk_aging"}.issubset(set(df.columns)):
        hcpo_present = bool((~df["inventory_risk_lcpo_only"]).any())  # heuristic: if any row is not LCPO-only, assume some HCPO
        lcpo_only_flag = bool(df["inventory_risk_lcpo_only"].any()) and not hcpo_present
    else:
        hcpo_present = False
        lcpo_only_flag = False
    df["hcpo_present"] = hcpo_present
    df["lcpo_only"] = lcpo_only_flag
    # inventory_data_missing when neither HCPO nor LCPO signals observed (no inventory columns present)
    inv_cols_present = any(c in df.columns for c in ["inventory_risk_lcpo_only", "inventory_risk_overstock", "inventory_risk_oos", "inventory_risk_aging"])
    df["inventory_data_missing"] = not inv_cols_present
    # Aggregate inventory risk flag
    inv_flags = [c for c in ["inventory_risk_overstock", "inventory_risk_oos", "inventory_risk_aging"] if c in df.columns]
    df["inventory_risk_flag"] = df[inv_flags].any(axis=1) if inv_flags else False
    # Contra placeholders
    for c, default in [("active_agreement_count", 0), ("contra_active_flag", False), ("days_to_next_expiry", pd.NA), ("funding_type", pd.NA)]:
        if c not in df.columns:
            df[c] = default

    # Buyability reason derivation (priority): LISTING_BLOCKED > SUPPRESSED > NO_RETAIL > NO_INVENTORY > COMPLIANCE > UNKNOWN
    reason_col = "buyability_reason_code"
    if reason_col not in df.columns:
        # Derive from boolean flags if present
        reasons = []
        cols = {c: (c in df.columns) for c in [
            "listing_blocked", "suppressed_from_website", "no_retail_offer", "no_inventory_offer", "compliance_block"
        ]}
        def _derive_reason(r: pd.Series) -> str:
            try:
                if cols.get("listing_blocked") and bool(r.get("listing_blocked")):
                    return "LISTING_BLOCKED"
                if cols.get("suppressed_from_website") and bool(r.get("suppressed_from_website")):
                    return "SUPPRESSED"
                if cols.get("no_retail_offer") and bool(r.get("no_retail_offer")):
                    return "NO_RETAIL"
                if cols.get("no_inventory_offer") and bool(r.get("no_inventory_offer")):
                    return "NO_INVENTORY"
                if cols.get("compliance_block") and bool(r.get("compliance_block")):
                    return "COMPLIANCE"
            except Exception:
                pass
            return "UNKNOWN"
        df[reason_col] = df.apply(_derive_reason, axis=1)
    # Reason family mirrors reason for now
    if "buyability_reason_family" not in df.columns:
        df["buyability_reason_family"] = df.get("buyability_reason_code", "UNKNOWN")

    # Short item name canonical (best-effort)
    if "short_item_name_canonical" not in df.columns:
        if "item_name" in df.columns:
            s = df["item_name"].astype("string").fillna("")
            df["short_item_name_canonical"] = s.str.slice(0, 60)
        else:
            df["short_item_name_canonical"] = pd.NA

    # Apply DQ flags (includes new flags and composite is_valid_row)
    df = apply_dq_flags(df)

    # Outputs
    out_dir = Path(output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "phase1_validated.parquet"
    csv_view_path = out_dir / "phase1_validated_view.csv"
    report_path = out_dir / "phase1_validation_report.json"
    attention_csv = out_dir / "phase1_attention_required.csv"
    severity_audit_csv = out_dir / "phase1_severity_audit.csv"
    fatal_log = out_dir / "phase1_fatal_errors.log"
    inv_audit = out_dir / "phase1_inventory_audit.json"
    contra_audit = out_dir / "phase1_contra_audit.json"
    runlog_path = out_dir / "phase1_run_log.json"

    # Write parquet
    df.to_parquet(parquet_path, index=False)

    # Write curated CSV view
    view_cols: List[str] = [
        "vendor_id",
        "vendor_name",
        "asin_id",
        "item_name",
        "brand",
        "category",
        "category_canonical",
        "archetype_gcc",
        "category_confidence_score",
        "sub_category",
        "canonical_week_id",
        "week_start_date",
        "week_end_date",
        "gms_value",
        "ordered_units",
        "net_ppm_value",
        "gv_value",
    ]
    dq_cols = [c for c in df.columns if str(c).startswith("dq_")]
    existing_view_cols = [c for c in view_cols if c in df.columns]
    df_view = df[existing_view_cols + dq_cols].copy()
    # ensure CSV friendly types
    for c in ["week_start_date", "week_end_date"]:
        if c in df_view.columns:
            df_view[c] = df_view[c].astype("string")
    df_view.to_csv(csv_view_path, index=False, encoding="utf-8-sig")

    # Validation report
    report = build_report(df, schema)
    write_report(report, str(report_path))
    # Attention CSV (High/Medium severity rows)
    write_attention_csv(df, str(attention_csv))

    # Severity audit CSV: asin, vendor_id, all dq_* flags, dq_severity, is_valid_row
    try:
        dq_cols = [c for c in df.columns if str(c).startswith("dq_")]
        keep_ids = [c for c in ["asin_id", "vendor_id", "canonical_week_id", "week_start_date"] if c in df.columns]
        audit_cols = keep_ids + dq_cols + [c for c in ["dq_severity", "is_valid_row"] if c in df.columns]
        audit_df = df[audit_cols].copy()
        # Ensure friendly date types
        if "week_start_date" in audit_df.columns:
            audit_df["week_start_date"] = audit_df["week_start_date"].astype("string")
        audit_df.to_csv(severity_audit_csv, index=False, encoding="utf-8-sig")
    except Exception:
        # Best-effort: don't fail pipeline on audit write
        pass

    # Organize archive directory and move deprecated artifacts if present
    try:
        archive_dir = out_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        deprecated = [
            out_dir / "phase1_output.parquet",
            out_dir / "phase1_output.csv",
            out_dir / "phase1_summary.json",
        ]
        for p in deprecated:
            if p.exists():
                target = archive_dir / p.name
                # Avoid overwrite by timestamp suffix
                if target.exists():
                    from datetime import datetime as _dt
                    ts = _dt.utcnow().strftime("%Y%m%d%H%M%S")
                    target = archive_dir / f"{p.stem}_{ts}{p.suffix}"
                p.replace(target)
    except Exception:
        pass

    # Inventory and Contra audits (best-effort)
    try:
        import json as _json
        inv_summary = {
            "counts": {
                "lcpo_only": int(df.get("inventory_risk_lcpo_only", pd.Series(False, index=df.index)).sum()),
                "overstock": int(df.get("inventory_risk_overstock", pd.Series(False, index=df.index)).sum()),
                "oos": int(df.get("inventory_risk_oos", pd.Series(False, index=df.index)).sum()),
                "aging": int(df.get("inventory_risk_aging", pd.Series(False, index=df.index)).sum()),
            }
        }
        inv_audit.write_text(_json.dumps(inv_summary, indent=2), encoding="utf-8")
    except Exception:
        pass
    try:
        import json as _json
        contra_summary = {
            "counts": {
                "active_rows": int(df.get("contra_active_flag", pd.Series(False, index=df.index)).sum()),
                "with_agreement": int((df.get("active_agreement_count", pd.Series(0, index=df.index)) > 0).sum()),
            }
        }
        contra_audit.write_text(_json.dumps(contra_summary, indent=2), encoding="utf-8")
    except Exception:
        pass

    # Fatal errors log if any Critical severity flags
    try:
        from .dq_checks import classify_row_severity
        sev = classify_row_severity(df)
        if sev.isin(["Critical"]).any():
            fatal_log.write_text("Critical data quality errors detected. Review and re-run.\n", encoding="utf-8")
            # Abort scoring per requirements
            raise RuntimeError("Phase 1 Critical DQ present; aborting")
    except Exception:
        pass

    # Run log
    run_log = {
        "input": str(Path(input_path).resolve()),
        "rows": int(len(df)),
        "outputs": {
            "parquet": str(parquet_path.resolve()),
            "csv_view": str(csv_view_path.resolve()),
            "validation_report": str(report_path.resolve()),
            "id_issues": str((out_dir / "phase1_id_issues.csv").resolve()),
            "unbuyable": str((out_dir / "phase1_unbuyable.csv").resolve()),
            "week_completeness": str((out_dir / "phase1_week_completeness.parquet").resolve()),
            "attention_csv": str(attention_csv.resolve()),
            "inventory_audit": str(inv_audit.resolve()),
            "contra_audit": str(contra_audit.resolve()),
            "fatal_errors_log": str(fatal_log.resolve()),
        },
        "dq_counts": {c: int(df[c].sum()) for c in dq_cols},
    }
    import json
    with open(runlog_path, "w", encoding="utf-8") as f:
        json.dump(run_log, f, indent=2)

    # framework.md automatic updater (append section)
    try:
        from datetime import datetime as _dt
        fw = Path("framework.md")
        ts = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []
        lines.append("\n### Phase 1f Patch Completed\n")
        lines.append(f"Timestamp: {ts}\n")
        # Dimensions presence summary (best-effort)
        dim_summary = []
        try:
            dim_summary.append(f"dim_asin: {int(len(dim_asin)) if 'dim_asin' in locals() and not dim_asin.empty else 0}")
            dim_summary.append(f"dim_vendor: {int(len(dim_vendor)) if 'dim_vendor' in locals() and not dim_vendor.empty else 0}")
            dim_summary.append(f"dim_category: {int(len(dim_category)) if 'dim_category' in locals() and not dim_category.empty else 0}")
            dim_summary.append(f"dim_buyability: {int(len(dim_buyability)) if 'dim_buyability' in locals() and not dim_buyability.empty else 0}")
        except Exception:
            pass
        if dim_summary:
            lines.append("Dimensions loaded: " + ", ".join(dim_summary) + "\n")
        # DQ flags introduced and counts
        dq_counts_line = ", ".join([f"{k}={v}" for k, v in run_log.get("dq_counts", {}).items()])
        lines.append("DQ flags counts: " + dq_counts_line + "\n")
        # Fallbacks used hints
        fallbacks = []
        if df.get("dq_buyability_missing", pd.Series([False]*len(df))).any():
            fallbacks.append("buyability_missing")
        if df.get("inventory_data_missing", pd.Series([False]*len(df))).any():
            fallbacks.append("inventory_missing")
        if df.get("dq_unknown_category", pd.Series([False]*len(df))).any():
            fallbacks.append("category_unknown")
        if fallbacks:
            lines.append("Fallbacks used: " + ", ".join(fallbacks) + "\n")
        # Severity counts
        try:
            from .dq_checks import aggregate_severity_counts
            sev_counts = aggregate_severity_counts(df)
            lines.append("Severity counts: " + ", ".join([f"{k}={v}" for k, v in sev_counts.items()]) + "\n")
        except Exception:
            pass
        # Artifact paths
        lines.append("Artifacts:\n")
        for k, v in run_log.get("outputs", {}).items():
            lines.append(f"- {k}: {v}\n")
        with fw.open("a", encoding="utf-8") as fh:
            fh.writelines(lines)
    except Exception:
        pass

    return df


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 ingestion loader (Overhauled)")
    p.add_argument("--input", required=True, help="Path to input CSV or Parquet")
    p.add_argument("--schema", required=True, help="Path to schema YAML file")
    p.add_argument("--column_map", required=False, help="Path to column alias YAML file (phase1_column_map.yaml)")
    p.add_argument("--calendar_config", required=False, help="Path to calendar config YAML (phase1_calendar.yaml)")
    p.add_argument("--output", required=True, help="Output directory for Brightlight Phase 1 outputs")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    load_phase1(
        input_path=args.input,
        schema_path=args.schema,
        column_map_path=args.column_map,
        calendar_config=args.calendar_config,
        output_base=args.output,
    )


if __name__ == "__main__":
    main()
