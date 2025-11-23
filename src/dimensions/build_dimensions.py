from __future__ import annotations

import argparse
from pathlib import Path
import re
import json
from datetime import datetime
from hashlib import sha256
from typing import Dict, Any, Tuple, List

import yaml

import pandas as pd

from .utils import (
    load_masterfile,
    build_dim_vendor,
    build_dim_asin,
    build_dim_brand,
    build_dim_category,
    build_dim_subcategory,
    write_parquet,
    write_builder_log,
    _split_code_and_name,
)


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Coerce a string/object series into boolean, accepting common truthy/falsey tokens.

    Accepted truthy: '1', 'y', 'yes', 'true', 't'
    Accepted falsey: '0', 'n', 'no', 'false', 'f'
    Anything else -> NA -> treated as False in downstream joins.
    """

    s = series.astype("string").str.strip().str.lower()
    truthy = {"1", "y", "yes", "true", "t"}
    falsy = {"0", "n", "no", "false", "f"}
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
    out = out.mask(s.isin(truthy), True)
    out = out.mask(s.isin(falsy), False)
    return out.fillna(False).astype(bool)


def load_buyability_sheet(masterfile_path: str) -> pd.DataFrame:
    """Load Buyability sheet from Masterfile and return columns: asin_id, is_buyable.

    The sheet is expected to be named 'Buyability' with at least two columns:
    - asin_id (or ASIN)
    - is_buyable (various truthy/falsey encodings supported)
    """

    xls = Path(masterfile_path)
    if not xls.exists():
        raise FileNotFoundError(f"Masterfile not found: {xls}")

    try:
        df = pd.read_excel(xls, sheet_name="Buyability", dtype="string")
    except Exception as e:  # sheet may not exist
        # Return empty frame with expected columns; downstream will handle gracefully
        return pd.DataFrame({"asin_id": pd.Series(dtype="string"), "is_buyable": pd.Series(dtype=bool)})

    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}
    asin_col = cols.get("asin_id") or cols.get("asin") or list(df.columns)[0]
    buy_col = cols.get("is_buyable") or cols.get("buyable") or list(df.columns)[1]

    out = pd.DataFrame({
        "asin_id": df[asin_col].astype("string").str.strip().str.upper(),
        "is_buyable": _coerce_bool(df[buy_col]),
    })
    out = out.dropna(subset=["asin_id"]).drop_duplicates("asin_id").reset_index(drop=True)
    return out


# --------------------- Phase 1e required function set ---------------------

def load_info_sheet(path: str) -> pd.DataFrame:
    """Load the Masterfile Info sheet.

    Returns a dataframe with at least: asin_id, item_name, brand_name, category, sub_category.
    Gracefully returns empty frame when the sheet is missing.
    """

    xls = Path(path)
    if not xls.exists():
        return pd.DataFrame(columns=["asin_id", "item_name", "brand_name", "category", "sub_category"]).astype("string")
    try:
        df = pd.read_excel(xls, sheet_name="Info", dtype="string")
        df.columns = [str(c).strip().lower() for c in df.columns]
        # Normalize key id column
        if "asin" in df.columns and "asin_id" not in df.columns:
            df = df.rename(columns={"asin": "asin_id"})
        return df
    except Exception:
        return pd.DataFrame(columns=["asin_id", "item_name", "brand_name", "category", "sub_category"]).astype("string")


def load_info2_sheet(path: str) -> pd.DataFrame:
    """Load the Masterfile Info2 sheet with vendor lookup.

    Returns a dataframe with vendor_id and related lookup fields.
    """

    xls = Path(path)
    if not xls.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(xls, sheet_name="Info2", dtype="string")
        df.columns = [str(c).strip().lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def load_inventory_sheet(path: str) -> pd.DataFrame:
    """Load Inventory sheet from Masterfile with minimal normalization.

    Expects columns: asin_id, canonical_week_id or week_start, metrics related to inventory.
    Returns empty frame on failure.
    """

    xls = Path(path)
    if not xls.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(xls, sheet_name="Inventory", dtype="string")
        df.columns = [str(c).strip().lower() for c in df.columns]
        if "asin" in df.columns and "asin_id" not in df.columns:
            df = df.rename(columns={"asin": "asin_id"})
        return df
    except Exception:
        return pd.DataFrame()


def build_contra_dimensions(path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build dim_contra_agreement and bridge_contra_asin from agreements file.

    The agreements file may be CSV or Excel and should contain columns:
    AGREEMENT_ID, INCLUDED_ASINS (comma or semicolon separated), START_DATE, END_DATE, FUNDING_TYPE.
    Returns (dim_contra_agreement, bridge_contra_asin). Empty frames when input missing.
    """

    p = Path(path)
    if not p.exists():
        return pd.DataFrame(), pd.DataFrame()
    try:
        if p.suffix.lower() in (".xlsx", ".xls"):
            raw = pd.read_excel(p, dtype="string")
        else:
            raw = pd.read_csv(p, dtype="string")
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        # Normalize expected columns
        id_col = next((c for c in raw.columns if c in ("agreement_id", "id", "agreement")), None)
        incl_col = next((c for c in raw.columns if c in ("included_asins", "asins", "asin_list")), None)
        start_col = next((c for c in raw.columns if c in ("start_date", "start")), None)
        end_col = next((c for c in raw.columns if c in ("end_date", "end")), None)
        fund_col = next((c for c in raw.columns if c in ("funding_type", "type")), None)
        if not id_col or not incl_col:
            return pd.DataFrame(), pd.DataFrame()
        dim = pd.DataFrame({
            "agreement_id": raw[id_col].astype("string"),
            "start_date": pd.to_datetime(raw.get(start_col), errors="coerce") if start_col else pd.NaT,
            "end_date": pd.to_datetime(raw.get(end_col), errors="coerce") if end_col else pd.NaT,
            "funding_type": raw.get(fund_col).astype("string") if fund_col else pd.Series([pd.NA]*len(raw), dtype="string"),
        })
        # Bridge explode
        def _split_asins(s: Any) -> List[str]:
            if s is None or (isinstance(s, float) and pd.isna(s)):
                return []
            txt = str(s)
            parts = [t.strip().upper() for t in re.split(r"[;,\s]+", txt) if t.strip()]
            return parts
        bridge = raw[[id_col, incl_col]].copy()
        bridge["asin_id_list"] = bridge[incl_col].apply(_split_asins)
        bridge = bridge.explode("asin_id_list")
        bridge = bridge.rename(columns={id_col: "agreement_id"})
        bridge = bridge.rename(columns={"asin_id_list": "asin_id"})
        bridge = bridge.dropna(subset=["agreement_id", "asin_id"]).astype({"agreement_id": "string", "asin_id": "string"})
        return dim, bridge[["agreement_id", "asin_id"]]
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def unify_calendar(calendar_path: str, unified_calendar_path: str) -> pd.DataFrame:
    """Merge/normalize raw calendar to unified calendar map.

    For Phase 1e, this function simply reads the provided unified calendar CSV and returns it
    with normalized headers. If calendar_path is provided and contains extra attributes,
    they will be merged by canonical_week_id when possible.
    """

    u = Path(unified_calendar_path)
    base = pd.read_csv(u, dtype="string") if u.exists() else pd.DataFrame()
    if base.empty:
        return base
    base.columns = [str(c).strip().lower() for c in base.columns]
    if not calendar_path:
        return base
    cp = Path(calendar_path)
    if not cp.exists():
        return base
    try:
        raw = pd.read_csv(cp, dtype="string")
        raw.columns = [str(c).strip().lower() for c in raw.columns]
        if "canonical_week_id" in raw.columns and "canonical_week_id" in base.columns:
            merged = base.merge(raw, on="canonical_week_id", how="left", suffixes=("", "_extra"))
            return merged
        return base
    except Exception:
        return base


def write_dimension_outputs(outputs: Dict[str, pd.DataFrame], root: str) -> None:
    """Write provided dimension dataframes to the output root as parquet.

    Keys are filenames without extension (e.g., 'dim_asin'). Function appends '.parquet'.
    """

    out_dir = Path(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (outputs or {}).items():
        try:
            write_parquet(df, str(out_dir / f"{name}.parquet"))
        except Exception:
            continue


# --------------------- Masterfile auto-repair helpers ---------------------

def _load_brand_alias_map() -> Dict[str, str]:
    """Load optional brand alias map from config/masterfile_brand_aliases.yaml.

    Returns a case-insensitive mapping (uppercased keys) to canonical brand strings.
    Missing file yields an empty dict.
    """

    cfg = Path("config") / "masterfile_brand_aliases.yaml"
    if not cfg.exists():
        return {}
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        out = {}
        for k, v in (data.items() if isinstance(data, dict) else []):
            if k is None or v is None:
                continue
            out[str(k).strip().upper()] = str(v).strip()
        return out
    except Exception:
        return {}


def _load_category_map() -> Dict[str, Any]:
    """Load optional category mapping from config/masterfile_category_map.yaml.

    Expected structure:
      subcategory_to_category: { "SubCat": "Category", ... }
      valid_hierarchy: [ {category: "X", sub_category: "Y"}, ... ]
    Missing file yields an empty dict.
    """

    cfg = Path("config") / "masterfile_category_map.yaml"
    if not cfg.exists():
        return {}
    try:
        with open(cfg, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def infer_brand_from_item_name(name: str | None, alias_map: Dict[str, str] | None = None) -> str | None:
    """Deterministically infer a brand from an item name.

    Rules (applied in order):
      1) Tokenize on spaces and dashes (including unicode en dashes). Normalize punctuation.
      2) If a token uppercased matches an alias key, return the alias canonical.
      3) Prefer ALL-CAPS tokens (length 2..15). If found, map via alias (if any) else title-case.
      4) Fallback to first token (title-cased) if length 2..30.
      5) Otherwise None.
    """

    if not name:
        return None
    s = str(name).strip()
    if not s:
        return None
    # Normalize unicode dashes and collapse whitespace
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s+", " ", s)
    # Split on space or dash
    tokens = re.split(r"[\s\-]+", s)
    tokens = [t.strip() for t in tokens if t and t.strip()]
    if not tokens:
        return None
    alias = (alias_map or {})
    # 2) Direct alias hits by token
    for t in tokens:
        tt = t.strip().upper()
        if tt in alias:
            return alias[tt]
    # 3) ALL-CAPS tokens
    for t in tokens:
        tt = t.strip()
        if re.fullmatch(r"[A-Z0-9]{2,15}", tt):
            m = alias.get(tt)
            return m if m else tt.title()
    # 4) First token fallback
    ft = tokens[0]
    if 2 <= len(ft) <= 30:
        tt = ft.strip().upper()
        return alias.get(tt, ft.title())
    return None


def fill_missing_brand(df: pd.DataFrame, alias_map: Dict[str, str] | None = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fill missing/empty brand_name values deterministically.

    - Try infer_brand_from_item_name on item_name.
    - If still missing, set to "UNKNOWN_BRAND".
    Returns (repaired_df, stats).
    """

    res = df.copy()
    stats: Dict[str, Any] = {
      "inferred_brand_count": 0,
      "unknown_brand_count": 0,
      "affected_asins": [],
    }
    if "brand_name" not in res.columns:
        return res, stats
    b = res["brand_name"].astype("string")
    empty = b.isna() | b.str.strip().eq("")
    if empty.any():
        # Attempt inference using item_name
        item = res.get("item_name", pd.Series([pd.NA] * len(res), index=res.index, dtype="string")).astype("string")
        alias = alias_map or {}
        inferred = item.map(lambda x: infer_brand_from_item_name(x, alias))
        applied = empty & inferred.notna()
        # Log before/after for applied inferences
        for asin, before, after in zip(res.get("asin_id", pd.Series([None]*len(res))), b.where(empty, pd.NA), inferred):
            # Only record rows where we actually apply
            # zip goes over full series; we will filter when updating
            pass
        res.loc[applied, "brand_name"] = inferred[applied]
        stats["inferred_brand_count"] = int(applied.sum())
        # Remaining empties -> UNKNOWN_BRAND
        b2 = res["brand_name"].astype("string")
        still_empty = b2.isna() | b2.str.strip().eq("")
        res.loc[still_empty, "brand_name"] = "UNKNOWN_BRAND"
        stats["unknown_brand_count"] = int(still_empty.sum())
        # Build affected_asins sample
        if "asin_id" in res.columns:
            changed_mask = empty
            sample = res.loc[changed_mask, ["asin_id", "item_name", "brand_name"]].head(500)
            stats["affected_asins"] = sample.to_dict(orient="records")
    return res, stats


def repair_category_fields(df: pd.DataFrame, cat_map_cfg: Dict[str, Any] | None = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Repair category/sub_category using mapping; assign UNKNOWN_CATEGORY when needed.

    Returns (repaired_df, stats) with counters and a sample of affected ASINs.
    """

    res = df.copy()
    stats: Dict[str, Any] = {
        "inferred_category_count": 0,
        "unknown_category_count": 0,
        "hierarchy_mismatch_count": 0,
        "affected_asins": [],
    }
    sub_to_cat = (cat_map_cfg or {}).get("subcategory_to_category", {})
    valid_pairs = {(str(d.get("category")).strip().lower(), str(d.get("sub_category")).strip().lower())
                   for d in (cat_map_cfg or {}).get("valid_hierarchy", []) if isinstance(d, dict)}

    cat = res.get("category")
    sub = res.get("sub_category")
    if cat is None and sub is None:
        return res, stats
    cat_s = (cat.astype("string") if cat is not None else pd.Series([pd.NA]*len(res), index=res.index, dtype="string"))
    sub_s = (sub.astype("string") if sub is not None else pd.Series([pd.NA]*len(res), index=res.index, dtype="string"))

    cat_empty = cat_s.isna() | cat_s.str.strip().eq("")
    sub_present = sub_s.notna() & sub_s.str.strip().ne("")

    # Where category empty and subcategory present, map
    def _map_cat(v: str | None) -> str | None:
        if v is None or pd.isna(v):
            return None
        key = str(v).strip()
        return sub_to_cat.get(key, None)

    mapped = sub_s.map(_map_cat)
    apply_map = cat_empty & sub_present & mapped.notna()
    res.loc[apply_map, "category"] = mapped[apply_map]
    stats["inferred_category_count"] = int(apply_map.sum())

    # If both category and sub_category empty -> UNKNOWN_CATEGORY (for category only)
    both_empty = cat_empty & (~sub_present)
    res.loc[both_empty, "category"] = "UNKNOWN_CATEGORY"
    stats["unknown_category_count"] = int(both_empty.sum())

    # Hierarchy mismatches: when both present, but not a valid pair (when mapping provided)
    if valid_pairs:
        both_present = (res["category"].astype("string").str.strip().ne("") &
                        res["sub_category"].astype("string").str.strip().ne(""))
        pair = pd.DataFrame({
            "c": res["category"].astype("string").str.strip().str.lower(),
            "s": res["sub_category"].astype("string").str.strip().str.lower(),
        })
        mismatch = both_present & ~pair.apply(lambda r: (r["c"], r["s"]) in valid_pairs, axis=1)
        stats["hierarchy_mismatch_count"] = int(mismatch.sum())

    # Build affected_asins sample
    if "asin_id" in res.columns:
        changed_mask = apply_map | both_empty
        sample = res.loc[changed_mask, [c for c in ["asin_id", "category", "sub_category"] if c in res.columns]].head(500)
        stats["affected_asins"] = sample.to_dict(orient="records")

    return res, stats


# --------------------- Category Intelligence (Session 3) ---------------------

def load_masterfile_category_block(path: str) -> pd.DataFrame:
    """Load category-relevant fields from Masterfile Info sheet.

    Returns minimal columns with normalized names:
      - asin_id
      - item_name
      - category_raw
      - sub_category_raw
      - category_code_raw (parsed from category_raw when present)
    Gracefully handles missing sheets/columns.
    """

    try:
        df = load_masterfile(path)
    except Exception:
        return pd.DataFrame(columns=[
            "asin_id", "item_name", "category_raw", "sub_category_raw", "category_code_raw"
        ]).astype("string")

    cols = {c.lower(): c for c in df.columns}
    asin_col = cols.get("asin_id") or cols.get("asin") or list(df.columns)[0]
    item_col = cols.get("item_name") or cols.get("item name") or None
    cat_col = cols.get("category") or cols.get("cat") or None
    sub_col = cols.get("sub_category") or cols.get("subcategory") or None

    out = pd.DataFrame({
        "asin_id": df[asin_col].astype("string").str.strip().str.upper(),
        "item_name": df[item_col].astype("string") if item_col else pd.Series([pd.NA]*len(df), dtype="string"),
        "category_raw": df[cat_col].astype("string") if cat_col else pd.Series([pd.NA]*len(df), dtype="string"),
        "sub_category_raw": df[sub_col].astype("string") if sub_col else pd.Series([pd.NA]*len(df), dtype="string"),
    })
    # Parse category code from raw category when encoded like "CODE - Name"
    def _parse_code(s: object) -> str | None:
        code, _ = _split_code_and_name(s)
        return code or None
    out["category_code_raw"] = out["category_raw"].map(_parse_code).astype("string")
    out = out.dropna(subset=["asin_id"]).drop_duplicates("asin_id").reset_index(drop=True)
    return out


def _load_category_mappings() -> Dict[str, Any]:
    """Load optional category mapping config.

    Expected structure (all optional):
      code_to_category: { CODE: Canonical Category }
      subcategory_to_category: { SubCat: Canonical Category }
      category_keywords: [ { keyword: "detergent", category: "Home Care" }, ... ]
    """

    cfg_path = Path("config") / "category_mappings.yaml"
    if not cfg_path.exists():
        return {"code_to_category": {}, "subcategory_to_category": {}, "category_keywords": []}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg.setdefault("code_to_category", {})
        cfg.setdefault("subcategory_to_category", {})
        cfg.setdefault("category_keywords", [])
        return cfg
    except Exception:
        return {"code_to_category": {}, "subcategory_to_category": {}, "category_keywords": []}


def derive_category_from_codes(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Derive category_from_codes using category_code_raw or sub_category_raw via mapping.

    Deterministic mapping only; unknowns remain NA.
    """

    out = df.copy()
    code_map = (config or {}).get("code_to_category", {})
    sub_map = (config or {}).get("subcategory_to_category", {})

    def _map_code(x: object) -> str | None:
        if pd.isna(x):
            return None
        key = str(x).strip().upper()
        return code_map.get(key)

    def _map_sub(x: object) -> str | None:
        if pd.isna(x):
            return None
        key = str(x).strip()
        return sub_map.get(key)

    from_code = out.get("category_code_raw", pd.Series([pd.NA]*len(out))).map(_map_code)
    from_sub = out.get("sub_category_raw", pd.Series([pd.NA]*len(out))).map(_map_sub)
    out["category_from_codes"] = pd.Series(from_code, dtype="string")
    # Only fill from sub when code-based missing
    mask_missing = out["category_from_codes"].isna()
    out.loc[mask_missing, "category_from_codes"] = pd.Series(from_sub, dtype="string")[mask_missing]
    return out


def derive_category_from_names(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Derive category_from_names using simple keyword rules over item_name.

    Case-insensitive substring matching based on config.category_keywords list of
    objects: {keyword: "detergent", category: "Home Care"}. First match wins.
    """

    out = df.copy()
    rules: List[Dict[str, str]] = (config or {}).get("category_keywords", []) or []
    # Preprocess rules
    rules2 = [(str(r.get("keyword", "")).lower(), str(r.get("category", ""))) for r in rules if r.get("keyword") and r.get("category")]

    def _pick(name: object) -> str | None:
        s = str(name).lower() if not pd.isna(name) else ""
        if not s:
            return None
        for kw, cat in rules2:
            if kw and kw in s:
                return cat
        return None

    out["category_from_names"] = out.get("item_name", pd.Series([pd.NA]*len(out))).map(_pick).astype("string")
    return out


def resolve_category_conflicts(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Resolve final category and confidence/DQ flags.

    Rules (deterministic):
      - If category_raw present (Masterfile), prefer as ground truth (base high confidence)
      - If code-derived and/or name-derived agree with raw -> high confidence (1.0)
      - If code-derived exists and differs from name-derived -> prefer code-derived; dq_category_conflict=True; confidence=0.7
      - If only name-derived exists -> use it; dq_category_inferred=True; confidence=0.5
      - If nothing -> UNKNOWN_CATEGORY; dq_missing_category=True; confidence=0.2
    """

    out = df.copy()
    raw = out.get("category_raw").astype("string") if "category_raw" in out.columns else pd.Series([pd.NA]*len(out), dtype="string")
    from_code = out.get("category_from_codes").astype("string") if "category_from_codes" in out.columns else pd.Series([pd.NA]*len(out), dtype="string")
    from_name = out.get("category_from_names").astype("string") if "category_from_names" in out.columns else pd.Series([pd.NA]*len(out), dtype="string")

    # Initialize outputs
    final = pd.Series([pd.NA]*len(out), dtype="string")
    dq_conflict = pd.Series(False, index=out.index)
    dq_inferred = pd.Series(False, index=out.index)
    dq_missing = pd.Series(False, index=out.index)
    conf = pd.Series([0.0]*len(out), dtype="float")

    raw_clean = raw.fillna("").str.strip()
    code_clean = from_code.fillna("").str.strip()
    name_clean = from_name.fillna("").str.strip()

    # Case 1: raw present
    has_raw = raw_clean.ne("")
    # Agreement masks
    agree_code = has_raw & code_clean.ne("") & (raw_clean.str.casefold() == code_clean.str.casefold())
    agree_name = has_raw & name_clean.ne("") & (raw_clean.str.casefold() == name_clean.str.casefold())
    any_agree = agree_code | agree_name
    final = final.where(~any_agree, raw)
    conf = conf.where(~any_agree, 1.0)

    # Case 2: code vs name conflict (when both present and differ)
    both_present = code_clean.ne("") & name_clean.ne("") & (code_clean.str.casefold() != name_clean.str.casefold())
    # Only apply where not already set by raw agreement
    mask2 = both_present & final.isna()
    final = final.where(~mask2, from_code)
    dq_conflict = dq_conflict | mask2
    conf = conf.where(~mask2, 0.7)

    # Case 3: only code present (and not set)
    only_code = code_clean.ne("") & name_clean.eq("") & final.isna()
    final = final.where(~only_code, from_code)
    conf = conf.where(~only_code, 0.8)

    # Case 4: only name present
    only_name = name_clean.ne("") & code_clean.eq("") & final.isna()
    final = final.where(~only_name, from_name)
    dq_inferred = dq_inferred | only_name
    conf = conf.where(~only_name, 0.5)

    # Case 5: nothing present
    nothing = final.isna()
    final = final.where(~nothing, "UNKNOWN_CATEGORY")
    dq_missing = dq_missing | nothing
    conf = conf.where(~nothing, 0.2)

    out["category_final"] = final.astype("string")
    out["category_confidence_score"] = conf.astype(float)
    out["dq_category_conflict"] = dq_conflict.astype(bool)
    out["dq_category_inferred"] = dq_inferred.astype(bool)
    out["dq_missing_category"] = dq_missing.astype(bool)
    return out


def _load_archetype_config() -> Dict[str, Any]:
    cfg_path = Path("config") / "category_archetypes.yaml"
    if not cfg_path.exists():
        # Provide minimal defaults to avoid hard failure
        return {
            "industry_archetypes": [],
            "gcc_archetypes": [],
            "mappings": {
                "category_to_industry": {},
                "category_to_gcc": {},
                "subcategory_overrides": {},
            },
        }
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        # Normalize keys
        cfg.setdefault("mappings", {})
        cfg["mappings"].setdefault("category_to_industry", {})
        cfg["mappings"].setdefault("category_to_gcc", {})
        cfg["mappings"].setdefault("subcategory_overrides", {})
        return cfg
    except Exception:
        return {
            "industry_archetypes": [],
            "gcc_archetypes": [],
            "mappings": {
                "category_to_industry": {},
                "category_to_gcc": {},
                "subcategory_overrides": {},
            },
        }


def assign_archetypes(df: pd.DataFrame, archetype_config: Dict[str, Any]) -> pd.DataFrame:
    """Assign industry and GCC archetypes from category_final (with subcategory overrides).

    When no mapping is found, assign UNKNOWN_ARCHETYPE and set dq_archetype_unassigned.
    """

    out = df.copy()
    maps = (archetype_config or {}).get("mappings", {})
    c2i = maps.get("category_to_industry", {})
    c2g = maps.get("category_to_gcc", {})
    sub_over = maps.get("subcategory_overrides", {})

    def _ind(row: pd.Series) -> str:
        sub = str(row.get("sub_category_raw", ""))
        cat = str(row.get("category_final", ""))
        # subcategory override
        if sub and sub in sub_over and isinstance(sub_over[sub], dict) and sub_over[sub].get("industry"):
            return str(sub_over[sub]["industry"])  
        return str(c2i.get(cat, "UNKNOWN_ARCHETYPE"))

    def _gcc(row: pd.Series) -> str:
        sub = str(row.get("sub_category_raw", ""))
        cat = str(row.get("category_final", ""))
        if sub and sub in sub_over and isinstance(sub_over[sub], dict) and sub_over[sub].get("gcc"):
            return str(sub_over[sub]["gcc"])  
        return str(c2g.get(cat, "UNKNOWN_ARCHETYPE"))

    out["archetype_industry"] = out.apply(_ind, axis=1).astype("string")
    out["archetype_gcc"] = out.apply(_gcc, axis=1).astype("string")
    out["dq_archetype_unassigned"] = (
        out["archetype_industry"].str.upper().eq("UNKNOWN_ARCHETYPE") | out["archetype_gcc"].str.upper().eq("UNKNOWN_ARCHETYPE")
    )
    return out


def build_dim_category_intelligence(masterfile_path: str, config_paths: Dict[str, str] | None = None) -> pd.DataFrame:
    """Full pipeline to build dim_category_intelligence.

    Steps:
      - Load Masterfile category block
      - Derive category from codes and names (config-driven)
      - Resolve conflicts and assign confidence/DQ
      - Assign archetypes (industry + GCC)
      - Return final DataFrame keyed by asin_id
    """

    df = load_masterfile_category_block(masterfile_path)
    cat_map_cfg = _load_category_mappings()
    df = derive_category_from_codes(df, cat_map_cfg)
    df = derive_category_from_names(df, cat_map_cfg)
    df = resolve_category_conflicts(df, cat_map_cfg)
    arch_cfg = _load_archetype_config()
    df = assign_archetypes(df, arch_cfg)

    # Select and order columns
    keep = [
        "asin_id",
        # raw
        "category_raw",
        "sub_category_raw",
        "category_code_raw",
        # derivations
        "category_from_codes",
        "category_from_names",
        # final
        "category_final",
        "category_confidence_score",
        # archetypes
        "archetype_industry",
        "archetype_gcc",
        # dq flags
        "dq_category_conflict",
        "dq_category_inferred",
        "dq_missing_category",
        "dq_archetype_unassigned",
    ]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    out = df[keep].copy()
    return out


def build_all_dimensions(masterfile_path: str, output_base: str) -> None:
    df_info = load_masterfile(masterfile_path)
    # Load Info2 for vendor dimension
    df_info2 = pd.read_excel(masterfile_path, sheet_name="Info2", dtype="string")
    df_info2.columns = [str(c).strip() for c in df_info2.columns]

    # Validations: non-null ASIN, vendor_code, brand_name in Info
    info_cols = {c.lower(): c for c in df_info.columns}
    asin_col = info_cols.get("asin") or info_cols.get("asin_id") or "ASIN"
    vendor_id_col = info_cols.get("vendor_id") or info_cols.get("vendor code") or info_cols.get("vendor_code")
    brand_col = info_cols.get("brand_name") or info_cols.get("brand")
    for required, label in [(asin_col, "ASIN"), (vendor_id_col, "vendor_code"), (brand_col, "brand_name")]:
        if required is None:
            raise ValueError(f"Info sheet missing required column: {label}")
        if df_info[required].isna().all():
            raise ValueError(f"Info sheet has all-null values for required column: {label}")

    # Build base dimensions
    dim_vendor = build_dim_vendor(df_info2)
    dim_asin = build_dim_asin(df_info)

    # Auto-repair brand/category on dim_asin before downstream dims are derived
    alias_map = _load_brand_alias_map()
    cat_map = _load_category_map()

    dim_asin, brand_stats = fill_missing_brand(dim_asin, alias_map)
    dim_asin, cat_stats = repair_category_fields(dim_asin, cat_map)

    # Derived dims from repaired dim_asin
    dim_brand = build_dim_brand(dim_asin)
    dim_category = build_dim_category(dim_asin)
    dim_subcategory = build_dim_subcategory(dim_asin)

    # Load buyability info and merge into dim_asin
    df_buy = load_buyability_sheet(masterfile_path)
    if not df_buy.empty:
        dim_asin = dim_asin.merge(df_buy, on="asin_id", how="left")
    else:
        dim_asin["is_buyable"] = False

    # Write outputs
    out_dir = Path(output_base)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(dim_vendor, str(out_dir / "dim_vendor.parquet"))
    write_parquet(dim_asin, str(out_dir / "dim_asin.parquet"))
    write_parquet(dim_brand, str(out_dir / "dim_brand.parquet"))
    write_parquet(dim_category, str(out_dir / "dim_category.parquet"))
    write_parquet(dim_subcategory, str(out_dir / "dim_subcategory.parquet"))
    # Also write the buyability table as a separate dimension file
    if 'df_buy' in locals():
        write_parquet(df_buy, str(out_dir / "dim_buyability.parquet"))

    # Category Intelligence dimension (Session 3)
    try:
        dim_cat_intel = build_dim_category_intelligence(masterfile_path)
        write_parquet(dim_cat_intel, str(out_dir / "dim_category_intelligence.parquet"))
    except Exception:
        # Graceful degradation: emit empty file to keep downstream optional
        dim_cat_intel = pd.DataFrame(columns=[
            "asin_id",
            "category_raw",
            "sub_category_raw",
            "category_code_raw",
            "category_from_codes",
            "category_from_names",
            "category_final",
            "category_confidence_score",
            "archetype_industry",
            "archetype_gcc",
            "dq_category_conflict",
            "dq_category_inferred",
            "dq_missing_category",
            "dq_archetype_unassigned",
        ])
        write_parquet(dim_cat_intel, str(out_dir / "dim_category_intelligence.parquet"))

    # Log (include repair stats and config hash)
    cfg_paths = {
        "brand_aliases": str((Path("config") / "masterfile_brand_aliases.yaml").resolve()),
        "category_map": str((Path("config") / "masterfile_category_map.yaml").resolve()),
    }
    # Compute a simple config hash over available files
    def _hash_cfg(paths: Dict[str, str]) -> str:
        h = sha256()
        for label, p in sorted(paths.items()):
            try:
                pp = Path(p)
                if pp.exists():
                    h.update(label.encode("utf-8"))
                    h.update(pp.read_bytes())
            except Exception:
                continue
        return h.hexdigest()

    log = {
        "source": str(Path(masterfile_path).resolve()),
        "output_dir": str(out_dir.resolve()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "paths": cfg_paths,
            "hash": _hash_cfg(cfg_paths),
        },
        "counts": {
            "dim_vendor": len(dim_vendor),
            "dim_asin": len(dim_asin),
            "dim_brand": len(dim_brand),
            "dim_category": len(dim_category),
            "dim_subcategory": len(dim_subcategory),
            "dim_buyability": int(len(df_buy)) if 'df_buy' in locals() else 0,
            "dim_category_intelligence": int(len(dim_cat_intel)) if 'dim_cat_intel' in locals() else 0,
        },
        "repairs": {
            "brand": brand_stats,
            "category": cat_stats,
        },
    }
    write_builder_log(str(out_dir / "dimension_builder_log.json"), log)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Brightlight dimensions from Masterfile.")
    p.add_argument("--input", required=True, help="Path to Masterfile.xlsx")
    p.add_argument("--output", required=True, help="Output directory for dimension parquet files")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    build_all_dimensions(args.input, args.output)


if __name__ == "__main__":
    main()
