from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Tuple

import pandas as pd


# -----------------------------
# IO helpers
# -----------------------------

def load_masterfile(path: str) -> pd.DataFrame:
    """Load masterfile ASIN-level sheet "Info" only.

    Per spec, this function returns a single DataFrame for the Info sheet.
    Vendor master (Info2) will be loaded by the builder.
    """

    xls_path = Path(path)
    if not xls_path.exists():
        raise FileNotFoundError(f"Masterfile not found: {xls_path}")

    df_info = pd.read_excel(xls_path, sheet_name="Info", dtype="string")

    # Normalize headers
    df_info.columns = [str(c).strip() for c in df_info.columns]

    return df_info


def _norm_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _to_title(x: object) -> str:
    return _norm_str(x).title()


def _to_upper(x: object) -> str:
    return _norm_str(x).upper()


def _normalize_brand_id(x: object) -> str:
    s = _norm_str(x).lower()
    # Keep alnum only for stable ids
    return re.sub(r"[^a-z0-9]+", "", s)


def _split_code_and_name(text: object) -> Tuple[str, str]:
    """Parse strings like "123 - Electronics", "GL12 Electronics", "12_Electronics".

    Returns (code, name). If no code is detected, code is "" and name is titlecased.
    """

    s = _norm_str(text)
    if not s:
        return "", ""

    # Patterns: CODE - NAME, CODE NAME, CODE: NAME, CODE_NAME
    m = re.match(r"^\s*([A-Za-z0-9]+)[\s\-_:]+(.+)$", s)
    if m:
        code = m.group(1).upper()
        name = m.group(2).strip().title()
        return code, name

    # If not matched, try split at first space if token looks like code
    parts = s.split(maxsplit=1)
    if parts and re.fullmatch(r"[A-Za-z0-9]+", parts[0]):
        code = parts[0].upper()
        name = parts[1].strip().title() if len(parts) > 1 else ""
        return code, name

    # Fallback: all name
    return "", s.title()


# -----------------------------
# Dimension builders
# -----------------------------

def build_dim_vendor(df_master: pd.DataFrame) -> pd.DataFrame:
    """Build vendor dimension from Info2 sheet.

    Expected columns (case-insensitive): vendor_id, vendor_name
    - Deduplicate
    - Uppercase vendor_id and vendor_name
    """

    cols = {c.lower(): c for c in df_master.columns}
    vid_col = cols.get("vendor_id") or cols.get("vendor code") or cols.get("vendor_code") or list(df_master.columns)[0]
    vname_col = cols.get("vendor_name") or cols.get("vendor") or list(df_master.columns)[1]

    dim = df_master[[vid_col, vname_col]].copy()
    dim.columns = ["vendor_id", "vendor_name"]
    dim["vendor_id"] = dim["vendor_id"].map(_to_upper)
    dim["vendor_name"] = dim["vendor_name"].map(_to_upper)
    dim = dim.dropna(subset=["vendor_id", "vendor_name"]).drop_duplicates()
    dim = dim.reset_index(drop=True)
    return dim


def build_dim_asin(df_master: pd.DataFrame) -> pd.DataFrame:
    """Build ASIN dimension from Info sheet.

    Columns in output: asin_id, vendor_id, vendor_name, marketplace, item_name,
    short_item_name, brand_name (Title Case), category, sub_category.
    """

    cols = {c.lower(): c for c in df_master.columns}
    asin_col = cols.get("asin") or cols.get("asin_id") or "ASIN"
    asin2_col = cols.get("asin2")
    vendor_id_col = cols.get("vendor_id") or cols.get("vendor code") or cols.get("vendor_code")
    vendor_name_col = cols.get("vendor_name") or cols.get("vendor name")
    marketplace_col = cols.get("marketplace")
    item_name_col = cols.get("item_name") or cols.get("item name")
    short_item_name_col = cols.get("short_item_name") or cols.get("short item name")
    brand_col = cols.get("brand_name") or cols.get("brand")
    category_col = cols.get("category")
    subcategory_col = cols.get("sub_category") or cols.get("subcategory") or cols.get("sub category")

    required = [asin_col, vendor_id_col, brand_col]
    if any(r is None for r in required):
        missing = ["ASIN" if asin_col is None else None, "vendor_id/vendor_code" if vendor_id_col is None else None, "brand_name" if brand_col is None else None]
        raise ValueError(f"Missing required columns in Info sheet: {[m for m in missing if m]}")

    df = pd.DataFrame({
        "asin_id": df_master[asin_col].astype("string").str.strip(),
        "vendor_id": df_master[vendor_id_col].astype("string").str.strip().str.upper(),
        "vendor_name": df_master[vendor_name_col].astype("string").str.strip() if vendor_name_col else pd.Series([], dtype="string"),
        "marketplace": df_master[marketplace_col].astype("string").str.strip() if marketplace_col else pd.Series([], dtype="string"),
        "item_name": df_master[item_name_col].astype("string").str.strip() if item_name_col else pd.Series([], dtype="string"),
        "short_item_name": df_master[short_item_name_col].astype("string").str.strip() if short_item_name_col else pd.Series([], dtype="string"),
        "brand_name": df_master[brand_col].astype("string").map(_to_title),
        "category": df_master[category_col].astype("string").str.strip() if category_col else pd.Series([], dtype="string"),
        "sub_category": df_master[subcategory_col].astype("string").str.strip() if subcategory_col else pd.Series([], dtype="string"),
    })

    if asin2_col and asin2_col in df_master.columns:
        # Drop by simply ignoring in construction (per spec).
        pass

    df = df.dropna(subset=["asin_id", "vendor_id", "brand_name"]).drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def build_dim_brand(df_asin: pd.DataFrame) -> pd.DataFrame:
    """Unique brand_name with normalized id and display name.

    Output columns: brand_id, brand_display
    """

    brands = (
        df_asin["brand_name"].dropna().astype("string").map(_norm_str).replace("", pd.NA).dropna().drop_duplicates()
    )
    dim = pd.DataFrame({
        "brand_display": brands.map(_to_title),
    })
    dim["brand_id"] = dim["brand_display"].map(_normalize_brand_id)
    dim = dim[["brand_id", "brand_display"]].drop_duplicates().reset_index(drop=True)
    return dim


def build_dim_category(df_asin: pd.DataFrame) -> pd.DataFrame:
    """Build category dimension from df_asin['category'].

    Output: gl_code, category_name
    """

    if "category" not in df_asin.columns:
        return pd.DataFrame(columns=["gl_code", "category_name"])  # empty

    cats = (
        df_asin["category"].fillna("").astype("string").map(_norm_str).drop_duplicates()
    )
    parsed = cats.map(_split_code_and_name)
    dim = pd.DataFrame(parsed.tolist(), columns=["gl_code", "category_name"]).drop_duplicates().reset_index(drop=True)
    return dim


def build_dim_subcategory(df_asin: pd.DataFrame) -> pd.DataFrame:
    """Build subcategory dimension from df_asin['sub_category'].

    Output: subcat_code, subcat_name
    """

    if "sub_category" not in df_asin.columns:
        return pd.DataFrame(columns=["subcat_code", "subcat_name"])  # empty

    subcats = (
        df_asin["sub_category"].fillna("").astype("string").map(_norm_str).drop_duplicates()
    )
    parsed = subcats.map(_split_code_and_name)
    dim = pd.DataFrame(parsed.tolist(), columns=["subcat_code", "subcat_name"]).drop_duplicates().reset_index(drop=True)
    return dim


def write_parquet(df: pd.DataFrame, output_path: str) -> None:
    """Write a dataframe to parquet ensuring parent directories exist."""

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Use pyarrow
    df.to_parquet(out, index=False)


def write_builder_log(log_path: str, counts: dict) -> None:
    out = Path(log_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(counts, f, indent=2)
