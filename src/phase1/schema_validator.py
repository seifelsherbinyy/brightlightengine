from __future__ import annotations

from typing import Dict, Any, List

import pandas as pd
import yaml


def load_schema(schema_path: str) -> Dict[str, Any]:
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = yaml.safe_load(f) or {}
    # Normalize keys
    if "required_columns" not in schema:
        schema["required_columns"] = []
    if "dtypes" not in schema:
        schema["dtypes"] = {}
    if "numeric_fields" not in schema:
        schema["numeric_fields"] = []
    if "id_fields" not in schema:
        schema["id_fields"] = []
    if "valid_ranges" not in schema:
        schema["valid_ranges"] = {}
    return schema


def validate_required_columns(df: pd.DataFrame, schema: Dict[str, Any]):
    missing: List[str] = []
    for col in schema.get("required_columns", []):
        if col not in df.columns:
            missing.append(col)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def enforce_dtypes(df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
    """Coerce DataFrame columns to the dtypes specified in schema['dtypes'].

    Supported logical dtypes: 'string', 'int', 'float', 'bool'. Unknown dtypes are ignored.
    Numeric coercions use errors='coerce'.
    """

    result = df.copy()
    dtypes = schema.get("dtypes", {})
    for col, typ in dtypes.items():
        if col not in result.columns:
            continue
        if typ == "string":
            result[col] = result[col].astype("string")
        elif typ in ("float", "double"):
            result[col] = pd.to_numeric(result[col], errors="coerce")
        elif typ in ("int", "int64", "integer"):
            result[col] = pd.to_numeric(result[col], errors="coerce").astype("Int64")
        elif typ == "bool":
            # Basic truthy coercion
            result[col] = (
                result[col]
                .astype("string")
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
                .astype("boolean")
            )
        else:
            # Unknown mapping: leave as-is
            pass
    return result


def resolve_column_aliases(df: pd.DataFrame, column_map_yaml: str) -> pd.DataFrame:
    """Rename columns to canonical names using alias mapping from YAML file.

    YAML structure example:
      aliases:
        asin_id: ["asin", "asin id"]
        vendor_id: ["vendor", "vendor code"]
        canonical_week_id: ["week", "week_label"]
        gms_value: ["gms($)", "gms"]

    Rules:
      - header names are compared case-insensitively after strip().
      - if multiple aliases map to the same existing column, first match wins.
    """

    with open(column_map_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    aliases: Dict[str, List[str]] = (cfg.get("aliases") or {})

    current = {str(c).strip().lower(): c for c in df.columns}
    rename: Dict[str, str] = {}
    for canonical, alist in aliases.items():
        if not alist:
            continue
        for alias in alist:
            key = str(alias).strip().lower()
            if key in current:
                original = current[key]
                # avoid renaming if the column already has canonical name
                if original != canonical:
                    rename[original] = canonical
                break
    if rename:
        return df.rename(columns=rename)
    return df
