from __future__ import annotations

import re
from typing import Any, Set
from pathlib import Path
import yaml

import pandas as pd


ASIN_PATTERN = re.compile(r"[^A-Z0-9]+")
ASIN_FULL_RX = re.compile(r"^[A-Z0-9]{10}$")
VENDOR_ID_RX = re.compile(r"^[A-Z0-9][A-Z0-9\-]{1,31}$")  # permissive: 2-32 chars alnum/dash
_VENDOR_SET: Set[str] | None = None


def normalize_asin(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).strip().upper()
    return ASIN_PATTERN.sub("", s)


def normalize_vendor_id(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip().upper()


def _parse_week_like(text: str) -> tuple[int | None, int | None]:
    """Try parsing various week label patterns into (year, week).

    Supports formats like: 2025W36, 202536, 2025-36, W36 2025, 36 2025, etc.
    Returns (None, None) when not parseable.
    """

    s = str(text).strip().upper()
    if not s:
        return None, None

    # 2025W36 or 2025-W36
    m = re.match(r"^(20\d{2})\s*-?W\s*(\d{1,2})$", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    # 202536 or 2025-36 or 2025 36
    m = re.match(r"^(20\d{2})[\s\-_/]*(\d{1,2})$", s)
    if m:
        return int(m.group(1)), int(m.group(2))

    # W36 2025 or 36W2025
    m = re.match(r"^W?\s*(\d{1,2})[\sW\-_/]*(20\d{2})$", s)
    if m:
        return int(m.group(2)), int(m.group(1))

    return None, None


def normalize_week(x: Any) -> int:
    """Normalize week identifier to integer YYYYWW.

    If parsing fails, returns 0.
    """

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return 0
    if isinstance(x, (int, pd.IntegerDtype)):
        return int(x)
    s = str(x).strip()
    # If already like 202536 -> int
    if re.fullmatch(r"20\d{2}\d{2}", s):
        return int(s)
    year, week = _parse_week_like(s)
    if year is None or week is None:
        return 0
    week = max(1, min(53, int(week)))
    return int(f"{year}{week:02d}")


def normalize_category(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip().title()


def is_valid_asin(x: Any) -> bool:
    """Return True if value looks like a canonical ASIN: 10 uppercase alphanumerics.

    Whitespace is stripped and letters uppercased before validation.
    """

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    s = str(x).strip().upper()
    return bool(ASIN_FULL_RX.fullmatch(s))


def _load_vendor_set() -> Set[str]:
    global _VENDOR_SET
    if _VENDOR_SET is not None:
        return _VENDOR_SET

    # Load dim_vendor via config/dimension_paths.yaml
    dim_paths_cfg = Path("config") / "dimension_paths.yaml"
    vendors: Set[str] = set()
    try:
        if dim_paths_cfg.exists():
            with open(dim_paths_cfg, "r", encoding="utf-8") as f:
                dp = yaml.safe_load(f) or {}
            root = Path(dp.get("default_output_root", "C:\\Users\\selsherb\\Documents\\AVS-E\\CL\\Brightlight\\dimensions"))
            fn = (dp.get("files", {}) or {}).get("dim_vendor")
            if fn:
                p = root / fn
                if p.exists():
                    df = pd.read_parquet(p)
                    col = None
                    for c in df.columns:
                        if str(c).strip().lower() == "vendor_id":
                            col = c
                            break
                    if col:
                        vendors = set(df[col].astype("string").str.strip().str.upper().dropna().tolist())
    except Exception:
        # Fallback to empty set on any IO error
        vendors = set()

    _VENDOR_SET = vendors
    return _VENDOR_SET


def is_valid_vendor_id(x: Any) -> bool:
    """Return True if the value is a plausible vendor_id.

    Backward-compatible rules:
      - Primary: membership in dim_vendor set → valid.
      - Otherwise: passes a permissive alphanumeric/dash regex (2–32 chars) → valid.
    """

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return False
    s = str(x).strip().upper()
    if not s:
        return False
    vendors = _load_vendor_set()
    if s in vendors:
        return True
    return bool(VENDOR_ID_RX.fullmatch(s))


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def normalize_brand(x: Any) -> str:
    """Normalize brand name for Phase 1 ingestion.

    - Trim and collapse whitespace
    - Normalize dashes
    - Preserve natural casing (no forced title-case)
    """

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x).replace("\u2013", "-").replace("\u2014", "-")
    s = _collapse_ws(s)
    return s


def normalize_item_name(x: Any) -> str:
    """Normalize item name for Phase 1 ingestion.

    - Trim and collapse whitespace
    - Normalize unicode dashes and quotes
    - Remove control characters
    - Preserve natural casing
    """

    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    s = str(x)
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    # Remove control characters
    s = re.sub(r"[\x00-\x1F\x7F]", " ", s)
    s = _collapse_ws(s)
    return s


def detect_misplaced_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Detect misplaced IDs between asin_id and vendor_id.

    Flags dq_misplaced_id when:
      - asin_id fails ASIN pattern AND asin_id exists in vendor list, OR
      - vendor_id looks like an ASIN (10 uppercase alphanumerics).
    Returns a copy with 'dq_misplaced_id' boolean column added (default False).
    """

    res = df.copy()
    vendors = _load_vendor_set()
    asin_col = "asin_id" if "asin_id" in res.columns else ("asin" if "asin" in res.columns else None)
    if asin_col is None:
        res["dq_misplaced_id"] = False
        return res

    asin_series = res[asin_col].astype("string").str.strip().str.upper()
    vendor_series = res.get("vendor_id", pd.Series(["" for _ in range(len(res))], index=res.index)).astype("string").str.strip().str.upper()

    asin_is_valid = asin_series.map(lambda s: bool(ASIN_FULL_RX.fullmatch(s)))
    asin_in_vendor = asin_series.isin(vendors) if vendors else pd.Series(False, index=res.index)
    vendor_looks_asin = vendor_series.map(lambda s: bool(ASIN_FULL_RX.fullmatch(s)))

    res["dq_misplaced_id"] = (~asin_is_valid & asin_in_vendor) | vendor_looks_asin
    return res


def quarantine_misplaced_ids(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Write dq_misplaced_id rows to phase1_id_issues.csv and set is_valid_row=False for them.

    Returns a modified copy of df.
    """

    res = df.copy()
    issues = res[res.get("dq_misplaced_id", False) == True]  # noqa: E712
    try:
        out_path = Path(output_dir) / "phase1_id_issues.csv"
        if not issues.empty:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            issues.to_csv(out_path, index=False, encoding="utf-8-sig")
    except Exception:
        # best-effort; do not fail pipeline here
        pass

    if "is_valid_row" not in res.columns:
        res["is_valid_row"] = True
    res.loc[res.get("dq_misplaced_id", False) == True, "is_valid_row"] = False  # noqa: E712
    return res
