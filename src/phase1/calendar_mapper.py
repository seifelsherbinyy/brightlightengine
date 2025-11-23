from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Union, Dict, Any

import pandas as pd


# Default date formats; can be overridden by config
DATE_FORMATS: List[str] = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
]


@dataclass
class CalendarConfig:
    raw_week_column: Optional[str] = None
    raw_start_column: Optional[str] = None
    raw_end_column: Optional[str] = None
    unified_calendar_path: str = str(Path("data") / "reference" / "unified_calendar_map.csv")
    raw_calendar_map_path: Optional[str] = str(Path("data") / "reference" / "calendar_map.csv")
    date_formats: Optional[List[str]] = None

    @staticmethod
    def from_mapping(cfg: Dict[str, Any] | None) -> "CalendarConfig":
        cfg = cfg or {}
        # Merge nested block first
        cal = cfg.get("calendar") or {}
        # Allow flat keys for backward compatibility
        unified_path = cal.get("unified_calendar_path") or cfg.get("unified_calendar_path")
        raw_bridge = cal.get("raw_calendar_map_path") or cfg.get("raw_calendar_map_path") or cfg.get("raw_calendar_bridge_path")
        raw_week_col = cal.get("raw_week_column") or cfg.get("raw_week_column") or cfg.get("week_raw_column") or cfg.get("week_column")
        raw_start_col = cal.get("raw_start_column") or cfg.get("raw_start_column")
        raw_end_col = cal.get("raw_end_column") or cfg.get("raw_end_column")
        date_formats = cal.get("date_formats") or cfg.get("date_formats")

        inst = CalendarConfig(
            raw_week_column=raw_week_col,
            raw_start_column=raw_start_col,
            raw_end_column=raw_end_col,
            unified_calendar_path=unified_path or str(Path("data") / "reference" / "unified_calendar_map.csv"),
            raw_calendar_map_path=raw_bridge or str(Path("data") / "reference" / "calendar_map.csv"),
            date_formats=date_formats,
        )

        # Update module-level DATE_FORMATS if provided
        if inst.date_formats and isinstance(inst.date_formats, list):
            # Only keep string formats
            formats = [f for f in inst.date_formats if isinstance(f, str) and f.strip()]
            if formats:
                global DATE_FORMATS
                DATE_FORMATS = formats
        return inst


def load_unified_calendar(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Unified calendar not found: {p}")
    df = pd.read_csv(p, dtype="string")
    # normalize headers
    df.columns = [str(c).strip().lower() for c in df.columns]
    # standard column names used downstream
    rename = {
        "canonical_week_id": "canonical_week_id",
        "week_start": "week_start_date",
        "week_end": "week_end_date",
        "month_number": "month_number",
        "month_label": "month_label",
        "quarter_label": "quarter_label",
        "canonical_year": "calendar_year",
        "canonical_week_number": "calendar_week_number",
    }
    for k in list(rename.keys()):
        if k not in df.columns:
            # do not error; allow schema variants
            rename.pop(k, None)
    df = df.rename(columns=rename)
    # Basic existence check
    if "canonical_week_id" not in df.columns:
        raise ValueError("Unified calendar must contain 'canonical_week_id' column")
    return df


def load_raw_calendar_map(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        # Return empty mapping if not provided
        return pd.DataFrame(columns=["raw_week_label", "canonical_week_id"]).astype("string")
    df = pd.read_csv(p, dtype="string")
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Expect columns like: raw_week_label, week_label, week_start, week_end
    if "week_label" in df.columns and "canonical_week_id" not in df.columns:
        df = df.rename(columns={"week_label": "canonical_week_id"})
    # Keep only the two needed fields if present
    keep = [c for c in ("raw_week_label", "canonical_week_id") if c in df.columns]
    if keep:
        df = df[keep]
    return df


def _try_parse_date(text: str) -> Optional[str]:
    s = str(text).strip()
    if not s:
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except Exception:
            continue
    return None


def parse_raw_canonical_format(raw: str) -> Optional[str]:
    """Return canonical week id when input already matches ^\d{4}W\d{2}$"""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if pd.isna(s) or s == "":
        return None
    import re
    if re.fullmatch(r"^\d{4}W\d{2}$", s):
        return s
    return None


def parse_yyyy_dash_wnn(raw: str) -> Optional[str]:
    """Parse strings like '2025-W36' into '2025W36'"""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    import re
    m = re.fullmatch(r"^(\d{4})-W(\d{1,2})$", s)
    if not m:
        return None
    year, wk = int(m.group(1)), int(m.group(2))
    wk = max(1, min(53, wk))
    return f"{year}W{wk:02d}"


def parse_wnn_space_yyyy(raw: str) -> Optional[str]:
    """Parse strings like 'W36 2025' into '2025W36'"""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    import re
    m = re.fullmatch(r"^W\s*(\d{1,2})\s*(\d{4})$", s)
    if not m:
        return None
    wk, year = int(m.group(1)), int(m.group(2))
    wk = max(1, min(53, wk))
    return f"{year}W{wk:02d}"


def parse_date_range(raw: str) -> Optional[Tuple[str, str]]:
    """Parse date range strings; supports 'DD/MM/YYYY-DD/MM/YYYY' and 'YYYY-MM-DD–YYYY-MM-DD'.

    Returns (start_date_iso, end_date_iso) or None when not matched.
    """
    if raw is None:
        return None
    s = str(raw).strip().replace("\u2013", "-")
    import re
    m_iso = re.search(r"(20\d{2}-\d{2}-\d{2})\s*[-]\s*(20\d{2}-\d{2}-\d{2})", s)
    if m_iso:
        start_iso = _try_parse_date(m_iso.group(1))
        end_iso = _try_parse_date(m_iso.group(2))
        if start_iso and end_iso:
            return start_iso, end_iso
        return None
    m_dmy = re.search(r"(\d{1,2}/\d{1,2}/20\d{2})\s*[-]\s*(\d{1,2}/\d{1,2}/20\d{2})", s)
    if m_dmy:
        start_iso = _try_parse_date(m_dmy.group(1))
        end_iso = _try_parse_date(m_dmy.group(2))
        if start_iso and end_iso:
            return start_iso, end_iso
    return None


def parse_single_date(date_str: str) -> Optional[str]:
    """Try to parse a single date string to ISO format using configured DATE_FORMATS."""
    return _try_parse_date(date_str)


def _map_start_date_to_unified(start_date: str, unified_calendar: pd.DataFrame) -> Optional[str]:
    """Map an ISO start_date to canonical_week_id using unified calendar.

    First tries equality with week_start_date, then containment within [week_start_date, week_end_date].
    """
    if start_date is None:
        return None
    if unified_calendar is None or unified_calendar.empty:
        return None
    cal = unified_calendar
    # Ensure expected columns
    if "week_start_date" not in cal.columns or "week_end_date" not in cal.columns:
        return None
    # Direct match
    direct = cal.loc[cal["week_start_date"].astype(str) == start_date]
    if not direct.empty:
        return direct.iloc[0]["canonical_week_id"]
    # Containment
    try:
        # ISO date string comparison works lexicographically for <= if both are YYYY-MM-DD
        mask = (cal["week_start_date"].astype(str) <= start_date) & (cal["week_end_date"].astype(str) >= start_date)
        row = cal.loc[mask]
        if not row.empty:
            return row.iloc[0]["canonical_week_id"]
    except Exception:
        return None
    return None


def parse_raw_week(raw_str: str) -> Optional[str]:
    """Parse a variety of raw week formats into canonical_week_id string like '2025W36'.

    Accepts formats:
      - '2025W36'
      - '2025-W36'
      - 'W36 2025'
      - 'DD/MM/YYYY-DD/MM/YYYY'
      - 'YYYY-MM-DD–YYYY-MM-DD' (supports hyphen or en dash)

    Returns canonical_week_id or None when not parseable directly.
    """

    if raw_str is None:
        return None
    s = str(raw_str).strip()
    if not s:
        return None

    up = s.upper().replace("\u2013", "-")  # normalize en dash to hyphen
    # Order per spec
    v = parse_raw_canonical_format(up)
    if v:
        return v
    v = parse_yyyy_dash_wnn(up)
    if v:
        return v
    v = parse_wnn_space_yyyy(up)
    if v:
        return v
    # Date ranges -> derive week id from start date via ISO calendar
    rng = parse_date_range(up)
    if rng:
        start_iso, _ = rng
        try:
            dt = datetime.fromisoformat(start_iso)
            iso_year, iso_week, _ = dt.isocalendar()
            return f"{iso_year}W{int(iso_week):02d}"
        except Exception:
            return None
    return None


def derive_canonical_week_id(
    df: pd.DataFrame,
    config: Union[CalendarConfig, Dict[str, Any], None],
    unified_calendar: pd.DataFrame,
    raw_calendar_map: pd.DataFrame,
) -> pd.DataFrame:
    """Derive canonical_week_id and attach calendar attributes.

    Steps:
      1. Determine raw week column name from df/config.
      2. Use parse_raw_week to compute canonical_week_id.
      3. Fallback: join on raw_calendar_map when parse returns None.
      4. Join with unified calendar to enrich attributes.
      5. Set dq_unmapped_week flag for failures.
    """

    result = df.copy()
    # Normalize config
    if not isinstance(config, CalendarConfig):
        config = CalendarConfig.from_mapping(config if isinstance(config, dict) else {})

    # Determine raw week column
    raw_week_col = config.raw_week_column
    if raw_week_col not in result.columns if raw_week_col else True:
        raw_week_col = None
        for cand in ("week_raw", "week", "week_label", "raw_week", "date_range"):
            if cand in result.columns:
                raw_week_col = cand
                break

    # Try deriving from raw week column
    if raw_week_col is not None:
        parsed = result[raw_week_col].map(parse_raw_week).astype("string")
        result["canonical_week_id"] = parsed
    else:
        # No week column: attempt from single date columns if provided
        result["canonical_week_id"] = pd.Series([pd.NA] * len(result), dtype="string")
        date_col = None
        for cand in (config.raw_start_column, config.raw_end_column, "week_start", "start_date"):
            if cand and cand in result.columns:
                date_col = cand
                break
        if date_col:
            iso_dates = result[date_col].map(parse_single_date)
            # Map via unified calendar
            result["canonical_week_id"] = iso_dates.map(lambda d: _map_start_date_to_unified(d, unified_calendar))

    # Fallback mapping for None using raw_calendar_map
    if not raw_calendar_map.empty and "raw_week_label" in raw_calendar_map.columns:
        # Only attempt fallback when we have a raw week column to join on
        if raw_week_col is not None and raw_week_col in result.columns:
            fallback = raw_calendar_map.rename(columns={"raw_week_label": raw_week_col})
            fallback = fallback[[raw_week_col, "canonical_week_id"]].dropna()
            if not fallback.empty:
                result = result.merge(
                    fallback.drop_duplicates(subset=[raw_week_col]),
                    on=raw_week_col,
                    how="left",
                    suffixes=("", "_fb"),
                )
                # fill missing from fallback
                result["canonical_week_id"] = result["canonical_week_id"].fillna(result.get("canonical_week_id_fb"))
                if "canonical_week_id_fb" in result.columns:
                    result = result.drop(columns=["canonical_week_id_fb"])

    # Join with unified calendar
    cal = unified_calendar
    if not cal.empty and "canonical_week_id" in cal.columns:
        result = result.merge(
            cal.drop_duplicates(subset=["canonical_week_id"]),
            on="canonical_week_id",
            how="left",
        )

    # DQ flag: missing canonical id or failed join to unified (e.g., no week_start_date)
    dq_flag = (
        result["canonical_week_id"].isna()
        | result["canonical_week_id"].astype("string").str.strip().eq("")
        | (~result.get("week_start_date").notna() if "week_start_date" in result.columns else False)
    )
    # If dq_flag computation used a scalar False, ensure boolean Series
    if not isinstance(dq_flag, pd.Series):
        dq_flag = pd.Series(False, index=result.index)
    result["dq_unmapped_week"] = dq_flag.fillna(True)

    # Ensure seasonal/holiday enrichment columns exist with safe defaults
    # Required per Phase 1f: season_name, holiday_name, peak_week_flag, retail_event_flag
    if "season_name" not in result.columns:
        result["season_name"] = pd.Series([pd.NA] * len(result), index=result.index, dtype="string")
    if "holiday_name" not in result.columns:
        result["holiday_name"] = pd.Series([pd.NA] * len(result), index=result.index, dtype="string")
    if "peak_week_flag" not in result.columns:
        result["peak_week_flag"] = False
    else:
        # coerce to boolean
        try:
            result["peak_week_flag"] = result["peak_week_flag"].fillna(False).astype(bool)
        except Exception:
            result["peak_week_flag"] = False
    if "retail_event_flag" not in result.columns:
        result["retail_event_flag"] = False
    else:
        try:
            result["retail_event_flag"] = result["retail_event_flag"].fillna(False).astype(bool)
        except Exception:
            result["retail_event_flag"] = False
    return result
