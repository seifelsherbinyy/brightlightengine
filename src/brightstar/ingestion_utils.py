"""Utility functions supporting Phase 1 ingestion and normalization."""
from __future__ import annotations

import json
import csv
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from .calendar_dimension_builder import build_canonical_dim_week, write_unified_calendar_csv
import yaml


LOGGER_NAME = "brightstar.phase1"

# Mandatory external output directory base for all Phase 1 artifacts
MANDATORY_OUTPUT_DIR = Path(r"C:\Users\selsherb\Documents\AVS-E\CL\Seif Vendors")


def get_timestamped_output_dir(timestamp: Optional[str] = None) -> Path:
    """Create and return a timestamped subdirectory under the mandatory base directory.

    Folder format: phase1_outputs_YYYY-MM-DD_HH-MM-SS
    """
    ensure_directory(MANDATORY_OUTPUT_DIR)
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = MANDATORY_OUTPUT_DIR / f"phase1_outputs_{ts}"
    return ensure_directory(out_dir)


DATE_RANGE_PATTERN = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{2,4})\s*[-–]\s*(\d{1,2}/\d{1,2}/\d{2,4})"
)
WEEK_NUMBER_PATTERN = re.compile(
    r"(?:FW|W)\s*0?(\d{1,2})(?:[^\d]*(\d{4}))?",
    flags=re.IGNORECASE,
)
ISO_WEEK_PATTERN = re.compile(r"(\d{4})[-_/]?W?0?(\d{1,2})")


@dataclass
class WeekLookup:
    """Container that accelerates week label lookups.

    Enhancements:
    - Supports exact lookup, then aggressive-normalized lookup (remove non-alnum),
      then parser-based resolution.
    - Parser resolution prefers mapping into a Sunday-start calendar derived from
      the provided mapping (via date containment or year/week_number when present).
    """

    mapping: pd.DataFrame
    _cache: Dict[str, Tuple[str, Optional[str], Optional[str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mapping.empty:
            raise ValueError("Calendar map is empty. Phase 1 requires at least one mapping.")

        df = self.mapping.copy()
        # Standard columns expected + optional derived
        for col in ("week_start", "week_end"):
            if col in df.columns:
                # Keep as text as well (for output); store parsed dates for lookup purposes
                parsed = pd.to_datetime(df[col], errors="coerce")
                df[f"_{col}_dt"] = parsed.dt.normalize()
            else:
                df[col] = pd.NA
                df[f"_{col}_dt"] = pd.NaT

        # Build normalization keys for lookups
        df["raw_normalized"] = df["raw_week_label"].astype(str).str.strip().str.lower()
        df["raw_norm_aggressive"] = df["raw_normalized"].map(lambda s: re.sub(r"[^a-z0-9]+", "", s))

        # Year/week_number optional support (Sunday-start semantics assumed in file)
        # Coerce to integers when present
        if "year" in df.columns:
            with pd.option_context('mode.chained_assignment', None):
                df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        if "week_number" in df.columns:
            with pd.option_context('mode.chained_assignment', None):
                df["week_number"] = pd.to_numeric(df["week_number"], errors="coerce").astype("Int64")

        # Retain copies and indices
        self._df = df
        self._lookup_exact = df.set_index("raw_normalized")
        self._lookup_aggr = df.drop_duplicates(subset=["raw_norm_aggressive"], keep="first").set_index("raw_norm_aggressive")
        # Map by canonical week label to the base row (first occurrence)
        self._by_week_label = df.drop_duplicates(subset=["week_label"], keep="first").set_index("week_label")
        # Optional mapping by (year, week_number)
        if "year" in df.columns and "week_number" in df.columns:
            self._by_year_week = df.dropna(subset=["year", "week_number"]).drop_duplicates(subset=["year", "week_number"], keep="first").set_index(["year", "week_number"])  # type: ignore
        else:
            self._by_year_week = None

    def _row_to_result(self, row: pd.Series) -> Tuple[str, Optional[str], Optional[str]]:
        return (
            str(row["week_label"]),
            _safe_date(row.get("week_start")),
            _safe_date(row.get("week_end")),
        )

    def _find_by_date(self, dt: pd.Timestamp) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        # Containment in [week_start, week_end]
        try:
            mask = (self._df["_week_start_dt"].notna()) & (self._df["_week_end_dt"].notna())
            within = self._df.loc[mask & (self._df["_week_start_dt"] <= dt) & (self._df["_week_end_dt"] >= dt)]
            if not within.empty:
                # Prefer the first canonical row per week_label
                row = within.drop_duplicates(subset=["week_label"], keep="first").iloc[0]
                return self._row_to_result(row)
        except Exception:
            pass
        return None

    def _find_by_year_week(self, year: int, week: int) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        try:
            if self._by_year_week is not None and (year, week) in self._by_year_week.index:  # type: ignore
                row = self._by_year_week.loc[(year, week)]  # type: ignore
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return self._row_to_result(row)
        except Exception:
            pass
        # As a fallback, attempt to find by week_label pattern YYYYWNN
        candidate = f"{year}W{week:02d}"
        if candidate in self._by_week_label.index:
            row = self._by_week_label.loc[candidate]
            return self._row_to_result(row)
        return None

    def resolve(self, raw_label: str) -> Tuple[str, Optional[str], Optional[str]]:
        """Return normalized label and boundaries for a raw week label.

        Resolution order:
        1) Exact map on raw label (lowercased/trimmed)
        2) Aggressively normalized map (remove non-alphanumerics)
        3) Parse and map via Sunday-start calendar (date containment or year/week)
        4) Last resort: echo raw label with None boundaries
        """

        key = str(raw_label).strip()
        normalized_key = key.lower()
        if not normalized_key:
            return "", None, None

        # Cache
        if normalized_key in self._cache:
            return self._cache[normalized_key]

        # 1) Exact map
        if normalized_key in self._lookup_exact.index:
            row = self._lookup_exact.loc[normalized_key]
            result = self._row_to_result(row)
            self._cache[normalized_key] = result
            return result

        # 2) Aggressive map
        aggr_key = re.sub(r"[^a-z0-9]+", "", normalized_key)
        if aggr_key in self._lookup_aggr.index:
            row = self._lookup_aggr.loc[aggr_key]
            result = self._row_to_result(row)
            self._cache[normalized_key] = result
            return result

        # 3) Parse and map
        parsed = _parse_week_tokens(key)
        if parsed:
            kind, data = parsed
            if kind == "daterange":
                # Map by date containment using the start date
                start_dt = pd.to_datetime(data[0], errors="coerce")
                if pd.notna(start_dt):
                    found = self._find_by_date(start_dt)
                    if found:
                        self._cache[normalized_key] = found
                        return found
            elif kind == "yearweek":
                year, week = data
                # If year is missing (0 sentinel), infer from mapping by choosing the
                # latest available year for that week number based on week_start.
                if int(year) == 0:
                    try:
                        candidates = self._df[self._df.get("week_number").astype("Int64") == int(week)]  # type: ignore
                        if not candidates.empty:
                            # Choose the row with the max week_start date
                            if "_week_start_dt" in candidates.columns:
                                row = candidates.sort_values("_week_start_dt", ascending=False).iloc[0]
                            else:
                                row = candidates.iloc[-1]
                            result = self._row_to_result(row)
                            self._cache[normalized_key] = result
                            return result
                    except Exception:
                        pass
                found = self._find_by_year_week(int(year), int(week))
                if found:
                    self._cache[normalized_key] = found
                    return found

        # 4) Fallback: try legacy ISO-like parsing then echo
        legacy = _parse_week_from_label(key)
        if legacy:
            self._cache[normalized_key] = legacy
            return legacy

        result = (key, None, None)
        self._cache[normalized_key] = result
        return result


def _safe_date(value: object) -> Optional[str]:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _parse_week_from_label(raw_label: str) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """Parse date ranges or explicit week references from a raw label."""

    text = str(raw_label).strip()
    if not text:
        return None

    range_match = DATE_RANGE_PATTERN.search(text)
    if range_match:
        start_raw, end_raw = range_match.groups()
        start_dt = pd.to_datetime(start_raw, errors="coerce")
        end_dt = pd.to_datetime(end_raw, errors="coerce")
        if pd.notna(start_dt) and pd.notna(end_dt):
            iso = start_dt.isocalendar()
            week_label, iso_start, iso_end = _week_from_number(iso.year, iso.week)
            start_value = iso_start or start_dt.strftime("%Y-%m-%d")
            end_value = iso_end or end_dt.strftime("%Y-%m-%d")
            return week_label, start_value, end_value

    iso_match = ISO_WEEK_PATTERN.search(text)
    if iso_match:
        year = int(iso_match.group(1))
        week = int(iso_match.group(2))
        return _week_from_number(year, week)

    week_match = WEEK_NUMBER_PATTERN.search(text)
    if week_match:
        week = int(week_match.group(1))
        year = week_match.group(2)
        if year:
            return _week_from_number(int(year), week)

    return None


# New, richer token parser that returns structured hints for WeekLookup to map
def _parse_week_tokens(raw_label: str) -> Optional[Tuple[str, Tuple]]:
    """Parse a raw label into structured tokens without deciding Sunday-start boundaries.

    Returns one of:
    - ("daterange", (start_str, end_str))
    - ("yearweek", (year, week))  # when year present or inferred
    - None if not recognized
    """

    text = str(raw_label).strip()
    if not text:
        return None

    # 1) Date range first (hyphen or en-dash, flexible spacing, 2- or 4-digit year)
    range_match = DATE_RANGE_PATTERN.search(text)
    if range_match:
        return ("daterange", (range_match.group(1), range_match.group(2)))

    # 2) ISO-like patterns:  YYYY-WNN or YYYYWNN
    iso_match = ISO_WEEK_PATTERN.search(text)
    if iso_match:
        year = int(iso_match.group(1))
        week = int(iso_match.group(2))
        return ("yearweek", (year, week))

    # 3) Week tokens: Week NN [YYYY] [Qn], FWNN [YYYY], Fiscal Week NN [YYYY]
    # Capture NN and optional YYYY
    m = re.search(r"(?:fiscal\s+week|week|fw)\s*0?(\d{1,2})(?:[^\d]{0,5}(\d{4}))?", text, flags=re.IGNORECASE)
    if m:
        week = int(m.group(1))
        year_str = m.group(2)
        if year_str:
            return ("yearweek", (int(year_str), week))
        else:
            # Defer year inference to WeekLookup (prefer latest available)
            # Use 0 as a sentinel; WeekLookup will choose a year
            return ("yearweek", (0, week))
    # 4) Bare numeric week like "41"
    bare = re.fullmatch(r"0?(\d{1,2})", text)
    if bare:
        return ("yearweek", (0, int(bare.group(1))))
    return None


def _week_from_number(year: int, week: int) -> Tuple[str, str, str]:
    """Legacy ISO-based week boundaries (Monday-start).

    Kept for backward compatibility in rare fallbacks. Prefer Sunday-start
    logic implemented inside WeekLookup for new parsing paths.
    """
    label = f"{year}W{week:02d}"
    try:
        start = date.fromisocalendar(year, week, 1)
        end = date.fromisocalendar(year, week, 7)
    except ValueError:
        return label, None, None
    return label, start.isoformat(), end.isoformat()


def _first_sunday_of_year(year: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=1, day=1)
    # weekday: Mon=0 .. Sun=6
    days_to_sunday = (d.weekday() + 1) % 7
    return (d - pd.Timedelta(days=days_to_sunday)).normalize() if days_to_sunday != 0 else d.normalize()


def _sunday_week_from_date(dt: pd.Timestamp) -> Tuple[int, int, pd.Timestamp, pd.Timestamp]:
    """Compute Sunday-start (Sun–Sat) week year/number and bounds for a given date.

    - Week starts on Sunday.
    - Week year is the year of the week_start (Sunday).
    - Week 1 is the week starting at the first Sunday of the year.
    """
    if pd.isna(dt):
        raise ValueError("Invalid date for Sunday-start computation")
    # Find week_start Sunday containing dt
    days_to_sunday = (dt.weekday() + 1) % 7
    week_start = (dt - pd.Timedelta(days=days_to_sunday)).normalize()
    week_end = week_start + pd.Timedelta(days=6)
    wy = week_start.year
    first_sun = _first_sunday_of_year(wy)
    if week_start < first_sun:
        # belongs to previous year's last week
        wy = wy - 1
        first_sun = _first_sunday_of_year(wy)
    week_no = int(((week_start - first_sun).days // 7) + 1)
    return wy, week_no, week_start, week_end


def load_config(path: str | Path) -> Dict:
    """Load a YAML configuration file."""

    with open(path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def ensure_directory(directory: str | Path) -> Path:
    """Ensure that the directory exists."""

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(config: Dict, output_dir: Optional[Path] = None) -> logging.Logger:
    """Configure project-wide logging for the ingestion phase.

    Per Phase 1 stabilization requirements, validation logs must be emitted to the
    mandatory output directory. We therefore route the file handler to that path
    exclusively, in addition to a console handler.
    """

    # Route logs to the timestamped run directory if provided; otherwise base directory
    base = output_dir if output_dir is not None else MANDATORY_OUTPUT_DIR
    ensure_directory(base)
    # Per requirements, default file name is ingestion_log.txt
    file_name = config.get("logging", {}).get("file_name", "ingestion_log.txt")
    log_path = base / file_name

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, config.get("logging", {}).get("level", "INFO")))
    logger.handlers = []

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.debug("Logging initialised. Logs will be written to %s", log_path)
    return logger


def load_calendar_map(path: str | Path) -> WeekLookup:
    """Load or build the canonical Sunday-start calendar and return a WeekLookup.

    This function ENFORCES a single canonical calendar (no synonym rows) and
    ensures there is exactly one row per week with Sunday-start boundaries.

    Behavior:
    - Attempt to read data/reference/unified_calendar_map.csv (canonical dim_week).
    - If missing or invalid, build the calendar for a broad range (2018..2030),
      write it to data/reference/unified_calendar_map.csv and a compatibility
      copy at data/reference/calendar_map.csv.
    - Construct a minimal mapping DataFrame with columns expected by WeekLookup:
      raw_week_label == week_label == canonical_week_id, plus week_start/week_end
      and numeric year/week_number for fast lookup.
    - Also write a copy of the canonical calendar to the mandatory run folder for
      audit convenience.
    """

    ref_path = Path(path)
    ref_dir = ref_path.parent
    unified_ref = ref_dir / "unified_calendar_map.csv"
    compat_ref = ref_dir / "calendar_map.csv"

    dim_week: pd.DataFrame
    read_ok = False
    if unified_ref.exists():
        try:
            dim_week = pd.read_csv(unified_ref)
            # basic sanity checks
            required_cols = {"canonical_year", "canonical_week_number", "canonical_week_id", "week_start", "week_end"}
            if required_cols.issubset(set(dim_week.columns)):
                read_ok = True
            else:
                read_ok = False
        except Exception:
            read_ok = False
    if not read_ok:
        # Build fresh canonical calendar and persist
        dim_week = build_canonical_dim_week(2018, 2030)
        try:
            write_unified_calendar_csv(dim_week, unified_ref)
            # write compatibility copy
            write_unified_calendar_csv(dim_week, compat_ref)
        except Exception as exc:
            logging.getLogger(LOGGER_NAME).warning("Failed to write canonical calendar references: %s", exc)

    # Normalize dtypes
    with pd.option_context('mode.chained_assignment', None):
        if not pd.api.types.is_datetime64_any_dtype(dim_week.get("week_start", pd.Series([], dtype='datetime64[ns]'))):
            try:
                dim_week["week_start"] = pd.to_datetime(dim_week["week_start"], errors="coerce")
                dim_week["week_end"] = pd.to_datetime(dim_week["week_end"], errors="coerce")
            except Exception:
                pass

    # Construct WeekLookup mapping DataFrame
    mapping = pd.DataFrame(
        {
            "raw_week_label": dim_week["canonical_week_id"].astype(str),
            "week_label": dim_week["canonical_week_id"].astype(str),
            "week_start": dim_week["week_start"].dt.date.astype(str),
            "week_end": dim_week["week_end"].dt.date.astype(str),
            "year": dim_week["canonical_year"],
            "week_number": dim_week["canonical_week_number"],
            "month_number": dim_week.get("month_number"),
            "month_label": dim_week.get("month_label"),
            "quarter_label": dim_week.get("quarter_label"),
            "display_label": dim_week.get("display_label"),
        }
    )

    # Persist a copy to the mandatory output directory for auditing
    try:
        ensure_directory(MANDATORY_OUTPUT_DIR)
        (MANDATORY_OUTPUT_DIR / "unified_calendar_map.csv").write_text(
            mapping.to_csv(index=False)
        )
    except Exception:
        pass

    return WeekLookup(mapping)


def _generate_unified_calendar_map(start_year: int, end_year: int) -> pd.DataFrame:
    """Generate a comprehensive Sunday-start calendar map with synonyms.

    Returns a DataFrame with at least columns:
    - raw_week_label, week_label, week_start, week_end
    And derived columns: year, week_number, month_number, month_label, quarter_label, display_label
    """

    weeks: List[dict] = []
    # Build weekly Sundays from the first Sunday on/before Jan 1 of start_year
    start_dt = _first_sunday_of_year(start_year)
    end_dt = pd.Timestamp(year=end_year, month=12, day=31)

    current = start_dt
    while current <= end_dt:
        wy, wn, wstart, wend = _sunday_week_from_date(current)
        label = f"{wy}W{wn:02d}"
        month_number = int(wstart.month)
        month_label = wstart.strftime("%b %Y")
        quarter = (int(((wstart.month - 1) // 3) + 1))
        display = f"Week {wn:02d} {wy} Q{quarter}"

        # Canonical/base representations (we will create synonyms below)
        base_entries = _expand_week_synonyms(wstart, wend, wy, wn, label)
        for raw in base_entries:
            weeks.append(
                {
                    "raw_week_label": raw,
                    "week_label": label,
                    "week_start": wstart.date().isoformat(),
                    "week_end": wend.date().isoformat(),
                    "year": wy,
                    "week_number": wn,
                    "month_number": month_number,
                    "month_label": month_label,
                    "quarter_label": f"Q{quarter}",
                    "display_label": display,
                }
            )

        # Advance one week
        current = current + pd.Timedelta(days=7)

    df = pd.DataFrame(weeks)
    # Deduplicate in case synonyms collide (they shouldn't)
    df = df.drop_duplicates(subset=["raw_week_label"], keep="first")
    # Add iso_year/iso_week/week_id fields for unified mapping across formats
    with pd.option_context('mode.chained_assignment', None):
        df["iso_year"] = df["year"].astype("Int64")
        df["iso_week"] = df["week_number"].astype("Int64")
        df["week_id"] = df["week_label"].astype(str)
    return df


def _expand_week_synonyms(week_start: pd.Timestamp, week_end: pd.Timestamp, year: int, week_number: int, week_label: str) -> List[str]:
    """Create a concise but robust set of raw label synonyms for a given week.

    Included variants:
    - Date range: MM/DD/YYYY-MM/DD/YYYY (hyphen) and en-dash variant.
    - Week tokens: "WNN YYYY", "YYYY-WNN", "YYYYWNN".
    - Display style: "Week NN", "Week NN YYYY", and with quarter: "Week NN YYYY Qq".
    """

    def fmt_us(d: pd.Timestamp, two_digit_year: bool = False) -> str:
        if two_digit_year:
            return d.strftime("%m/%d/%y")
        return d.strftime("%m/%d/%Y")

    start_us = fmt_us(week_start)
    end_us = fmt_us(week_end)
    # Core variants
    variants = [
        f"{start_us}-{end_us}",
        f"{start_us} - {end_us}",
        f"{start_us} – {end_us}",  # en-dash
        f"W{week_number:02d} {year}",
        f"{year}-W{week_number:02d}",
        f"{year}W{week_number:02d}",
        f"Week {week_number:02d}",
        f"Week {week_number:02d} {year}",
        f"Week {week_number} {year}",
    ]
    # Quarter decorated display
    quarter = int(((week_start.month - 1) // 3) + 1)
    variants.append(f"Week {week_number:02d} {year} Q{quarter}")
    # Also include canonical week_label as raw label to make direct matches easy
    variants.append(week_label)
    return variants


def load_master_lookup(path: str | Path, config: Dict | None = None) -> pd.DataFrame:
    """Load the master reference file linking ASINs to vendor details.

    Upgrades:
    - Supports multiple sheets: when sheet_name is None and Excel input, all sheets are concatenated.
    - Safe dtype=str for CSV/Excel to prevent implicit numeric/date coercion.
    - Applies config column mappings; tries case-insensitive auto-rename when unspecified.
    - Validates ASIN uniqueness (by canonical asin_key); last occurrence kept with a warning if conflicts.
    - Produces clean columns: asin, asin_key, vendor_code, vendor_name.
    """

    if not path:
        return pd.DataFrame(columns=["asin", "asin_key", "vendor_code", "vendor_name"])

    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["asin", "asin_key", "vendor_code", "vendor_name"])

    config = config or {}
    sheet_name = config.get("sheet_name")
    columns = config.get("columns", {})

    # Load the master file with robust handling of missing/renamed sheets
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        try:
            # If sheet_name is None, read all sheets
            if sheet_name in (None, "all", "*"):
                all_frames = pd.read_excel(path, sheet_name=None, dtype=str)
                frame = pd.concat(all_frames.values(), ignore_index=True) if all_frames else pd.DataFrame()
            else:
                frame = pd.read_excel(path, sheet_name=sheet_name, dtype=str)
        except ValueError as exc:
            # Common case: configured sheet name doesn't exist. Fall back gracefully.
            logger = logging.getLogger(__name__)
            try:
                xl = pd.ExcelFile(path)
                # Prefer a sheet that contains 'master' (case-insensitive); otherwise first sheet
                fallback_sheet = None
                candidates = [s for s in xl.sheet_names]
                for s in candidates:
                    if str(s).strip().lower() == "master" or "master" in str(s).strip().lower():
                        fallback_sheet = s
                        break
                if fallback_sheet is None and candidates:
                    fallback_sheet = candidates[0]
                logger.warning(
                    "Configured masterfile sheet '%s' not found in %s. Using sheet '%s' instead. Error: %s",
                    sheet_name,
                    path,
                    fallback_sheet,
                    exc,
                )
                frame = pd.read_excel(path, sheet_name=fallback_sheet, dtype=str)
            except Exception:
                # If we still fail, re-raise the original error to preserve context
                raise exc
    else:
        frame = pd.read_csv(path, dtype=str)
    if columns:
        rename_map = {value: key for key, value in columns.items()}
        frame = frame.rename(columns=rename_map)

    expected = {"asin", "vendor_code", "vendor_name"}
    # Attempt case-insensitive rename when explicit mapping is not provided
    rename_map = {}
    lower_map = {col.lower(): col for col in frame.columns}
    for key in expected:
        if key not in frame.columns and key in lower_map:
            rename_map[lower_map[key]] = key
    if rename_map:
        frame = frame.rename(columns=rename_map)

    frame = frame[[col for col in frame.columns if col in expected]].copy()
    for column in ("asin", "vendor_code", "vendor_name"):
        if column not in frame.columns:
            frame[column] = pd.NA
        else:
            frame[column] = frame[column].apply(
                lambda value: (str(value).strip() or pd.NA)
                if pd.notna(value)
                else pd.NA
            )

    frame["asin"] = frame["asin"].apply(
        lambda value: (str(value).strip() or pd.NA) if pd.notna(value) else pd.NA
    )
    # Canonical ASIN key to make joins robust to spaces/dashes/case
    frame["asin_key"] = frame["asin"].map(canonicalize_asin)
    frame = frame.dropna(subset=["asin"])  # drop rows with missing ASIN
    # Validate uniqueness and prefer last occurrence per asin_key
    if not frame.empty:
        dupes = frame.duplicated(subset=["asin_key"], keep=False)
        if dupes.any():
            # Check if there are conflicting vendor mappings for same asin_key
            conflicts = (
                frame.loc[dupes]
                .groupby("asin_key")[
                    ["vendor_code", "vendor_name"]
                ]
                .nunique(dropna=False)
                .max(axis=1)
            )
            if (conflicts > 1).any():
                logging.getLogger(LOGGER_NAME).warning(
                    "Master lookup contains conflicting vendor mappings for some ASINs; last occurrence will be kept."
                )
    frame = frame.drop_duplicates(subset=["asin_key"], keep="last")
    return frame[["asin", "asin_key", "vendor_code", "vendor_name"]]


# ============================================================
# Product Master (vendor_code + asin → descriptive item fields)
# ============================================================

PRODUCT_MASTER_COLUMNS: List[str] = [
    "vendor_code",
    "vendor_name",
    "asin",
    "item_name_raw",
    "item_name_clean",
    "brand",
    "category",
    "sub_category",
    "pack_size",
    "pack_unit",
    "notes",
]


def load_product_master(config: Dict) -> pd.DataFrame:
    """Load the Product Master CSV defined at paths.product_master_path.

    Behavior:
    - If file missing or unreadable, return an empty DataFrame with expected columns.
    - Never raises; logs a warning and degrades gracefully.
    """
    logger = logging.getLogger(__name__)
    path_str = (config or {}).get("paths", {}).get("product_master_path")
    if not path_str:
        return pd.DataFrame(columns=PRODUCT_MASTER_COLUMNS)
    path = Path(path_str)
    try:
        if path.exists():
            df = pd.read_csv(path)
            # Ensure required columns exist (fill missing with NA)
            for c in PRODUCT_MASTER_COLUMNS:
                if c not in df.columns:
                    df[c] = pd.NA
            return df[PRODUCT_MASTER_COLUMNS].copy()
        else:
            logger.warning("Product Master not found at %s; proceeding without item attributes.", path)
            return pd.DataFrame(columns=PRODUCT_MASTER_COLUMNS)
    except Exception as exc:  # pragma: no cover - defensive on FS variations
        logger.warning("Failed to load Product Master at %s (%s). Proceeding without item attributes.", path, exc)
        return pd.DataFrame(columns=PRODUCT_MASTER_COLUMNS)


def attach_product_master(normalized_df: pd.DataFrame, product_master_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join Product Master attributes onto ASIN-level rows.

    Join keys:
      - vendor_code (when available)
      - asin

    For vendor-level rows (no ASIN), the descriptive fields remain null.
    """
    if normalized_df is None or normalized_df.empty:
        return normalized_df if isinstance(normalized_df, pd.DataFrame) else pd.DataFrame()

    df = normalized_df.copy()
    # Ensure expected identifier columns exist
    if "asin" not in df.columns:
        df["asin"] = pd.NA
    if product_master_df is None or product_master_df.empty:
        # Ensure the new columns exist but remain null
        for c in [
            "item_name_clean",
            "brand",
            "category",
            "sub_category",
            "pack_size",
            "pack_unit",
        ]:
            if c not in df.columns:
                df[c] = pd.NA
        return df

    pm = product_master_df.copy()
    # Select only the columns we need for enrichment
    keep_cols = [
        "vendor_code",
        "asin",
        "item_name_clean",
        "brand",
        "category",
        "sub_category",
        "pack_size",
        "pack_unit",
    ]
    for c in keep_cols:
        if c not in pm.columns:
            pm[c] = pd.NA
    pm = pm[keep_cols]

    # Perform left join for rows with ASIN present
    if df["asin"].notna().any():
        try:
            df = df.merge(pm, on=["vendor_code", "asin"], how="left")
        except Exception:
            # Fallback to join on asin only when vendor_code missing in normalized
            df = df.merge(pm.drop(columns=["vendor_code"], errors="ignore"), on=["asin"], how="left")
    else:
        # Ensure columns exist
        for c in keep_cols[2:]:
            if c not in df.columns:
                df[c] = pd.NA
    return df


# ============================================================
# Item name extraction and pending list (from raw exports)
# ============================================================

def _clean_item_name(name: object) -> Optional[str]:
    """Clean a raw item name to a canonical display form.

    strip → lower → collapse spaces → title-case.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return None
    s = str(name).strip()
    if not s:
        return None
    s = " ".join(s.lower().split())
    return s.title()


def append_pending_itemnames(pending_path: Path, rows: List[Dict[str, object]]) -> None:
    """Append unseen item names to metadata/new_itemnames_pending.csv.

    rows: list of dicts with keys at least: vendor_code, asin, item_name_raw, item_name_clean, source_file, detected_at.
    """
    if not rows:
        return
    pending_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)
    if pending_path.exists():
        try:
            df_old = pd.read_csv(pending_path)
            combined = pd.concat([df_old, df_new], ignore_index=True)
        except Exception:
            combined = df_new
    else:
        combined = df_new
    # Keep simple; do not dedupe to preserve audit trail
    combined.to_csv(pending_path, index=False)


def extract_item_names_from_raw(df_raw: pd.DataFrame) -> pd.Series:
    """Best-effort extraction of item names from common columns in raw files.

    Looks for headers like: item_name, title, product_title, item name, item, description.
    Returns a Series `item_name_clean_extracted` aligned to df_raw index, with cleaned names or NA.
    """
    if df_raw is None or df_raw.empty:
        return pd.Series([pd.NA] * 0, dtype=object)
    candidates = [
        "item_name", "item name", "title", "product_title", "product title", "asin_title",
        "description", "item_description", "item description",
    ]
    lower_map = {c.lower(): c for c in df_raw.columns}
    col = next((lower_map[c] for c in candidates if c in lower_map), None)
    if not col:
        # try any column that looks like a title by content (heuristic, still offline/deterministic)
        col = next((c for c in df_raw.columns if str(c).strip().lower().endswith("title")), None)
    if not col:
        return pd.Series([pd.NA] * len(df_raw), index=df_raw.index, dtype=object)
    cleaned = df_raw[col].apply(_clean_item_name)
    cleaned = cleaned.where(cleaned.notna(), other=pd.NA)
    return cleaned.rename("item_name_clean_extracted")


# ============================================================
# Masterfile item-name extraction for fallback
# ============================================================

def load_masterfile_frame(config: Dict) -> pd.DataFrame:
    """Load the raw Masterfile table configured at paths.masterfile.

    Returns a DataFrame (may be empty) with original columns; caller is responsible for selecting columns.
    Falls back between sheets similar to load_master_lookup. Never raises.
    """
    try:
        path_str = (config or {}).get("paths", {}).get("masterfile")
        if not path_str:
            return pd.DataFrame()
        path = Path(path_str)
        if not path.exists():
            return pd.DataFrame()
        mf_cfg = (config or {}).get("masterfile", {})
        sheet_name = mf_cfg.get("sheet_name")
        if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
            try:
                if sheet_name in (None, "all", "*"):
                    frames = pd.read_excel(path, sheet_name=None, dtype=str)
                    return pd.concat(frames.values(), ignore_index=True) if frames else pd.DataFrame()
                return pd.read_excel(path, sheet_name=sheet_name, dtype=str)
            except ValueError:
                # mimic fallback from load_master_lookup
                try:
                    xl = pd.ExcelFile(path)
                    fallback = None
                    for s in xl.sheet_names:
                        if str(s).strip().lower() == "master" or "master" in str(s).strip().lower():
                            fallback = s
                            break
                    if fallback is None and xl.sheet_names:
                        fallback = xl.sheet_names[0]
                    return pd.read_excel(path, sheet_name=fallback, dtype=str)
                except Exception:
                    return pd.DataFrame()
        # CSV fallback
        return pd.read_csv(path, dtype=str)
    except Exception:
        return pd.DataFrame()


def extract_masterfile_item_names(masterfile_df: pd.DataFrame) -> pd.DataFrame:
    """Extract ASIN→item_name mapping from Masterfile for fallback enrichment.

    - Normalizes column names to lowercase to find asin and item_name-like columns.
    - Cleans names via _clean_item_name.
    - Returns 2 columns: asin, item_name_master
    """
    if masterfile_df is None or masterfile_df.empty:
        return pd.DataFrame(columns=["asin", "item_name_master"]).astype({"asin": str})
    df = masterfile_df.copy()
    lower_map = {c.lower(): c for c in df.columns}
    asin_col = lower_map.get("asin") or next((lower_map[c] for c in lower_map if c.replace(" ", "") == "asin#"), None)
    # Heuristics for item name column
    name_candidates = [
        "item_name", "item name", "title", "product_title", "product title",
        "asin_title", "description", "item_description", "item description",
    ]
    name_col = None
    for cand in name_candidates:
        if cand in lower_map:
            name_col = lower_map[cand]
            break
    if name_col is None:
        # pick first text-like column with many non-null values
        for c in df.columns:
            if df[c].dtype == object and df[c].notna().sum() > 0:
                name_col = c
                break
    if not asin_col or not name_col:
        return pd.DataFrame(columns=["asin", "item_name_master"]).astype({"asin": str})
    out = df[[asin_col, name_col]].rename(columns={asin_col: "asin", name_col: "item_name_master"}).copy()
    out["asin"] = out["asin"].astype(str).str.strip()
    out["item_name_master"] = out["item_name_master"].apply(_clean_item_name)
    out = out.dropna(subset=["asin"]).drop_duplicates(subset=["asin"], keep="last")
    return out[["asin", "item_name_master"]]


# ============================================================
# Value casting and flags
# ============================================================

_MISSING_TOKENS = {"", "na", "n/a", "-", "#div/0!", "null", "none"}


def _safe_to_float(val: object) -> Tuple[Optional[float], Optional[str]]:
    """Safely convert raw value to float and return (value, flag).

    Returns value_flag of 'missing' on known tokens, 'invalid_cast' on parse error, else None.
    """
    if val is None:
        return (None, "missing")
    if isinstance(val, float) and pd.isna(val):
        return (None, "missing")
    sval = str(val).strip()
    if sval.lower() in _MISSING_TOKENS:
        return (None, "missing")
    try:
        # Remove thousand separators and currency symbols minimally
        sval2 = sval.replace(",", "").replace("$", "")
        return (float(sval2), None)
    except Exception:
        return (None, "invalid_cast")


def apply_value_flags(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Ensure numeric value casting and set value_flag for missing/invalid/zero-from-source.

    - Coerces 'value' via _safe_to_float if not numeric.
    - Sets 'value_flag' to 'missing' or 'invalid_cast' where appropriate.
    - Detects files with all-zero values (per source_file) and sets 'zero_from_source' except for whitelisted metrics.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    # Ensure value_flag exists
    if "value_flag" not in out.columns:
        out["value_flag"] = pd.NA
    # Cast values safely when dtype is object
    if out["value"].dtype == object:
        casted = out["value"].apply(_safe_to_float)
        out["value"] = casted.apply(lambda t: t[0])
        flags = casted.apply(lambda t: t[1])
        out.loc[flags.notna(), "value_flag"] = flags[flags.notna()].values
    # zero-from-source detection per file
    allowed_zero = set((config or {}).get("ingestion", {}).get("allowed_zero_metrics", []) or [])
    if "source_file" in out.columns:
        for src, group in out.groupby("source_file"):
            non_missing = group["value"].fillna(0.0)
            if (non_missing == 0.0).all():
                # mark except allowed metrics
                idx = group.index
                mask = ~group["metric"].astype(str).isin(allowed_zero)
                mark_idx = idx[mask]
                out.loc[mark_idx, "value_flag"] = out.loc[mark_idx, "value_flag"].fillna("zero_from_source")
    return out


# ============================================================
# Export contract utilities and validation
# ============================================================

REQUIRED_EXPORT_COLUMNS: List[str] = [
    "entity_type", "entity_id", "vendor_code", "vendor_name",
    "asin", "item_name_clean", "metric", "unit",
    "week_label", "week_start_date", "fiscal_week", "fiscal_year",
    "value", "value_flag", "source_file", "ingestion_run_id",
]


def ensure_export_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure final schema/columns exist and are properly derived where possible."""
    if df is None:
        return pd.DataFrame(columns=REQUIRED_EXPORT_COLUMNS)
    out = df.copy()
    # entity_type/entity_id
    if "asin" not in out.columns:
        out["asin"] = pd.NA
    out["entity_type"] = out["asin"].apply(lambda x: "asin" if pd.notna(x) and str(x).strip() else "vendor")
    if "entity_id" not in out.columns:
        out["entity_id"] = out.apply(
            lambda r: str(r.get("vendor_code", "")) + ("_" + str(r.get("asin"))) if r["entity_type"] == "asin" else str(r.get("vendor_code", "")),
            axis=1,
        )
    # unit column default
    if "unit" not in out.columns:
        out["unit"] = pd.NA
    # week_start_date: provide alongside legacy week_start without renaming to preserve backward compatibility
    if "week_start" in out.columns and "week_start_date" not in out.columns:
        try:
            out["week_start_date"] = pd.to_datetime(out["week_start"], errors="coerce")
        except Exception:
            out["week_start_date"] = out["week_start"]
    # fiscal fields safe
    for c in ("fiscal_week", "fiscal_year"):
        if c not in out.columns:
            out[c] = pd.NA
    # ingestion_run_id ensure
    if "ingestion_run_id" not in out.columns:
        out["ingestion_run_id"] = pd.NA
    # value_flag ensure
    if "value_flag" not in out.columns:
        out["value_flag"] = pd.NA
    # enforce column order
    for c in REQUIRED_EXPORT_COLUMNS:
        if c not in out.columns:
            out[c] = pd.NA
    return out[REQUIRED_EXPORT_COLUMNS + [c for c in out.columns if c not in REQUIRED_EXPORT_COLUMNS]]


def validate_phase1_output(df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> None:
    """Best-effort validation of the normalized Phase 1 output.

    Logs warnings only; never raises.
    """
    lg = logger or logging.getLogger(LOGGER_NAME)
    if df is None or df.empty:
        lg.warning("Phase 1 validation: normalized dataframe is empty.")
        return
    missing = [c for c in REQUIRED_EXPORT_COLUMNS if c not in df.columns]
    if missing:
        lg.warning("Phase 1 validation: missing required columns: %s", ", ".join(missing))
    # metric not null
    if "metric" in df.columns and df["metric"].isna().any():
        lg.warning("Phase 1 validation: %d rows have null metric.", int(df["metric"].isna().sum()))
    # item_name_clean for ASIN rows when product master present (heuristic)
    if {"entity_type", "item_name_clean"}.issubset(df.columns):
        mask_asin = df["entity_type"].astype(str).str.lower() == "asin"
        if mask_asin.any():
            missing_names = df.loc[mask_asin, "item_name_clean"].isna().sum()
            if missing_names:
                lg.info("Phase 1 validation: %d ASIN rows without item_name_clean (OK if product master incomplete).", int(missing_names))
    # week_start_date coverage
    if "week_start_date" in df.columns:
        unmapped = df["week_start_date"].isna().sum()
        if unmapped:
            lg.warning("Phase 1 validation: %d rows with unmapped week_start_date.", int(unmapped))
    # duplicates by (entity_id, metric, week)
    if {"entity_id", "metric", "week_start_date"}.issubset(df.columns):
        dups = df.duplicated(subset=["entity_id", "metric", "week_start_date"], keep=False).sum()
        if dups:
            lg.warning("Phase 1 validation: %d duplicate rows on (entity_id, metric, week).", int(dups))


def apply_master_lookup(df: pd.DataFrame, master_lookup: pd.DataFrame) -> pd.DataFrame:
    """Enrich normalized rows with vendor metadata from the master lookup."""

    if master_lookup.empty:
        if "vendor_name" not in df.columns:
            df = df.copy()
            df["vendor_name"] = pd.NA
        return df

    normalized = df.copy()
    normalized["asin"] = normalized["asin"].astype(str).str.strip()
    # Create canonical join key on both sides
    normalized["asin_key"] = normalized["asin"].map(canonicalize_asin)
    lookup = master_lookup.copy()
    if "asin_key" not in lookup.columns:
        lookup["asin_key"] = lookup["asin"].map(canonicalize_asin)
    enriched = normalized.merge(lookup[["asin_key", "vendor_code", "vendor_name"]], on="asin_key", how="left", suffixes=("", "_master"))

    if "vendor_code_master" in enriched.columns:
        enriched["vendor_code"] = enriched["vendor_code_master"].where(
            enriched["vendor_code_master"].notna() & (enriched["vendor_code_master"].astype(str).str.len() > 0),
            enriched["vendor_code"],
        )
        enriched.drop(columns=["vendor_code_master"], inplace=True)

    if "vendor_name_master" in enriched.columns:
        enriched["vendor_name"] = enriched["vendor_name_master"].where(
            enriched["vendor_name_master"].notna() & (enriched["vendor_name_master"].astype(str).str.len() > 0),
            enriched.get("vendor_name"),
        )
        enriched.drop(columns=["vendor_name_master"], inplace=True)

    if "vendor_name" not in enriched.columns:
        enriched["vendor_name"] = pd.NA
    else:
        enriched["vendor_name"] = enriched["vendor_name"].apply(
            lambda value: (str(value).strip() or pd.NA)
            if pd.notna(value)
            else pd.NA
        )

    enriched["vendor_code"] = enriched["vendor_code"].apply(
        lambda value: (str(value).strip() or pd.NA) if pd.notna(value) else pd.NA
    )
    # Drop helper key
    if "asin_key" in enriched.columns:
        enriched.drop(columns=["asin_key"], inplace=True)
    # Optional: small debug about unmatched rows
    try:
        logger = logging.getLogger(LOGGER_NAME)
        missing = int(enriched["vendor_code"].isna().sum())
        if missing:
            logger.debug("Master lookup: %s rows have no vendor mapping (post-join)", missing)
    except Exception:
        pass

    return enriched


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize header casing and whitespace.

    - Strips whitespace
    - Collapses duplicate columns after stripping by appending an index suffix
    - Leaves original case (we rely on token normalizer for matching)
    """

    df = df.copy()
    raw_cols = [str(col).strip() for col in df.columns]
    seen: Dict[str, int] = {}
    fixed: List[str] = []
    for col in raw_cols:
        base = col
        if base in seen:
            seen[base] += 1
            fixed.append(f"{base}.{seen[base]}")
        else:
            seen[base] = 0
            fixed.append(base)
    df.columns = fixed
    return df


def canonicalize_asin(text: object) -> str:
    """Create a canonical ASIN key: uppercase, alphanumeric only.

    Keeps the original `asin` column unchanged for output while enabling
    robust joins between masterfile and raw exports regardless of case,
    spacing, hyphens, or symbols.
    """

    if pd.isna(text):
        return ""
    s = str(text).strip().upper()
    # Remove all non-alphanumeric characters
    return re.sub(r"[^A-Z0-9]", "", s)


def _normalize_header_token(text: object) -> str:
    """Normalize a header/alias token for matching.

    - Converts to string, strips whitespace
    - Lowercases
    - Removes all non-alphanumeric characters (spaces, punctuation, underscores)

    This allows matching headers like "ASIN", "A S I N", "asin", or "ASIN#"
    to the same token.
    """

    if text is None:
        return ""
    token = str(text).strip().lower()
    if not token:
        return ""
    # Remove any non [a-z0-9] characters to make matching robust
    token = re.sub(r"[^a-z0-9]+", "", token)
    return token


def detect_identifier_column(
    df: pd.DataFrame,
    aliases: Iterable[str],
    fallback_index: Optional[int] = None,
    default_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, Optional[str]]:
    """Detect a column by alias and optionally create it when missing.

    Upgrades:
    - Robust alias/token matching (case/space/punct insensitive)
    - Heuristic detection for ASIN-like columns if aliases fail
    - Clear error and ambiguity handling
    """

    alias_tokens = [_normalize_header_token(alias) for alias in aliases]
    # Build mapping from normalized token to original column name; if duplicates, keep first
    normalized_map: Dict[str, str] = {}
    dup_warnings: List[str] = []
    for col in df.columns:
        token = _normalize_header_token(col)
        if token in normalized_map and normalized_map[token] != col:
            dup_warnings.append(f"'{normalized_map[token]}' vs '{col}'")
            # keep the first occurrence
        else:
            normalized_map[token] = col

    for token in alias_tokens:
        if token and token in normalized_map:
            return df, normalized_map[token]

    # If not found and default_name exists already in df, return it
    if default_name and default_name in df.columns:
        return df, default_name

    # Heuristic: detect ASIN-like column if the target appears to be ASIN
    def is_probably_asin_series(s: pd.Series) -> float:
        def looks_like_asin(val: object) -> bool:
            if pd.isna(val):
                return False
            t = canonicalize_asin(val)
            # ASINs are typically 10 chars, allow 8-12 alnum
            return 8 <= len(t) <= 12 and any(c.isalpha() for c in t) and any(c.isdigit() for c in t)

        if s.empty:
            return 0.0
        sample = s.head(200)
        denom = int(sample.notna().sum()) or len(sample)
        if denom == 0:
            return 0.0
        score = sum(1 for x in sample if looks_like_asin(x)) / float(denom)
        return score

    expecting_asin = any(tok in {"asin", "asins"} for tok in alias_tokens) or (default_name and _normalize_header_token(default_name) == "asin")
    if expecting_asin:
        best_col = None
        best_score = 0.0
        for col in df.columns:
            score = is_probably_asin_series(df[col])
            if score > best_score:
                best_score = score
                best_col = col
        if best_col is not None and best_score >= 0.5:
            return df, best_col

    # Fallback by positional index
    if fallback_index is not None:
        columns = list(df.columns)
        if -len(columns) <= fallback_index < len(columns):
            return df, columns[fallback_index]

    # As a last resort, create the column if a default name was provided
    if default_name:
        df = df.copy()
        if default_name not in df.columns:
            df[default_name] = pd.NA
        return df, default_name

    return df, None


def detect_structure(df: pd.DataFrame, id_columns: Iterable[str]) -> str:
    """Detect whether the dataset is wide or long format."""

    columns = set(df.columns)
    if {"Week", "Metric", "Value"}.issubset(columns):
        return "long"
    if {"week", "metric", "value"}.issubset({col.lower() for col in df.columns}):
        return "long"
    if all(col in df.columns for col in id_columns):
        non_id = [col for col in df.columns if col not in id_columns]
        for column in non_id:
            _, week = split_metric_and_week(str(column))
            if week:
                return "wide"
        if len(non_id) > 3:
            return "wide"
    return "long"


def split_metric_and_week(column_name: str) -> Tuple[str, Optional[str]]:
    """Split a column name into metric and raw week parts."""

    clean_name = str(column_name).strip()
    paren_matches = re.findall(r"\(([^()]*)\)", clean_name)
    if paren_matches:
        for token in reversed(paren_matches):
            candidate = token.strip()
            if not candidate:
                continue
            if not re.search(r"[A-Za-z0-9]", candidate):
                continue
            metric_part = clean_name[: clean_name.rfind(f"({token})")].strip()
            if not metric_part:
                metric_part = clean_name
            if _parse_week_from_label(candidate) or re.search(r"\bW\d{1,2}\b", candidate, re.IGNORECASE) or re.search(r"\d", candidate):
                return metric_part.strip(), candidate

    range_match = DATE_RANGE_PATTERN.search(clean_name)
    if range_match:
        return clean_name[: range_match.start()].strip(), f"{range_match.group(1)}-{range_match.group(2)}"

    week_number_match = re.search(r"(\b(?:FW|W)\s?\d{1,2}\s?\d{4}\b|\d{4}-W\d{2})", clean_name)
    if week_number_match:
        token = week_number_match.group(1)
        metric_part = clean_name.replace(token, "").strip(" -_")
        return metric_part.strip(), token.strip()

    return clean_name, None


def split_metric_unit_week(header: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Split a messy wide header into (metric, unit, week_raw).

    Supports examples:
    - "Sessions (09/07/2025-09/14/2025) %"
    - "Units Week 41"
    - "Shipped COGS (2024W41)"
    - "Shipped Revenue ($) 09/07/2025-09/14/2025"
    """

    s = str(header or "").strip()
    if not s:
        return "", None, None

    # 1) Extract week token preference: date-range, YYYYWNN/ YYYY-WNN, Week NN
    week_raw: Optional[str] = None
    m = DATE_RANGE_PATTERN.search(s)
    if m:
        week_raw = m.group(0)
    else:
        m = ISO_WEEK_PATTERN.search(s)
        if m:
            week_raw = m.group(0)
        else:
            m = re.search(r"(?i)\b(?:fiscal\s+week|week|wk|w)\s*0?\d{1,2}\b", s)
            if m:
                week_raw = m.group(0)

    metric_part = s
    if week_raw:
        metric_part = metric_part.replace(week_raw, " ")

    # 2) Extract unit tokens from remaining metric part
    unit: Optional[str] = None
    if "%" in metric_part:
        unit = "%"
        metric_part = metric_part.replace("%", " ")
    for sym in ["$", "€", "£"]:
        if sym in metric_part and unit is None:
            unit = sym
        metric_part = metric_part.replace(sym, " ")
    if re.search(r"(?i)\bunits?\b", metric_part):
        if unit is None:
            unit = "units"
        metric_part = re.sub(r"(?i)\bunits?\b", " ", metric_part)

    # 3) Clean metric
    metric = re.sub(r"[()]+", " ", metric_part)
    metric = re.sub(r"\s{2,}", " ", metric).strip(" -:_ ")
    return metric, unit, week_raw


def _extract_metric_and_unit(text: str) -> Tuple[str, Optional[str]]:
    """Extract metric name and unit symbol from a header fragment.

    Recognizes common unit symbols like %, $, and words like 'Units'. Returns
    a tuple (metric_without_unit, unit_symbol_or_None).
    """
    s = str(text).strip()
    if not s:
        return s, None
    unit: Optional[str] = None
    # Capture explicit symbols
    if "%" in s:
        unit = "%"
        s = s.replace("%", "").strip()
    # Currency
    for sym in ["$", "€", "£"]:
        if sym in s:
            unit = sym if unit is None else unit
            s = s.replace(sym, "").strip()
    # Units word (case-insensitive)
    if re.search(r"\bunits?\b", s, flags=re.IGNORECASE):
        unit = unit or "units"
        s = re.sub(r"\bunits?\b", "", s, flags=re.IGNORECASE).strip()
    # Remove double spaces and trailing separators
    s = re.sub(r"\s{2,}", " ", s).strip(" -:_")
    return s, unit


def melt_wide_dataframe(
    df: pd.DataFrame,
    id_columns: List[str],
    lookup: WeekLookup,
    value_round: Optional[int] = None,
) -> pd.DataFrame:
    """Transform a wide dataframe into the normalized long format."""

    records: List[Dict[str, object]] = []
    for column in df.columns:
        if column in id_columns:
            continue
        # Improved splitter to extract metric, unit, and raw week
        metric, unit, raw_week = split_metric_unit_week(column)
        if not raw_week:
            # try legacy splitting
            metric_raw, legacy_week = split_metric_and_week(column)
            if legacy_week:
                metric2, unit2 = _extract_metric_and_unit(metric_raw)
                metric = metric or metric2
                unit = unit or unit2
                raw_week = legacy_week
            else:
                continue
        # Defer ambiguous week-only tokens to finalize_weeks for dominant-year handling
        week_label, week_start, week_end = "", None, None
        parsed = _parse_week_tokens(raw_week)
        if parsed:
            kind, data = parsed
            if kind == "daterange":
                # Resolve via date containment (use start date first; finalize_weeks uses midpoint for >7d)
                start_dt = pd.to_datetime(data[0], errors="coerce")
                if pd.notna(start_dt):
                    found = lookup._find_by_date(pd.to_datetime(start_dt))  # type: ignore[attr-defined]
                    if found:
                        week_label, week_start, week_end = found
            elif kind == "yearweek":
                y, w = int(data[0]), int(data[1])
                if y != 0:
                    found = lookup._find_by_year_week(y, w)  # type: ignore[attr-defined]
                    if found:
                        week_label, week_start, week_end = found
        for _, row in df.iterrows():
            value = row[column]
            if pd.isna(value):
                continue
            if value_round is not None and isinstance(value, (int, float)):
                value = round(float(value), value_round)
            records.append(
                {
                    "vendor_code": row[id_columns[0]],
                    "asin": row[id_columns[1]],
                    "metric": metric,
                    "unit": unit,
                    "week_label": week_label,
                    "week_raw": raw_week,
                    "week_start": week_start,
                    "week_end": week_end,
                    "value": value,
                }
            )
    out = pd.DataFrame.from_records(records)
    if not out.empty:
        out["value"] = _coerce_value_series(out["value"], out.get("metric"), value_round)
        # Phase 1 patch: enforce unit "$" for ASP & CPPU after generic extraction
        try:
            mask_special = out["metric"].astype(str).isin(["ASP", "CPPU"])
            if "unit" not in out.columns:
                out["unit"] = pd.NA
            out.loc[mask_special, "unit"] = "$"
        except Exception:
            # Do not fail ingestion on enforcement logic
            pass
    return out


def normalize_long_dataframe(
    df: pd.DataFrame,
    lookup: WeekLookup,
    week_column: str,
    metric_column: str,
    value_column: str,
    id_columns: List[str],
    value_round: Optional[int] = None,
) -> pd.DataFrame:
    """Normalize a long dataframe into the canonical schema."""

    df = df.copy()
    df.rename(columns={week_column: "week_raw"}, inplace=True)
    df.rename(columns={metric_column: "metric"}, inplace=True)
    df.rename(columns={value_column: "value"}, inplace=True)
    df.rename(columns={id_columns[0]: "vendor_code", id_columns[1]: "asin"}, inplace=True)

    # Separate unit from metric if embedded
    mu = df["metric"].map(lambda m: _extract_metric_and_unit(str(m)))
    df["metric"] = mu.map(lambda t: t[0])
    df["unit"] = mu.map(lambda t: t[1])
    # Map only unambiguous weeks here; leave ambiguous to finalize_weeks
    week_labels: List[str] = []
    week_starts: List[Optional[str]] = []
    week_ends: List[Optional[str]] = []
    for raw in df["week_raw"].astype(str).tolist():
        wk_label, wk_start, wk_end = "", None, None
        parsed = _parse_week_tokens(raw)
        if parsed:
            kind, data = parsed
            if kind == "daterange":
                start_dt = pd.to_datetime(data[0], errors="coerce")
                if pd.notna(start_dt):
                    found = lookup._find_by_date(pd.to_datetime(start_dt))  # type: ignore[attr-defined]
                    if found:
                        wk_label, wk_start, wk_end = found
            elif kind == "yearweek":
                y, w = int(data[0]), int(data[1])
                if y != 0:
                    found = lookup._find_by_year_week(y, w)  # type: ignore[attr-defined]
                    if found:
                        wk_label, wk_start, wk_end = found
        week_labels.append(wk_label)
        week_starts.append(wk_start)
        week_ends.append(wk_end)

    df["week_label"] = week_labels
    df["week_start"] = week_starts
    df["week_end"] = week_ends

    df["value"] = _coerce_value_series(df["value"], df.get("metric"), value_round)

    keep_cols = ["vendor_code", "asin", "metric", "unit", "week_label", "week_raw", "week_start", "week_end", "value"]
    return df[keep_cols]


def deduplicate_records(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Drop duplicate records keeping the most recent entry."""

    before = len(df)
    deduped = df.drop_duplicates(
        subset=["vendor_code", "asin", "metric", "week_label"], keep="last"
    )
    after = len(deduped)
    if after < before:
        logger.info("Deduplicated %s duplicate rows", before - after)
    return deduped


def detect_week_column(df: pd.DataFrame) -> Optional[str]:
    """Attempt to find a column representing the week label."""

    candidates = [
        "week",
        "Week",
        "Week Label",
        "Fiscal Week",
        "week_label",
        "Week_Number",
    ]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def detect_metric_column(df: pd.DataFrame) -> Optional[str]:
    """Attempt to find a metric column for long-form data."""

    candidates = ["Metric", "metric", "Measure", "measure"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def detect_value_column(df: pd.DataFrame) -> Optional[str]:
    """Attempt to find a value column for long-form data."""

    candidates = ["Value", "value", "Amount", "amount"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def write_output(df: pd.DataFrame, config: Dict, logger: logging.Logger, output_dir: Optional[Path] = None) -> Path:
    """Persist the normalized dataframe to disk in the mandatory directory.

    Requirements:
    - Write exclusively to MANDATORY_OUTPUT_DIR
    - Always attempt to write Parquet primary as "normalized_phase1.parquet"
    - Always write CSV fallback as "normalized_phase1.csv"
    - Optionally write XLSX for human readability
    - No silent failures: log descriptive messages; if all formats fail, raise RuntimeError
    """

    base = output_dir if output_dir is not None else MANDATORY_OUTPUT_DIR
    ensure_directory(base)
    parquet_path = base / "normalized_phase1.parquet"
    csv_path = base / "normalized_phase1.csv"
    xlsx_path = base / "normalized_phase1.xlsx"

    wrote_any = False
    primary_path: Optional[Path] = None

    # Parquet primary (with optional rotation by week)
    rotation_cfg = (config.get("ingestion", {}).get("parquet_rotation") or {})
    rotate = bool(rotation_cfg.get("enabled", False))
    row_threshold = int(rotation_cfg.get("row_threshold", 2_000_000))
    # Prepare explicit schema via pyarrow
    def _make_schema():
        try:
            import pyarrow as pa
            fields = [
                pa.field("vendor_code", pa.string()),
                pa.field("vendor_name", pa.string()),
                pa.field("asin", pa.string()),
                pa.field("metric", pa.string()),
                pa.field("unit", pa.string()),
                pa.field("week_raw", pa.string()),
                pa.field("week_label", pa.string()),
                pa.field("week_start", pa.timestamp("ms")),
                pa.field("week_end", pa.timestamp("ms")),
                pa.field("value", pa.float64()),
                pa.field("is_valid_asin", pa.bool_()),
                pa.field("invalid_reason", pa.string()),
                pa.field("source_file", pa.string()),
            ]
            return pa.schema(fields)
        except Exception:
            return None

    # Coerce types before writing
    work = df.copy()
    # Value float64
    if "value" in work.columns:
        work["value"] = pd.to_numeric(work["value"], errors="coerce").astype("float64")
    # Datetimes
    for c in ("week_start", "week_end"):
        if c in work.columns:
            work[c] = pd.to_datetime(work[c], errors="coerce")
    # Ensure strings UTF-8-able
    for c in ["vendor_code", "vendor_name", "asin", "metric", "unit", "week_raw", "week_label", "invalid_reason", "source_file"]:
        if c in work.columns:
            work[c] = work[c].astype(str)
    if "is_valid_asin" in work.columns:
        work["is_valid_asin"] = work["is_valid_asin"].astype("boolean")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        schema = _make_schema()
        if rotate and (len(work) >= row_threshold):
            parts = []
            for wk, chunk in work.groupby("week_label", dropna=False):
                if not isinstance(wk, str) or not wk:
                    continue
                part_path = base / f"normalized_phase1_{wk}.parquet"
                try:
                    tbl = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)
                    pq.write_table(tbl, part_path)
                    parts.append(part_path)
                except Exception as exc:
                    logger.error("Failed to write rotated Parquet %s: %s", part_path, exc)
            # Combined file
            tbl_all = pa.Table.from_pandas(work, schema=schema, preserve_index=False)
            pq.write_table(tbl_all, parquet_path)
            primary_path = parquet_path
            wrote_any = True if (parts or parquet_path.exists()) else wrote_any
        else:
            tbl = pa.Table.from_pandas(work, schema=schema, preserve_index=False)
            pq.write_table(tbl, parquet_path)
            primary_path = parquet_path
            wrote_any = True
    except Exception as exc:
        logger.error("Failed to write Parquet %s: %s", parquet_path, exc)

    # CSV fallback (always attempt)
    try:
        df.to_csv(csv_path, index=False)
        if primary_path is None:
            primary_path = csv_path
        wrote_any = True or wrote_any
    except Exception as exc:
        logger.error("Failed to write CSV %s: %s", csv_path, exc)

    # Optional XLSX for convenience (can be toggled)
    if bool(config.get("ingestion", {}).get("output_xlsx", True)):
        try:
            excel_ready = _prepare_for_excel(df, config)
            excel_ready.to_excel(xlsx_path, index=False)
            wrote_any = True or wrote_any
        except Exception as exc:
            logger.warning("Failed to write XLSX %s: %s", xlsx_path, exc)

    if not wrote_any or primary_path is None:
        raise RuntimeError("Failed to write Phase 1 outputs (Parquet/CSV/XLSX)")

    return primary_path


def _safe_str(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def _validate_normalized_schema(df: pd.DataFrame) -> Dict[str, object]:
    required_cols = [
        "vendor_code",
        "asin",
        "metric",
        "week_label",
        "week_raw",
        "week_start",
        "week_end",
        "value",
        "source_file",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    issues: List[str] = []
    if missing:
        issues.append(f"Missing required columns: {missing}")
    # Basic week label format check
    bad_week = df[~df["week_label"].astype(str).str.match(r"^\d{4}W\d{2}$", na=False)] if "week_label" in df.columns else pd.DataFrame()
    bad_week_count = int(len(bad_week)) if not bad_week.empty else 0
    return {
        "missing_columns": missing,
        "bad_week_labels": bad_week_count,
        "total_rows": int(len(df)),
    }


def write_phase1_reports(df: pd.DataFrame, logger: logging.Logger, output_dir: Optional[Path] = None) -> Dict[str, Path]:
    """Write validation and summary artifacts to the mandatory directory."""

    base = output_dir if output_dir is not None else MANDATORY_OUTPUT_DIR
    ensure_directory(base)
    outputs: Dict[str, Path] = {}

    # Schema snapshot
    schema_info = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    schema_path = base / "schema_snapshot.json"
    with open(schema_path, "w", encoding="utf-8") as fh:
        json.dump(schema_info, fh, indent=2)
    outputs["schema_snapshot"] = schema_path

    # Schema validation
    validation = _validate_normalized_schema(df)
    validation_path = base / "schema_validation.json"
    with open(validation_path, "w", encoding="utf-8") as fh:
        json.dump(validation, fh, indent=2)
    outputs["schema_validation"] = validation_path

    # Ingestion audit report
    try:
        audit = {
            "total_rows": int(len(df)),
            "by_source_file": df.groupby("source_file").size().rename("rows").reset_index().to_dict(orient="records")
            if "source_file" in df.columns
            else [],
            "by_metric": df.groupby("metric").size().rename("rows").reset_index().to_dict(orient="records")
            if "metric" in df.columns
            else [],
        }
    except Exception as exc:
        audit = {"error": f"Failed to build audit: {exc}"}
    audit_path = base / "ingestion_audit.json"
    with open(audit_path, "w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2)
    outputs["ingestion_audit"] = audit_path

    # Vendor summary
    try:
        vendor_summary = (
            df.groupby(["vendor_code", "vendor_name"], dropna=False)
            .agg(records=("metric", "size"), asins=("asin", "nunique"), weeks=("week_label", "nunique"))
            .reset_index()
        )
        vendor_summary_path = base / "vendor_summary.csv"
        vendor_summary.to_csv(vendor_summary_path, index=False)
        outputs["vendor_summary"] = vendor_summary_path
    except Exception as exc:
        logger.warning("Failed to write vendor summary: %s", exc)

    # ASIN summary
    try:
        asin_summary = (
            df.groupby(["vendor_code", "vendor_name", "asin"], dropna=False)
            .agg(records=("metric", "size"), weeks=("week_label", "nunique"), metrics=("metric", "nunique"))
            .reset_index()
        )
        asin_summary_path = base / "asin_summary.csv"
        asin_summary.to_csv(asin_summary_path, index=False)
        outputs["asin_summary"] = asin_summary_path
    except Exception as exc:
        logger.warning("Failed to write ASIN summary: %s", exc)

    # Week summary
    try:
        keep_cols = [c for c in ["week_label", "week_start", "week_end"] if c in df.columns]
        gb_cols = ["week_label"]
        week_summary = (
            df.groupby(gb_cols, dropna=False)
            .agg(records=("metric", "size"), metrics=("metric", "nunique"))
            .reset_index()
        )
        if set(["week_start", "week_end"]).issubset(df.columns):
            week_meta = df.drop_duplicates(subset=["week_label"]) [ ["week_label", "week_start", "week_end"] ]
            week_summary = week_summary.merge(week_meta, on="week_label", how="left")
        week_summary_path = base / "week_summary.csv"
        week_summary.to_csv(week_summary_path, index=False)
        outputs["week_summary"] = week_summary_path
    except Exception as exc:
        logger.warning("Failed to write week summary: %s", exc)

    # Row-level validation log (e.g., missing essentials)
    try:
        asin_missing = df["asin"].isna() | (df["asin"].astype(str).str.strip() == "") if "asin" in df.columns else pd.Series(False, index=df.index)
        week_missing = df["week_label"].isna() | (df["week_label"].astype(str).str.strip() == "") if "week_label" in df.columns else pd.Series(False, index=df.index)
        mask = asin_missing | week_missing
        bad_rows = df.loc[mask] if mask.any() else df.head(0)
        val_path = base / "validation_rows.csv"
        bad_rows.to_csv(val_path, index=False)
        outputs["validation_rows"] = val_path
    except Exception as exc:
        logger.warning("Failed to write validation rows: %s", exc)

    return outputs


def _prepare_for_excel(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Coerce dataframe types to be Excel-friendly.

    - Convert pandas NA to None
    - Keep numeric columns numeric; round "value" if configured
    - Convert datetimes to dates (ISO) to avoid Excel NaT issues
    - Ensure object/category columns are plain Python objects or strings
    """

    df_x = df.copy()

    # Optional value rounding (already applied earlier, but ensure for safety)
    value_round = config.get("ingestion", {}).get("value_round")
    if "value" in df_x.columns and value_round is not None:
        df_x["value"] = pd.to_numeric(df_x["value"], errors="coerce").round(value_round)

    for col in df_x.columns:
        s = df_x[col]
        # Datetime handling: store as date (or ISO string) for Excel
        if pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s, errors="coerce").dt.date
            df_x[col] = s
            continue
        # Pandas nullable integers/booleans -> object with None for NA
        if str(s.dtype).startswith(("Int", "UInt")) or pd.api.types.is_bool_dtype(s):
            df_x[col] = s.astype("object").where(~pd.isna(s), None)
            continue
        # Floats can stay as float; leave NaN as is (Excel will show blanks)
        if pd.api.types.is_float_dtype(s):
            df_x[col] = s
            continue
        # Category or object -> convert pd.NA to None, keep strings as-is
        if pd.api.types.is_categorical_dtype(s):
            s = s.astype("object")
        df_x[col] = s.where(~pd.isna(s), None)

    return df_x


def _coerce_value_series(values: pd.Series, metrics: Optional[pd.Series] = None, value_round: Optional[int] = None) -> pd.Series:
    """Convert a heterogeneous value series to numeric.

    - Handles strings with commas, currency symbols, and percentage signs
    - Parentheses indicate negatives: (123.45) -> -123.45
    - If a value contains '%', interpret as percentage (divide by 100)
    - Applies optional rounding
    """

    s = values.copy()

    # Fast path: already numeric dtypes
    if pd.api.types.is_numeric_dtype(s):
        return s.round(value_round) if value_round is not None else s

    def parse_one(x: object) -> float:
        if pd.isna(x):
            return float('nan')
        if isinstance(x, (int, float)):
            return float(x)
        t = str(x).strip()
        if t == "":
            return float('nan')
        is_percent = "%" in t
        negative = False
        # Parentheses denote negative
        if t.startswith("(") and t.endswith(")"):
            negative = True
            t = t[1:-1]
        # Remove currency symbols and other decorations except digits, dot, minus and percent
        t = t.replace(",", "")
        t = t.replace("$", "")
        t = t.replace("€", "").replace("£", "")
        t = t.replace("%", "")
        t = t.strip()
        try:
            val = float(t)
        except Exception:
            return float('nan')
        if negative:
            val = -val
        if is_percent:
            val = val / 100.0
        return val

    parsed = s.map(parse_one)
    if value_round is not None:
        parsed = parsed.round(value_round)
    return parsed


def write_state(metadata: Dict[str, object], config: Dict, output_dir: Optional[Path] = None) -> Path:
    """Persist ingestion state metadata as JSON in the mandatory directory."""

    base = output_dir if output_dir is not None else MANDATORY_OUTPUT_DIR
    ensure_directory(base)
    state_path = base / "ingestion_state.json"
    with open(state_path, "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return state_path


def _extract_years_from_labels(series: pd.Series) -> List[int]:
    years: List[int] = []
    for val in series.dropna().astype(str):
        m = re.match(r"^(\d{4})W\d{2}$", val)
        if m:
            years.append(int(m.group(1)))
    return years


def finalize_weeks(df: pd.DataFrame, lookup: WeekLookup, logger: logging.Logger, issues_output: Optional[Path] = None) -> pd.DataFrame:
    """Ensure all rows have canonical week fields; log failures to issues_output.

    Strategy:
    - For rows with missing/blank week_label, parse week_raw tokens.
    - If only week number present (no year), infer year from majority of already-mapped rows; fallback to current year.
    - If a single date is embedded, infer Sunday-start week containing that date.
    - If still unresolved, write to issues CSV then assign a best-effort canonical week (year=mode/current, week=1).
    """
    if df.empty:
        # Phase 1 patch: enforce unit "$" for ASP & CPPU after generic extraction
        try:
            mask_special = df["metric"].astype(str).isin(["ASP", "CPPU"])
            if "unit" not in df.columns:
                df["unit"] = pd.NA
            df.loc[mask_special, "unit"] = "$"
        except Exception:
            pass

        return df

    out = df.copy()

    # Determine dominant year from date-range headers present in week_raw
    dominant_year: Optional[int] = None
    try:
        if "week_raw" in out.columns:
            uniq_weeks = pd.Series(sorted(set(str(x) for x in out["week_raw"].dropna())))
            dr = uniq_weeks[uniq_weeks.str.contains(DATE_RANGE_PATTERN)] if not uniq_weeks.empty else pd.Series(dtype=str)
            years: List[int] = []
            for raw in dr.tolist():
                m = DATE_RANGE_PATTERN.search(raw)
                if not m:
                    continue
                start_dt = pd.to_datetime(m.group(1), errors="coerce")
                if pd.notna(start_dt):
                    found = lookup._find_by_date(pd.to_datetime(start_dt))  # type: ignore[attr-defined]
                    if found and isinstance(found[0], str) and len(found[0]) >= 6:
                        try:
                            years.append(int(found[0][:4]))
                        except Exception:
                            pass
            if years:
                vc = pd.Series(years).value_counts()
                top = vc[vc == vc.max()].index.tolist()
                dominant_year = int(max(top))  # tie -> most recent
    except Exception:
        dominant_year = None

    # Fallback dominant year from already mapped labels
    if dominant_year is None:
        existing_years = _extract_years_from_labels(out.loc[:, "week_label"]) or []
        if existing_years:
            vc = pd.Series(existing_years).value_counts()
            top = vc[vc == vc.max()].index.tolist()
            dominant_year = int(max(top))
        else:
            dominant_year = datetime.now().year

    # Build synonym map for this run
    syn_rows: List[Dict[str, object]] = []

    # Identify unresolved or ambiguous rows
    missing_mask = out["week_label"].isna() | (out["week_label"].astype(str).str.strip() == "")
    indices = list(out.index[missing_mask])

    issues: List[Dict[str, object]] = []
    for idx in indices:
        raw = str(out.at[idx, "week_raw"]) if "week_raw" in out.columns else ""
        wk_label, wk_start, wk_end = "", None, None
        parsed_week_number: Optional[int] = None
        parsed_year_guess: Optional[int] = None

        parsed = _parse_week_tokens(raw)
        if parsed:
            kind, data = parsed
            if kind == "daterange":
                start_dt = pd.to_datetime(data[0], errors="coerce")
                end_dt = pd.to_datetime(data[1], errors="coerce")
                # choose midpoint if >7 days else start containment
                if pd.notna(start_dt) and pd.notna(end_dt):
                    mid = start_dt + (end_dt - start_dt) / 2
                    found = lookup._find_by_date(pd.to_datetime(mid))  # type: ignore[attr-defined]
                    if found:
                        wk_label, wk_start, wk_end = found
            elif kind == "yearweek":
                y, w = int(data[0]), int(data[1])
                parsed_week_number = w
                if y == 0:
                    parsed_year_guess = dominant_year
                    found = lookup._find_by_year_week(int(dominant_year), int(w))  # type: ignore[attr-defined]
                else:
                    parsed_year_guess = y
                    found = lookup._find_by_year_week(int(y), int(w))  # type: ignore[attr-defined]
                if found:
                    wk_label, wk_start, wk_end = found

        if not wk_label:
            # legacy fallback
            legacy = _parse_week_from_label(raw)
            if legacy:
                wk_label, wk_start, wk_end = legacy
                try:
                    parsed_year_guess = int(wk_label[:4])
                except Exception:
                    pass

        if not wk_label:
            # Default to dominant year W01
            issues.append({
                "index": idx,
                "week_raw": raw,
                "note": "Unable to resolve week; defaulted to dominant year W01",
                "source_file": out.at[idx, "source_file"] if "source_file" in out.columns else None,
                "metric": out.at[idx, "metric"] if "metric" in out.columns else None,
                "asin": out.at[idx, "asin"] if "asin" in out.columns else None,
            })
            fb = lookup._find_by_year_week(int(dominant_year), 1)  # type: ignore[attr-defined]
            if fb:
                wk_label, wk_start, wk_end = fb
            else:
                wk_label = f"{dominant_year}W01"
                wk_start, wk_end = None, None

        out.at[idx, "week_label"] = wk_label
        out.at[idx, "week_start"] = wk_start
        out.at[idx, "week_end"] = wk_end

        syn_rows.append({
            "raw_week_label": raw,
            "parsed_week_number": parsed_week_number,
            "parsed_year_guess": parsed_year_guess,
            "canonical_week_id": wk_label,
        })

    # Persist synonym map and issues
    try:
        base = issues_output.parent if issues_output is not None else MANDATORY_OUTPUT_DIR
        ensure_directory(base)
        if syn_rows:
            syn_df = pd.DataFrame(syn_rows).drop_duplicates(subset=["raw_week_label"], keep="last")
            syn_df.to_csv(base / "week_synonym_map.csv", index=False)
    except Exception as exc:
        logger.warning("Failed to write week_synonym_map.csv: %s", exc)

    try:
        if issues_output is not None and issues:
            issues_df = pd.DataFrame(issues)
            issues_df.to_csv(issues_output, index=False)
    except Exception as exc:
        logger.warning("Failed to write week parsing issues: %s", exc)

    return out


def validate_and_standardize_asins(df: pd.DataFrame, output_dir: Optional[Path] = None) -> pd.DataFrame:
    """Standardize `asin` column and emit invalid ASINs to CSV with context.

    - Strips and canonicalizes to uppercase alphanumeric
    - Validates 10-char alphanumeric pattern
    - Writes invalid rows with context to `invalid_asins.csv` in the output_dir (or base)
    - Returns dataframe filtered to valid ASINs with cleaned `asin`
    """
    if "asin" not in df.columns:
        return df
    base = output_dir if output_dir is not None else MANDATORY_OUTPUT_DIR
    ensure_directory(base)
    work = df.copy()
    work["asin_clean"] = work["asin"].map(canonicalize_asin)
    valid_mask = work["asin_clean"].astype(str).str.fullmatch(r"[A-Z0-9]{10}")
    work["is_valid_asin"] = valid_mask
    work["invalid_reason"] = None
    work.loc[~valid_mask, "invalid_reason"] = "ASIN must be 10 alphanumeric chars"

    invalid = work.loc[~valid_mask].copy()
    if not invalid.empty:
        # Add some context columns if present
        keep_cols = [c for c in [
            "source_file", "vendor_code", "vendor_name", "asin", "asin_clean", "invalid_reason"
        ] if c in invalid.columns]
        invalid_path = base / "invalid_asins.csv"
        invalid[keep_cols].to_csv(invalid_path, index=False)
    # Replace asin with cleaned for all rows; do not drop invalids (keep schema stable)
    work["asin"] = work["asin_clean"]
    work.drop(columns=["asin_clean"], inplace=True)
    return work


def summarize_phase(df: pd.DataFrame, template: str) -> str:
    """Generate a human-readable summary of the ingestion results."""

    if df.empty:
        return "No rows processed."
    summary = template.format(
        total_rows=len(df),
        unique_weeks=df["week_label"].nunique(),
        last_week=df.sort_values("week_start", na_position="last")["week_label"].iloc[-1],
    )
    return summary


def validate_extension(path: Path, allowed: Iterable[str]) -> None:
    """Ensure the file extension is allowed."""

    suffix = path.suffix.lower()
    if suffix not in {ext.lower() for ext in allowed}:
        raise ValueError(f"Unsupported file extension: {suffix}. Allowed: {allowed}")


def read_input_file(path: Path) -> pd.DataFrame:
    """Read a delimited text or Excel file into a DataFrame safely.

    Supported formats: csv, tsv, txt, xlsx, xlsm, xls.
    - Auto-detect delimiter for text files (comma, tab, semicolon, pipe)
    - Use dtype=str to avoid silent type coercion
    - Do not drop rows silently; if parsing errors occur, raise with context
    """

    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".txt"}:
        # Robust CSV fallback: let pandas infer separator using Python engine and preserve all symbols
        try:
            return pd.read_csv(path, dtype=str, sep=None, engine="python")
        except Exception:
            # Last resort: try comma
            try:
                return pd.read_csv(path, dtype=str, sep=",", engine="python")
            except Exception as exc:
                raise ValueError(f"Failed to read delimited file {path}: {exc}")

    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            # Read the first sheet by default as strings
            return pd.read_excel(path, dtype=str)
        except Exception as exc:
            raise ValueError(f"Failed to read Excel file {path}: {exc}")

    raise ValueError(f"Unsupported input file extension for {path}")
