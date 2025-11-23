"""Canonical Sunday-start calendar (dim_week) builder for Phase 1 ingestion.

This module produces a single, clean weekly calendar with exactly one row per
week using Sunday as the first day of the week and Saturday as the last day.

Columns:
- canonical_year (int)
- canonical_week_number (int)
- canonical_week_id (str, format YYYYWNN)
- week_start (datetime64[ns], normalized to 00:00:00)
- week_end (datetime64[ns], normalized to 00:00:00)
- month_number (int) — from week_start
- month_label (str) — e.g., "Sep 2025"
- quarter_label (str) — e.g., "Q3"
- display_label (str) — e.g., "2025 Week 41"

The calendar is deterministic and suitable as the single source of truth for
week mapping across ingestion.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def _first_sunday_on_or_before(day: pd.Timestamp) -> pd.Timestamp:
    # pandas weekday: Monday=0 ... Sunday=6
    # We want the previous (or same) Sunday => days_back = (weekday + 1) % 7
    weekday = int(day.weekday())
    days_back = (weekday + 1) % 7
    return (day - pd.Timedelta(days=days_back)).normalize()


def _first_sunday_of_year(year: int) -> pd.Timestamp:
    return _first_sunday_on_or_before(pd.Timestamp(year=year, month=1, day=1))


def build_canonical_dim_week(start_year: int = 2018, end_year: int = 2030) -> pd.DataFrame:
    """Build a Sunday-start canonical weekly calendar.

    The week numbering restarts at 1 on the first Sunday of each canonical_year.
    """

    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    rows: list[dict] = []
    cur = _first_sunday_of_year(start_year)
    end_dt = pd.Timestamp(year=end_year, month=12, day=31).normalize()

    # Include boundary years to handle the case when the first Sunday for start_year
    # falls in the previous calendar year (e.g., 2017-12-31 for 2018).
    first_sunday_by_year = {y: _first_sunday_of_year(y) for y in range(start_year - 1, end_year + 2)}

    while cur <= end_dt:
        week_start = cur
        week_end = cur + pd.Timedelta(days=6)
        canonical_year = int(week_start.year)
        week1 = first_sunday_by_year.get(canonical_year) or _first_sunday_of_year(canonical_year)
        canonical_week_number = int(((week_start - week1).days // 7) + 1)
        canonical_week_id = f"{canonical_year}W{canonical_week_number:02d}"

        rows.append(
            {
                "canonical_year": canonical_year,
                "canonical_week_number": canonical_week_number,
                "canonical_week_id": canonical_week_id,
                "week_start": week_start,
                "week_end": week_end,
                "month_number": int(week_start.month),
                "month_label": week_start.strftime("%b %Y"),
                "quarter_label": f"Q{((week_start.month - 1)//3) + 1}",
                "display_label": f"{canonical_year} Week {canonical_week_number:02d}",
            }
        )

        cur = cur + pd.Timedelta(days=7)

    df = pd.DataFrame(rows)

    # Validations
    if not df["canonical_week_id"].is_unique:
        raise AssertionError("Duplicate canonical_week_id in generated calendar")
    if not (df["week_end"] - df["week_start"]).eq(pd.Timedelta(days=6)).all():
        raise AssertionError("Non 7-day weeks detected in generated calendar")
    if df["week_start"].isna().any() or df["week_end"].isna().any():
        raise AssertionError("NaT week_start/week_end detected in generated calendar")

    return df


def write_unified_calendar_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["week_start"] = out["week_start"].dt.date.astype(str)
    out["week_end"] = out["week_end"].dt.date.astype(str)
    out.to_csv(path, index=False)
