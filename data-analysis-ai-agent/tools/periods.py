"""
Time periods: turning text like "2026-Q1" into exact date ranges.

Used by:
- change_analysis.py: to split the data into "Period A" and "Period B"
- charts.py: to group line charts per day/week/month/quarter/year

THE KEY IDEA: a period is a HALF-OPEN date range [start, end)
    start  = the first moment INSIDE the period   (2026-01-01 00:00)
    end    = the first moment AFTER the period    (2026-04-01 00:00)
A row belongs to the period when  start <= date < end.

Why not "end = 2026-03-31"? Because a sale at 2026-03-31 18:30 is later
than 2026-03-31 00:00 and would be wrongly excluded. Half-open ranges have
no gaps and no overlaps: Q1 ends exactly where Q2 begins.
"""

import re
from dataclasses import dataclass

import pandas as pd

from tools.errors import ToolError
from tools.inspection import column_kind
from tools.utils import check_columns_exist

# Readable period names -> pandas "period" codes (used by line charts).
PERIODS = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}

# A period counts as "complete" only if the dataset covers at least this
# share of its days. Example: data starting on 5 January covers 27 of
# January's 31 days = 87%, which is above 80%, so January counts as complete.
# (This is a documented rule of thumb, not a law of nature.)
MIN_COVERAGE = 0.8

# Patterns for the accepted text formats. "(\d{4})" captures 4 digits (a year).
YEAR_PATTERN = re.compile(r"^(\d{4})$")                 # "2026"
QUARTER_PATTERN = re.compile(r"^(\d{4})-?Q([1-4])$")    # "2026-Q1" or "2026Q1"
MONTH_PATTERN = re.compile(r"^(\d{4})-(\d{1,2})$")      # "2026-03"
DAY_PATTERN = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2})$")  # "2026-03-15"


# @dataclass writes the boring parts of a class for us (__init__, printing,
# comparing). frozen=True makes it read-only: a period can't be changed by accident.
@dataclass(frozen=True)
class Period:
    label: str            # how we show it to people, e.g. "2026-Q1"
    kind: str             # "year", "quarter", "month", "day" or "custom"
    start: pd.Timestamp   # first moment inside the period
    end: pd.Timestamp     # first moment AFTER the period (not included)

    @property
    def last_day(self) -> pd.Timestamp:
        """The last calendar day inside the period (for display)."""
        return self.end - pd.Timedelta(days=1)

    def to_dict(self) -> dict:
        """JSON-safe description, e.g. {"label": "2026-Q1", "start": "2026-01-01", "end": "2026-03-31"}."""
        return {"label": self.label, "kind": self.kind,
                "start": self.start.date().isoformat(), "end": self.last_day.date().isoformat()}


def parse_period(spec) -> Period:
    """
    Turn a period description into a Period.

    Accepted forms:
        "2026"        a whole year
        "2026-Q1"     a quarter (Q1 = Jan-Mar, Q2 = Apr-Jun, ...)
        "2026-03"     a month
        "2026-03-15"  a single day
        {"start": "2026-01-10", "end": "2026-02-20"}   custom range, BOTH days included
    """
    if isinstance(spec, Period):
        return spec
    if isinstance(spec, dict):
        return _parse_custom(spec)
    if not isinstance(spec, str):
        raise ToolError(f"Period must be text like '2026-Q1' or a dict with start/end, got {spec!r}")

    text = spec.strip().upper()
    try:
        if match := YEAR_PATTERN.match(text):
            # ":=" (the "walrus operator") assigns AND tests in one step.
            year = int(match.group(1))
            start = pd.Timestamp(year=year, month=1, day=1)
            return Period(text, "year", start, start + pd.DateOffset(years=1))
        if match := QUARTER_PATTERN.match(text):
            year, quarter = int(match.group(1)), int(match.group(2))
            start = pd.Timestamp(year=year, month=3 * (quarter - 1) + 1, day=1)  # Q2 -> month 4
            return Period(f"{year}-Q{quarter}", "quarter", start, start + pd.DateOffset(months=3))
        if match := MONTH_PATTERN.match(text):
            year, month = int(match.group(1)), int(match.group(2))
            start = pd.Timestamp(year=year, month=month, day=1)
            return Period(f"{year}-{month:02d}", "month", start, start + pd.DateOffset(months=1))
        if match := DAY_PATTERN.match(text):
            start = pd.Timestamp(year=int(match.group(1)), month=int(match.group(2)),
                                 day=int(match.group(3)))
            return Period(start.date().isoformat(), "day", start, start + pd.Timedelta(days=1))
    except ValueError as error:  # e.g. month 13 or 30 February
        raise ToolError(f"Invalid period '{spec}': {error}")
    raise ToolError(
        f"Unrecognised period '{spec}'. Use '2026', '2026-Q1', '2026-03', '2026-03-15' "
        "or {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}."
    )


def _parse_custom(spec: dict) -> Period:
    """A custom range: the user gives the first and the LAST day (both included)."""
    if "start" not in spec or "end" not in spec:
        raise ToolError("A custom period needs both 'start' and 'end', e.g. {'start': '2026-01-01', 'end': '2026-01-31'}.")
    try:
        start = pd.Timestamp(spec["start"]).normalize()  # normalize() = set the time to 00:00
        last_day = pd.Timestamp(spec["end"]).normalize()
    except (ValueError, TypeError) as error:
        raise ToolError(f"Invalid custom period {spec}: {error}")
    if last_day < start:
        raise ToolError(f"Custom period ends ({last_day.date()}) before it starts ({start.date()}).")
    label = spec.get("label") or f"{start.date()} to {last_day.date()}"
    # +1 day turns the INCLUDED last day into our EXCLUDED end moment.
    return Period(label, "custom", start, last_day + pd.Timedelta(days=1))


def periods_overlap(a: Period, b: Period) -> bool:
    """True if two periods share any moment in time."""
    return a.start < b.end and b.start < a.end


def check_date_column(df: pd.DataFrame, date_column: str) -> None:
    """Make sure the column exists and contains real dates (not text)."""
    check_columns_exist(df, date_column)
    kind = column_kind(df[date_column])
    if kind != "datetime":
        raise ToolError(
            f"Column '{date_column}' is {kind}, not dates. "
            "Convert it first with clean_data (convert_types)."
        )


def filter_period(df: pd.DataFrame, date_column: str, period) -> pd.DataFrame:
    """Return only the rows whose date falls inside the period (start <= date < end)."""
    period = parse_period(period)
    check_date_column(df, date_column)
    dates = df[date_column]
    # Rows with a missing date compare as False, so they are never included.
    return df[(dates >= period.start) & (dates < period.end)]


def period_coverage(df: pd.DataFrame, date_column: str, period) -> dict:
    """
    How completely does the dataset cover this period?

    Two checks:
    1. Day coverage: the share of the period's days that lie inside the
       dataset's overall date range (first date to last date).
       Example: the data ends on 10 May, and we ask about Q2 (1 Apr - 30 Jun):
       only 40 of 91 days are covered -> 44% -> INCOMPLETE.
    2. Month coverage: does every calendar month in the period have at
       least one row? A quarter with an empty month is suspicious.

    Limitation: we can only see the rows we were given. If the dataset is a
    sample, "complete" means "complete within this file".
    """
    period = parse_period(period)
    check_date_column(df, date_column)
    dates = df[date_column].dropna()
    rows = filter_period(df, date_column, period)

    period_days = (period.end - period.start).days
    if dates.empty:
        covered_days = 0
    else:
        first, last = dates.min().normalize(), dates.max().normalize()
        covered_start = max(period.start, first)
        covered_end = min(period.end, last + pd.Timedelta(days=1))
        covered_days = max(0, (covered_end - covered_start).days)
    coverage = covered_days / period_days

    # Every month start inside the period, e.g. Q1 -> Jan 1, Feb 1, Mar 1.
    expected_months = pd.date_range(period.start.replace(day=1), period.last_day, freq="MS")
    expected = {m.strftime("%Y-%m") for m in expected_months}
    present = set(rows[date_column].dt.strftime("%Y-%m"))
    missing_months = sorted(expected - present)

    issues = []
    if coverage < MIN_COVERAGE:
        issues.append(
            f"The dataset covers only {coverage:.0%} of the days in {period.label} "
            f"(data runs from {dates.min().date() if not dates.empty else '-'} "
            f"to {dates.max().date() if not dates.empty else '-'})."
        )
    if missing_months and len(expected) > 1:
        issues.append(f"{period.label} has no rows at all in month(s): {', '.join(missing_months)}.")

    return {
        "period": period.label,
        "rows": len(rows),
        "coverage_percent": round(100 * coverage, 1),
        "months_expected": len(expected),
        "months_with_data": len(expected) - len(missing_months),
        "is_complete": not issues,
        "issues": issues,
    }
