"""Tests for tools/periods.py: parsing periods, filtering rows, and coverage checks."""

import pandas as pd
import pytest

from tools import ToolError
from tools.periods import (
    Period,
    filter_period,
    parse_period,
    period_coverage,
    periods_overlap,
)


@pytest.mark.parametrize("text, label, start, last_day", [
    ("2026", "2026", "2026-01-01", "2026-12-31"),
    ("2026-Q1", "2026-Q1", "2026-01-01", "2026-03-31"),
    ("2026q4", "2026-Q4", "2026-10-01", "2026-12-31"),       # lower case, no dash
    ("2026-02", "2026-02", "2026-02-01", "2026-02-28"),
    ("2024-2", "2024-02", "2024-02-01", "2024-02-29"),        # leap year
    ("2026-03-15", "2026-03-15", "2026-03-15", "2026-03-15"),
])
def test_parse_text_periods(text, label, start, last_day):
    period = parse_period(text)
    assert period.label == label
    assert period.start == pd.Timestamp(start)
    assert period.last_day == pd.Timestamp(last_day)


def test_custom_period_includes_both_days():
    period = parse_period({"start": "2026-01-10", "end": "2026-01-20"})
    assert period.kind == "custom"
    assert period.start == pd.Timestamp("2026-01-10")
    assert period.end == pd.Timestamp("2026-01-21")       # end is EXCLUSIVE: the next day
    assert period.to_dict()["end"] == "2026-01-20"         # but shown as the last included day


@pytest.mark.parametrize("bad, message", [
    ("2026-13", "Invalid period"),
    ("2026-02-30", "Invalid period"),
    ("2026-Q5", "Unrecognised"),
    ("Q1 2026", "Unrecognised"),
    ("last month", "Unrecognised"),
    (2026, "must be text"),
    ({"start": "2026-01-01"}, "needs both"),
    ({"start": "2026-02-01", "end": "2026-01-01"}, "before it starts"),
    ({"start": "soon", "end": "later"}, "Invalid custom period"),
])
def test_invalid_periods(bad, message):
    with pytest.raises(ToolError, match=message):
        parse_period(bad)


def test_half_open_boundaries():
    # A sale late on 31 March belongs to Q1; a sale at midnight on 1 April belongs to Q2.
    df = pd.DataFrame({"when": pd.to_datetime(["2026-03-31 18:30", "2026-04-01 00:00"])})
    assert len(filter_period(df, "when", "2026-Q1")) == 1
    assert len(filter_period(df, "when", "2026-Q2")) == 1
    assert filter_period(df, "when", "2026-Q1")["when"].iloc[0].hour == 18


def test_rows_without_date_are_never_included():
    df = pd.DataFrame({"when": pd.to_datetime(["2026-01-05", None])})
    assert len(filter_period(df, "when", "2026")) == 1


def test_text_dates_are_rejected():
    df = pd.DataFrame({"when": ["2026-01-05"]})
    with pytest.raises(ToolError, match="Convert it first"):
        filter_period(df, "when", "2026")


def test_overlap():
    q1, q2 = parse_period("2026-Q1"), parse_period("2026-Q2")
    assert not periods_overlap(q1, q2)                     # touching is not overlapping
    assert periods_overlap(q1, parse_period("2026-03"))
    assert periods_overlap(parse_period("2026"), q2)


def test_coverage_complete_quarters(sales_df):
    q1 = period_coverage(sales_df, "Date", "2026-Q1")
    assert q1["rows"] == 15
    assert q1["coverage_percent"] == 95.6                  # data starts 5 Jan: 86 of 90 days
    assert q1["months_with_data"] == 3 and q1["is_complete"]


def test_coverage_incomplete_year(sales_df):
    year = period_coverage(sales_df, "Date", "2026")
    assert not year["is_complete"]
    assert year["months_expected"] == 12 and year["months_with_data"] == 6
    assert any("2026-07" in issue for issue in year["issues"])


def test_coverage_detects_an_empty_month_inside_the_data_range():
    dates = pd.to_datetime(["2026-01-01", "2026-01-31", "2026-03-01", "2026-03-31"])
    df = pd.DataFrame({"d": dates})                        # February has no rows
    coverage = period_coverage(df, "d", "2026-Q1")
    assert coverage["coverage_percent"] == 100.0           # the date RANGE covers the quarter...
    assert not coverage["is_complete"]                     # ...but a month is empty
    assert "2026-02" in coverage["issues"][0]


def test_period_object_passes_through():
    period = parse_period("2026-Q1")
    assert parse_period(period) is period
    assert isinstance(period, Period)
