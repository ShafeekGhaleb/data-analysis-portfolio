"""Tests for inspect_data and the to_python helper."""

import json

import numpy as np
import pandas as pd

from tools import inspect_data, load_file
from tools.utils import to_python


def test_inspect_sales_data(sales_csv_path):
    report = inspect_data(load_file(sales_csv_path))
    assert report["n_rows"] == 30
    assert report["n_columns"] == 8
    assert report["duplicate_rows"] == 0
    assert report["total_missing_values"] == 0
    kinds = {col["name"]: col["kind"] for col in report["columns"]}
    assert kinds["Revenue_EUR"] == "numeric"
    assert kinds["Region"] == "text"
    assert len(report["sample_rows"]) == 5


def test_inspect_finds_missing_values(messy_csv_path):
    report = inspect_data(load_file(messy_csv_path))
    by_name = {col["name"]: col for col in report["columns"]}
    assert by_name["quantity"]["missing"] == 1
    assert by_name["payment_method"]["missing"] == 1
    assert report["total_missing_values"] == 3


def test_report_is_json_safe(messy_csv_path):
    # The whole point of our design: the result must convert to JSON
    # (that's how it will be sent to the LLM). json.dumps raises an error if not.
    report = inspect_data(load_file(messy_csv_path))
    text = json.dumps(report)
    assert "NaN" not in text  # NaN is not valid JSON; missing must be null


def test_to_python_conversions():
    assert to_python(np.int64(5)) == 5 and type(to_python(np.int64(5))) is int
    assert to_python(np.float64(1 / 3)) == 0.3333
    assert to_python(np.nan) is None
    assert to_python(pd.NaT) is None
    assert to_python(float("inf")) is None
    assert to_python(pd.Timestamp("2026-01-05")) == "2026-01-05T00:00:00"
    assert to_python(np.bool_(True)) is True
