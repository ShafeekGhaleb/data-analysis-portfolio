"""
Tests for load_file.

How to read a test:
- Each function starting with "test_" is one test. pytest runs them all.
- `assert <condition>` means "this MUST be true". If it's false, the test fails.
- `with pytest.raises(ToolError, match="..."):` means "the code inside MUST
  raise a ToolError whose message contains this text". Testing that errors
  happen correctly is just as important as testing success.
- `tmp_path` is a built-in pytest fixture: a fresh, empty temporary folder,
  deleted afterwards. Perfect for creating test files.
"""

import pandas as pd
import pytest

from tools import ToolError, load_file


def test_load_real_sales_csv(sales_csv_path):
    df = load_file(sales_csv_path)
    assert df.shape == (30, 8)  # shape = (rows, columns)
    assert list(df.columns)[:3] == ["Date", "Region", "Channel"]


def test_load_excel_file(tmp_path, small_df):
    # Create a real .xlsx file, then check we read back exactly the same table.
    path = tmp_path / "data.xlsx"
    small_df.to_excel(path, index=False)
    loaded = load_file(path)
    # assert_frame_equal compares two tables cell by cell.
    # check_dtype=False: Excel stores 100.0 as 100, so pandas reads it back
    # as a whole number (int) instead of a decimal (float). The VALUES are equal.
    pd.testing.assert_frame_equal(loaded, small_df, check_dtype=False)


def test_load_semicolon_csv(tmp_path):
    # Many European programs save CSVs with ";" instead of ",".
    path = tmp_path / "euro.csv"
    path.write_text("name;value\nA;1\nB;2\n", encoding="utf-8")
    df = load_file(path)
    assert list(df.columns) == ["name", "value"]
    assert df["value"].sum() == 3


def test_load_latin1_csv(tmp_path):
    # "Müller" saved in the old latin-1 encoding: not valid utf-8.
    path = tmp_path / "old.csv"
    path.write_bytes("name,city\nMüller,Köln\n".encode("latin-1"))
    df = load_file(path)
    assert df.loc[0, "name"] == "Müller"


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(ToolError, match="File not found"):
        load_file(tmp_path / "does_not_exist.csv")


def test_unsupported_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ToolError, match="Unsupported file type"):
        load_file(path)


def test_empty_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ToolError, match="empty"):
        load_file(path)


def test_header_only_file_is_empty(tmp_path):
    path = tmp_path / "header_only.csv"
    path.write_text("a,b,c\n")
    with pytest.raises(ToolError, match="0 rows"):
        load_file(path)


def test_missing_excel_sheet(tmp_path, small_df):
    path = tmp_path / "data.xlsx"
    small_df.to_excel(path, index=False)
    with pytest.raises(ToolError, match="sheet"):
        load_file(path, sheet_name="NoSuchSheet")
