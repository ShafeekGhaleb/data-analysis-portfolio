"""
Small helper functions shared by several tools.

The main problem solved here:
Pandas and NumPy use their OWN number types (numpy.int64, numpy.float64,
NaN for "missing", Timestamp for dates). These cannot be turned into JSON
directly. Later, every tool result is sent to the LLM as JSON, so all our
tools return only plain Python types: int, float, str, bool, None, list, dict.
"""

import math

import numpy as np
import pandas as pd

from tools.errors import ToolError


def to_python(value, decimals: int = 4):
    """
    Convert one pandas/NumPy value into a plain Python value.

    Examples:
        numpy.int64(5)       -> 5
        numpy.float64(2.5)   -> 2.5
        float('nan') / NaT   -> None      (JSON has no "NaN"; None = "missing")
        Timestamp(...)       -> "2026-01-05T00:00:00"  (text in ISO format)
    """
    # pd.isna() is True for every kind of "missing": None, NaN, NaT, pd.NA.
    # We check that value is not a list/array first, because pd.isna() on a
    # list returns a list instead of True/False.
    if not isinstance(value, (list, tuple, dict, np.ndarray)) and pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isinf(value):  # infinity is also not allowed in JSON
            return None
        # round(): avoid ugly numbers like 971.0233333333.
        # "+ 0.0" turns the confusing "negative zero" (-0.0) into a normal 0.0.
        return round(value, decimals) + 0.0
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (str, list, dict)):
        return value
    # Anything else (rare): fall back to its text form.
    return str(value)


def check_columns_exist(df: pd.DataFrame, columns) -> None:
    """
    Raise a helpful ToolError if any requested column is not in the table.

    We list the available columns in the message. That is very useful later:
    if the LLM misspells a column, it can read the message and correct itself.
    """
    if isinstance(columns, str):  # allow a single name as well as a list
        columns = [columns]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ToolError(
            f"Column(s) not found: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def check_not_empty(df: pd.DataFrame) -> None:
    """Raise a ToolError if the table has no rows."""
    if df.empty:
        raise ToolError("The dataset is empty (0 rows), so there is nothing to analyse.")
