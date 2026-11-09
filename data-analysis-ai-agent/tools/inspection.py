"""
Tool 2: inspect_data. Describes a dataset's structure and quality.

This is the FIRST thing a good analyst (and our agent) does with new data:
"How many rows? Which columns? What type is each? Is anything missing?"

The result is a dictionary, NOT the data itself. Later, only this summary
is sent to the LLM. It is small, cheap, and keeps the raw data private.
"""

import json

import pandas as pd

from tools.utils import check_not_empty, to_python


def column_kind(series: pd.Series) -> str:
    """
    Give a simple, human-friendly type name for one column.

    Pandas has many detailed type names (int64, float64, str, object,
    datetime64[ns], bool, ...). For analysis we only need 4 broad kinds.
    A "Series" is one column of a DataFrame.
    """
    # Check boolean first: pandas also counts True/False as numbers.
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "text"


def inspect_data(df: pd.DataFrame, sample_rows: int = 5) -> dict:
    """
    Summarise a DataFrame: size, columns, types, missing values, duplicates.

    Returns a dictionary like:
    {
      "n_rows": 30, "n_columns": 8, "duplicate_rows": 0,
      "total_missing_values": 0,
      "columns": [ {"name": "Region", "kind": "text", "missing": 0, ...}, ... ],
      "sample_rows": [ {...first row...}, ... ]
    }
    """
    check_not_empty(df)

    columns_info = []
    # df.columns is the list of column names; df[name] gives that column.
    for name in df.columns:
        series = df[name]
        missing = int(series.isna().sum())  # isna() -> True/False per row; sum counts the Trues
        info = {
            "name": str(name),
            "kind": column_kind(series),
            "pandas_dtype": str(series.dtype),
            "missing": missing,
            "missing_percent": round(100 * missing / len(df), 2),
            "unique_values": int(series.nunique()),  # number of different values (ignores missing)
            # The first 3 different values, to show what the column looks like.
            "examples": [to_python(v) for v in series.dropna().unique()[:3]],
        }
        columns_info.append(info)

    return {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        # duplicated() marks rows that are exact copies of an earlier row.
        "duplicate_rows": int(df.duplicated().sum()),
        "total_missing_values": int(df.isna().sum().sum()),  # sum per column, then sum all
        "columns": columns_info,
        # A neat trick: pandas' own to_json() already converts NaN -> null and
        # dates -> text, and json.loads() turns that text back into plain
        # Python dicts. This gives clean, JSON-safe sample rows.
        "sample_rows": json.loads(
            df.head(sample_rows).to_json(orient="records", date_format="iso")
        ),
    }
