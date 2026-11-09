"""
Tool 4: statistics. Summary statistics, grouping, and correlation.

These functions produce EVERY number the agent will report. The LLM will
never calculate numbers itself; it only reads the results of these
functions. That is what makes the agent's answers trustworthy.
"""

import pandas as pd

from tools.errors import ToolError
from tools.inspection import column_kind
from tools.utils import check_columns_exist, check_not_empty, to_python


def describe_statistics(df: pd.DataFrame, columns: list | None = None) -> dict:
    """
    Summary statistics for each column, depending on its kind:

    - numeric:  count, mean, median, std, min, quartiles, max, sum
    - text/boolean: count, number of unique values, most common values
    - datetime: earliest and latest date

    Beginner notes on the numeric measures:
    - mean   = the average (sum / count); sensitive to extreme values
    - median = the middle value when sorted; robust to extreme values
    - std    = standard deviation: how spread out the values are
    - q1/q3  = 25% / 75% of values are below this number
    """
    check_not_empty(df)
    if columns:
        check_columns_exist(df, columns)
    targets = columns or list(df.columns)

    result = {"numeric": {}, "categorical": {}, "datetime": {}}
    for name in targets:
        series = df[name]
        kind = column_kind(series)
        base = {"count": int(series.notna().sum()), "missing": int(series.isna().sum())}

        if kind == "numeric":
            result["numeric"][str(name)] = {
                **base,  # "**" copies all keys of `base` into this dictionary
                "mean": to_python(series.mean()),
                "median": to_python(series.median()),
                "std": to_python(series.std()),
                "min": to_python(series.min()),
                "q1": to_python(series.quantile(0.25)),
                "q3": to_python(series.quantile(0.75)),
                "max": to_python(series.max()),
                "sum": to_python(series.sum()),
            }
        elif kind == "datetime":
            result["datetime"][str(name)] = {
                **base,
                "earliest": to_python(series.min()),
                "latest": to_python(series.max()),
            }
        else:  # text or boolean
            # value_counts() counts how often each value appears, most common first.
            top = series.value_counts().head(5)
            result["categorical"][str(name)] = {
                **base,
                "unique_values": int(series.nunique()),
                "top_values": {str(k): int(v) for k, v in top.items()},
            }
    return result


# Allowed aggregation methods and whether they need a numeric value column.
AGGREGATIONS = {
    "sum": True, "mean": True, "median": True, "std": True,
    "min": True, "max": True,
    "count": False,    # counts rows: works for any column (or no column)
    "nunique": False,  # counts different values: works for any column
}


def group_and_aggregate(df: pd.DataFrame, group_by, value_column: str | None = None,
                        method: str = "sum", sort_descending: bool = True,
                        top_n: int | None = None) -> dict:
    """
    Answer questions like "total revenue by region" or "average price per category".

    Example:
        group_and_aggregate(df, group_by="Region", value_column="Revenue_EUR", method="sum")

    For "sum" and "count", each group also gets its percentage of the total.
    """
    check_not_empty(df)
    if method not in AGGREGATIONS:
        raise ToolError(f"Unknown method '{method}'. Choose one of {list(AGGREGATIONS)}")

    group_cols = [group_by] if isinstance(group_by, str) else list(group_by)
    check_columns_exist(df, group_cols)

    if value_column is None:
        if method != "count":
            raise ToolError(f"Method '{method}' needs a value_column.")
    else:
        check_columns_exist(df, value_column)
        if AGGREGATIONS[method] and column_kind(df[value_column]) != "numeric":
            raise ToolError(
                f"Column '{value_column}' is not numeric, so '{method}' is impossible. "
                "Try cleaning it with convert_types, or use 'count'/'nunique'."
            )

    # groupby() splits the table into groups (e.g. one per region).
    # dropna=False keeps a group for rows where the group value is missing,
    # so those rows are not silently ignored.
    grouped = df.groupby(group_cols, dropna=False)
    if value_column is None:
        values = grouped.size()  # rows per group
    else:
        values = grouped[value_column].agg(method)  # e.g. .agg("sum")

    # Percentages only make sense for "sum" and "count" (you can't add up averages).
    # The total is computed BEFORE top_n cuts groups off, so percentages
    # always mean "share of the whole dataset".
    with_share = method in ("sum", "count")
    total = values.sum() if with_share else None

    values = values.sort_values(ascending=not sort_descending)
    if top_n:
        values = values.head(top_n)

    rows = []
    # .items() gives (group key, value) pairs. With several group columns,
    # the key is a tuple like ("North", "Online").
    for key, value in values.items():
        keys = key if isinstance(key, tuple) else (key,)
        row = {str(col): to_python(k) for col, k in zip(group_cols, keys)}
        row["value"] = to_python(value)
        if with_share:
            row["percent_of_total"] = to_python(100 * value / total, 2) if total else None
        rows.append(row)

    return {
        "group_by": [str(c) for c in group_cols],
        "value_column": value_column,
        "method": method,
        "n_groups": len(rows),
        "total": to_python(total) if with_share else None,
        "rows": rows,
    }


def _strength_label(r: float) -> str:
    """Turn a correlation number (-1 to 1) into words a beginner can read."""
    size = abs(r)
    if size >= 0.7:
        strength = "strong"
    elif size >= 0.4:
        strength = "moderate"
    elif size >= 0.2:
        strength = "weak"
    else:
        return "no clear relationship"
    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction}"


def correlation(df: pd.DataFrame, columns: list | None = None, method: str = "pearson") -> dict:
    """
    How strongly do numeric columns move together?

    The correlation coefficient r is between -1 and +1:
      +1 -> when one goes up, the other always goes up (perfect positive)
       0 -> no linear relationship
      -1 -> when one goes up, the other always goes down (perfect negative)

    method:
      "pearson"  -> measures straight-line relationships (the classic choice)
      "spearman" -> measures any steadily increasing/decreasing relationship;
                    less affected by outliers

    Remember: correlation is NOT causation.
    """
    check_not_empty(df)
    if method not in ("pearson", "spearman"):
        raise ToolError("method must be 'pearson' or 'spearman'.")
    if columns:
        check_columns_exist(df, columns)
        not_numeric = [c for c in columns if column_kind(df[c]) != "numeric"]
        if not_numeric:
            raise ToolError(f"Correlation needs numeric columns; these are not: {not_numeric}")
        numeric_cols = list(columns)
    else:
        numeric_cols = [c for c in df.columns if column_kind(df[c]) == "numeric"]

    if len(numeric_cols) < 2:
        raise ToolError(
            f"Correlation needs at least 2 numeric columns; found {len(numeric_cols)}: {numeric_cols}"
        )

    matrix = df[numeric_cols].corr(method=method)

    # Each pair once: (A, B) but not (B, A), and not (A, A).
    pairs = []
    for i, col_a in enumerate(numeric_cols):        # enumerate gives (position, value)
        for col_b in numeric_cols[i + 1:]:          # only columns AFTER col_a
            r = matrix.loc[col_a, col_b]
            if pd.isna(r):  # happens when a column never changes (constant)
                continue
            pairs.append({
                "column_a": str(col_a),
                "column_b": str(col_b),
                "r": to_python(r, 3),
                "interpretation": _strength_label(r),
            })
    # Strongest relationships first (by size, ignoring the sign).
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)

    return {
        "method": method,
        "columns": [str(c) for c in numeric_cols],
        "pairs": pairs,
        "matrix": {
            str(a): {str(b): to_python(matrix.loc[a, b], 3) for b in numeric_cols}
            for a in numeric_cols
        },
    }
