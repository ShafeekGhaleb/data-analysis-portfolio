"""
Tool 3: clean_data. Small, safe cleaning steps plus one function that runs them.

Two golden rules used in every function below:

1. NEVER change the user's original table. Each function works on a COPY
   (df.copy()) and returns a new table. The original can always be compared
   or restored.

2. ALWAYS explain what changed. Each function returns TWO things:
       (new_table, step_report)
   The step_report is a dictionary describing what was done. Together the
   reports form a "cleaning log", which goes into the final analysis report
   so the user can trust (and check) the cleaning.
"""

import re

import pandas as pd

from tools.errors import ToolError
from tools.inspection import column_kind
from tools.utils import check_columns_exist, to_python

# A converted column must have at least this share of values that convert
# successfully. 0.9 = 90%. This stops us from turning a real text column into
# numbers just because a few values happen to look numeric.
CONVERSION_THRESHOLD = 0.9

# Regular expression ("regex"): a pattern for matching text.
# This one matches numbers with thousands separators, like "1,200" or
# "-12,345.67". It does NOT match "3,50" (a European decimal comma), which we
# deliberately leave alone rather than guess wrongly.
THOUSANDS_PATTERN = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")

# Characters removed before trying to read a value as a number.
CURRENCY_AND_SPACES = re.compile(r"[€$£\s]")

# What a date must look like: digits, separator, digits, separator, digits
# (e.g. "2026-01-05", "05/01/2026", "5.1.2026"), optionally followed by a time.
# "(?: ... )" groups parts of the pattern without "capturing" them.
DATE_PATTERN = r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$"


def _report(step: str, description: str, before: pd.DataFrame, after: pd.DataFrame,
            details: dict | None = None) -> dict:
    """Build the standard report dictionary that every cleaning step returns."""
    return {
        "step": step,
        "description": description,
        "rows_before": len(before),
        "rows_after": len(after),
        "details": details or {},  # "x or {}" -> use {} if x is None
    }


# ---------------------------------------------------------------------------
# Individual cleaning steps
# ---------------------------------------------------------------------------

def strip_whitespace(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Remove spaces at the start/end of text values, e.g. "  North " -> "North".
    Empty texts ("" or "   ") become missing values, because that is what
    they really are.
    """
    result = df.copy()
    changed = {}  # column name -> number of values changed
    for name in result.columns:
        if column_kind(result[name]) != "text":
            continue  # only text columns can have extra spaces
        original = result[name]
        # .map() runs a small function on every value. "lambda v: ..." is a
        # tiny unnamed function. We only strip values that really are strings.
        stripped = original.map(lambda v: v.strip() if isinstance(v, str) else v)
        stripped = stripped.replace("", pd.NA)  # empty text -> missing
        # Count values that changed, ignoring cells missing in both versions.
        n_changed = int((original.ne(stripped) & ~(original.isna() & stripped.isna())).sum())
        if n_changed:
            result[name] = stripped
            changed[str(name)] = n_changed
    return result, _report(
        "strip_whitespace",
        f"Trimmed extra spaces in {sum(changed.values())} text value(s).",
        df, result, {"values_changed_per_column": changed},
    )


def standardize_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Make column names tidy: " Unit Price (EUR) " -> "unit_price_eur".
    Lowercase, words joined by "_", no special characters.
    """
    result = df.copy()
    renamed = {}
    new_names = []
    for name in result.columns:
        new = str(name).strip().lower()
        new = re.sub(r"[^0-9a-z]+", "_", new)  # anything not a letter/digit -> "_"
        new = new.strip("_") or "column"       # remove "_" at the ends; never empty
        # If two columns would get the same name, add a number: "price_2".
        base, counter = new, 2
        while new in new_names:
            new = f"{base}_{counter}"
            counter += 1
        new_names.append(new)
        if new != name:
            renamed[str(name)] = new
    result.columns = new_names
    return result, _report(
        "standardize_column_names",
        f"Renamed {len(renamed)} column(s) to a tidy format.",
        df, result, {"renamed": renamed},
    )


def remove_duplicates(df: pd.DataFrame, subset: list | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Remove rows that are exact copies of an earlier row (the first copy is kept).

    subset: optionally, only compare these columns, e.g. ["order_id"]
            to treat rows with the same order ID as duplicates.
    """
    if subset:
        check_columns_exist(df, subset)
    # drop_duplicates() returns a NEW table without the copies.
    # reset_index(drop=True) renumbers rows 0, 1, 2, ... with no gaps.
    result = df.drop_duplicates(subset=subset).reset_index(drop=True)
    removed = len(df) - len(result)
    return result, _report(
        "remove_duplicates",
        f"Removed {removed} duplicate row(s).",
        df, result, {"rows_removed": removed, "compared_columns": subset or "all"},
    )


def _try_numeric(series: pd.Series) -> pd.Series:
    """Try to read text values as numbers. Values that fail become missing (NaN)."""
    def clean_one(value):
        if not isinstance(value, str):
            return value
        text = CURRENCY_AND_SPACES.sub("", value)  # "€ 1,200.50" -> "1,200.50"
        if THOUSANDS_PATTERN.match(text):
            text = text.replace(",", "")            # "1,200.50"  -> "1200.50"
        return text
    # errors="coerce" means: if a value can't be converted, use NaN instead
    # of crashing. We then COUNT those failures to decide if it's safe.
    return pd.to_numeric(series.map(clean_one), errors="coerce")


def _try_datetime(series: pd.Series, dayfirst: bool) -> pd.Series:
    """Try to read text values as dates. Values that fail become missing (NaT)."""
    # pandas' date reader is very generous: it reads "May" as a date, and
    # even reads the price "3,50" as 1 March 2050! (Our tests caught that.)
    # So a value only counts as a date if it LOOKS like one: three groups
    # of digits separated by "-", "/" or ".", e.g. 2026-01-05 or 05/01/2026.
    # Other values become missing and are counted against the 90% threshold.
    texts = series.astype("string")
    looks_like_date = texts.str.contains(DATE_PATTERN, na=False)
    if not looks_like_date.any():
        return pd.Series(pd.NaT, index=series.index)
    series = series.where(looks_like_date)  # keep date-like values; others -> missing
    # format="mixed" lets each value have its own format ("2026-01-05", "05/01/2026").
    return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=dayfirst)


def convert_types(df: pd.DataFrame, columns: list | None = None,
                  dayfirst: bool = False) -> tuple[pd.DataFrame, dict]:
    """
    Turn text columns that really contain numbers or dates into proper types.

    Why it matters: "€1,200" stored as TEXT cannot be summed or averaged.
    After conversion it becomes the number 1200.0.

    A column is converted only if at least 90% of its non-missing values
    convert successfully. Values that fail become missing, and we report
    how many, so nothing is lost silently.

    dayfirst: set True for dates written day-first, e.g. "05/01/2026" = 5 January.
    """
    result = df.copy()
    if columns:
        check_columns_exist(df, columns)
    targets = columns or list(result.columns)
    converted = {}
    for name in targets:
        series = result[name]
        if column_kind(series) != "text":
            continue  # already numbers/dates: nothing to do
        non_missing = series.notna().sum()
        if non_missing == 0:
            continue
        # Try numbers first, then dates.
        for new_kind, attempt in (("numeric", _try_numeric(series)),
                                  ("datetime", _try_datetime(series, dayfirst))):
            success_rate = attempt.notna().sum() / non_missing
            if success_rate >= CONVERSION_THRESHOLD:
                result[name] = attempt
                failed_mask = series.notna() & attempt.isna()
                converted[str(name)] = {
                    "to": new_kind,
                    "values_made_missing": int(failed_mask.sum()),
                    # Show the problem values so a human can check them.
                    "problem_values": [to_python(v) for v in series[failed_mask].unique()[:5]],
                }
                break  # converted: don't try the next type
    return result, _report(
        "convert_types",
        f"Converted {len(converted)} text column(s) to numbers or dates.",
        df, result, {"converted_columns": converted},
    )


MISSING_STRATEGIES = {"drop_rows", "fill_mean", "fill_median", "fill_mode", "fill_value"}


def handle_missing(df: pd.DataFrame, strategy: str = "drop_rows",
                   columns: list | None = None, fill_value=None) -> tuple[pd.DataFrame, dict]:
    """
    Deal with missing values using one strategy:

    - "drop_rows":   delete rows that have a missing value (in `columns`, or any column)
    - "fill_mean":   replace missing numbers with the column average
    - "fill_median": replace missing numbers with the middle value (safer with outliers)
    - "fill_mode":   replace missing values with the most common value (works for text too)
    - "fill_value":  replace missing values with `fill_value`, e.g. 0 or "Unknown"

    There is no single "correct" choice. It depends on the data and the
    question, which is why this is NOT done automatically: the user (or later
    the agent, explaining its reasoning) must choose.
    """
    if strategy not in MISSING_STRATEGIES:
        raise ToolError(f"Unknown strategy '{strategy}'. Choose one of {sorted(MISSING_STRATEGIES)}")
    if strategy == "fill_value" and fill_value is None:
        raise ToolError("Strategy 'fill_value' needs a fill_value, e.g. 0 or 'Unknown'.")
    if columns:
        check_columns_exist(df, columns)

    result = df.copy()
    # Columns to treat: those requested, or every column that has missing values.
    targets = columns or [c for c in result.columns if result[c].isna().any()]

    if strategy == "drop_rows":
        result = result.dropna(subset=targets).reset_index(drop=True)
        removed = len(df) - len(result)
        return result, _report(
            "handle_missing", f"Dropped {removed} row(s) with missing values.",
            df, result, {"strategy": strategy, "columns": [str(c) for c in targets]},
        )

    filled, skipped = {}, {}
    for name in targets:
        series = result[name]
        n_missing = int(series.isna().sum())
        if n_missing == 0:
            continue
        if strategy in ("fill_mean", "fill_median") and column_kind(series) != "numeric":
            skipped[str(name)] = "not numeric: mean/median impossible"
            continue
        if strategy == "fill_mean":
            value = series.mean()
        elif strategy == "fill_median":
            value = series.median()
        elif strategy == "fill_mode":
            modes = series.mode()  # can return several values if tied; take the first
            if modes.empty:
                skipped[str(name)] = "column is completely empty"
                continue
            value = modes.iloc[0]
        else:
            value = fill_value
        result[name] = series.fillna(value)
        filled[str(name)] = {"filled": n_missing, "with": to_python(value)}

    return result, _report(
        "handle_missing",
        f"Filled missing values in {len(filled)} column(s) using '{strategy}'.",
        df, result, {"strategy": strategy, "filled": filled, "skipped": skipped},
    )


# ---------------------------------------------------------------------------
# The main tool: run several steps in order
# ---------------------------------------------------------------------------

# The default "safe" steps. Order matters:
#   1. strip spaces first, so " 1,200 " can become a number and " Bob" == "Bob";
#   2. convert types next, so "1,200" and "1200" become the same value;
#   3. remove duplicates last, so copies hidden by 1 and 2 are now detected.
# Handling missing values is NOT a default step (see handle_missing above).
DEFAULT_STEPS = ["strip_whitespace", "convert_types", "remove_duplicates"]

# A "dispatch table": maps a step NAME (text) to the FUNCTION to call.
# This is how text from the LLM ("remove_duplicates") will later become
# a real function call, without long if/elif chains.
STEP_FUNCTIONS = {
    "strip_whitespace": strip_whitespace,
    "standardize_column_names": standardize_column_names,
    "convert_types": convert_types,
    "remove_duplicates": remove_duplicates,
}


def clean_data(df: pd.DataFrame, steps: list | None = None,
               missing_strategy: str | None = None,
               fill_value=None) -> tuple[pd.DataFrame, list[dict]]:
    """
    Run several cleaning steps in order and collect a cleaning log.

    Example:
        cleaned, log = clean_data(df)                                # safe defaults
        cleaned, log = clean_data(df, missing_strategy="fill_median")  # + fill gaps

    Returns (cleaned_table, cleaning_log), where cleaning_log is a list of
    step reports, one per step.
    """
    steps = steps or DEFAULT_STEPS
    unknown = [s for s in steps if s not in STEP_FUNCTIONS]
    if unknown:
        raise ToolError(f"Unknown cleaning step(s) {unknown}. Available: {list(STEP_FUNCTIONS)}")

    log = []
    current = df
    for step_name in steps:
        function = STEP_FUNCTIONS[step_name]   # look up the function by its name
        current, report = function(current)    # run it; keep its output for the next step
        log.append(report)

    if missing_strategy:
        current, report = handle_missing(current, missing_strategy, fill_value=fill_value)
        log.append(report)

    return current, log
