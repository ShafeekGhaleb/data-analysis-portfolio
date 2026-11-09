"""
The "tools" package: the agent's hands.

Every file in this folder contains plain Python functions that do ONE data
job each (load, inspect, clean, calculate statistics, ...).

Important design rule: these functions know NOTHING about AI. They can be
used and tested on their own. Later (Stage 4), we will describe them to the
LLM so it can ask us to run them.

This file (__init__.py) turns the folder into a "package", so other code can
write `from tools import load_file`. The lines below re-export the most
important functions, so users don't need to know which file each one is in.
"""

from tools.errors import ToolError
from tools.data_loading import load_file
from tools.inspection import inspect_data
from tools.cleaning import (
    clean_data,
    convert_types,
    handle_missing,
    remove_duplicates,
    standardize_column_names,
    strip_whitespace,
)
from tools.statistics import correlation, describe_statistics, group_and_aggregate
from tools.charts import create_chart, create_waterfall_chart
from tools.change_analysis import explain_change
from tools.periods import filter_period, parse_period, period_coverage

# __all__ lists the public names of this package (what `from tools import *` gives).
__all__ = [
    "ToolError",
    "load_file",
    "inspect_data",
    "clean_data",
    "convert_types",
    "handle_missing",
    "remove_duplicates",
    "standardize_column_names",
    "strip_whitespace",
    "describe_statistics",
    "group_and_aggregate",
    "correlation",
    "create_chart",
    "create_waterfall_chart",
    "explain_change",
    "parse_period",
    "filter_period",
    "period_coverage",
]
