"""
conftest.py: shared test setup ("fixtures") used by all test files.

A FIXTURE is a function that prepares something a test needs (a small
table, a file path, ...). A test asks for it simply by using the fixture's
name as a parameter, e.g. `def test_x(small_df):`. pytest finds this file
automatically and runs the fixture before the test.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Path(__file__) is this file; .parent is the "tests" folder; .parent again
# is the project folder. This works no matter where pytest is started from.
PROJECT_DIR = Path(__file__).parent.parent
SAMPLE_DIR = PROJECT_DIR / "sample_data"   # "/" joins paths with Path objects


@pytest.fixture
def sales_csv_path() -> Path:
    """The real (clean) sales dataset from the portfolio project."""
    return SAMPLE_DIR / "sales_data.csv"


@pytest.fixture
def messy_csv_path() -> Path:
    """A deliberately messy dataset: spaces, € signs, duplicates, gaps, bad values."""
    return SAMPLE_DIR / "messy_orders.csv"


@pytest.fixture
def small_df() -> pd.DataFrame:
    """A tiny table where we can calculate every correct answer by hand."""
    return pd.DataFrame({
        "region": ["North", "North", "South", "South", "East"],
        "channel": ["Online", "Store", "Online", "Online", "Store"],
        "units": [10, 20, 30, 40, 50],
        "revenue": [100.0, 200.0, 300.0, 400.0, 500.0],
        "cost": [50.0, 90.0, 160.0, 210.0, 240.0],
    })


@pytest.fixture
def df_with_missing() -> pd.DataFrame:
    """A tiny table with missing values (np.nan / None) to test handle_missing."""
    return pd.DataFrame({
        "city": ["Berlin", None, "Berlin", "Munich"],
        "score": [10.0, np.nan, 30.0, 100.0],
    })
