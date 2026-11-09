"""
Tests for tools/change_analysis.py (explain_change).

Most tests use TINY hand-made datasets where we can work out the correct
answer with pencil and paper. Each scenario isolates ONE effect:

    volume-only : same prices, same product blend, more units
    price-only  : same units, higher prices
    mix-only    : same total units and prices, different product blend
"""

import json

import numpy as np
import pandas as pd
import pytest

from tools import ToolError, explain_change
from tools.change_analysis import auto_select_dimensions, is_nested, reconciliation_tolerance


# ---------------------------------------------------------------------------
# Helper: build a small sales table for January (A) and February (B)
# ---------------------------------------------------------------------------

def make_sales(rows_a, rows_b):
    """
    rows_a / rows_b: lists of (product, region, units, price).
    Revenue = units * price. Dates are spread over each whole month, so the
    periods count as complete (no coverage warnings get in the way).
    """
    records = []
    for i, (product, region, units, price) in enumerate(rows_a):
        records.append(("2026-01-01", i, product, region, units, price))
    for i, (product, region, units, price) in enumerate(rows_b):
        records.append(("2026-02-28", -i, product, region, units, price))
    df = pd.DataFrame(records, columns=["base", "offset", "product", "region", "units", "price"])
    df["date"] = pd.to_datetime(df["base"]) + pd.to_timedelta(df["offset"], unit="D")
    df["revenue"] = df["units"] * df["price"]
    return df.drop(columns=["base", "offset"])


def run(df, **kwargs):
    """explain_change with price/volume/mix switched on and A=Jan, B=Feb."""
    # dimensions are given explicitly: on tiny tables, automatic selection
    # treats "product" (3 values in 4 rows) as an ID column and skips it.
    params = dict(metric="revenue", date_column="date", period_a="2026-01", period_b="2026-02",
                  units_column="units", item_column="product", price_column="price",
                  dimensions=["product", "region"])
    params.update(kwargs)
    return explain_change(df, **params)


BASE_A = [("X", "North", 10, 5.0), ("Y", "South", 20, 2.0)]   # revenue 50 + 40 = 90


# ---------------------------------------------------------------------------
# Price / volume / mix scenarios
# ---------------------------------------------------------------------------

def test_volume_only_change():
    # Every product sells exactly twice as many units at the same price.
    result = run(make_sales(BASE_A, [("X", "North", 20, 5.0), ("Y", "South", 40, 2.0)]))
    effects = result["price_volume_mix"]["effects"]
    assert result["summary"]["change"] == 90.0
    assert effects["volume"] == 90.0        # (60 - 30) units x 3.00 average price
    assert effects["mix"] == 0.0            # same blend (1/3 X, 2/3 Y)
    assert effects["price"] == 0.0          # same prices
    assert result["reconciled"]


def test_price_only_change():
    # Same units; X costs 6 instead of 5, Y costs 2.50 instead of 2.
    result = run(make_sales(BASE_A, [("X", "North", 10, 6.0), ("Y", "South", 20, 2.5)]))
    effects = result["price_volume_mix"]["effects"]
    assert result["summary"]["change"] == 20.0
    assert effects["price"] == 20.0         # 10 x 1.00 + 20 x 0.50
    assert effects["volume"] == 0.0
    assert effects["mix"] == 0.0
    items = {i["item"]: i for i in result["price_volume_mix"]["items"]}
    assert items["X"]["price_change_percent"] == 20.0
    assert items["Y"]["price_change_percent"] == 25.0


def test_mix_only_change():
    # Still 30 units at the same prices, but the blend moves to the expensive X.
    result = run(make_sales(BASE_A, [("X", "North", 20, 5.0), ("Y", "South", 10, 2.0)]))
    effects = result["price_volume_mix"]["effects"]
    assert result["summary"]["change"] == 30.0  # 120 - 90
    assert effects["mix"] == 30.0               # (20x5 + 10x2) - 30 x 3.00
    assert effects["volume"] == 0.0
    assert effects["price"] == 0.0


def test_new_and_discontinued_products():
    # Y disappears, Z appears; X is unchanged.
    df = make_sales(BASE_A, [("X", "North", 10, 5.0), ("Z", "West", 5, 10.0)])
    result = run(df)
    pvm = result["price_volume_mix"]
    assert pvm["effects"]["new_items"] == 50.0            # Z: 5 x 10
    assert pvm["effects"]["discontinued_items"] == -40.0  # Y: 20 x 2 lost
    assert pvm["effects"]["volume"] == pvm["effects"]["mix"] == pvm["effects"]["price"] == 0.0
    assert pvm["counts"] == {"continuing": 1, "new": 1, "discontinued": 1}
    product = {c["group"]: c["status"] for c in result["drivers"]["product"]["contributions"]}
    assert product == {"Z": "new", "Y": "discontinued", "X": "existing"}
    assert any("new in 2026-02" in w and "Z" in w for w in result["warnings"])
    assert any("discontinued" in w and "Y" in w for w in result["warnings"])


def test_no_continuing_products():
    result = run(make_sales([("X", "N", 10, 5.0)], [("Z", "N", 3, 10.0)]))
    pvm = result["price_volume_mix"]
    assert pvm["effects"]["new_items"] == 30.0 and pvm["effects"]["discontinued_items"] == -50.0
    assert pvm["notes"]                                    # explains that nothing continued
    assert result["reconciled"]


# ---------------------------------------------------------------------------
# Reconciliation: every breakdown must add up to the total change
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_everything_reconciles_on_random_data(seed):
    # Random products (some only in one period), random units and prices.
    rng = np.random.default_rng(seed)  # a "seed" makes the random numbers repeatable
    def rows(products):
        return [(p, rng.choice(["N", "S", "E"]), int(rng.integers(1, 50)),
                 float(rng.integers(100, 5000)) / 100) for p in products for _ in range(3)]
    df = make_sales(rows(["A", "B", "C", "D"]), rows(["B", "C", "D", "E"]))
    result = run(df, dimensions=["product", "region"], top_n=2)

    change = result["summary"]["change"]
    assert result["reconciled"] and all(c["passed"] for c in result["reconciliation"]["checks"])
    # Contributions (top 2 + "others") add up to the change, per dimension.
    for driver in result["drivers"].values():
        listed = sum(c["change"] for c in driver["contributions"])
        others = driver["others"]["change"] if driver["others"] else 0
        assert listed + others == pytest.approx(change, abs=1e-3)
    # Volume + mix + price + new + discontinued = change.
    assert sum(result["price_volume_mix"]["effects"].values()) == pytest.approx(change, abs=1e-3)


def test_tolerance_grows_with_size():
    assert reconciliation_tolerance(10, 20) == 1e-6                 # small totals: absolute floor
    assert reconciliation_tolerance(1e12, 2e12) == pytest.approx(2000.0)  # 1e-9 x 2e12


# ---------------------------------------------------------------------------
# Missing, invalid and negative values
# ---------------------------------------------------------------------------

def test_missing_metric_value_is_excluded_and_reported():
    df = make_sales(BASE_A, [("X", "North", 10, 5.0), ("Y", "South", 20, 2.0)])
    df.loc[df.index[-1], "revenue"] = np.nan
    result = run(df)
    assert result["summary"]["period_b_total"] == 50.0    # the missing value is left out...
    assert any("no revenue value" in w for w in result["warnings"])   # ...and reported
    assert result["price_volume_mix"]["status"] == "not_reliable"     # PVM refuses
    assert result["price_volume_mix"]["effects"] is None


def test_missing_units_make_pvm_not_reliable():
    df = make_sales(BASE_A, [("X", "North", 10, 5.0)])
    df.loc[0, "units"] = np.nan
    pvm = run(df)["price_volume_mix"]
    assert pvm["status"] == "not_reliable"
    assert any("missing value" in reason for reason in pvm["reasons"])


def test_missing_dimension_values_get_their_own_group():
    df = make_sales(BASE_A, [("X", None, 10, 5.0), ("Y", "South", 20, 2.0)])
    result = run(df)
    groups = [c["group"] for c in result["drivers"]["region"]["contributions"]]
    assert "(missing)" in groups
    assert result["reconciled"]
    assert any("grouped as '(missing)'" in w for w in result["warnings"])


def test_inconsistent_price_column_blocks_pvm():
    df = make_sales(BASE_A, [("X", "North", 10, 5.0), ("Y", "South", 20, 2.0)])
    df["price"] = df["price"] * 1.5          # e.g. prices in another currency
    pvm = run(df)["price_volume_mix"]
    assert pvm["status"] == "not_reliable"
    assert any("satisfy revenue ≈ units × price" in r for r in pvm["reasons"])


def test_small_rounding_in_price_column_is_accepted():
    df = make_sales(BASE_A, [("X", "North", 10, 5.0), ("Y", "South", 20, 2.0)])
    df.loc[0, "revenue"] += 0.004            # less than a cent of rounding
    assert run(df)["price_volume_mix"]["status"] == "ok"


def test_negative_revenue_is_reported_and_pvm_refused():
    # A refund in February.
    df = make_sales(BASE_A, [("X", "North", 10, 5.0), ("Y", "South", 20, 2.0), ("X", "North", 1, 5.0)])
    df.loc[df.index[-1], "revenue"] = -5.0
    result = run(df)
    assert result["summary"]["period_b_total"] == 85.0
    assert result["reconciled"]                            # contributions still add up
    assert any("negative revenue" in w for w in result["warnings"])
    assert result["price_volume_mix"]["status"] == "not_reliable"


def test_negative_units_make_pvm_not_reliable():
    df = make_sales(BASE_A, [("X", "North", -2, 5.0)])
    pvm = run(df)["price_volume_mix"]
    assert pvm["status"] == "not_reliable"
    assert any("negative units" in r for r in pvm["reasons"])


def test_revenue_without_units_is_caught():
    df = make_sales(BASE_A, [("X", "North", 0, 5.0)])
    df.loc[df.index[-1], "revenue"] = 30.0   # revenue but zero units
    df["price"] = np.nan                     # no listed price to check against
    pvm = run(df, price_column=None)["price_volume_mix"]
    assert pvm["status"] == "not_reliable"
    assert any("In 2026-02" in r and "zero units" in r for r in pvm["reasons"])


# ---------------------------------------------------------------------------
# Empty and unusual periods
# ---------------------------------------------------------------------------

def test_empty_period_is_refused_with_the_data_range():
    df = make_sales(BASE_A, BASE_A)
    with pytest.raises(ToolError, match="has no rows. The data covers 2026-01-01 to 2026-02-28"):
        run(df, period_b="2026-05")


def test_zero_change_has_no_shares():
    result = run(make_sales(BASE_A, BASE_A))
    assert result["summary"]["change"] == 0.0
    assert all(c["share_of_change"] is None for c in result["drivers"]["product"]["contributions"])
    assert any("almost) zero" in w for w in result["warnings"])


def test_zero_start_total_has_no_percentage():
    result = run(make_sales([("X", "N", 0, 5.0)], [("X", "N", 4, 5.0)]), units_column=None,
                 item_column=None, price_column=None)
    assert result["summary"]["change_percent"] is None
    assert any("percentage change is undefined" in w for w in result["warnings"])


def test_small_samples_and_incomplete_periods_are_flagged(sales_df):
    result = explain_change(sales_df, "Revenue_EUR", "Date", "2026-05",
                            {"start": "2026-06-01", "end": "2026-07-31"})
    assert not result["data_quality"]["period_b"]["is_complete"]
    assert any("covers only 48%" in w for w in result["warnings"])
    assert any("fewer than 5 rows" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs, message", [
    ({"aggregate": "mean"}, "not additive"),
    ({"metric": "region"}, "not numeric"),
    ({"metric": None}, "needs a metric"),
    ({"period_b": "2026"}, "overlap"),
    ({"item_column": None}, "BOTH units_column and item_column"),
    ({"dimensions": ["revenue"]}, "cannot be a dimension"),
    ({"dimensions": ["regoin"]}, "Available columns"),
    ({"top_n": 0}, "top_n"),
    ({"period_a": "January"}, "Unrecognised period"),
])
def test_invalid_requests(kwargs, message):
    with pytest.raises(ToolError, match=message):
        run(make_sales(BASE_A, BASE_A), **kwargs)


def test_text_date_column_is_refused():
    df = make_sales(BASE_A, BASE_A)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    with pytest.raises(ToolError, match="clean_data"):
        run(df)


def test_count_aggregate_works_without_metric():
    df = make_sales(BASE_A, BASE_A + [("Y", "South", 1, 2.0)])
    result = explain_change(df, None, "date", "2026-01", "2026-02", aggregate="count")
    assert result["summary"]["change"] == 1.0
    assert result["price_volume_mix"] is None


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

def test_auto_selection_skips_ids_and_numbers():
    df = pd.DataFrame({
        "order_id": [f"A{i}" for i in range(10)],   # unique per row: an ID, not a segment
        "region": ["N", "S"] * 5,                    # a real segment
        "only_one": ["x"] * 10,                       # 1 group: explains nothing
        "amount": range(10),                          # numeric: a metric, not a segment
    })
    assert auto_select_dimensions(df, exclude=[]) == ["region"]


def test_nested_dimensions(sales_df):
    assert is_nested(sales_df, "Category", "Product")      # each product has one category
    assert not is_nested(sales_df, "Region", "Channel")


# ---------------------------------------------------------------------------
# The real sales data (regression: these numbers were checked by hand)
# ---------------------------------------------------------------------------

def test_real_sales_q1_vs_q2(sales_df):
    result = explain_change(sales_df, "Revenue_EUR", "Date", "2026-Q1", "2026-Q2",
                            units_column="Units", item_column="Product",
                            price_column="Unit_Price_EUR", top_n=8)
    summary = result["summary"]
    assert (summary["period_a_total"], summary["period_b_total"]) == (11586.6, 17544.1)
    assert summary["change"] == 5957.5 and summary["change_percent"] == 51.42

    region = {c["group"]: c["change"] for c in result["drivers"]["Region"]["contributions"]}
    assert region == {"North": 3276.0, "South": 2851.2, "East": -628.5, "West": 458.8}
    assert result["dimension_ranking"][0] == "Channel"       # Online = 74% of the change

    combos = {c["group"]: c for c in result["top_combinations"]}
    assert combos["East / Online"]["change"] == -1756.5
    assert combos["West / USB-C Hub"]["status"] == "discontinued"
    assert result["skipped_nested_pairs"] == ["Category x Product"]

    pvm = result["price_volume_mix"]
    assert pvm["effects"] == {"volume": 6293.2294, "mix": -335.7294, "price": 0.0,
                              "new_items": 0.0, "discontinued_items": 0.0}
    assert (pvm["units_a"], pvm["units_b"]) == (591, 912)
    assert result["reconciled"]
    assert "not WHY" in result["causation_note"]
    json.dumps(result)                                         # JSON-safe for the future LLM


# ---------------------------------------------------------------------------
# Remaining guardrails (found with a coverage report)
# ---------------------------------------------------------------------------

def test_text_units_column_blocks_pvm():
    df = make_sales(BASE_A, BASE_A)
    df["units"] = df["units"].astype(str) + " pcs"          # "10 pcs" is text, not a number
    pvm = run(df)["price_volume_mix"]
    assert pvm["status"] == "not_reliable"
    assert any("'units' is not numeric" in r for r in pvm["reasons"])


def test_negative_price_blocks_pvm():
    df = make_sales(BASE_A, BASE_A)
    df.loc[0, "price"] = -5.0
    pvm = run(df)["price_volume_mix"]
    assert any("negative price" in r for r in pvm["reasons"])


def test_pvm_with_count_is_refused():
    with pytest.raises(ToolError, match="only works with aggregate='sum'"):
        run(make_sales(BASE_A, BASE_A), aggregate="count")


def test_rows_without_dates_are_reported():
    df = make_sales(BASE_A, BASE_A)
    df.loc[0, "date"] = pd.NaT
    result = run(df)
    assert result["data_quality"]["rows_without_date"] == 1
    assert any("have no date" in w for w in result["warnings"])


def test_failed_reconciliation_is_never_hidden(monkeypatch):
    # Safety net: pretend the allowed tolerance is NEGATIVE, so every check fails.
    # monkeypatch temporarily replaces a function during this one test only.
    import tools.change_analysis as module
    monkeypatch.setattr(module, "reconciliation_tolerance", lambda *totals: -1.0)
    result = run(make_sales(BASE_A, [("X", "North", 20, 5.0), ("Y", "South", 40, 2.0)]))
    assert result["reconciled"] is False
    assert any("RECONCILIATION FAILED" in w for w in result["warnings"])
