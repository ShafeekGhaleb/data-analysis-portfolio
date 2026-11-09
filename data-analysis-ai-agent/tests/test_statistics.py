"""
Tests for the statistics tools.

The most important tests here compare our results on the real sales data
with the numbers that were checked independently earlier in this project
(total revenue €29,130.70, North €8,427.80, ...).
"""

import pytest

from tools import ToolError, correlation, describe_statistics, group_and_aggregate, load_file


# ---------- describe_statistics ----------

def test_describe_numeric_by_hand(small_df):
    stats = describe_statistics(small_df)["numeric"]["revenue"]
    # revenue = 100, 200, 300, 400, 500 -> easy to check by hand
    assert stats["count"] == 5
    assert stats["mean"] == 300.0
    assert stats["median"] == 300.0
    assert stats["min"] == 100.0 and stats["max"] == 500.0
    assert stats["sum"] == 1500.0
    assert stats["q1"] == 200.0 and stats["q3"] == 400.0
    assert stats["std"] == pytest.approx(158.1139, abs=1e-4)  # sample standard deviation


def test_describe_categorical(small_df):
    stats = describe_statistics(small_df, columns=["region"])
    region = stats["categorical"]["region"]
    assert region["unique_values"] == 3
    assert region["top_values"] == {"North": 2, "South": 2, "East": 1}
    assert stats["numeric"] == {}  # we only asked for one column


def test_describe_real_sales(sales_csv_path):
    stats = describe_statistics(load_file(sales_csv_path))
    assert stats["numeric"]["Revenue_EUR"]["sum"] == pytest.approx(29130.70)
    assert stats["numeric"]["Units"]["sum"] == 1503
    assert stats["numeric"]["Revenue_EUR"]["mean"] == pytest.approx(971.0233)


# ---------- group_and_aggregate ----------

def test_group_sum_with_percentages(small_df):
    result = group_and_aggregate(small_df, "region", "revenue", "sum")
    rows = result["rows"]
    assert [r["region"] for r in rows] == ["South", "East", "North"]  # biggest first
    assert rows[0]["value"] == 700.0
    assert rows[0]["percent_of_total"] == pytest.approx(46.67)
    assert result["total"] == 1500.0


def test_group_mean_has_no_percentages(small_df):
    result = group_and_aggregate(small_df, "region", "revenue", "mean")
    assert "percent_of_total" not in result["rows"][0]
    assert result["total"] is None


def test_group_count_without_value_column(small_df):
    result = group_and_aggregate(small_df, "channel", method="count")
    assert {r["channel"]: r["value"] for r in result["rows"]} == {"Online": 3, "Store": 2}


def test_group_by_two_columns(small_df):
    result = group_and_aggregate(small_df, ["region", "channel"], "revenue", "sum")
    first = result["rows"][0]
    assert first == {"region": "South", "channel": "Online", "value": 700.0, "percent_of_total": 46.67}


def test_top_n_keeps_percentages_of_whole_dataset(small_df):
    result = group_and_aggregate(small_df, "region", "revenue", "sum", top_n=1)
    assert result["n_groups"] == 1
    assert result["rows"][0]["percent_of_total"] == pytest.approx(46.67)


def test_real_sales_matches_portfolio_numbers(sales_csv_path):
    df = load_file(sales_csv_path)
    regions = group_and_aggregate(df, "Region", "Revenue_EUR", "sum")
    assert {r["Region"]: r["value"] for r in regions["rows"]} == pytest.approx(
        {"North": 8427.8, "West": 7748.2, "South": 6771.2, "East": 6183.5}
    )
    channels = group_and_aggregate(df, "Channel", "Revenue_EUR", "sum")
    online = channels["rows"][0]
    assert online["Channel"] == "Online"
    assert online["percent_of_total"] == pytest.approx(68.92)
    top = group_and_aggregate(df, "Product", "Revenue_EUR", "sum", top_n=1)["rows"][0]
    assert top["Product"] == "Office Chair" and top["value"] == pytest.approx(9282.0)


def test_group_errors(small_df):
    with pytest.raises(ToolError, match="Available columns"):
        group_and_aggregate(small_df, "Regoin", "revenue")
    with pytest.raises(ToolError, match="not numeric"):
        group_and_aggregate(small_df, "region", "channel", "sum")
    with pytest.raises(ToolError, match="Unknown method"):
        group_and_aggregate(small_df, "region", "revenue", "average")
    with pytest.raises(ToolError, match="needs a value_column"):
        group_and_aggregate(small_df, "region", method="sum")


# ---------- correlation ----------

def test_perfect_correlation(small_df):
    result = correlation(small_df, columns=["units", "revenue"])
    pair = result["pairs"][0]
    assert pair["r"] == 1.0  # revenue is exactly 10 x units
    assert pair["interpretation"] == "strong positive"


def test_correlation_uses_numeric_columns_and_sorts(small_df):
    result = correlation(small_df)
    assert result["columns"] == ["units", "revenue", "cost"]
    strengths = [abs(p["r"]) for p in result["pairs"]]
    assert strengths == sorted(strengths, reverse=True)
    assert len(result["pairs"]) == 3  # 3 columns -> 3 unique pairs


def test_spearman_and_negative():
    import pandas as pd
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [100, 10, 5, 1]})
    pair = correlation(df, method="spearman")["pairs"][0]
    assert pair["r"] == -1.0
    assert pair["interpretation"] == "strong negative"


def test_correlation_errors(small_df):
    with pytest.raises(ToolError, match="at least 2 numeric"):
        correlation(small_df[["region", "units"]])
    with pytest.raises(ToolError, match="numeric columns"):
        correlation(small_df, columns=["region", "units"])
