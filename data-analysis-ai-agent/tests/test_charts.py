"""
Tests for create_chart.

A test cannot "look" at a picture the way a person does, so we test what we can:
1. A real PNG file is created and is readable as an image.
2. The NUMBERS behind the chart (the returned "data") are correct.
3. Bad requests give clear ToolErrors.
4. No figures are left open in memory.
(Looking at the charts with our own eyes is still part of the job: see the
example charts produced by examples/stage2_demo.py.)
"""

import json

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from tools import ToolError, clean_data, correlation, create_chart, group_and_aggregate, load_file

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"  # the first 8 bytes of EVERY PNG file


@pytest.fixture
def sales_df(sales_csv_path):
    """The sales data, cleaned so that 'Date' is a real date column."""
    cleaned, _ = clean_data(load_file(sales_csv_path))
    return cleaned


def assert_valid_png(path):
    """Check that a file exists, really is a PNG, and has a sensible size."""
    with open(path, "rb") as file:            # "rb" = read raw bytes
        assert file.read(8) == PNG_SIGNATURE
    image = mpimg.imread(path)                 # load the picture as an array of pixels
    height, width = image.shape[:2]
    assert width > 500 and height > 300


# ---------- Every chart type works ----------

@pytest.mark.parametrize("request_args", [
    {"chart_type": "bar", "x": "Region", "y": "Revenue_EUR"},
    {"chart_type": "line", "x": "Date", "y": "Revenue_EUR", "period": "month"},
    {"chart_type": "pie", "x": "Channel", "y": "Revenue_EUR"},
    {"chart_type": "histogram", "x": "Revenue_EUR"},
    {"chart_type": "scatter", "x": "Unit_Price_EUR", "y": "Revenue_EUR"},
])
def test_every_chart_type_creates_a_png(sales_df, tmp_path, request_args):
    result = create_chart(sales_df, output_dir=tmp_path, **request_args)
    assert_valid_png(result["path"])
    assert result["chart_type"] == request_args["chart_type"]
    assert result["title"]                           # a default title was created
    json.dumps(result)                               # result must be JSON-safe (for the LLM later)
    assert plt.get_fignums() == []                   # no figure left open in memory


# ---------- Bar ----------

def test_bar_numbers_match_statistics_tool(sales_df, tmp_path):
    result = create_chart(sales_df, "bar", x="Region", y="Revenue_EUR", output_dir=tmp_path)
    expected = group_and_aggregate(sales_df, "Region", "Revenue_EUR", "sum")["rows"]
    assert result["data"] == expected                # chart and statistics agree exactly
    assert result["data"][0] == {"Region": "North", "value": 8427.8, "percent_of_total": 28.93}
    assert result["orientation"] == "vertical"


def test_bar_long_names_become_horizontal(sales_df, tmp_path):
    result = create_chart(sales_df, "bar", x="Product", y="Revenue_EUR", output_dir=tmp_path)
    assert result["orientation"] == "horizontal"     # "Wireless Mouse" is too long for vertical labels


def test_bar_top_n_and_count(sales_df, tmp_path):
    result = create_chart(sales_df, "bar", x="Product", top_n=2, output_dir=tmp_path)
    assert result["aggregate"] == "count"            # no y -> count rows
    assert result["groups_shown"] == 2 and result["groups_total"] == 6


def test_bar_mean(small_df, tmp_path):
    result = create_chart(small_df, "bar", x="region", y="revenue", aggregate="mean", output_dir=tmp_path)
    assert {r["region"]: r["value"] for r in result["data"]} == {"South": 350.0, "East": 500.0, "North": 150.0}


# ---------- Line ----------

def test_line_monthly_totals(sales_df, tmp_path):
    result = create_chart(sales_df, "line", x="Date", y="Revenue_EUR", period="month", output_dir=tmp_path)
    assert result["n_points"] == 6                   # January to June
    first = result["data"][0]
    assert first["x"].startswith("2026-01-01")
    # January rows: 462.5 + 256 + 897 + 539 + 952 = 3106.5 (checked by hand)
    assert first["value"] == pytest.approx(3106.5)
    total = sum(point["value"] for point in result["data"])
    assert total == pytest.approx(29130.7)           # nothing lost or counted twice


def test_line_without_period_uses_every_date(sales_df, tmp_path):
    result = create_chart(sales_df, "line", x="Date", y="Revenue_EUR", output_dir=tmp_path)
    assert result["n_points"] == sales_df["Date"].nunique()
    xs = [point["x"] for point in result["data"]]
    assert xs == sorted(xs)                          # in time order


def test_line_errors(sales_df, small_df, tmp_path):
    with pytest.raises(ToolError, match="convert it first"):
        create_chart(sales_df, "line", x="Region", y="Revenue_EUR", output_dir=tmp_path)
    with pytest.raises(ToolError, match="only works with dates"):
        create_chart(small_df, "line", x="units", y="revenue", period="month", output_dir=tmp_path)
    with pytest.raises(ToolError, match="Unknown period"):
        create_chart(sales_df, "line", x="Date", y="Revenue_EUR", period="decade", output_dir=tmp_path)


# ---------- Pie ----------

def test_pie_shares(sales_df, tmp_path):
    result = create_chart(sales_df, "pie", x="Channel", y="Revenue_EUR", output_dir=tmp_path)
    shares = {r["Channel"]: r["percent_of_total"] for r in result["data"]}
    assert shares == {"Online": 68.92, "Store": 31.08}
    assert result["total"] == pytest.approx(29130.7)


def test_pie_folds_small_slices_into_other(tmp_path):
    # 8 categories -> the 5 biggest + "Other" (6 slices maximum).
    df = pd.DataFrame({"letter": list("ABCDEFGH"), "amount": [80, 70, 60, 50, 40, 30, 20, 10]})
    result = create_chart(df, "pie", x="letter", y="amount", output_dir=tmp_path)
    assert len(result["data"]) == 6
    other = result["data"][-1]
    assert other["letter"] == "Other"
    assert other["value"] == 30 + 20 + 10
    assert other["grouped_categories"] == 3
    assert sum(r["percent_of_total"] for r in result["data"]) == pytest.approx(100, abs=0.05)


def test_pie_errors(small_df, tmp_path):
    with pytest.raises(ToolError, match="'sum' or 'count'"):
        create_chart(small_df, "pie", x="region", y="revenue", aggregate="mean", output_dir=tmp_path)
    negative = pd.DataFrame({"k": ["a", "b"], "v": [5, -3]})
    with pytest.raises(ToolError, match="negative"):
        create_chart(negative, "pie", x="k", y="v", output_dir=tmp_path)


# ---------- Histogram ----------

def test_histogram_counts_every_value(sales_df, tmp_path):
    result = create_chart(sales_df, "histogram", x="Revenue_EUR", bins=8, output_dir=tmp_path)
    assert len(result["counts"]) == 8
    assert len(result["bin_edges"]) == 9             # 8 bars have 9 boundaries
    assert sum(result["counts"]) == 30               # every row lands in exactly one bin
    assert result["median"] == 924.5
    assert result["min"] == 256.0 and result["max"] == 2142.0


def test_histogram_errors(sales_df, tmp_path):
    with pytest.raises(ToolError, match="numeric column"):
        create_chart(sales_df, "histogram", x="Region", output_dir=tmp_path)
    with pytest.raises(ToolError, match="bins"):
        create_chart(sales_df, "histogram", x="Units", bins=1, output_dir=tmp_path)


# ---------- Scatter ----------

def test_scatter_correlation_matches_statistics_tool(sales_df, tmp_path):
    result = create_chart(sales_df, "scatter", x="Unit_Price_EUR", y="Revenue_EUR", output_dir=tmp_path)
    expected = correlation(sales_df, ["Unit_Price_EUR", "Revenue_EUR"])["pairs"][0]
    assert result["pearson_r"] == expected["r"] == 0.732
    assert result["interpretation"] == "strong positive"
    assert result["n_points"] == 30


def test_scatter_color_by_limits_groups(tmp_path):
    df = pd.DataFrame({
        "x": range(10), "y": range(10),
        "team": ["A", "A", "A", "B", "B", "C", "C", "D", "E", "F"],
    })
    result = create_chart(df, "scatter", x="x", y="y", color_by="team", output_dir=tmp_path)
    # The 3 biggest teams keep their names; D, E, F become "Other".
    assert result["groups"] == {"A": 3, "B": 2, "C": 2, "Other": 3}


def test_scatter_errors(sales_df, tmp_path):
    with pytest.raises(ToolError, match="both x and y"):
        create_chart(sales_df, "scatter", x="Units", output_dir=tmp_path)
    with pytest.raises(ToolError, match="numeric columns"):
        create_chart(sales_df, "scatter", x="Region", y="Units", output_dir=tmp_path)


# ---------- General behaviour ----------

def test_general_errors(sales_df, tmp_path):
    with pytest.raises(ToolError, match="Unknown chart_type"):
        create_chart(sales_df, "radar", x="Region", output_dir=tmp_path)
    with pytest.raises(ToolError, match="Available columns"):
        create_chart(sales_df, "bar", x="Regoin", y="Revenue_EUR", output_dir=tmp_path)
    with pytest.raises(ToolError, match="Unknown aggregate"):
        create_chart(sales_df, "bar", x="Region", y="Revenue_EUR", aggregate="total", output_dir=tmp_path)


def test_figure_is_closed_even_after_an_error(sales_df, tmp_path):
    # The error happens AFTER the figure was created; try/finally must still close it.
    with pytest.raises(ToolError):
        create_chart(sales_df, "histogram", x="Region", output_dir=tmp_path)
    assert plt.get_fignums() == []


def test_output_folder_and_file_names(sales_df, tmp_path):
    nested = tmp_path / "reports" / "charts"         # does not exist yet
    result = create_chart(sales_df, "bar", x="Region", y="Revenue_EUR", output_dir=nested,
                          title="Revenue by Region: H1 2026")
    assert result["path"].endswith("revenue-by-region-h1-2026.png")  # safe name made from the title
    assert_valid_png(result["path"])

    custom = create_chart(sales_df, "bar", x="Region", y="Revenue_EUR",
                          output_dir=tmp_path, filename="my_chart.jpg")
    assert custom["path"].endswith("my_chart.png")   # always saved as PNG


def test_missing_category_is_shown_not_hidden(tmp_path):
    df = pd.DataFrame({"city": ["Berlin", None, "Berlin"], "sales": [1, 2, 3]})
    result = create_chart(df, "bar", x="city", y="sales", output_dir=tmp_path)
    assert len(result["data"]) == 2                  # Berlin AND the missing-city group
