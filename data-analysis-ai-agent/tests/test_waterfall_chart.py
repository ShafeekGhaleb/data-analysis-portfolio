"""Tests for create_waterfall_chart (the bridge chart for explain_change results)."""

import copy

import matplotlib.pyplot as plt
import pytest

from tools import ToolError, create_waterfall_chart, explain_change

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def analysis(sales_df):
    return explain_change(sales_df, "Revenue_EUR", "Date", "2026-Q1", "2026-Q2",
                          units_column="Units", item_column="Product",
                          price_column="Unit_Price_EUR", top_n=8)


def is_png(path) -> bool:
    with open(path, "rb") as file:
        return file.read(8) == PNG_SIGNATURE


def test_region_bridge(analysis, tmp_path):
    chart = create_waterfall_chart(analysis, "Region", output_dir=tmp_path)
    assert is_png(chart["path"]) and chart["reconciled"]
    steps = chart["steps"]
    assert [s["label"] for s in steps] == ["2026-Q1", "North", "South", "East", "West", "2026-Q2"]
    assert steps[0]["value"] == 11586.6 and steps[-1]["value"] == 17544.1
    # Each floating bar starts exactly where the previous one ended.
    for previous, current in zip(steps[1:-2], steps[2:-1]):
        assert current["start"] == pytest.approx(previous["end"])
    assert steps[-2]["end"] == pytest.approx(17544.1)     # the last step lands on the Q2 total
    assert steps[3]["kind"] == "decrease"                 # East went down
    assert plt.get_fignums() == []


def test_price_volume_mix_bridge(analysis, tmp_path):
    chart = create_waterfall_chart(analysis, "price_volume_mix", output_dir=tmp_path)
    labels = [s["label"] for s in chart["steps"]]
    # New/discontinued are 0 here, so they are not drawn; volume/mix/price always are.
    assert labels == ["2026-Q1", "Volume", "Mix", "Price", "2026-Q2"]
    assert chart["steps"][1]["value"] == 6293.2294


def test_small_groups_fold_into_other(analysis, tmp_path):
    chart = create_waterfall_chart(analysis, "Product", max_steps=3, output_dir=tmp_path)
    middle = chart["steps"][1:-1]
    assert [s["label"] for s in middle] == ["Keyboard", "Office Chair", "Other (4)"]
    assert sum(s["value"] for s in middle) == pytest.approx(5957.5)


def test_new_and_discontinued_steps_appear_when_present(tmp_path):
    import pandas as pd
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-31", "2026-02-01", "2026-02-28"]),
        "item": ["X", "Y", "X", "Z"], "units": [10, 20, 10, 5], "rev": [50.0, 40.0, 50.0, 50.0],
    })
    result = explain_change(df, "rev", "date", "2026-01", "2026-02", dimensions=["item"],
                            units_column="units", item_column="item")
    chart = create_waterfall_chart(result, "price_volume_mix", output_dir=tmp_path)
    assert [s["label"] for s in chart["steps"]][-3:-1] == ["New items", "Discontinued"]


def test_unreliable_pvm_is_not_drawn(sales_df, tmp_path):
    broken = sales_df.copy()
    broken.loc[0, "Units"] = None                          # makes PVM not reliable
    result = explain_change(broken, "Revenue_EUR", "Date", "2026-Q1", "2026-Q2",
                            units_column="Units", item_column="Product")
    with pytest.raises(ToolError, match="not reliable"):
        create_waterfall_chart(result, "price_volume_mix", output_dir=tmp_path)
    assert plt.get_fignums() == []


def test_tampered_numbers_are_refused(analysis, tmp_path):
    # If the numbers do not lead from total A to total B, refuse to draw a misleading chart.
    tampered = copy.deepcopy(analysis)
    tampered["summary"]["period_b_total"] += 100
    with pytest.raises(ToolError, match="refusing to draw"):
        create_waterfall_chart(tampered, "Region", output_dir=tmp_path)


@pytest.mark.parametrize("breakdown, kwargs, message", [
    ("Regoin", {}, "Unknown breakdown"),
    ("Region", {"max_steps": 1}, "max_steps"),
    ("Region", {"max_steps": 20}, "max_steps"),
])
def test_invalid_requests(analysis, tmp_path, breakdown, kwargs, message):
    with pytest.raises(ToolError, match=message):
        create_waterfall_chart(analysis, breakdown, output_dir=tmp_path, **kwargs)


def test_needs_an_analysis_and_pvm(sales_df, tmp_path):
    with pytest.raises(ToolError, match="needs the result of explain_change"):
        create_waterfall_chart({"rows": []}, "Region", output_dir=tmp_path)
    without_pvm = explain_change(sales_df, "Revenue_EUR", "Date", "2026-Q1", "2026-Q2")
    with pytest.raises(ToolError, match="no price/volume/mix"):
        create_waterfall_chart(without_pvm, "price_volume_mix", output_dir=tmp_path)
