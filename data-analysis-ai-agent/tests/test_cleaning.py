"""Tests for the cleaning tools."""

import pandas as pd
import pytest

from tools import (
    ToolError,
    clean_data,
    convert_types,
    handle_missing,
    load_file,
    remove_duplicates,
    standardize_column_names,
    strip_whitespace,
)


def test_original_table_is_never_changed(messy_csv_path):
    df = load_file(messy_csv_path)
    snapshot = df.copy()
    clean_data(df, missing_strategy="drop_rows")
    pd.testing.assert_frame_equal(df, snapshot)  # still identical: golden rule 1


def test_strip_whitespace():
    df = pd.DataFrame({"name": ["  Bob ", "Ann", "   "], "n": [1, 2, 3]})
    result, report = strip_whitespace(df)
    assert result["name"].tolist()[:2] == ["Bob", "Ann"]
    assert pd.isna(result.loc[2, "name"])  # blank text became missing
    assert report["details"]["values_changed_per_column"] == {"name": 2}


def test_standardize_column_names():
    df = pd.DataFrame(columns=[" Unit Price (EUR) ", "Region", "region"])
    result, report = standardize_column_names(df)
    assert list(result.columns) == ["unit_price_eur", "region", "region_2"]
    assert report["details"]["renamed"]["Region"] == "region"


def test_remove_duplicates():
    df = pd.DataFrame({"id": [1, 1, 2], "v": ["a", "a", "b"]})
    result, report = remove_duplicates(df)
    assert len(result) == 2
    assert report["details"]["rows_removed"] == 1


def test_remove_duplicates_by_subset():
    df = pd.DataFrame({"id": [1, 1, 2], "v": ["a", "CHANGED", "b"]})
    result, _ = remove_duplicates(df, subset=["id"])
    assert result["id"].tolist() == [1, 2]


def test_convert_currency_text_to_numbers():
    # 20 good values + 1 bad one = 95% convertible: above the 90% threshold.
    df = pd.DataFrame({"price": ["€1,200.50", "€ 35.00", "$7", "£2"] * 5 + ["unknown"]})
    result, report = convert_types(df)
    assert result["price"].tolist()[:4] == [1200.50, 35.0, 7.0, 2.0]
    assert pd.isna(result.loc[20, "price"])  # "unknown" -> missing...
    info = report["details"]["converted_columns"]["price"]
    assert info["values_made_missing"] == 1          # ...and it is REPORTED
    assert info["problem_values"] == ["unknown"]


def test_too_many_bad_values_means_no_conversion():
    # Only 80% convertible (below 90%): the tool refuses rather than
    # turning lots of values into missing.
    df = pd.DataFrame({"price": ["€10", "€20", "€30", "€40", "n/a"]})
    result, report = convert_types(df)
    assert result["price"].tolist() == ["€10", "€20", "€30", "€40", "n/a"]
    assert report["details"]["converted_columns"] == {}


def test_partly_date_like_column_is_not_converted():
    df = pd.DataFrame({"code": ["2026-01-05", "A-17", "B-22", "C-01"]})
    result, _ = convert_types(df)
    assert result["code"].tolist() == ["2026-01-05", "A-17", "B-22", "C-01"]


def test_european_decimal_comma_is_not_misread():
    # "3,50" must NOT silently become 350. Since most values fail, the
    # column stays as text instead of being converted wrongly.
    df = pd.DataFrame({"price": ["3,50", "2,99", "10,00"]})
    result, report = convert_types(df)
    assert result["price"].tolist() == ["3,50", "2,99", "10,00"]
    assert report["details"]["converted_columns"] == {}


def test_real_text_column_is_not_converted():
    df = pd.DataFrame({"city": ["Berlin", "Munich", "May", "Hamburg"]})
    result, report = convert_types(df)
    assert result["city"].tolist() == ["Berlin", "Munich", "May", "Hamburg"]


def test_convert_dates_including_mixed_formats():
    df = pd.DataFrame({"d": ["2026-01-05", "2026/01/15", "2026-02-01"]})
    result, report = convert_types(df)
    assert pd.api.types.is_datetime64_any_dtype(result["d"])
    assert result.loc[1, "d"] == pd.Timestamp("2026-01-15")


def test_dayfirst_dates():
    df = pd.DataFrame({"d": ["05/01/2026", "20/01/2026"]})
    result, _ = convert_types(df, dayfirst=True)
    assert result.loc[0, "d"] == pd.Timestamp("2026-01-05")  # 5 January, not 1 May


@pytest.mark.parametrize(
    # parametrize = run the same test several times with different inputs.
    "strategy, expected_scores",
    [
        ("fill_mean", [10.0, 140 / 3, 30.0, 100.0]),
        ("fill_median", [10.0, 30.0, 30.0, 100.0]),
        ("fill_value", [10.0, 0.0, 30.0, 100.0]),
    ],
)
def test_fill_strategies(df_with_missing, strategy, expected_scores):
    result, _ = handle_missing(df_with_missing, strategy, columns=["score"], fill_value=0)
    assert result["score"].tolist() == pytest.approx(expected_scores)


def test_fill_mode_works_for_text(df_with_missing):
    result, _ = handle_missing(df_with_missing, "fill_mode", columns=["city"])
    assert result["city"].tolist() == ["Berlin", "Berlin", "Berlin", "Munich"]


def test_fill_mean_skips_text_columns(df_with_missing):
    result, report = handle_missing(df_with_missing, "fill_mean")
    assert "city" in report["details"]["skipped"]
    assert result["score"].isna().sum() == 0


def test_drop_rows(df_with_missing):
    result, report = handle_missing(df_with_missing, "drop_rows")
    assert len(result) == 3
    assert report["rows_after"] == 3


def test_bad_strategy_and_bad_column(df_with_missing):
    with pytest.raises(ToolError, match="Unknown strategy"):
        handle_missing(df_with_missing, "magic")
    with pytest.raises(ToolError, match="Available columns"):
        handle_missing(df_with_missing, "fill_mean", columns=["scroe"])


def test_full_cleaning_of_messy_file(messy_csv_path):
    df = load_file(messy_csv_path)
    cleaned, log = clean_data(df)

    # 12 rows -> 11: the hidden duplicate order 1002 ("  Bob Jones " vs
    # "Bob Jones") is found only AFTER whitespace was stripped.
    assert len(cleaned) == 11
    assert cleaned["order_id"].is_unique
    assert pd.api.types.is_numeric_dtype(cleaned["unit_price"])
    assert pd.api.types.is_datetime64_any_dtype(cleaned["order_date"])
    assert cleaned.loc[0, "unit_price"] == 1200.50
    assert [step["step"] for step in log] == ["strip_whitespace", "convert_types", "remove_duplicates"]


def test_unknown_step():
    with pytest.raises(ToolError, match="Unknown cleaning step"):
        clean_data(pd.DataFrame({"a": [1]}), steps=["make_it_nice"])
