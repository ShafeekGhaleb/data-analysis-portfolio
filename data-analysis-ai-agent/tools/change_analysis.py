"""
Tool 6: explain_change. Explains WHERE a metric changed between two periods.

Question it answers: "Revenue went from A to B. Which segments (regions,
products, channels, ...) made up that change, and was it driven by selling
more units, by prices, or by a different product mix?"

=====================================================================
PART 1: CONTRIBUTION ANALYSIS (works for any additive metric)
=====================================================================

For an ADDITIVE metric (a total built by adding rows: revenue, units,
number of orders), the total is the sum of its groups:

    Total_A = sum over groups g of  Value_A(g)
    Total_B = sum over groups g of  Value_B(g)

So the change splits EXACTLY into one piece per group:

    Change = Total_B - Total_A = sum over g of ( Value_B(g) - Value_A(g) )
                                                 |___ contribution(g) ___|

    share_of_change(g) = 100 * contribution(g) / Change

Shares can be above 100% or negative: if North grows +120 and East falls
-20, the total change is +100, so North = 120% and East = -20%.

Averages and ratios (average price, conversion rate) are NOT additive:
the average of a whole is not the sum of the groups' averages. That is why
this tool only accepts aggregate="sum" or "count".

=====================================================================
PART 2: PRICE / VOLUME / MIX (needs units and an item column)
=====================================================================

Notation, for each item i (e.g. each product) and period A or B:
    U(i) = units sold        R(i) = revenue        P(i) = R(i) / U(i)  (average price)

Items are split into three sets:
    C = continuing items: sold in BOTH periods
    N = new items:        sold only in B
    D = discontinued:     sold only in A

For the continuing items C:
    U_A = sum of U_A(i)                  total units in A
    U_B = sum of U_B(i)                  total units in B
    avg_A = (sum of R_A(i)) / U_A        average price per unit in A

    VOLUME = (U_B - U_A) * avg_A
        "What if we had sold the new number of units, at A's average price?"

    MIX    = sum[ U_B(i) * P_A(i) ]  -  U_B * avg_A
        "With B's units but A's prices: how much does the BLEND of items
         alone change revenue?" Positive = customers moved to pricier items.

    PRICE  = sum[ U_B(i) * ( P_B(i) - P_A(i) ) ]
        "How much did each item's own price change earn, on B's units?"

    NEW ITEMS          = + revenue in B of items in N
    DISCONTINUED ITEMS = - revenue in A of items in D

Why these add up EXACTLY to the total change (the algebra):
    VOLUME + MIX  = sum[ U_B(i) * P_A(i) ] - U_A * avg_A
                  = sum[ U_B(i) * P_A(i) ] - R_A(C)
    + PRICE       = sum[ U_B(i) * P_B(i) ] - R_A(C)  =  R_B(C) - R_A(C)
    + NEW - DISC  = R_B(all) - R_A(all)              =  total change   ✓

Using P(i) = R(i) / U(i) (instead of a listed price) guarantees R = U * P
for every item, so the identity holds even when an item's price varied
within a period. The listed price column is used only to CHECK that the
units and revenue columns are consistent.

=====================================================================
PART 3: RECONCILIATION TOLERANCE
=====================================================================

Computers store decimals approximately: 0.1 + 0.2 = 0.30000000000000004.
Adding many numbers in a different order can differ by tiny amounts. So
"adds up exactly" is checked as:

    |sum of parts - total change|  <=  max(1e-6, 1e-9 * scale)

where scale = the largest of |Total_A|, |Total_B| and 1. In plain words:
differences below one millionth of a unit (or a billionth of the totals,
for huge numbers) are rounding noise; anything bigger is a real error.
Checks run on the UNROUNDED numbers. The returned numbers are rounded to
4 decimals for readability, so adding the rounded values by hand can be
off by about 0.0001 per value.

=====================================================================
IMPORTANT: CONTRIBUTION IS NOT CAUSATION
=====================================================================
"East contributed -628.50" is a calculated fact about WHERE the change
happened. It does not say WHY. Every result carries this reminder.
"""

from itertools import combinations

import pandas as pd

from tools.errors import ToolError
from tools.inspection import column_kind
from tools.periods import check_date_column, filter_period, parse_period, period_coverage, periods_overlap
from tools.utils import check_columns_exist, check_not_empty, to_python

# --- Documented thresholds and limits (one place, easy to review) ---
ABS_TOLERANCE = 1e-6          # reconciliation: absolute rounding noise allowed
REL_TOLERANCE = 1e-9          # reconciliation: relative rounding noise allowed
SMALL_SAMPLE_ROWS = 5         # fewer rows than this = "small sample" warning
MIN_DIMENSION_GROUPS = 2      # auto-chosen dimensions need 2..20 groups
MAX_DIMENSION_GROUPS = 20
MAX_ID_LIKE_RATIO = 0.5       # >50% unique values = an ID or a name, not a segment
MAX_COMBINATION_DIMENSIONS = 4  # pairs are built from at most the first 4 dimensions
PRICE_REL_TOLERANCE = 0.01    # a row is consistent if |R - U*P| <= 1% of R ...
PRICE_ABS_TOLERANCE = 0.01    # ... or <= 0.01 (one cent), whichever is larger
MIN_CONSISTENT_SHARE = 0.95   # at least 95% of rows must be consistent
MAX_LISTED_GROUPS = 10        # longest list of names we return (new groups, ...)
ADDITIVE_AGGREGATES = ("sum", "count")
MISSING_LABEL = "(missing)"

CAUSATION_NOTE = (
    "Contributions show WHERE the change happened, not WHY. They are calculated "
    "facts, not proven business causes. Possible causes must be presented as "
    "hypotheses to investigate."
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def reconciliation_tolerance(*totals) -> float:
    """The allowed rounding noise for numbers of this size (see PART 3 above)."""
    scale = max([abs(float(t)) for t in totals] + [1.0])
    return max(ABS_TOLERANCE, REL_TOLERANCE * scale)


def _check(name: str, parts_sum: float, expected: float, tolerance: float) -> dict:
    """One reconciliation check: do the parts add up to the expected total?"""
    difference = float(parts_sum) - float(expected)
    return {"check": name, "difference": difference, "passed": abs(difference) <= tolerance}


def _total(frame: pd.DataFrame, metric: str | None, aggregate: str) -> float:
    """Sum of the metric (missing values skipped), or the number of rows for "count"."""
    if aggregate == "count":
        return float(len(frame))
    return float(frame[metric].sum())


def _labels(frame: pd.DataFrame, dims: list) -> pd.DataFrame:
    """Group labels as text; missing values get a visible label instead of vanishing."""
    labels = frame[dims].astype(object)
    return labels.where(labels.notna(), MISSING_LABEL)


def is_nested(df: pd.DataFrame, dim_1: str, dim_2: str) -> bool:
    """
    True if one column is fully determined by the other (a hierarchy).

    Example: each Product belongs to exactly one Category, so knowing the
    product tells you the category. Combining them adds no information.
    groupby(x)[y].nunique() counts how many different y values each x has;
    if that is never more than 1, y depends entirely on x.
    """
    labels = _labels(df, [dim_1, dim_2])
    one_way = labels.groupby(dim_1)[dim_2].nunique().max() <= 1
    other_way = labels.groupby(dim_2)[dim_1].nunique().max() <= 1
    return bool(one_way or other_way)


def auto_select_dimensions(df: pd.DataFrame, exclude: list) -> list:
    """
    Pick the columns that look like useful segments.

    Limitation: on TINY datasets (a handful of rows), real segments can look
    ID-like (3 products in 4 rows = 75% unique) and are skipped. In that
    case, pass `dimensions` explicitly.

    A good dimension is a text/boolean column with 2 to 20 different values
    that is not ID-like. Example: "Region" (4 values in 30 rows) is a
    segment; "customer" (11 values in 11 rows) is an identifier: grouping
    by it explains nothing.
    """
    chosen = []
    for name in df.columns:
        if name in exclude or column_kind(df[name]) not in ("text", "boolean"):
            continue
        n_unique = df[name].nunique()
        n_present = int(df[name].notna().sum())
        if not MIN_DIMENSION_GROUPS <= n_unique <= MAX_DIMENSION_GROUPS:
            continue
        if n_present and n_unique / n_present > MAX_ID_LIKE_RATIO:
            continue
        chosen.append(name)
    return chosen


# ---------------------------------------------------------------------------
# Contributions per dimension
# ---------------------------------------------------------------------------

def contributions_by(a: pd.DataFrame, b: pd.DataFrame, dims: list, metric: str | None,
                     aggregate: str, total_change: float, tolerance: float, top_n: int) -> dict:
    """
    Split the total change into one contribution per group.

    dims can be one column (["Region"]) or two (["Region", "Channel"]) for
    combinations like "East / Online".
    """
    tables = {}
    for name, frame in (("a", a), ("b", b)):
        labels = _labels(frame, dims)
        keys = [labels[d] for d in dims]  # group by these label columns
        if aggregate == "count":
            values = frame.groupby(keys).size().astype(float)
        else:
            # sum() skips missing values; a group of only-missing values sums to 0.
            values = frame[metric].groupby(keys).sum()
        tables[name] = values
        tables[f"rows_{name}"] = frame.groupby(keys).size()

    # Put A and B side by side. A group missing from one period gets 0 there.
    table = pd.concat(tables, axis=1).fillna(0)
    table["change"] = table["b"] - table["a"]
    table["abs_change"] = table["change"].abs()
    # Biggest movement first. "mergesort" is a STABLE sort: ties keep their
    # (alphabetical) order, so results are identical every time we run.
    table = table.sort_values("abs_change", ascending=False, kind="mergesort")

    shares_meaningful = abs(total_change) > tolerance
    entries = []
    for key, row in table.iterrows():
        key = key if isinstance(key, tuple) else (key,)
        rows_a, rows_b = int(row["rows_a"]), int(row["rows_b"])
        status = "new" if rows_a == 0 else "discontinued" if rows_b == 0 else "existing"
        entry = {"group": " / ".join(str(k) for k in key)}
        if len(dims) > 1:
            entry.update({str(d): str(k) for d, k in zip(dims, key)})
        entry.update({
            "a": to_python(row["a"]), "b": to_python(row["b"]),
            "change": to_python(row["change"]),
            "share_of_change": to_python(100 * row["change"] / total_change, 2) if shares_meaningful else None,
            "rows_a": rows_a, "rows_b": rows_b, "status": status,
            # A group is a small sample if, in a period where it exists, it has < 5 rows.
            "small_sample": any(0 < n < SMALL_SAMPLE_ROWS for n in (rows_a, rows_b)),
        })
        entries.append(entry)

    top, rest = entries[:top_n], entries[top_n:]
    rest_change = float(table["change"].iloc[top_n:].sum())
    others = None
    if rest:
        others = {
            "groups": len(rest),
            "a": to_python(table["a"].iloc[top_n:].sum()),
            "b": to_python(table["b"].iloc[top_n:].sum()),
            "change": to_python(rest_change),
            "share_of_change": to_python(100 * rest_change / total_change, 2) if shares_meaningful else None,
        }

    gross = float(table["abs_change"].sum())  # total movement, ignoring direction
    return {
        "dimensions": [str(d) for d in dims],
        "n_groups": len(entries),
        "contributions": top,
        "others": others,
        "gross_movement": to_python(gross),
        # Concentration: how much of all movement comes from the single biggest group.
        # 1.0 = one group explains everything; low = change spread across many groups.
        "concentration": to_python(table["abs_change"].iloc[0] / gross, 3) if gross > 0 else None,
        "new_groups": [e["group"] for e in entries if e["status"] == "new"][:MAX_LISTED_GROUPS],
        "discontinued_groups": [e["group"] for e in entries if e["status"] == "discontinued"][:MAX_LISTED_GROUPS],
        "small_sample_groups": sum(e["small_sample"] for e in entries),
        "_reconciliation": _check(" x ".join(map(str, dims)), table["change"].sum(), total_change, tolerance),
    }


# ---------------------------------------------------------------------------
# Price / volume / mix
# ---------------------------------------------------------------------------

def _pvm_reliability_problems(periods: dict, metric: str, units: str,
                              item: str, price: str | None) -> list:
    """
    Check that units, revenue (and price) are compatible BEFORE decomposing.

    periods: {"2026-Q1": rows_of_A, "2026-Q2": rows_of_B}
    Returns a list of problems; an empty list means "safe to decompose".
    """
    rows = pd.concat(periods.values())
    problems = []
    for col in [metric, units] + ([price] if price else []):
        if column_kind(rows[col]) != "numeric":
            problems.append(f"Column '{col}' is not numeric.")
    if problems:
        return problems  # the checks below need numbers

    needed = [metric, units, item] + ([price] if price else [])
    n_missing = int(rows[needed].isna().any(axis=1).sum())
    if n_missing:
        problems.append(f"{n_missing} row(s) have a missing value in {needed}. "
                        "Clean or fill them first; dropping them silently would bias the split.")
    if (rows[units] < 0).any():
        problems.append(f"{int((rows[units] < 0).sum())} row(s) have negative units "
                        "(returns?). Units must be positive to define a price per unit.")
    if (rows[metric] < 0).any():
        problems.append(f"{int((rows[metric] < 0).sum())} row(s) have negative {metric} "
                        "(refunds?). Price/volume/mix needs non-negative revenue.")
    if price:
        if (rows[price] < 0).any():
            problems.append(f"{int((rows[price] < 0).sum())} row(s) have a negative price.")
        complete = rows.dropna(subset=[metric, units, price])
        if len(complete):
            expected = complete[units] * complete[price]
            allowed = (complete[metric].abs() * PRICE_REL_TOLERANCE).clip(lower=PRICE_ABS_TOLERANCE)
            consistent = ((complete[metric] - expected).abs() <= allowed).mean()
            if consistent < MIN_CONSISTENT_SHARE:
                problems.append(
                    f"Only {consistent:.0%} of rows satisfy {metric} ≈ {units} × {price} "
                    f"(within 1%); at least {MIN_CONSISTENT_SHARE:.0%} is required. "
                    "The columns may use different units, currencies or include discounts."
                )
    # Within EACH period, an item with revenue must have units (otherwise price = R / 0).
    # Checking both periods together would hide a problem: units sold in A
    # would mask a "revenue but 0 units" item in B. (A test caught exactly that.)
    for period_label, frame in periods.items():
        clean = frame.dropna(subset=[metric, units, item])
        per_item = clean.groupby(clean[item].astype(str))[[metric, units]].sum()
        bad = per_item[(per_item[units] == 0) & (per_item[metric] != 0)]
        if len(bad):
            problems.append(f"In {period_label}, item(s) {list(bad.index[:5])} have revenue but zero "
                            "units, so a price per unit cannot be calculated.")
    return problems


def price_volume_mix(a: pd.DataFrame, b: pd.DataFrame, metric: str, units: str, item: str,
                     price: str | None, expected_change: float, tolerance: float,
                     labels: tuple = ("period A", "period B")) -> dict:
    """
    Decompose the change into volume, mix, price, new and discontinued items (see PART 2).

    expected_change is the REAL total change of the metric. The five effects
    are reconciled against it (not against a subtotal computed here), so if
    any row were ever silently left out, the check would fail.
    """
    problems = _pvm_reliability_problems({labels[0]: a, labels[1]: b}, metric, units, item, price)
    if problems:
        return {
            "status": "not_reliable",
            "reasons": problems,
            "effects": None,
            "explanation": "No price/volume/mix numbers are returned, because with these data "
                           "problems they could be misleading.",
        }

    def per_item(frame):
        grouped = frame.groupby(frame[item].astype(str))[[metric, units]].sum()
        return grouped[grouped[units] > 0]  # an item with 0 units was not really sold

    ga, gb = per_item(a), per_item(b)
    continuing = ga.index.intersection(gb.index)
    new = gb.index.difference(ga.index)
    discontinued = ga.index.difference(gb.index)

    units_a_c = ga.loc[continuing, units].sum()
    units_b_c = gb.loc[continuing, units].sum()
    notes = []
    if units_a_c > 0:
        price_a = ga[metric] / ga[units]   # P_A(i) for every item
        price_b = gb[metric] / gb[units]   # P_B(i)
        avg_a = ga.loc[continuing, metric].sum() / units_a_c
        volume = (units_b_c - units_a_c) * avg_a
        mix = (gb.loc[continuing, units] * price_a[continuing]).sum() - units_b_c * avg_a
        price_effect = (gb.loc[continuing, units] * (price_b[continuing] - price_a[continuing])).sum()
    else:
        price_a = price_b = None
        volume = mix = price_effect = 0.0
        notes.append("No item was sold in both periods, so the whole change is new/discontinued items.")

    new_effect = gb.loc[new, metric].sum()
    discontinued_effect = -ga.loc[discontinued, metric].sum()
    effects = {"volume": volume, "mix": mix, "price": price_effect,
               "new_items": new_effect, "discontinued_items": discontinued_effect}

    check = _check("price_volume_mix", sum(effects.values()), expected_change, tolerance)

    items = []
    for name in sorted(ga.index.union(gb.index)):
        status = "continuing" if name in continuing else "new" if name in new else "discontinued"
        pa = to_python(ga.loc[name, metric] / ga.loc[name, units]) if name in ga.index else None
        pb = to_python(gb.loc[name, metric] / gb.loc[name, units]) if name in gb.index else None
        items.append({
            "item": name, "status": status,
            "units_a": to_python(ga.loc[name, units]) if name in ga.index else 0,
            "units_b": to_python(gb.loc[name, units]) if name in gb.index else 0,
            "avg_price_a": pa, "avg_price_b": pb,
            "price_change_percent": to_python(100 * (pb / pa - 1), 2) if pa and pb else None,
        })

    units_a_all, units_b_all = ga[units].sum(), gb[units].sum()
    return {
        "status": "ok",
        "item_column": item,
        "effects": {k: to_python(v) for k, v in effects.items()},
        "total_change": to_python(expected_change),
        "units_a": to_python(units_a_all),
        "units_b": to_python(units_b_all),
        "units_change_percent": to_python(100 * (units_b_all / units_a_all - 1), 2) if units_a_all else None,
        "average_price_a": to_python(ga[metric].sum() / units_a_all) if units_a_all else None,
        "average_price_b": to_python(gb[metric].sum() / units_b_all) if units_b_all else None,
        "items": items[:MAX_DIMENSION_GROUPS],
        "counts": {"continuing": len(continuing), "new": len(new), "discontinued": len(discontinued)},
        "notes": notes,
        "_reconciliation": check,
    }


# ---------------------------------------------------------------------------
# The main tool
# ---------------------------------------------------------------------------

def explain_change(df: pd.DataFrame, metric: str | None, date_column: str, period_a, period_b,
                   dimensions: list | None = None, aggregate: str = "sum",
                   units_column: str | None = None, item_column: str | None = None,
                   price_column: str | None = None, top_n: int = 5) -> dict:
    """
    Explain how `metric` changed from period_a to period_b.

    Example:
        explain_change(df, metric="Revenue_EUR", date_column="Date",
                       period_a="2026-Q1", period_b="2026-Q2",
                       units_column="Units", item_column="Product",
                       price_column="Unit_Price_EUR")

    dimensions:   columns to break the change down by (chosen automatically if empty)
    aggregate:    "sum" (add up the metric) or "count" (count rows; metric may be None)
    units_column, item_column (+ optional price_column): turn on price/volume/mix
    top_n:        how many groups to list per dimension (the rest become "others")
    """
    # ---------- 1. Validate the request ----------
    check_not_empty(df)
    if aggregate not in ADDITIVE_AGGREGATES:
        raise ToolError(
            f"aggregate='{aggregate}' is not additive: a total's {aggregate} is not the sum of "
            "its groups, so it cannot be split into contributions. Use 'sum' or 'count', or "
            "compare the two periods side by side instead."
        )
    if aggregate == "sum":
        if not metric:
            raise ToolError("aggregate='sum' needs a metric column, e.g. 'Revenue_EUR'.")
        check_columns_exist(df, metric)
        if column_kind(df[metric]) != "numeric":
            raise ToolError(f"Metric '{metric}' is {column_kind(df[metric])}, not numeric. "
                            "Convert it with clean_data, or use aggregate='count'.")
    check_date_column(df, date_column)
    if not isinstance(top_n, int) or not 1 <= top_n <= MAX_DIMENSION_GROUPS:
        raise ToolError(f"top_n must be a whole number from 1 to {MAX_DIMENSION_GROUPS}.")

    pa, pb = parse_period(period_a), parse_period(period_b)
    if periods_overlap(pa, pb):
        raise ToolError(f"Periods {pa.label} and {pb.label} overlap; a row would be counted in both.")

    pvm_requested = bool(units_column or item_column or price_column)
    if pvm_requested:
        if not (units_column and item_column):
            raise ToolError("Price/volume/mix needs BOTH units_column and item_column "
                            "(the column that identifies products); price_column is optional.")
        if aggregate != "sum":
            raise ToolError("Price/volume/mix only works with aggregate='sum' of a revenue metric.")
        check_columns_exist(df, [c for c in (units_column, item_column, price_column) if c])

    special = {c for c in (metric, date_column, units_column, price_column) if c}
    auto_chosen = dimensions is None
    if auto_chosen:
        dimensions = auto_select_dimensions(df, exclude=list(special))
    else:
        check_columns_exist(df, dimensions)
        clash = special.intersection(dimensions)
        if clash:
            raise ToolError(f"{sorted(clash)} cannot be a dimension (it is the metric, date, units or price).")

    # ---------- 2. Split the data into the two periods ----------
    a = filter_period(df, date_column, pa)
    b = filter_period(df, date_column, pb)
    dates = df[date_column].dropna()
    for period, rows in ((pa, a), (pb, b)):
        if rows.empty:
            raise ToolError(
                f"Period {period.label} ({period.start.date()} to {period.last_day.date()}) has no rows. "
                f"The data covers {dates.min().date()} to {dates.max().date()}."
            )

    # ---------- 3. Totals ----------
    total_a, total_b = _total(a, metric, aggregate), _total(b, metric, aggregate)
    change = total_b - total_a
    tolerance = reconciliation_tolerance(total_a, total_b)
    warnings = []

    # ---------- 4. Data quality ----------
    quality = {"rows_without_date": int(df[date_column].isna().sum())}
    if quality["rows_without_date"]:
        warnings.append(f"{quality['rows_without_date']} row(s) have no date and are excluded from both periods.")
    for key, period, rows in (("period_a", pa, a), ("period_b", pb, b)):
        coverage = period_coverage(df, date_column, period)
        missing_metric = int(rows[metric].isna().sum()) if aggregate == "sum" else 0
        negative = int((rows[metric] < 0).sum()) if aggregate == "sum" else 0
        quality[key] = {**coverage, "missing_metric_values": missing_metric, "negative_metric_values": negative}
        warnings.extend(coverage["issues"])
        if len(rows) < SMALL_SAMPLE_ROWS:
            warnings.append(f"{period.label} has only {len(rows)} row(s): results are fragile.")
        if missing_metric:
            warnings.append(f"{period.label}: {missing_metric} row(s) have no {metric} value; "
                            "they are left out of the totals.")
        if negative:
            warnings.append(f"{period.label}: {negative} row(s) have a negative {metric} "
                            "(returns or refunds?). They are included and reduce the totals.")
    if abs(change) <= tolerance:
        warnings.append("The total change is (almost) zero, so 'share of change' is not meaningful; "
                        "look at the offsetting gains and losses instead.")
    if total_a == 0:
        warnings.append(f"{pa.label} total is 0, so the percentage change is undefined.")

    # ---------- 5. Contributions per dimension ----------
    drivers, checks = {}, []
    for dim in dimensions:
        result = contributions_by(a, b, [dim], metric, aggregate, change, tolerance, top_n)
        checks.append(result.pop("_reconciliation"))
        drivers[str(dim)] = result
        n_missing = int(pd.concat([a[dim], b[dim]]).isna().sum())
        if n_missing:
            warnings.append(f"{dim}: {n_missing} row(s) have no value; they are grouped as '{MISSING_LABEL}'.")
        if result["new_groups"]:
            warnings.append(f"{dim}: new in {pb.label} (no rows in {pa.label}): {result['new_groups']}.")
        if result["discontinued_groups"]:
            warnings.append(f"{dim}: no rows in {pb.label} (discontinued?): {result['discontinued_groups']}.")
        if result["small_sample_groups"]:
            warnings.append(
                f"{dim}: {result['small_sample_groups']} of {result['n_groups']} groups have fewer than "
                f"{SMALL_SAMPLE_ROWS} rows in a period; treat group-level results as indicative only."
            )

    # Rank dimensions: the one where the change is most concentrated tells the clearest story.
    ranking = sorted(drivers, key=lambda d: drivers[d]["concentration"] or 0, reverse=True)

    # ---------- 6. Two-dimension combinations (e.g. Region x Channel) ----------
    combos, skipped_pairs = [], []
    for pair in combinations(dimensions[:MAX_COMBINATION_DIMENSIONS], 2):
        if is_nested(df, *pair):
            # e.g. Category x Product: every product has ONE category, so
            # "Electronics / Keyboard" would only repeat "Keyboard".
            skipped_pairs.append(" x ".join(map(str, pair)))
            continue
        result = contributions_by(a, b, list(pair), metric, aggregate, change, tolerance, top_n)
        checks.append(result["_reconciliation"])
        combos.extend(result["contributions"])
    combos.sort(key=lambda e: abs(e["change"]), reverse=True)

    # ---------- 7. Price / volume / mix ----------
    pvm = None
    if pvm_requested:
        pvm = price_volume_mix(a, b, metric, units_column, item_column, price_column,
                               change, tolerance, labels=(pa.label, pb.label))
        if pvm["status"] == "ok":
            checks.append(pvm.pop("_reconciliation"))
        else:
            warnings.append("Price/volume/mix was NOT calculated: " + " ".join(pvm["reasons"]))

    all_passed = all(c["passed"] for c in checks)
    if not all_passed:
        warnings.append("RECONCILIATION FAILED: the parts do not add up to the total. Do not use this breakdown.")

    return {
        "metric": metric,
        "aggregate": aggregate,
        "period_a": pa.to_dict(),
        "period_b": pb.to_dict(),
        "summary": {
            "period_a_total": to_python(total_a),
            "period_b_total": to_python(total_b),
            "change": to_python(change),
            "change_percent": to_python(100 * change / total_a, 2) if total_a else None,
            "rows_a": len(a),
            "rows_b": len(b),
        },
        "dimensions": {"used": [str(d) for d in dimensions], "chosen_automatically": auto_chosen},
        "drivers": drivers,
        "dimension_ranking": ranking,
        "top_combinations": combos[:top_n],
        "skipped_nested_pairs": skipped_pairs,
        "price_volume_mix": pvm,
        "reconciliation": {
            "tolerance": tolerance,
            "checks": [{**c, "difference": to_python(c["difference"], 12)} for c in checks],
            "all_passed": all_passed,
        },
        "reconciled": all_passed,
        "data_quality": quality,
        "warnings": warnings,
        "causation_note": CAUSATION_NOTE,
    }
