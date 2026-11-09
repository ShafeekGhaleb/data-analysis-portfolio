"""
Tool 5: create_chart. Draws a chart with Matplotlib and saves it as a PNG file.

Supported chart types, and the question each one answers:

    bar        "How do categories compare?"         (revenue by region)
    line       "How does something change over time?" (revenue per month)
    pie        "How is a whole split into parts?"   (share of sales by channel)
    histogram  "How are the values distributed?"    (how big are most orders?)
    scatter    "Do two numbers move together?"      (unit price vs revenue)

THREE IMPORTANT DESIGN IDEAS

1. Charts never calculate their own numbers. Bar and pie charts reuse
   group_and_aggregate() from statistics.py, which is already tested.
   So a chart can never disagree with the statistics: same function,
   same numbers.

2. The LLM cannot SEE images. So besides saving the PNG, create_chart
   RETURNS the plotted numbers as a dictionary. Later the agent reads those
   numbers to describe the chart truthfully ("North is highest with 8,427.80").

3. Honest, readable design. Bars start at zero, there are few colours,
   labels are written in text colours (not series colours), and pies have
   at most 6 slices. The colours were checked with a colour-blindness
   validator.
"""

import re
from pathlib import Path

# Matplotlib can draw to a window on screen OR straight to a file. "Agg" is
# the file-only "backend". We need it because the agent runs on servers with
# no screen. It MUST be chosen before pyplot is imported.
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402  (must come after matplotlib.use)
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from tools.errors import ToolError  # noqa: E402
from tools.inspection import column_kind  # noqa: E402
from tools.statistics import AGGREGATIONS, describe_strength, group_and_aggregate  # noqa: E402
from tools.utils import check_columns_exist, check_not_empty, to_python  # noqa: E402

# ---------------------------------------------------------------------------
# Style settings, kept in ONE place so every chart looks consistent
# ---------------------------------------------------------------------------

SURFACE = "#fcfcfb"         # chart background (an off-white)
TEXT_PRIMARY = "#0b0b0b"    # titles
TEXT_SECONDARY = "#52514e"  # axis labels, tick labels, value labels
GRID = "#e4e3df"            # light grey for gridlines and axis lines
OTHER_GREY = "#a3a29c"      # for the "Other" group

# Category colours in a FIXED order: the 1st group always gets blue, the 2nd
# orange, and so on. This order was checked so that neighbouring colours stay
# distinguishable for colour-blind readers.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
PRIMARY = PALETTE[0]  # charts with a single series use blue

CHART_TYPES = ("bar", "line", "pie", "histogram", "scatter")

MAX_BAR_CATEGORIES = 15   # more bars than this become unreadable
MAX_PIE_SLICES = 6        # more slices than this: nobody can compare them
MAX_SCATTER_GROUPS = 3    # more colours than this can't be told apart reliably
MAX_RETURNED_POINTS = 100  # limit on how many data points we send back (for the LLM later)

# Readable time periods for line charts -> pandas "period" codes.
PERIODS = {"day": "D", "week": "W", "month": "M", "quarter": "Q", "year": "Y"}
# How dates are written on the x-axis for each period ("%b" = short month name).
DATE_FORMATS = {"day": "%d %b", "week": "%d %b", "month": "%b %Y", "quarter": "%b %Y", "year": "%Y"}

# Words used in automatic titles, e.g. "Total Revenue_EUR by Region".
AGGREGATE_WORDS = {
    "sum": "Total", "mean": "Average", "median": "Median", "std": "Spread (std) of",
    "min": "Minimum", "max": "Maximum", "count": "Number of rows", "nunique": "Distinct",
}

DPI = 150                 # "dots per inch": image sharpness
FIGURE_SIZE = (8, 4.5)    # inches: 8 x 4.5 -> 1200 x 675 pixels at 150 dpi


# ---------------------------------------------------------------------------
# The main tool
# ---------------------------------------------------------------------------

def create_chart(df: pd.DataFrame, chart_type: str, x: str, y: str | None = None,
                 aggregate: str = "sum", color_by: str | None = None,
                 period: str | None = None, bins: int = 20, top_n: int | None = None,
                 title: str | None = None, output_dir="outputs",
                 filename: str | None = None) -> dict:
    """
    Draw one chart, save it as PNG, and return a description of what was drawn.

    Parameters (which ones matter depends on chart_type):
        x          bar/pie: the category column;  line: the date/number column;
                   histogram: the numeric column; scatter: the horizontal numbers
        y          the numeric value column (bar/line/pie/scatter). For bar/line/pie,
                   leave it out to simply COUNT rows.
        aggregate  how to combine values per group: sum, mean, median, count, ...
        color_by   scatter only: a category column to colour the dots (max 3 groups)
        period     line only, for dates: "day", "week", "month", "quarter", "year"
        bins       histogram only: how many bars to split the range into
        top_n      bar only: show only the N largest groups
        title      chart title (a sensible title is made automatically if empty)
        output_dir folder to save into (created if it doesn't exist)
        filename   file name; made from the title if empty

    Returns a dict such as:
        {"chart_type": "bar", "title": "...", "path": "outputs/....png", "data": [...]}
    """
    # --- 1. Check the request before drawing anything ---
    check_not_empty(df)
    if chart_type not in CHART_TYPES:
        raise ToolError(f"Unknown chart_type '{chart_type}'. Choose one of {list(CHART_TYPES)}")
    # [x, y, color_by] minus the ones that are None -> the columns we need.
    check_columns_exist(df, [c for c in (x, y, color_by) if c])
    if aggregate not in AGGREGATIONS:
        raise ToolError(f"Unknown aggregate '{aggregate}'. Choose one of {list(AGGREGATIONS)}")

    # --- 2. Create an empty figure (the whole image) with one "axes" (the plot area) ---
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    fig.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    # try/finally: whatever happens (even an error), the "finally" part runs
    # and closes the figure. Unclosed figures stay in memory, and an agent
    # making hundreds of charts would slowly fill up the computer's memory.
    try:
        # --- 3. Draw: each chart type has its own small helper function ---
        if chart_type == "bar":
            default_title, data = _draw_bar(ax, df, x, y, aggregate, top_n)
        elif chart_type == "line":
            default_title, data = _draw_line(ax, df, x, y, aggregate, period)
        elif chart_type == "pie":
            default_title, data = _draw_pie(ax, df, x, y, aggregate)
        elif chart_type == "histogram":
            default_title, data = _draw_histogram(ax, df, x, bins)
        else:  # "scatter"
            default_title, data = _draw_scatter(ax, df, x, y, color_by)

        # --- 4. Title, then save the PNG ---
        title = title or default_title
        ax.set_title(title, loc="left", fontsize=13, fontweight="bold",
                     color=TEXT_PRIMARY, pad=14)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)  # create folder(s) if needed
        path = output_dir / (filename or f"{_slugify(title)}.png")
        if path.suffix.lower() != ".png":
            path = path.with_suffix(".png")
        # bbox_inches="tight" trims empty margins around the chart.
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=SURFACE)
    finally:
        plt.close(fig)

    return {"chart_type": chart_type, "title": title, "path": str(path), **data}


# ---------------------------------------------------------------------------
# One drawing helper per chart type.
# Each returns (default_title, data_description_dict).
# ---------------------------------------------------------------------------

def _draw_bar(ax, df, x, y, aggregate, top_n):
    """Bar chart: one bar per category, biggest first."""
    method = aggregate if y else "count"
    # Reuse our tested statistics function: the chart shows EXACTLY these numbers.
    limit = min(top_n or MAX_BAR_CATEGORIES, MAX_BAR_CATEGORIES)
    full = group_and_aggregate(df, x, y, method)
    rows = full["rows"][:limit]
    labels = [_label(r[x]) for r in rows]
    values = [r["value"] if r["value"] is not None else 0 for r in rows]

    # Long or many category names don't fit side by side, so in that case
    # we draw HORIZONTAL bars, where names are easy to read.
    horizontal = len(labels) > 6 or max(len(label) for label in labels) > 12
    if horizontal:
        # barh draws from the bottom up, so reverse the lists to put the biggest on top.
        bars = ax.barh(labels[::-1], values[::-1], height=0.55, color=PRIMARY)
        _style_axes(ax, value_axis="x")
        ax.set_xlabel(_value_label(y, method), color=TEXT_SECONDARY)
    else:
        bars = ax.bar(labels, values, width=0.55, color=PRIMARY)
        _style_axes(ax, value_axis="y")
        ax.set_ylabel(_value_label(y, method), color=TEXT_SECONDARY)
        ax.set_xlabel(x, color=TEXT_SECONDARY)

    # Write each value at the end of its bar (bar_label is built into Matplotlib).
    ax.bar_label(bars, labels=[_format_number(v) for v in (values[::-1] if horizontal else values)],
                 padding=3, color=TEXT_SECONDARY, fontsize=9)
    ax.margins(x=0.08) if horizontal else ax.margins(y=0.1)  # room for those labels

    title = f"{AGGREGATE_WORDS[method]} {y or ''} by {x}".replace("  ", " ")
    data = {"x": x, "y": y, "aggregate": method, "orientation": "horizontal" if horizontal else "vertical",
            "groups_shown": len(rows), "groups_total": full["n_groups"], "data": rows}
    return title, data


def _draw_line(ax, df, x, y, aggregate, period):
    """Line chart: values in order of x (usually dates), optionally grouped per period."""
    kind = column_kind(df[x])
    if kind not in ("datetime", "numeric"):
        raise ToolError(
            f"Line charts need dates or numbers on the x-axis, but '{x}' is {kind}. "
            "If it contains dates stored as text, convert it first with clean_data/convert_types."
        )
    if y and column_kind(df[y]) != "numeric":
        raise ToolError(f"Column '{y}' is not numeric, so it can't be drawn as a line.")
    if period:
        if period not in PERIODS:
            raise ToolError(f"Unknown period '{period}'. Choose one of {list(PERIODS)}")
        if kind != "datetime":
            raise ToolError(f"'period' only works with dates, but '{x}' is {kind}.")

    data = df.dropna(subset=[x])
    keys = data[x]
    if period:
        # to_period("M") turns 2026-01-17 into "January 2026"; .start_time turns that
        # back into a date (2026-01-01), so all January rows share one key.
        keys = keys.dt.to_period(PERIODS[period]).dt.start_time

    method = aggregate if y else "count"
    grouped = data.groupby(keys)[y].agg(method) if y else data.groupby(keys).size()
    grouped = grouped.sort_index()  # sort by time / x value, NOT by size

    # Show dots on each point only when there are few points (otherwise it's cluttered).
    ax.plot(grouped.index, grouped.values, color=PRIMARY, linewidth=2,
            solid_capstyle="round", solid_joinstyle="round",
            marker="o" if len(grouped) <= 24 else None, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5)
    _style_axes(ax, value_axis="y")
    ax.set_ylabel(_value_label(y, method), color=TEXT_SECONDARY)
    ax.set_xlabel(x + (f" (per {period})" if period else ""), color=TEXT_SECONDARY)

    if kind == "datetime" and period and len(grouped) <= 24:
        # Grouped by period with few points: put one tick on EACH point and
        # label it clearly for that period, e.g. "Jan 2026" for months.
        ax.set_xticks(grouped.index)
        ax.xaxis.set_major_formatter(mdates.DateFormatter(DATE_FORMATS[period]))
    elif kind == "datetime":
        # Otherwise let Matplotlib choose the ticks; ConciseDateFormatter
        # writes them compactly ("Feb", "Mar", ...).
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    # Label only the LAST value. One clear label beats a number on every point.
    ax.annotate(_format_number(grouped.iloc[-1]), (grouped.index[-1], grouped.iloc[-1]),
                xytext=(6, 0), textcoords="offset points", va="center",
                color=TEXT_SECONDARY, fontsize=9)

    points = [{"x": to_python(k), "value": to_python(v)} for k, v in grouped.items()]
    title = f"{AGGREGATE_WORDS[method]} {y or ''} over {x}".replace("  ", " ")
    if period:
        title += f" (per {period})"
    return title, {"x": x, "y": y, "aggregate": method, "period": period,
                   "n_points": len(points), "data": points[:MAX_RETURNED_POINTS]}


def _draw_pie(ax, df, x, y, aggregate):
    """Donut chart: each category's share of the whole (max 6 slices)."""
    method = aggregate if y else "count"
    if method not in ("sum", "count"):
        raise ToolError("Pie charts show parts of a whole, so aggregate must be 'sum' or 'count'.")
    rows = group_and_aggregate(df, x, y, method)["rows"]
    if any((r["value"] or 0) < 0 for r in rows):
        raise ToolError("Pie charts can't show negative values. Use a bar chart instead.")

    # Too many slices: keep the biggest 5 and add the rest together as "Other".
    if len(rows) > MAX_PIE_SLICES:
        keep = rows[:MAX_PIE_SLICES - 1]
        rest = rows[MAX_PIE_SLICES - 1:]
        other = {x: "Other", "value": sum(r["value"] for r in rest),
                 "percent_of_total": round(sum(r["percent_of_total"] for r in rest), 2),
                 "grouped_categories": len(rest)}
        rows = keep + [other]

    values = [r["value"] for r in rows]
    colors = [OTHER_GREY if r[x] == "Other" else PALETTE[i] for i, r in enumerate(rows)]
    # Every slice gets its NAME and PERCENTAGE written next to it, so the reader
    # never has to match colours to a legend.
    labels = [f"{_label(r[x])}\n{r['percent_of_total']:.1f}%" for r in rows]

    ax.pie(values, labels=labels, colors=colors, startangle=90, counterclock=False,
           labeldistance=1.18, textprops={"color": TEXT_SECONDARY, "fontsize": 9, "ha": "center"},
           # width=0.38 makes it a donut; the background-coloured edge leaves
           # a small gap between slices so they are clearly separated.
           wedgeprops={"width": 0.38, "edgecolor": SURFACE, "linewidth": 2})
    ax.set_aspect("equal")  # keep the circle round
    total = sum(values)
    ax.text(0, 0, f"{_format_number(total)}\ntotal", ha="center", va="center",
            color=TEXT_PRIMARY, fontsize=11)

    title = f"Share of {y or 'rows'} by {x}"
    return title, {"x": x, "y": y, "aggregate": method, "total": to_python(total), "data": rows}


def _draw_histogram(ax, df, x, bins):
    """Histogram: how many values fall into each range ("bin") of a numeric column."""
    if column_kind(df[x]) != "numeric":
        raise ToolError(f"Histograms need a numeric column, but '{x}' is {column_kind(df[x])}.")
    if not isinstance(bins, int) or not 2 <= bins <= 100:
        raise ToolError("bins must be a whole number between 2 and 100.")
    values = df[x].dropna()
    if values.empty:
        raise ToolError(f"Column '{x}' has no values to draw.")

    # ax.hist returns: counts per bin, the bin edges (boundaries), and the bar shapes.
    counts, edges, _ = ax.hist(values, bins=bins, color=PRIMARY,
                               edgecolor=SURFACE, linewidth=1.5)  # thin gaps between bars
    median = values.median()
    # A vertical reference line at the median, labelled in text colour.
    ax.axvline(median, color=TEXT_SECONDARY, linewidth=1.5)
    ax.margins(y=0.15)  # empty space above the tallest bar, for the label
    # xycoords=("data", "axes fraction"): x is a data value (the median), but y
    # is a position in the plot area (1 = the very top), so the label always
    # sits at the top, above the bars.
    ax.annotate(f"median {_format_number(median)}", (median, 1), xycoords=("data", "axes fraction"),
                xytext=(5, -12), textcoords="offset points", color=TEXT_SECONDARY, fontsize=9)
    _style_axes(ax, value_axis="y")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _format_number(v)))  # "1,000" not "1000"
    ax.set_xlabel(x, color=TEXT_SECONDARY)
    ax.set_ylabel("Number of rows", color=TEXT_SECONDARY)

    return f"Distribution of {x}", {
        "x": x, "bins": bins, "n_values": int(values.size),
        "mean": to_python(values.mean()), "median": to_python(median),
        "min": to_python(values.min()), "max": to_python(values.max()),
        "bin_edges": [to_python(e) for e in edges],
        "counts": [int(c) for c in counts],
    }


def _draw_scatter(ax, df, x, y, color_by):
    """Scatter plot: one dot per row, x against y. Optionally coloured by a category."""
    if not y:
        raise ToolError("Scatter plots need both x and y (two numeric columns).")
    for col in (x, y):
        if column_kind(df[col]) != "numeric":
            raise ToolError(f"Scatter plots need numeric columns, but '{col}' is {column_kind(df[col])}.")

    data = df.dropna(subset=[x, y])
    if len(data) < 2:
        raise ToolError("Scatter plots need at least 2 rows with both values present.")

    groups_info = None
    if color_by:
        # The 3 most common groups get their own colour; all others become grey "Other".
        top_groups = list(data[color_by].value_counts().index[:MAX_SCATTER_GROUPS])
        names = data[color_by].where(data[color_by].isin(top_groups), "Other")
        order = top_groups + (["Other"] if (names == "Other").any() else [])
        for i, group in enumerate(order):
            part = data[names == group]
            ax.scatter(part[x], part[y], s=45, label=_label(group),
                       color=OTHER_GREY if group == "Other" else PALETTE[i],
                       edgecolors=SURFACE, linewidths=1)  # white ring separates overlapping dots
        ax.legend(frameon=False, labelcolor=TEXT_SECONDARY, title=color_by,
                  title_fontsize=9, fontsize=9)
        groups_info = {_label(g): int((names == g).sum()) for g in order}
    else:
        ax.scatter(data[x], data[y], s=45, color=PRIMARY, edgecolors=SURFACE, linewidths=1)

    _style_axes(ax, value_axis="both")
    ax.set_xlabel(x, color=TEXT_SECONDARY)
    ax.set_ylabel(y, color=TEXT_SECONDARY)

    # The LLM can't see the dots, so we also return the correlation in numbers and words.
    r = data[x].corr(data[y])
    return f"{y} vs {x}", {
        "x": x, "y": y, "color_by": color_by, "n_points": len(data),
        "pearson_r": to_python(r, 3),
        "interpretation": describe_strength(r) if pd.notna(r) else "not defined (a column is constant)",
        "groups": groups_info,
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _style_axes(ax, value_axis: str) -> None:
    """Quiet, consistent styling: no box, light gridlines on the value axis only."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)       # remove the top and right border lines
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)
    if value_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _format_number(v)))
    if value_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: _format_number(v)))
    ax.set_axisbelow(True)  # draw gridlines BEHIND the bars/lines, not on top


def _format_number(value) -> str:
    """12345.6 -> '12,346'; 18.5 -> '18.5'; 3 -> '3'. Short and readable for labels."""
    value = float(value)
    if abs(value) >= 1000 or value == int(value):
        return f"{value:,.0f}"   # ":,.0f" = thousands separator, 0 decimals
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def _label(value) -> str:
    """Category names as text; missing values get a visible name instead of disappearing."""
    return "(missing)" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)


def _value_label(y, method) -> str:
    """Axis label for the value axis, e.g. 'Revenue_EUR (sum)' or 'Number of rows'."""
    return f"{y} ({method})" if y else "Number of rows"


def _slugify(text: str) -> str:
    """Turn a title into a safe file name: 'Total Revenue by Region' -> 'total-revenue-by-region'."""
    slug = re.sub(r"[^0-9a-z]+", "-", text.lower()).strip("-")
    return slug[:80] or "chart"
