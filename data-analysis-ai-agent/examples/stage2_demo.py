"""
Stage 2 demo: create one chart of every type from the sales data.

Run it from the project folder:
    python examples/stage2_demo.py

The PNG files are saved in the "outputs" folder.
"""

import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from tools import clean_data, create_chart, load_file  # noqa: E402

OUTPUT_DIR = PROJECT_DIR / "outputs"


def main() -> None:
    sales = load_file(PROJECT_DIR / "sample_data" / "sales_data.csv")

    # The "Date" column is loaded as TEXT. A line chart needs real dates,
    # so we clean first: convert_types turns it into a datetime column.
    # This is the same order the agent will follow: load -> clean -> chart.
    sales, _ = clean_data(sales)

    # Each dictionary is one chart request: the same shape of request the
    # LLM will send in Stage 4 when it asks us to run "create_chart".
    requests = [
        {"chart_type": "bar", "x": "Region", "y": "Revenue_EUR", "aggregate": "sum"},
        {"chart_type": "bar", "x": "Product", "y": "Revenue_EUR", "aggregate": "sum"},
        {"chart_type": "line", "x": "Date", "y": "Revenue_EUR", "aggregate": "sum", "period": "month"},
        {"chart_type": "pie", "x": "Channel", "y": "Revenue_EUR"},
        {"chart_type": "histogram", "x": "Revenue_EUR", "bins": 8},
        {"chart_type": "scatter", "x": "Unit_Price_EUR", "y": "Revenue_EUR", "color_by": "Category"},
    ]

    for request in requests:
        # "**request" unpacks the dictionary into named arguments:
        # create_chart(sales, chart_type="bar", x="Region", ...)
        result = create_chart(sales, output_dir=OUTPUT_DIR, **request)
        print(f"\n✅ {result['chart_type']:<9} -> {Path(result['path']).name}")
        # Print what the agent will "know" about the chart (without the long lists).
        summary = {k: v for k, v in result.items() if k not in ("path", "bin_edges")}
        print(json.dumps(summary, ensure_ascii=False, default=str)[:400])


if __name__ == "__main__":
    main()
