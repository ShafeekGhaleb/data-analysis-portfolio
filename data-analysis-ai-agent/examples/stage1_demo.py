"""
Stage 1 demo: use the data toolkit by hand, exactly the way the AI agent
will use it later (load -> inspect -> clean -> analyse).

Run it from the project folder:
    python examples/stage1_demo.py
"""

import json
import sys
from pathlib import Path

# Make "import tools" work when this file is run directly: add the project
# folder (the parent of "examples") to the list of places Python searches.
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from tools import (  # noqa: E402  (import after the sys.path change, on purpose)
    ToolError,
    clean_data,
    correlation,
    describe_statistics,
    group_and_aggregate,
    inspect_data,
    load_file,
)

SAMPLE_DIR = PROJECT_DIR / "sample_data"


def show(title: str, data) -> None:
    """Print a heading and then the data as neatly indented JSON."""
    print(f"\n=== {title} ===")
    # ensure_ascii=False prints "€" instead of "€".
    print(json.dumps(data, indent=2, ensure_ascii=False))


def main() -> None:
    # ---------- Part 1: a messy file ----------
    messy = load_file(SAMPLE_DIR / "messy_orders.csv")
    report = inspect_data(messy, sample_rows=0)
    show("1. Messy file: quality check",
         {k: report[k] for k in ("n_rows", "duplicate_rows", "total_missing_values")})
    show("   Column kinds BEFORE cleaning",
         {c["name"]: c["kind"] for c in report["columns"]})

    cleaned, log = clean_data(messy)
    show("2. Cleaning log", [step["description"] for step in log])
    show("   Conversion details", log[1]["details"]["converted_columns"])
    show("   Column kinds AFTER cleaning",
         {c["name"]: c["kind"] for c in inspect_data(cleaned, sample_rows=0)["columns"]})

    # ---------- Part 2: analyse the real sales data ----------
    sales = load_file(SAMPLE_DIR / "sales_data.csv")
    show("3. Revenue statistics", describe_statistics(sales, ["Revenue_EUR"])["numeric"])
    show("4. Revenue by region", group_and_aggregate(sales, "Region", "Revenue_EUR", "sum")["rows"])
    show("5. Correlations", correlation(sales, ["Units", "Unit_Price_EUR", "Revenue_EUR"])["pairs"])

    # ---------- Part 3: a friendly error ----------
    try:
        group_and_aggregate(sales, "Regoin", "Revenue_EUR")  # misspelled on purpose
    except ToolError as error:
        print(f"\n=== 6. Friendly error message ===\n{error}")


# This block runs only when the file is executed directly
# (python examples/stage1_demo.py), not when it is imported by other code.
if __name__ == "__main__":
    main()
