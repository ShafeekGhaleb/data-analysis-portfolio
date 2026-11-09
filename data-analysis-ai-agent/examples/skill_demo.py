"""
Stage 2.5 demo: the Metric Change Explainer on the real sales data.

This script plays the role the LLM agent will play later: it follows the
workflow in skills/explain-metric-change/SKILL.md step by step, calling the
Python tools directly. (No LLM and no API key are involved.)

Run it from the project folder:
    python examples/skill_demo.py              # charts saved in outputs/
    python examples/skill_demo.py my_folder    # charts saved in my_folder/
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from tools import (  # noqa: E402
    ToolError,
    clean_data,
    create_waterfall_chart,
    explain_change,
    inspect_data,
    load_file,
)


def money(value) -> str:
    """Format a number as euros with a sign, e.g. +€3,276.00 / −€628.50."""
    sign = "+" if value > 0 else "−" if value < 0 else ""
    return f"{sign}€{abs(value):,.2f}"


def main(output_dir: Path) -> None:
    # Workflow step 1: inspect.
    raw = load_file(PROJECT_DIR / "sample_data" / "sales_data.csv")
    kinds = {c["name"]: c["kind"] for c in inspect_data(raw, sample_rows=0)["columns"]}
    print(f"Step 1  inspect: Date is '{kinds['Date']}', so it must be cleaned first.")

    # Workflow step 2: clean (Date text -> real dates).
    sales, _ = clean_data(raw)
    print("Step 2  clean:   Date converted to real dates.")

    # Workflow steps 3-4: choose periods and run the analysis.
    result = explain_change(
        sales, metric="Revenue_EUR", date_column="Date",
        period_a="2026-Q1", period_b="2026-Q2",
        units_column="Units", item_column="Product", price_column="Unit_Price_EUR",
        top_n=8,
    )
    print("Step 3-4 explain_change(Revenue_EUR, 2026-Q1 -> 2026-Q2)")

    # Workflow step 5: never use a breakdown that does not reconcile.
    print(f"Step 5  reconciled: {result['reconciled']} "
          f"({len(result['reconciliation']['checks'])} checks, "
          f"tolerance {result['reconciliation']['tolerance']:.2e})")
    if not result["reconciled"]:
        print("STOP: reconciliation failed.")
        return

    s = result["summary"]
    print("\n=== WHAT CHANGED ===")
    print(f"Revenue: €{s['period_a_total']:,.2f} -> €{s['period_b_total']:,.2f} "
          f"= {money(s['change'])} ({s['change_percent']:+.1f}%)")

    print("\n=== WHERE IT CHANGED (dimensions ranked by concentration) ===")
    for dim in result["dimension_ranking"]:
        driver = result["drivers"][dim]
        parts = ", ".join(f"{c['group']} {money(c['change'])} ({c['share_of_change']:.0f}%)"
                          for c in driver["contributions"])
        print(f"{dim:<9} concentration {driver['concentration']:.2f}: {parts}")

    print("\n=== BIGGEST TWO-DIMENSION MOVEMENTS ===")
    for combo in result["top_combinations"]:
        flag = f"  [{combo['status']}]" if combo["status"] != "existing" else ""
        print(f"{combo['group']:<24} {money(combo['change']):>12}{flag}")
    print(f"(skipped nested pairs: {result['skipped_nested_pairs']})")

    pvm = result["price_volume_mix"]
    print("\n=== TYPE OF CHANGE (price / volume / mix) ===")
    for name, value in pvm["effects"].items():
        print(f"{name:<19} {money(value):>12}")
    print(f"Units {pvm['units_a']} -> {pvm['units_b']} ({pvm['units_change_percent']:+.1f}%); "
          f"average price per unit €{pvm['average_price_a']:.2f} -> €{pvm['average_price_b']:.2f}")

    # Workflow step 6: read every warning.
    print("\n=== DATA CAVEATS ===")
    for warning in result["warnings"]:
        print("-", warning)
    print("-", result["causation_note"])

    # Workflow step 7: visualise.
    print("\n=== CHARTS ===")
    for breakdown in (result["dimension_ranking"][0], "Region", "Product", "price_volume_mix"):
        chart = create_waterfall_chart(result, breakdown, output_dir=output_dir)
        print(f"{breakdown:<17} -> {chart['path']}")

    # The guardrails in action: requests the tool refuses or flags.
    print("\n=== GUARDRAILS ===")
    for description, kwargs in [
        ("Average is not additive", {"aggregate": "mean"}),
        ("Period with no data", {"period_b": "2026-Q3"}),
    ]:
        try:
            explain_change(sales, "Revenue_EUR", "Date",
                           **{"period_a": "2026-Q1", "period_b": "2026-Q2", **kwargs})
        except ToolError as error:
            print(f"{description}: refused -> {error}")

    incomplete = explain_change(sales, "Revenue_EUR", "Date", "2026-05",
                                {"start": "2026-06-01", "end": "2026-07-31"})
    print("Incomplete period: warned ->", incomplete["warnings"][0])


if __name__ == "__main__":
    # sys.argv holds the command-line words: ["examples/skill_demo.py", "my_folder"].
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_DIR / "outputs")
