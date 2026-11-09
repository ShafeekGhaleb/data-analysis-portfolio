# 🤖 Data Analysis AI Agent

An AI agent that accepts CSV/Excel files, understands analysis requests written
in plain English, cleans and analyses the data with Python and Pandas, creates
charts, and writes a professional report.

> **Status:** 🚧 In development. Stages 1, 2 and 2.5 (data toolkit, charts, Metric Change
> Explainer tools) are complete. The LLM agent itself starts at Stage 3.

## How it works (architecture)

```
User ──► Interface ──► Agent loop ──► LLM (Claude): understands, plans, explains
                            │
                            └──► Tools (Python/Pandas): load, inspect, clean,
                                 statistics, charts, report
```

**Key design principle:** the LLM never calculates numbers itself. Every number
comes from tested Python tools. The LLM decides *which* tool to use and explains
the results.

## Development stages

| Stage | Description | Status |
|---|---|---|
| 0 | Project setup | ✅ Done |
| 1 | Data toolkit: load, inspect, clean, statistics (+ tests) | ✅ Done |
| 2 | Charts: bar, line, pie, histogram, scatter (+ tests) | ✅ Done |
| 2.5 | Metric Change Explainer: periods, contribution analysis, price/volume/mix, waterfall chart, SKILL.md (+ tests) | ✅ Done (Python tools) |
| 3 | First LLM call | ⏳ Next |
| 4 | Tool use | ⬜ |
| 5 | Agent loop | ⬜ |
| 6 | Reports | ⬜ |
| 7 | Command-line app | ⬜ |
| 8 | Quality and safety | ⬜ |
| 9 | Portfolio polish | ⬜ |
| 10 | Streamlit web app (optional) | ⬜ |

## Project structure

```
data-analysis-ai-agent/
├── tools/                  # The agent's "hands": plain Python functions
│   ├── errors.py           # ToolError: one clear error type for all tools
│   ├── utils.py            # Helpers: JSON-safe values, column checks
│   ├── data_loading.py     # load_file: CSV / Excel -> DataFrame
│   ├── inspection.py       # inspect_data: size, types, missing values
│   ├── cleaning.py         # clean_data + individual cleaning steps
│   ├── statistics.py       # describe_statistics, group_and_aggregate, correlation
│   ├── charts.py           # create_chart + create_waterfall_chart (PNG)
│   ├── periods.py          # "2026-Q1" -> exact date ranges, coverage checks
│   └── change_analysis.py  # explain_change: contributions + price/volume/mix
├── skills/
│   └── explain-metric-change/SKILL.md   # instructions for the future LLM agent
├── tests/                  # 156 automated tests (pytest)
├── sample_data/            # A clean sales dataset and a deliberately messy one
├── examples/               # stage1_demo.py, stage2_demo.py, skill_demo.py
├── outputs/                # Generated charts (not uploaded to GitHub)
├── requirements.txt        # Libraries to install
├── pyproject.toml          # pytest settings
└── .env.example            # API key template (needed from Stage 3)
```

## Getting started

```bash
cd data-analysis-ai-agent

# 1. Create a virtual environment (a private box of libraries for this project)
python -m venv .venv

# 2. Activate it
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install the libraries
pip install -r requirements.txt

# 4. Run the tests
pytest

# 5. Try the toolkit and the charts
python examples/stage1_demo.py
python examples/stage2_demo.py   # saves 6 example charts in outputs/
python examples/skill_demo.py    # Metric Change Explainer: Q1 vs Q2 + 4 waterfall charts
```

## Tools available so far

| Tool | What it does |
|---|---|
| `load_file(path)` | Reads `.csv` (any separator/encoding) or `.xlsx`, with clear errors |
| `inspect_data(df)` | Rows, columns, kinds, missing values, duplicates, sample rows |
| `clean_data(df)` | Trims spaces, converts text to numbers/dates, removes duplicates, and returns a cleaning log |
| `handle_missing(df, strategy)` | Drop rows, or fill with mean / median / mode / a value |
| `describe_statistics(df)` | Mean, median, std, quartiles, top values, date ranges |
| `group_and_aggregate(df, ...)` | e.g. total revenue by region, with % of total |
| `correlation(df)` | Pearson / Spearman correlations, explained in words |
| `explain_change(df, metric, date_column, period_a, period_b, ...)` | Splits a change between two periods into exact, reconciled contributions by segment, plus price / volume / mix effects; flags small samples, missing values, new/discontinued segments and incomplete periods |
| `create_waterfall_chart(analysis, breakdown)` | Bridge chart from period A's total to period B's total |
| `create_chart(df, chart_type, x, y, ...)` | Bar, line, pie (donut), histogram or scatter chart saved as PNG; also returns the plotted numbers so the agent can describe the chart accurately |

## Skills

A **tool** is a Python function (the agent's hands). A **skill** is a set of
instructions telling the LLM *when* and *how* to use tools and how to present
the result (the agent's know-how).

| Skill | Tools it uses | Status |
|---|---|---|
| [`explain-metric-change`](skills/explain-metric-change/SKILL.md) | `explain_change`, `create_waterfall_chart`, `inspect_data`, `clean_data` | Python tools ✅ implemented and tested. LLM use ⏳ after Stages 4–5 (tool registry, agent loop, skill loader) |

**Example (Q1 → Q2 2026, sales sample):** revenue +€5,957.50 (+51.4%). All of the
growth came from **volume** (+€6,293.23, units +54%); **mix** was −€335.73 and
**price** €0.00. Every breakdown is checked to add up exactly to the total change.
These figures show *where* the change happened, not *why*.
