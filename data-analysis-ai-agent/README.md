# 🤖 Data Analysis AI Agent

An AI agent that accepts CSV/Excel files, understands analysis requests written
in plain English, cleans and analyses the data with Python and Pandas, creates
charts, and writes a professional report.

> **Status:** 🚧 In development. Stage 1 (data toolkit) is complete.

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
| 2 | Charts | ⏳ Next |
| 3 | First LLM call | ⬜ |
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
│   └── statistics.py       # describe_statistics, group_and_aggregate, correlation
├── tests/                  # 48 automated tests (pytest)
├── sample_data/            # A clean sales dataset and a deliberately messy one
├── examples/stage1_demo.py # Run the toolkit by hand
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

# 5. Try the toolkit
python examples/stage1_demo.py
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
