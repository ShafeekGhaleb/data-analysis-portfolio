"""
Integration tests for the explain-metric-change skill.

"Integration" = several parts working TOGETHER:
- the SKILL.md instructions are well-formed and only mention tools that exist;
- the whole workflow (load -> clean -> explain -> charts) runs end to end;
- the demo script runs as a real program.

Note: these tests do NOT test an LLM. The agent loop and skill loader do not
exist yet (Stages 4-5). They test that the skill's instructions and the
Python tools fit together.
"""

import re
import subprocess
import sys

import tools
from tools import clean_data, create_waterfall_chart, explain_change, load_file

from conftest import PROJECT_DIR, SAMPLE_DIR

SKILL_DIR = PROJECT_DIR / "skills" / "explain-metric-change"
SKILL_FILE = SKILL_DIR / "SKILL.md"


def read_front_matter(text: str) -> dict:
    """
    Read the small header between the two '---' lines at the top of SKILL.md.
    Each line is 'key: value'. (A tiny parser, so we don't need a YAML library.)
    """
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, "SKILL.md must start with a '---' header block"
    fields = {}
    for line in match.group(1).splitlines():
        key, _, value = line.partition(":")
        fields[key.strip()] = value.strip()
    return fields


def test_skill_header_is_valid():
    header = read_front_matter(SKILL_FILE.read_text(encoding="utf-8"))
    assert header["name"] == SKILL_DIR.name == "explain-metric-change"
    assert re.fullmatch(r"[a-z0-9-]+", header["name"])        # lowercase-and-dashes only
    assert 50 < len(header["description"]) <= 1024            # short enough to always show the LLM


def test_skill_only_mentions_tools_that_exist():
    # Every `backticked_name` that looks like one of our tool functions must exist.
    text = SKILL_FILE.read_text(encoding="utf-8")
    mentioned = set(re.findall(r"`([a-z_]+)`", text))
    tool_like = {name for name in mentioned
                 if name.startswith(("explain_", "create_", "inspect_", "clean_", "group_", "parse_", "period_"))}
    assert {"explain_change", "create_waterfall_chart", "inspect_data", "clean_data"} <= tool_like
    missing = tool_like - set(tools.__all__)
    assert not missing, f"SKILL.md mentions tools that do not exist: {missing}"


def test_skill_is_honest_about_its_status():
    text = SKILL_FILE.read_text(encoding="utf-8")
    assert "cannot use this skill on its own yet" in text
    assert "Contribution is not causation" in text
    assert "No unsupported recommendations" in text


def test_full_workflow_end_to_end(tmp_path):
    # The workflow the SKILL.md describes, run by Python instead of an LLM.
    sales, _ = clean_data(load_file(SAMPLE_DIR / "sales_data.csv"))
    result = explain_change(sales, "Revenue_EUR", "Date", "2026-Q1", "2026-Q2",
                            units_column="Units", item_column="Product",
                            price_column="Unit_Price_EUR")
    assert result["reconciled"]
    best = result["dimension_ranking"][0]
    for breakdown in (best, "price_volume_mix"):
        chart = create_waterfall_chart(result, breakdown, output_dir=tmp_path)
        assert chart["reconciled"]
    assert len(list(tmp_path.glob("*.png"))) == 2


def test_demo_script_runs(tmp_path):
    # Run the demo as a separate program, exactly like a user would.
    completed = subprocess.run(
        [sys.executable, str(PROJECT_DIR / "examples" / "skill_demo.py"), str(tmp_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert "reconciled: True" in completed.stdout
    assert "+€5,957.50 (+51.4%)" in completed.stdout
    # Channel (the best-ranked dimension) + Region + Product + price/volume/mix.
    assert len(list(tmp_path.glob("waterfall-*.png"))) == 4
