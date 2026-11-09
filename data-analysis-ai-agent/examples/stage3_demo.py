"""
Stage 3 demo: ask Claude to describe a dataset from its safe summary.

Three modes:
    python examples/stage3_demo.py             DRY RUN (default): build the request, estimate
                                               the cost, check the budget. Nothing is sent.
    python examples/stage3_demo.py --simulate  as above, then a FAKE client returns a prepared
                                               answer, to show how responses are handled.
    python examples/stage3_demo.py --live      really sends ONE request (needs an API key).
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from agent.config import has_credentials, load_dotenv_file, load_settings  # noqa: E402
from agent.llm_client import LLMClient, LLMError  # noqa: E402
from agent.pricing import PRICES_VERIFIED_ON, price_for  # noqa: E402
from agent.privacy import safe_dataset_summary, safe_json  # noqa: E402
from agent.prompts import build_dataset_messages  # noqa: E402
from agent.testing import FakeAnthropicClient, fake_response  # noqa: E402
from tools import load_file  # noqa: E402

QUESTION = ("Describe this dataset in plain English, then suggest three analyses that a data "
            "toolkit with statistics, grouping, correlation, charts and period-comparison "
            "tools could perform on it. Do not state any numbers that are not in the summary.")


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--simulate", action="store_true")
    mode.add_argument("--live", action="store_true")
    args = parser.parse_args()

    load_dotenv_file()
    settings = load_settings()
    price = price_for(settings.model)
    print(f"Settings: model={settings.model}, effort={settings.effort}, max_tokens={settings.max_tokens}, "
          f"retries={settings.max_retries}, budget=${settings.budget_usd:.2f}")
    print(f"Prices (verified {PRICES_VERIFIED_ON}): ${price.input}/M input, ${price.output}/M output")

    # 1. Our tools run locally; only a structure-only summary leaves the computer.
    df = load_file(PROJECT_DIR / "sample_data" / "sales_data.csv")
    summary_json = safe_json(safe_dataset_summary(df))
    messages = build_dataset_messages(summary_json, QUESTION)

    if args.live:
        if not has_credentials():
            print("\nNo API key configured: the live request was NOT sent.")
            return
        client, label = None, "LIVE"
    elif args.simulate:
        client = FakeAnthropicClient([fake_response(
            text="(SIMULATED) This dataset has 30 rows and 8 columns about sales ...",
            input_tokens=1100, output_tokens=420)])
        label = "SIMULATED (fake client - nothing sent)"
    else:
        client, label = None, "DRY RUN"

    # log_path=None in dry-run/simulate: nothing real happened, so nothing to log.
    llm = LLMClient(settings, client=client, log_path=None if not args.live else
                    PROJECT_DIR / "logs" / "llm_usage.jsonl")
    request = llm.build_request(messages)

    print(f"\n=== 1. THE REQUEST ({label}) ===")
    shown = {**request, "system": request["system"][:120] + " ...",
             "messages": [{"role": "user", "content": [
                 {"type": "text", "text": b["text"][:160] + (" ..." if len(b["text"]) > 160 else "")}
                 for b in request["messages"][0]["content"]]}]}
    print(json.dumps(shown, indent=2))
    print(f"Data block: {len(summary_json):,} characters of JSON; contains no rows and no cell values.")

    print("\n=== 2. BEFORE SENDING: estimate and budget check ===")
    estimate = llm.preflight(request)
    print(f"Estimated input tokens (safe upper bound): {estimate['estimated_input_tokens']:,}")
    print(f"Maximum possible cost: ${estimate['max_cost_estimate_usd']:.6f} "
          f"(input estimate + {settings.max_tokens:,} output tokens) x {settings.attempts} attempts")
    print(f"Budget: ${llm.budget.remaining_usd:.2f} remaining -> request ALLOWED")

    if not (args.simulate or args.live):
        print("\nDry run: nothing was sent. Use --simulate to see response handling.")
        return

    print(f"\n=== 3. SEND + RESPONSE ({label}) ===")
    try:
        result = llm.ask(messages)
    except LLMError as error:
        print(f"LLMError [{error.kind}]: {error}")
        return
    print(result.text)
    print(f"\nstop_reason={result.stop_reason}  complete={result.complete}  request_id={result.request_id}")
    for warning in result.warnings:
        print("WARNING:", warning)

    u = result.usage
    print("\n=== 4. AFTER: real token counts and cost estimates ===")
    print(f"Input tokens:  {u['input_tokens']:,} (our upper estimate was {result.estimated_input_tokens:,})")
    print(f"Output tokens: {u['output_tokens']:,} (includes any hidden thinking)")
    print(f"Usage-based cost estimate: ${u['usage_cost_estimate_usd']:.6f} "
          f"(= {u['input_tokens']:,} x ${price.input}/M + {u['output_tokens']:,} x ${price.output}/M)")
    print(f"Maximum estimate was:      ${result.max_cost_estimate_usd:.6f}")
    print(f"Session: ${llm.budget.spent_estimate_usd:.6f} estimated spent, ${llm.budget.remaining_usd:.6f} left")
    print("Actual billed cost: see the Anthropic Console (estimates can differ from billing).")


if __name__ == "__main__":
    main()
