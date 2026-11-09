"""
ONE real API call. DISABLED by default: it only runs when you set
RUN_LIVE_TESTS=1 AND an API key is configured.

    RUN_LIVE_TESTS=1 pytest -m live

Cost: one short request, at most about $0.14 worst case (usually far less).
"""

import os

import pytest

from agent.config import has_credentials, load_dotenv_file, load_settings
from agent.llm_client import LLMClient

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(os.environ.get("RUN_LIVE_TESTS") != "1",
                       reason="live API tests are disabled (set RUN_LIVE_TESTS=1 to enable)"),
]


def test_one_real_request(tmp_path):
    load_dotenv_file()
    if not has_credentials():
        pytest.skip("no API key configured")
    llm = LLMClient(load_settings(), log_path=tmp_path / "usage.jsonl")
    result = llm.ask([{"role": "user", "content": "Reply with exactly the word: ready"}])
    assert result.complete, result.warnings
    assert "ready" in result.text.lower()
    assert result.usage["input_tokens"] <= result.estimated_input_tokens
    assert result.usage_cost_estimate_usd <= result.max_cost_estimate_usd
