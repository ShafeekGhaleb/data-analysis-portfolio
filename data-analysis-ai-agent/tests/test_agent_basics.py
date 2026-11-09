"""
Tests for the Stage 3 building blocks that need no SDK: settings, prices,
token/cost arithmetic and the budget tracker.
"""

import pytest

from agent.budget import BudgetExceededError, BudgetTracker
from agent.config import ConfigError, has_credentials, load_settings
from agent.pricing import (
    PRICE_TABLE,
    PricingError,
    estimate_input_tokens,
    estimate_max_cost,
    price_for,
    round_usd,
    usage_cost,
)
from agent.testing import fake_response


# ---------- settings ----------

def test_default_settings_match_the_stage3_decisions():
    settings = load_settings(env={})                        # an empty environment -> defaults
    assert settings.model == "claude-sonnet-5-5"
    assert settings.budget_usd == 1.00
    assert settings.effort == "low"
    assert settings.max_tokens == 4000
    assert settings.attempts == 3                           # 1 try + 2 automatic retries


def test_model_is_configurable():
    settings = load_settings(env={"ANTHROPIC_MODEL": "claude-opus-5-5", "LLM_EFFORT": "MEDIUM"})
    assert settings.model == "claude-opus-5-5" and settings.effort == "medium"


@pytest.mark.parametrize("env, message", [
    ({"ANTHROPIC_MODEL": "claude-unknown-9"}, "no verified price"),
    ({"LLM_EFFORT": "extreme"}, "LLM_EFFORT"),
    ({"LLM_MAX_TOKENS": "0"}, "LLM_MAX_TOKENS"),
    ({"LLM_MAX_TOKENS": "lots"}, "not a valid number"),
    ({"LLM_BUDGET_USD": "100"}, "at most 20.00"),           # a typo can't allow $100
    ({"LLM_BUDGET_USD": "-1"}, "above 0"),
    ({"LLM_MAX_RETRIES": "9"}, "LLM_MAX_RETRIES"),
])
def test_invalid_settings_are_refused(env, message):
    with pytest.raises(ConfigError, match=message):
        load_settings(env=env)


def test_has_credentials_never_reveals_the_value():
    secret = "sk-" + "ant-" + "FAKE-VALUE-FOR-TEST"         # built at runtime: no key-like literal in files
    assert has_credentials({"ANTHROPIC_API_KEY": secret}) is True
    assert has_credentials({"ANTHROPIC_API_KEY": ""}) is False
    assert has_credentials({}) is False


# ---------- prices (verified against the official pricing page) ----------

def test_verified_prices():
    sonnet = price_for("claude-sonnet-5-5")
    assert (sonnet.input, sonnet.output, sonnet.cache_write_5m, sonnet.cache_read) == (2.00, 10.00, 2.50, 0.10)
    opus = price_for("claude-opus-5-5")
    assert (opus.input, opus.output) == (4.00, 20.00)
    assert set(PRICE_TABLE) == {"claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"}
    with pytest.raises(PricingError, match="No verified price"):
        price_for("gpt-anything")


# ---------- token and cost arithmetic ----------

def test_input_estimate_counts_bytes_plus_overhead():
    assert estimate_input_tokens("abcd") == 4 + 200
    assert estimate_input_tokens("€") == 3 + 200            # "€" is 3 bytes in UTF-8: never underestimated


def test_maximum_cost_formula():
    # 2,000 input x $2/M + 4,000 output x $10/M = 0.004 + 0.040 = $0.044 per attempt
    assert estimate_max_cost("claude-sonnet-5-5", 2000, 4000) == pytest.approx(0.044)
    assert estimate_max_cost("claude-sonnet-5-5", 2000, 4000, attempts=3) == pytest.approx(0.132)


def test_haiku_long_prompt_tier_is_refused():
    with pytest.raises(PricingError, match="higher price tier"):
        estimate_max_cost("claude-haiku-5-5", 150_000, 1000)


def test_usage_cost_from_real_token_counts():
    usage = fake_response(input_tokens=1_000, output_tokens=500, cache_creation=2_000, cache_read=10_000).usage
    result = usage_cost("claude-sonnet-5-5", usage)
    parts = result["cost_parts_usd"]
    assert parts["input"] == pytest.approx(0.002)           # 1,000 x $2/M
    assert parts["output"] == pytest.approx(0.005)          # 500 x $10/M
    assert parts["cache_write"] == pytest.approx(0.005)     # 2,000 x $2.50/M
    assert parts["cache_read"] == pytest.approx(0.001)      # 10,000 x $0.10/M
    assert result["usage_cost_estimate_usd"] == pytest.approx(0.013)


def test_missing_usage_fields_count_as_zero():
    usage = fake_response().usage
    usage.cache_creation_input_tokens = None
    assert usage_cost("claude-sonnet-5-5", usage)["cache_creation_input_tokens"] == 0


def test_rounding_never_understates():
    assert round_usd(0.0000001) == 0.000001
    assert round_usd(0.123456) == 0.123456


# ---------- budget ----------

def test_budget_check_and_record():
    budget = BudgetTracker(1.00)
    budget.check(0.99)                                      # fits: no error
    budget.record(0.30)
    assert budget.remaining_usd == pytest.approx(0.70)
    with pytest.raises(BudgetExceededError, match="refused before sending"):
        budget.check(0.71)                                  # worst case does not fit
    budget.check(0.70)                                      # exactly fits


def test_budget_must_be_positive():
    with pytest.raises(ValueError):
        BudgetTracker(0)
