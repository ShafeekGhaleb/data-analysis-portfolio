"""
Tests for LLMClient, using the FAKE client: no key, no network, no cost.

Covers: budget enforcement BEFORE sending, cost recording AFTER, every error
type, refusals, incomplete answers, the exact request we send, and the log.
"""

import json

import anthropic
import httpx2
import pytest

from agent.budget import BudgetTracker
from agent.config import load_settings
from agent.llm_client import LLMClient, LLMError
from agent.prompts import SYSTEM_PROMPT, build_dataset_messages
from agent.testing import FakeAnthropicClient, fake_response

MESSAGES = build_dataset_messages('{"n_rows":30}', "Describe the data.")
REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def make_client(outcomes, budget=1.00, log_path=None, **env):
    settings = load_settings(env={"LLM_BUDGET_USD": str(budget), **env})
    fake = FakeAnthropicClient(outcomes)
    return LLMClient(settings, client=fake, log_path=log_path), fake


def api_error(cls, status):
    """Build a real SDK error object, as the SDK would raise it."""
    return cls(f"HTTP {status}", response=httpx2.Response(status, request=REQUEST), body=None)


# ---------- the request we send ----------

def test_request_contents():
    llm, fake = make_client([fake_response()])
    llm.ask(MESSAGES)
    sent = fake.messages.calls[0]
    assert sent["model"] == "claude-sonnet-5-5"
    assert sent["max_tokens"] == 4000
    assert sent["output_config"] == {"effort": "low"}
    assert sent["system"] == SYSTEM_PROMPT
    assert sent["messages"] == MESSAGES
    # Things we deliberately do NOT send:
    assert "temperature" not in sent        # rejected by current models
    assert "fallbacks" not in sent          # refusal fallback: OFF (your decision)
    assert "thinking" not in sent           # adaptive thinking is this model's default


# ---------- budget: checked BEFORE every request ----------

def test_request_that_could_exceed_the_budget_is_never_sent():
    # Worst case is about $0.13 (see test_agent_basics); a $0.05 budget cannot cover it.
    llm, fake = make_client([fake_response()], budget=0.05)
    with pytest.raises(LLMError) as error:
        llm.ask(MESSAGES)
    assert error.value.kind == "budget"
    assert fake.messages.calls == []        # PROOF: the request never reached the client
    assert llm.budget.spent_estimate_usd == 0


def test_budget_runs_out_after_several_requests():
    # Realistic answers: the full 4,000 output tokens each (the most max_tokens allows).
    # Each really costs 900 x $2/M + 4,000 x $10/M = $0.0418.
    # The worst case checked BEFORE each request is about $0.13 (3 attempts).
    answers = [fake_response(input_tokens=900, output_tokens=4000) for _ in range(5)]
    llm, fake = make_client(answers, budget=0.20)
    llm.ask(MESSAGES)                       # $0.20 left >= worst case -> sent; $0.0418 spent
    llm.ask(MESSAGES)                       # $0.158 left >= worst case -> sent; $0.0836 spent
    with pytest.raises(LLMError) as error:
        llm.ask(MESSAGES)                   # $0.116 left <  worst case -> REFUSED, not sent
    assert error.value.kind == "budget"
    assert len(fake.messages.calls) == 2
    assert llm.budget.spent_estimate_usd == pytest.approx(0.0836)


@pytest.mark.parametrize("budget", [0.15, 0.30, 0.50, 1.00])
def test_spending_can_never_exceed_the_cap(budget):
    # Keep asking until refused. Because every real answer costs at most the worst
    # case that was checked first, the total can never pass the cap.
    answers = [fake_response(input_tokens=900, output_tokens=4000) for _ in range(40)]
    llm, _ = make_client(answers, budget=budget)
    while True:
        try:
            llm.ask(MESSAGES)
        except LLMError:
            break
    assert llm.budget.spent_estimate_usd <= budget


def test_impossible_usage_is_flagged():
    # max_tokens is a hard cap, so more output than max_tokens should never happen.
    llm, _ = make_client([fake_response(output_tokens=25_000)])
    result = llm.ask(MESSAGES)
    assert any("more output tokens than max_tokens" in w for w in result.warnings)


def test_usage_and_cost_recorded_after_response():
    llm, _ = make_client([fake_response(input_tokens=1_100, output_tokens=420)])
    result = llm.ask(MESSAGES)
    assert result.usage["input_tokens"] == 1_100 and result.usage["output_tokens"] == 420
    assert result.usage_cost_estimate_usd == pytest.approx(0.0064)   # 1,100x$2/M + 420x$10/M
    assert llm.budget.spent_estimate_usd == pytest.approx(0.0064)
    # The maximum (before) is always at least the usage-based estimate (after).
    assert result.max_cost_estimate_usd >= result.usage_cost_estimate_usd


def test_low_input_estimate_is_flagged():
    llm, _ = make_client([fake_response(input_tokens=999_999)])
    result = llm.ask(MESSAGES)
    assert any("estimate was too LOW" in w for w in result.warnings)


# ---------- response handling ----------

def test_complete_answer_ignores_thinking_blocks():
    llm, _ = make_client([fake_response(text="The answer.", thinking="secret reasoning")])
    result = llm.ask(MESSAGES)
    assert result.complete and not result.refused
    assert result.text == "The answer."
    assert result.request_id == "req_SIMULATED_0001"


@pytest.mark.parametrize("stop_reason", ["max_tokens", "model_context_window_exceeded", "pause_turn"])
def test_incomplete_answers_are_flagged(stop_reason):
    llm, _ = make_client([fake_response(text="Half an ans", stop_reason=stop_reason)])
    result = llm.ask(MESSAGES)
    assert result.complete is False
    assert any("INCOMPLETE" in w for w in result.warnings)


def test_refusal_is_reported_honestly():
    llm, _ = make_client([fake_response(text="", stop_reason="refusal", refusal_category="cyber")])
    result = llm.ask(MESSAGES)
    assert result.refused and not result.complete
    assert result.refusal_category == "cyber"
    assert result.text == ""
    assert any("declined" in w for w in result.warnings)


def test_unknown_stop_reason_is_not_treated_as_complete():
    llm, _ = make_client([fake_response(stop_reason="something_new")])
    assert llm.ask(MESSAGES).complete is False


# ---------- errors ----------

@pytest.mark.parametrize("error, kind", [
    (api_error(anthropic.AuthenticationError, 401), "authentication"),
    (api_error(anthropic.PermissionDeniedError, 403), "permission"),
    (api_error(anthropic.NotFoundError, 404), "not_found"),
    (api_error(anthropic.RateLimitError, 429), "rate_limit"),
    (api_error(anthropic.BadRequestError, 400), "bad_request"),
    (api_error(anthropic.InternalServerError, 500), "server_error"),
    (anthropic.APIConnectionError(request=REQUEST), "connection"),
    (ValueError("strange"), "unexpected"),
])
def test_errors_become_clear_llm_errors(error, kind):
    llm, _ = make_client([error])
    with pytest.raises(LLMError) as raised:
        llm.ask(MESSAGES)
    assert raised.value.kind == kind
    assert llm.budget.spent_estimate_usd == 0   # failed requests are not charged...


def test_timeout_charges_the_worst_case():
    llm, _ = make_client([anthropic.APITimeoutError(request=REQUEST)])
    with pytest.raises(LLMError) as raised:
        llm.ask(MESSAGES)
    assert raised.value.kind == "timeout"
    # ...except timeouts: the server may have processed them, so the reserved worst case is counted.
    assert llm.budget.spent_estimate_usd > 0.1


def test_missing_key_means_nothing_is_sent(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    llm = LLMClient(load_settings(env={}), client=None, log_path=None)  # no fake: real path
    with pytest.raises(LLMError) as raised:
        llm.ask(MESSAGES)
    assert raised.value.kind == "missing_credentials"


# ---------- the usage log: metadata only ----------

def test_log_contains_metadata_only(tmp_path, monkeypatch):
    secret = "sk-" + "ant-" + "FAKE-SECRET-123"
    monkeypatch.setenv("ANTHROPIC_API_KEY", secret)
    log = tmp_path / "usage.jsonl"
    llm, _ = make_client([fake_response(text="ANSWER-TEXT-XYZ"), api_error(anthropic.BadRequestError, 400)],
                         log_path=log)
    llm.ask(build_dataset_messages('{"col":"DATA-VALUE-ABC"}', "QUESTION-TEXT-QRS"))
    with pytest.raises(LLMError):
        llm.ask(MESSAGES)

    text = log.read_text()
    for forbidden in (secret, "ANSWER-TEXT-XYZ", "DATA-VALUE-ABC", "QUESTION-TEXT-QRS", "Rules:"):
        assert forbidden not in text
    entries = [json.loads(line) for line in text.splitlines()]
    assert [e["outcome"] for e in entries] == ["ok", "error"]
    assert entries[0]["request_id"] == "req_SIMULATED_0001"
    assert {"timestamp_utc", "model", "input_tokens", "output_tokens",
            "usage_cost_estimate_usd", "max_cost_estimate_usd"} <= set(entries[0])


def test_logger_refuses_content_fields(tmp_path):
    from agent.usage_log import UsageLogger
    with pytest.raises(ValueError, match="non-metadata"):
        UsageLogger(tmp_path / "x.jsonl").write(model="m", prompt="secret prompt")
