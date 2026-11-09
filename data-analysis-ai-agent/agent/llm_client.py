"""
LLMClient: the only place where our program talks to Claude.

The life of one request (method ask):

  1. BUILD     the request: model, max_tokens, effort, system prompt, messages
  2. ESTIMATE  input tokens (safe upper bound) and the MAXIMUM possible cost
  3. CHECK     the budget. If the worst case does not fit, stop here:
               nothing is sent and nothing is spent
  4. SEND      through the official Anthropic SDK (which adds the API key itself)
  5. HANDLE    errors: turn every SDK error into one clear LLMError
  6. RECORD    the real token usage and the usage-based cost estimate
  7. INTERPRET the stop reason: complete answer, cut off, or refused
  8. LOG       metadata only (no prompt, no answer, no key)

DEPENDENCY INJECTION: the SDK client is passed IN (`client=`). Tests pass a
fake client, so the whole flow is tested without a key, network or cost.
"""

from dataclasses import dataclass, field
from pathlib import Path

from agent.budget import BudgetExceededError, BudgetTracker
from agent.config import PROJECT_DIR, Settings, has_credentials
from agent.pricing import estimate_input_tokens, estimate_max_cost, usage_cost
from agent.prompts import SYSTEM_PROMPT, request_text
from agent.usage_log import UsageLogger

DEFAULT_LOG_PATH = PROJECT_DIR / "logs" / "llm_usage.jsonl"

# Stop reasons that mean "the answer is finished".
COMPLETE_STOP_REASONS = {"end_turn", "stop_sequence"}
# Stop reasons that mean "the answer was cut off or interrupted".
INCOMPLETE_STOP_REASONS = {"max_tokens", "model_context_window_exceeded", "pause_turn", "tool_use"}


class LLMError(Exception):
    """
    One error type for every problem talking to Claude.
    `kind` tells the program what happened; the message never contains the key.
    """
    def __init__(self, kind: str, message: str, status_code: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status_code = status_code


@dataclass
class LLMResponse:
    text: str                       # the answer text (thinking is never included)
    model: str
    stop_reason: str | None
    complete: bool                  # True only for a normally finished answer
    refused: bool                   # True if Claude's safety system declined
    refusal_category: str | None
    request_id: str | None
    estimated_input_tokens: int     # our safe upper estimate (before sending)
    max_cost_estimate_usd: float    # worst case (before sending)
    usage: dict                     # real token counts + usage-based cost estimate (after)
    warnings: list = field(default_factory=list)

    @property
    def usage_cost_estimate_usd(self) -> float:
        return self.usage["usage_cost_estimate_usd"]


class LLMClient:
    def __init__(self, settings: Settings, client=None, budget: BudgetTracker | None = None,
                 log_path: Path | None = DEFAULT_LOG_PATH):
        self.settings = settings
        self._client = client            # None -> the real SDK client is created on first use
        self.budget = budget or BudgetTracker(settings.budget_usd)
        self.logger = UsageLogger(log_path)

    # ---------- step 1: build ----------
    def build_request(self, messages: list, system: str = SYSTEM_PROMPT) -> dict:
        """The exact keyword arguments given to client.messages.create(...)."""
        return {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "system": system,
            "messages": messages,
            # Effort controls how much the model thinks. Thinking itself is
            # adaptive by default on this model, so we do not send `thinking`.
            "output_config": {"effort": self.settings.effort},
        }

    # ---------- steps 2-3: estimate and check ----------
    def preflight(self, request: dict) -> dict:
        """Estimate the worst case and check the budget. Raises BudgetExceededError if it doesn't fit."""
        estimated_input = estimate_input_tokens(request_text(request["system"], request["messages"]))
        max_cost = estimate_max_cost(self.settings.model, estimated_input, request["max_tokens"],
                                     attempts=self.settings.attempts)
        self.budget.check(max_cost)
        return {"estimated_input_tokens": estimated_input, "max_cost_estimate_usd": max_cost}

    # ---------- the whole flow ----------
    def ask(self, messages: list, system: str = SYSTEM_PROMPT) -> LLMResponse:
        request = self.build_request(messages, system)
        try:
            estimate = self.preflight(request)
        except BudgetExceededError as error:
            self.logger.write(model=self.settings.model, outcome="refused_by_budget",
                              error_type="BudgetExceededError")
            raise LLMError("budget", str(error)) from error

        client = self._get_client()
        try:
            response = client.messages.create(**request)                   # step 4: send
        except Exception as error:                                         # step 5: errors
            raise self._handle_error(error, estimate) from error

        usage = usage_cost(self.settings.model, response.usage)            # step 6: record
        self.budget.record(usage["usage_cost_estimate_usd"], label="response")

        result = self._interpret(response, estimate, usage)                # step 7: interpret
        self.logger.write(                                                 # step 8: log
            model=result.model, outcome="ok" if result.complete else "incomplete_or_refused",
            request_id=result.request_id, stop_reason=result.stop_reason,
            input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"],
            cache_creation_input_tokens=usage["cache_creation_input_tokens"],
            cache_read_input_tokens=usage["cache_read_input_tokens"],
            estimated_input_tokens=estimate["estimated_input_tokens"],
            max_cost_estimate_usd=estimate["max_cost_estimate_usd"],
            usage_cost_estimate_usd=usage["usage_cost_estimate_usd"],
        )
        return result

    # ---------- helpers ----------
    def _get_client(self):
        """Create the real SDK client only when first needed (and only if a key exists)."""
        if self._client is None:
            if not has_credentials():
                raise LLMError("missing_credentials",
                               "No API key is configured. Nothing was sent. (Stage 3: the key is "
                               "added only after the offline tests are approved.)")
            import anthropic  # the official SDK; it reads the key from the environment itself
            self._client = anthropic.Anthropic(timeout=self.settings.timeout_seconds,
                                               max_retries=self.settings.max_retries)
        return self._client

    def _handle_error(self, error: Exception, estimate: dict) -> LLMError:
        """Map SDK errors (most specific first) to LLMError. Never include request content or keys."""
        import anthropic
        status = getattr(error, "status_code", None)
        if isinstance(error, anthropic.APITimeoutError):
            # The server may have processed (and billed) the request before the
            # timeout, so we conservatively charge the reserved worst case.
            self.budget.record(estimate["max_cost_estimate_usd"], label="timeout (worst case reserved)")
            kind, text = "timeout", "The request timed out after all retries. Its worst-case cost was counted against the budget."
        elif isinstance(error, anthropic.AuthenticationError):
            kind, text = "authentication", "The API key was rejected (invalid, revoked or missing)."
        elif isinstance(error, anthropic.PermissionDeniedError):
            kind, text = "permission", "The API key is not allowed to use this model or feature."
        elif isinstance(error, anthropic.NotFoundError):
            kind, text = "not_found", f"Model '{self.settings.model}' or the endpoint was not found. Check ANTHROPIC_MODEL."
        elif isinstance(error, anthropic.RateLimitError):
            kind, text = "rate_limit", "Rate limit reached even after automatic retries. Wait a minute and try again."
        elif isinstance(error, anthropic.BadRequestError):
            kind, text = "bad_request", "The API rejected the request as invalid (HTTP 400)."
        elif isinstance(error, anthropic.APIStatusError):
            kind, text = "server_error", f"The API returned an error (HTTP {status}) after automatic retries."
        elif isinstance(error, anthropic.APIConnectionError):
            kind, text = "connection", "Could not connect to the API (network problem)."
        else:
            kind, text = "unexpected", f"Unexpected error: {type(error).__name__}."
        self.logger.write(model=self.settings.model, outcome="error", error_type=type(error).__name__,
                          status_code=status, estimated_input_tokens=estimate["estimated_input_tokens"],
                          max_cost_estimate_usd=estimate["max_cost_estimate_usd"])
        return LLMError(kind, text, status_code=status)

    def _interpret(self, response, estimate: dict, usage: dict) -> LLMResponse:
        """Turn the SDK response into our LLMResponse, with honest flags."""
        stop_reason = getattr(response, "stop_reason", None)
        # Only "text" blocks are the answer; "thinking" blocks are internal reasoning.
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        refused = stop_reason == "refusal"
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None) if refused and details else None

        warnings = []
        if stop_reason in INCOMPLETE_STOP_REASONS:
            warnings.append(f"INCOMPLETE answer (stop_reason={stop_reason}). Do not treat it as a full answer.")
        elif refused:
            warnings.append("Claude's safety system declined this request (stop_reason=refusal).")
        elif stop_reason not in COMPLETE_STOP_REASONS:
            warnings.append(f"Unknown stop_reason '{stop_reason}'; treated as incomplete.")
        actual_input = (usage["input_tokens"] + usage["cache_creation_input_tokens"]
                        + usage["cache_read_input_tokens"])
        if usage["output_tokens"] > self.settings.max_tokens:
            warnings.append("Usage reports more output tokens than max_tokens allows; the cost "
                            "may exceed the maximum estimate. Please report this.")
        if actual_input > estimate["estimated_input_tokens"]:
            warnings.append("Input token estimate was too LOW; the budget check was not conservative "
                            "enough. Please report this.")

        return LLMResponse(
            text=text, model=getattr(response, "model", self.settings.model), stop_reason=stop_reason,
            complete=stop_reason in COMPLETE_STOP_REASONS, refused=refused, refusal_category=category,
            request_id=getattr(response, "_request_id", None),
            estimated_input_tokens=estimate["estimated_input_tokens"],
            max_cost_estimate_usd=estimate["max_cost_estimate_usd"], usage=usage, warnings=warnings,
        )
