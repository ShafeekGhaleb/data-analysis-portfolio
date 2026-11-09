"""
Prices and token arithmetic.

WHAT IS A TOKEN?
The model reads and writes text in small pieces called tokens (roughly
3/4 of an English word). You pay per token:
    input tokens  = everything we SEND (system prompt + messages)
    output tokens = everything Claude WRITES, including its hidden thinking

TWO KINDS OF COST NUMBER IN THIS PROJECT (never mix them up):
1. MAXIMUM cost estimate (BEFORE a request): the worst case if Claude used
   every output token we allow. Used to decide whether a request may be sent.
2. USAGE-BASED cost estimate (AFTER a response): the token counts Anthropic
   reports, multiplied by our price table.
Neither is the ACTUAL BILLED cost. Only your Anthropic Console billing page
shows that (it can differ, e.g. through account discounts or price changes).

Prices verified on 2026-11-09 against the official pages:
    https://platform.claude.com/docs/en/about-claude/pricing
    https://platform.claude.com/docs/en/about-claude/models/overview
Facts from the official thinking documentation that our formulas rely on:
- "max_tokens is a hard cap on total output for the request, thinking and
  response text combined."
- Thinking tokens are "billed as output tokens", and usage.output_tokens is
  "the inclusive, authoritative total used for billing".
"""

import math
from dataclasses import dataclass

PRICES_VERIFIED_ON = "2026-11-09"
PRICING_SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"

PER_MILLION = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """US dollars per ONE MILLION tokens."""
    input: float
    output: float
    cache_write_5m: float   # writing a prompt into the 5-minute cache
    cache_read: float       # reading a cached prompt (much cheaper)
    max_prompt_tokens: int | None = None  # above this, the price is different (see notes)


# Only models listed here can be used: we refuse any model whose price we
# do not know, because then the budget could not be protected.
PRICE_TABLE = {
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_write_5m=2.50, cache_read=0.10),
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00, cache_write_5m=5.00, cache_read=0.20),
    # Haiku 5.5 is cheaper for prompts up to 100,000 tokens; longer prompts cost
    # 5x more. We only allow it below that limit (checked in estimate_max_cost).
    "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50, cache_write_5m=0.125, cache_read=0.01,
                                   max_prompt_tokens=100_000),
}

# Fixed allowance for the request's structure (roles, separators, ...) that is
# not part of our visible text. Generous on purpose.
REQUEST_OVERHEAD_TOKENS = 200


class PricingError(Exception):
    """Raised when a cost cannot be calculated safely."""


def price_for(model: str) -> ModelPrice:
    """Look up a model's prices, or refuse clearly."""
    if model not in PRICE_TABLE:
        raise PricingError(
            f"No verified price for model '{model}'. Known models: {sorted(PRICE_TABLE)}. "
            "Add its price (from the official pricing page) to agent/pricing.py first."
        )
    return PRICE_TABLE[model]


def estimate_input_tokens(text: str) -> int:
    """
    A SAFE UPPER estimate of how many input tokens `text` will use.

    Rule: count the text's UTF-8 BYTES (1 English letter = 1 byte; "€" = 3
    bytes) and assume each token covers at least one byte. Real tokens
    usually cover 3-4 English characters, so the true count is normally far
    lower. Overestimating input is cheap insurance: at $2 per million
    tokens, 3,000 extra tokens cost $0.006.

    After every response we compare this estimate with the real count and
    flag it if it was ever too low (see LLMClient).
    """
    return len(text.encode("utf-8")) + REQUEST_OVERHEAD_TOKENS


def cost_of(tokens: int, price_per_million: float) -> float:
    """tokens x (price per million / 1,000,000)."""
    return tokens * price_per_million / PER_MILLION


def estimate_max_cost(model: str, input_tokens: int, max_tokens: int, attempts: int = 1) -> float:
    """
    The MOST a request could cost, before sending it:

        per attempt = input_tokens x input price  +  max_tokens x output price
        maximum     = per attempt x attempts

    max_tokens is a hard cap on ALL output (thinking + answer), so Claude can
    never be billed for more output than this. `attempts` covers the SDK's
    automatic retries: a request that times out might still have been
    processed (and billed) by the server before being sent again.
    """
    price = price_for(model)
    if price.max_prompt_tokens and input_tokens > price.max_prompt_tokens:
        raise PricingError(
            f"Prompt estimate {input_tokens:,} tokens is above {price.max_prompt_tokens:,}, where "
            f"'{model}' uses a higher price tier that this project does not model."
        )
    per_attempt = cost_of(input_tokens, price.input) + cost_of(max_tokens, price.output)
    return per_attempt * attempts


def usage_cost(model: str, usage) -> dict:
    """
    The usage-based cost estimate AFTER a response, from the token counts
    Anthropic reported in response.usage.

    usage.input_tokens counts only the UNCACHED input; cached input is
    reported separately (cache_creation_input_tokens, cache_read_input_tokens)
    and priced differently. A missing field (None) counts as 0.
    """
    price = price_for(model)
    tokens = {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
    }
    parts = {
        "input": cost_of(tokens["input_tokens"], price.input),
        "output": cost_of(tokens["output_tokens"], price.output),
        "cache_write": cost_of(tokens["cache_creation_input_tokens"], price.cache_write_5m),
        "cache_read": cost_of(tokens["cache_read_input_tokens"], price.cache_read),
    }
    return {**tokens, "cost_parts_usd": parts, "usage_cost_estimate_usd": sum(parts.values())}


def round_usd(value: float) -> float:
    """Round UP to the next 1/1,000,000 of a dollar, so displayed costs are never understated."""
    return math.ceil(value * PER_MILLION) / PER_MILLION
