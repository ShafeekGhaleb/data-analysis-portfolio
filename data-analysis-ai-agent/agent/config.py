"""
Settings for talking to Claude, read from environment variables / the .env file.

The .env file (git-ignored) holds your personal settings and, later, the API
key. This module reads the SETTINGS. It only ever checks whether a key EXISTS;
it never reads, returns, prints or logs the key's value. The official SDK
reads the key itself when it sends a request.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from agent.pricing import PRICE_TABLE

PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "ANTHROPIC_MODEL": "claude-sonnet-5-5",  # your Stage 3 decision
    "LLM_EFFORT": "low",                      # how much the model thinks (see below)
    "LLM_MAX_TOKENS": "4000",                 # hard cap on output per request (thinking + answer)
    "LLM_BUDGET_USD": "1.00",                 # session spending cap (your decision)
    "LLM_TIMEOUT_SECONDS": "60",
    "LLM_MAX_RETRIES": "2",                   # the SDK's automatic retries
}

# Effort = how much the model may think before answering. More thinking =
# more output tokens = more cost. For a simple question, "low" is enough.
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
MAX_TOKENS_LIMIT = 64_000      # we never allow more per request in this project
BUDGET_HARD_CEILING = 20.00    # a typo like "100" must not silently allow $100

# The environment variables the SDK accepts as credentials. We only test
# whether one is SET; we never look at its value.
CREDENTIAL_VARIABLES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


class ConfigError(Exception):
    """A setting is missing or invalid. The message never contains secrets."""


@dataclass(frozen=True)
class Settings:
    model: str
    effort: str
    max_tokens: int
    budget_usd: float
    timeout_seconds: float
    max_retries: int

    @property
    def attempts(self) -> int:
        """The first try plus the SDK's automatic retries."""
        return 1 + self.max_retries


def load_dotenv_file(path: Path | None = None) -> None:
    """
    Copy settings from .env into the environment, WITHOUT overriding values
    that are already set (so a real environment variable always wins).
    """
    from dotenv import load_dotenv  # imported here so tests don't need a .env file
    load_dotenv(path or PROJECT_DIR / ".env", override=False)


def load_settings(env: dict | None = None) -> Settings:
    """
    Build validated Settings from `env` (default: the real environment).

    Passing a plain dict makes this easy to test without touching the real
    environment or any .env file.
    """
    source = os.environ if env is None else env
    raw = {name: str(source.get(name) or default).strip() for name, default in DEFAULTS.items()}

    model = raw["ANTHROPIC_MODEL"]
    if model not in PRICE_TABLE:
        raise ConfigError(f"ANTHROPIC_MODEL='{model}' has no verified price, so the budget "
                          f"cannot be protected. Choose one of {sorted(PRICE_TABLE)}.")
    effort = raw["LLM_EFFORT"].lower()
    if effort not in EFFORT_LEVELS:
        raise ConfigError(f"LLM_EFFORT='{effort}' is invalid. Choose one of {EFFORT_LEVELS}.")

    try:
        max_tokens = int(raw["LLM_MAX_TOKENS"])
        budget = float(raw["LLM_BUDGET_USD"])
        timeout = float(raw["LLM_TIMEOUT_SECONDS"])
        retries = int(raw["LLM_MAX_RETRIES"])
    except ValueError as error:
        raise ConfigError(f"A numeric LLM setting is not a valid number: {error}")

    if not 1 <= max_tokens <= MAX_TOKENS_LIMIT:
        raise ConfigError(f"LLM_MAX_TOKENS must be between 1 and {MAX_TOKENS_LIMIT:,}.")
    if not 0 < budget <= BUDGET_HARD_CEILING:
        raise ConfigError(f"LLM_BUDGET_USD must be above 0 and at most {BUDGET_HARD_CEILING:.2f}.")
    if not 1 <= timeout <= 600:
        raise ConfigError("LLM_TIMEOUT_SECONDS must be between 1 and 600.")
    if not 0 <= retries <= 5:
        raise ConfigError("LLM_MAX_RETRIES must be between 0 and 5.")

    return Settings(model=model, effort=effort, max_tokens=max_tokens, budget_usd=budget,
                    timeout_seconds=timeout, max_retries=retries)


def has_credentials(env: dict | None = None) -> bool:
    """True if an API credential variable is set. The value itself is never read out."""
    source = os.environ if env is None else env
    return any(bool(source.get(name)) for name in CREDENTIAL_VARIABLES)
