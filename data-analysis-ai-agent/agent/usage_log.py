"""
A metadata-only usage log: one JSON line per request in logs/llm_usage.jsonl.

ALLOWED fields only (enforced by ALLOWED_FIELDS): when, which model, how
many tokens, the cost estimates, the request ID and the outcome.
NEVER logged: prompts, answers, dataset content, error message text, keys.
(Error messages are not logged because an API error text could quote parts
of the request.)
"""

import json
from datetime import datetime, timezone
from pathlib import Path

ALLOWED_FIELDS = {
    "timestamp_utc", "model", "outcome", "request_id", "stop_reason",
    "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens",
    "estimated_input_tokens", "max_cost_estimate_usd", "usage_cost_estimate_usd",
    "error_type", "status_code",
}


class UsageLogger:
    def __init__(self, path: Path | None):
        """path=None turns logging off (used by most tests)."""
        self.path = Path(path) if path else None

    def write(self, **fields) -> dict:
        unknown = set(fields) - ALLOWED_FIELDS
        if unknown:
            # A programming mistake must never put content into the log.
            raise ValueError(f"Refusing to log non-metadata fields: {sorted(unknown)}")
        entry = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as file:   # "a" = append
                file.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry
