"""
The "agent" package: everything that talks to the LLM.

Stage 3 contents (the "phone line" to Claude):
    config.py      settings from .env (model, effort, limits) - never the key's value
    pricing.py     the price table and all token/cost arithmetic
    budget.py      the session spending cap, checked BEFORE every request
    privacy.py     builds a safe, minimal dataset summary (no rows, no personal data)
    prompts.py     the system prompt and how data is placed in a request
    usage_log.py   writes a metadata-only log line per request
    llm_client.py  LLMClient: puts all of the above together around the SDK
    testing.py     a fake Anthropic client, so tests never need a key or money

Not built yet (later stages): tool registry (Stage 4), agent loop and
skill loader (Stage 5).
"""
