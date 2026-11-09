"""
A FAKE Anthropic client for tests and the offline demo.

It behaves like the real SDK client in the one way we use it,
client.messages.create(**request), but returns prepared answers instead of
calling the internet. So tests are free, fast, repeatable and need no key.

Every response it returns is clearly marked as SIMULATED in its request ID.
"""

from types import SimpleNamespace


def fake_response(text: str = "Simulated answer.", stop_reason: str = "end_turn",
                  input_tokens: int = 900, output_tokens: int = 150,
                  cache_creation: int = 0, cache_read: int = 0, thinking: str | None = None,
                  refusal_category: str | None = None, model: str = "claude-sonnet-5-5"):
    """Build an object shaped like the SDK's Message (only the fields we read)."""
    content = []
    if thinking is not None:
        content.append(SimpleNamespace(type="thinking", thinking=thinking))
    if text:
        content.append(SimpleNamespace(type="text", text=text))
    return SimpleNamespace(
        model=model, content=content, stop_reason=stop_reason,
        stop_details=(SimpleNamespace(type="refusal", category=refusal_category, explanation=None)
                      if stop_reason == "refusal" else None),
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens,
                              cache_creation_input_tokens=cache_creation,
                              cache_read_input_tokens=cache_read),
        _request_id="req_SIMULATED_0001",
    )


class _FakeMessages:
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = []                      # every request we were given, for tests to inspect

    def create(self, **request):
        self.calls.append(request)
        if not self._outcomes:
            raise AssertionError("FakeAnthropicClient: no more prepared responses")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):   # a prepared error: raise it like the SDK would
            raise outcome
        return outcome


class FakeAnthropicClient:
    """Usage: FakeAnthropicClient([fake_response(...), SomeError(...), ...])"""
    def __init__(self, outcomes):
        self.messages = _FakeMessages(outcomes)
