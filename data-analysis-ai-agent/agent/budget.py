"""
The session spending cap.

THE RULE: a request may be sent only if its MAXIMUM possible cost fits in
what is left of the budget. We check this BEFORE sending, because once a
request is sent, its tokens are spent; checking afterwards would be too late.

"Session" = one BudgetTracker, i.e. one run of the program. The cap resets
when the program starts again. For a limit across all runs, also set a
monthly spend limit in the Anthropic Console.
"""

from dataclasses import dataclass, field


class BudgetExceededError(Exception):
    """Raised BEFORE a request is sent, when it could exceed the remaining budget."""


@dataclass
class BudgetTracker:
    limit_usd: float
    spent_estimate_usd: float = 0.0          # sum of usage-based estimates so far
    requests_recorded: int = 0
    history: list = field(default_factory=list)

    def __post_init__(self):
        if not self.limit_usd > 0:
            raise ValueError("The budget limit must be greater than 0.")

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.limit_usd - self.spent_estimate_usd)

    def check(self, max_cost_usd: float) -> None:
        """Refuse a request whose worst case does not fit. Nothing is sent if this raises."""
        if max_cost_usd > self.remaining_usd:
            raise BudgetExceededError(
                f"Request refused before sending: it could cost up to ${max_cost_usd:.4f}, "
                f"but only ${self.remaining_usd:.4f} of the ${self.limit_usd:.2f} session budget "
                "is left. Lower max_tokens, shorten the input, or raise LLM_BUDGET_USD."
            )

    def record(self, cost_usd: float, label: str = "response") -> None:
        """Add a cost after a request (usage-based estimate, or a reserved worst case)."""
        self.spent_estimate_usd += cost_usd
        self.requests_recorded += 1
        self.history.append({"label": label, "cost_usd": cost_usd})
