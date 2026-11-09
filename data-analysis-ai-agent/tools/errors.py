"""
One custom error type for all our tools.

Why do we need our own error?
- When something goes wrong (file not found, column does not exist, ...),
  we want a CLEAR message that a human (and later the LLM) can understand
  and act on, e.g. "Column 'Revnue' not found. Available columns: ...".
- Having one error type means the agent loop (Stage 5) can catch
  `ToolError`, send the message back to the LLM, and let it try again,
  instead of the whole program crashing.
"""


class ToolError(Exception):
    """Raised when a tool cannot do its job. The message explains why."""

    # "class ToolError(Exception)" means: ToolError is a kind of Exception.
    # We don't need to add any code: inheriting is enough to create a new,
    # recognisable error type. "pass" is not even needed because of this docstring.
