"""The shared LLM engine — the only place that calls Claude.

Both the kid tutor and the parent counselor use this same engine; only the
system prompt differs. The Anthropic client is imported lazily and injectable,
so this module imports (and unit-tests run) without the `anthropic` package or
an API key — tests pass a fake client.
"""

from __future__ import annotations

from typing import Any

from chat.config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL


class ChatEngine:
    """Thin wrapper over the Claude Messages API."""

    def __init__(self, client: Any | None = None, model: str = DEFAULT_MODEL) -> None:
        self._client = client
        self.model = model

    def _get_client(self) -> Any:
        """Lazily create a real Anthropic client (reads ANTHROPIC_API_KEY)."""
        if self._client is None:
            from anthropic import Anthropic  # imported here so tests need no key/pkg

            self._client = Anthropic()
        return self._client

    def generate(
        self,
        system: str,
        messages: list[dict[str, str]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Send one request to Claude and return the concatenated text reply."""
        resp = self._get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )
