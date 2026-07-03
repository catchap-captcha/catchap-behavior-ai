"""Chat (LLM) configuration.

The chatbot's conversational ability comes from Claude (an LLM) and works from
day one — it does not need collected data. Personalization (telling a child what
they're weak at) does need the learning data, but the chat itself does not.
"""

from __future__ import annotations

# Default model. Opus 4.8 is the most capable; for high-volume kid chat you may
# switch to a cheaper tier (claude-haiku-4-5 / claude-sonnet-5) — that's a cost
# decision, so it's left configurable rather than hardcoded downstream.
DEFAULT_MODEL = "claude-opus-4-8"

# Chat replies are short; 1024 is plenty and keeps latency/cost low.
DEFAULT_MAX_TOKENS = 1024

# Roles the chatbot can play (different persona + data exposure + safety).
ROLE_KID = "kid"        # 냥냥이 — child-facing tutor
ROLE_PARENT = "parent"  # 상담사 — parent-facing counselor
ROLES = (ROLE_KID, ROLE_PARENT)
