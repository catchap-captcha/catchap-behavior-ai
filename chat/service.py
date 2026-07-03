"""Chat orchestration — the public entry point.

    reply(role, user_message, history, learning_result)
      = pick persona(role) + inject learning context + send history to Claude

The learning analysis (learning.diagnose) is done elsewhere and passed in as
`learning_result`; this layer only turns it into context and drives the LLM.
The chat works even with `learning_result=None` (no data yet) — it just stays
general instead of personalized.
"""

from __future__ import annotations

from chat.config import ROLE_KID, ROLE_PARENT, ROLES
from chat.context import build_learning_context
from chat.engine import ChatEngine
from chat.prompts import system_for
from learning.models import DiagnoseResult


def reply(
    role: str,
    user_message: str,
    history: list[dict[str, str]] | None = None,
    learning_result: DiagnoseResult | None = None,
    engine: ChatEngine | None = None,
) -> str:
    """Generate one chatbot reply.

    Args:
        role: "kid" (냥냥이) or "parent" (상담사).
        user_message: the new user message.
        history: prior turns [{"role":"user"/"assistant","content":...}, ...].
            The LLM is stateless, so the full history is resent each call.
        learning_result: this student's diagnosis, or None if no data yet.
        engine: LLM engine (injectable for tests); a default is created if None.

    Returns:
        The assistant's text reply.
    """
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; use one of {ROLES}")

    context = build_learning_context(learning_result, role)
    system = system_for(role, context)
    messages = list(history or []) + [{"role": "user", "content": user_message}]

    engine = engine or ChatEngine()
    return engine.generate(system, messages)


# convenience wrappers
def kid_reply(user_message, history=None, learning_result=None, engine=None) -> str:
    """냥냥이(아이용) 답변."""
    return reply(ROLE_KID, user_message, history, learning_result, engine)


def parent_reply(user_message, history=None, learning_result=None, engine=None) -> str:
    """상담사(학부모용) 답변."""
    return reply(ROLE_PARENT, user_message, history, learning_result, engine)
