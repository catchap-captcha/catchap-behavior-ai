"""LLM chatbot (kid tutor 냥냥이 + parent counselor 상담사).

Public entry: :func:`chat.service.reply` (also `kid_reply` / `parent_reply`).
Conversational ability comes from Claude and works without collected data;
personalization uses the learning diagnosis when available. See
``docs/CHATBOT.md`` for the design.
"""

from chat.config import ROLE_KID, ROLE_PARENT, ROLES
from chat.engine import ChatEngine
from chat.service import kid_reply, parent_reply, reply

__all__ = ["reply", "kid_reply", "parent_reply", "ChatEngine", "ROLE_KID", "ROLE_PARENT", "ROLES"]
