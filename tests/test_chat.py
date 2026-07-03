"""Tests for the LLM chatbot package.

Uses a fake Claude client (records the call, returns canned text) so the whole
suite runs without the `anthropic` package or an API key.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from chat import ChatEngine, kid_reply, parent_reply, reply
from chat.context import NO_DATA_NOTE, build_learning_context
from chat.prompts import system_for
from learning.models import DiagnoseResult, MasteryResult, WeakConcept


# --------------------------------------------------------------------------- #
# fake Claude client
# --------------------------------------------------------------------------- #
@dataclass
class _Block:
    type: str
    text: str


@dataclass
class _Resp:
    content: list


class FakeClient:
    """Records the last messages.create call and returns a fixed reply."""

    def __init__(self, reply_text="안녕! 🐱"):
        self.reply_text = reply_text
        self.last_call = None

    class _Messages:
        def __init__(self, outer):
            self.outer = outer

        def create(self, **kwargs):
            self.outer.last_call = kwargs
            return _Resp(content=[_Block("text", self.outer.reply_text)])

    @property
    def messages(self):
        return FakeClient._Messages(self)


def _fake_engine(text="안녕! 🐱"):
    client = FakeClient(text)
    return ChatEngine(client=client), client


def _diagnosis():
    return DiagnoseResult(
        student_id="haeun",
        mastery={
            "FRACTION_ADD": MasteryResult("FRACTION_ADD", 0.42, 8, 3, 0.33, False),
            "DECIMAL_CMP": MasteryResult("DECIMAL_CMP", 0.90, 8, 7, 0.33, False),
        },
        weak_concepts=[WeakConcept("FRACTION_ADD", 0.42, 0.61, "최근 5문제 중 4개 오답")],
        diagnostic_needed=["SHAPES"],
        recommendations=[],
    )


# --------------------------------------------------------------------------- #
# context builder
# --------------------------------------------------------------------------- #
def test_context_none_returns_no_data_note():
    assert build_learning_context(None, "kid") == NO_DATA_NOTE


def test_context_kid_hides_numbers():
    ctx = build_learning_context(_diagnosis(), "kid")
    assert "FRACTION_ADD" in ctx
    assert "42%" not in ctx          # numbers hidden from the child
    assert "DECIMAL_CMP" in ctx      # strength surfaced


def test_context_parent_shows_numbers():
    ctx = build_learning_context(_diagnosis(), "parent")
    assert "42%" in ctx              # parent gets detail
    assert "최근 5문제" in ctx


# --------------------------------------------------------------------------- #
# prompt selection
# --------------------------------------------------------------------------- #
def test_system_for_differs_by_role():
    kid = system_for("kid", "ctx")
    parent = system_for("parent", "ctx")
    assert "냥냥이" in kid and "반말" in kid
    assert "상담사" in parent and "존댓말" in parent
    assert "ctx" in kid and "ctx" in parent


# --------------------------------------------------------------------------- #
# service.reply
# --------------------------------------------------------------------------- #
def test_reply_returns_text_and_sends_history():
    engine, client = _fake_engine("힌트 줄게 🐱")
    history = [
        {"role": "user", "content": "안녕"},
        {"role": "assistant", "content": "안녕 하은아!"},
    ]
    out = kid_reply("그림찾기 어려워", history=history, learning_result=_diagnosis(), engine=engine)
    assert out == "힌트 줄게 🐱"

    call = client.last_call
    # full history + new message resent (LLM is stateless)
    assert len(call["messages"]) == 3
    assert call["messages"][-1] == {"role": "user", "content": "그림찾기 어려워"}
    # kid persona + injected learning context in the system prompt
    assert "냥냥이" in call["system"]
    assert "FRACTION_ADD" in call["system"]


def test_kid_and_parent_use_different_system():
    ke, kc = _fake_engine()
    pe, pc = _fake_engine()
    kid_reply("도와줘", learning_result=_diagnosis(), engine=ke)
    parent_reply("우리 아이 어때요?", learning_result=_diagnosis(), engine=pe)
    assert "냥냥이" in kc.last_call["system"]
    assert "상담사" in pc.last_call["system"]


def test_reply_without_data_stays_general():
    engine, client = _fake_engine()
    kid_reply("안녕", engine=engine)  # no learning_result
    assert NO_DATA_NOTE in client.last_call["system"]


def test_unknown_role_raises():
    with pytest.raises(ValueError):
        reply("teacher", "hi", engine=_fake_engine()[0])
