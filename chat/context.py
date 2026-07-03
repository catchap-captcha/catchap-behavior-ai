"""Turn the learning analysis into a fact block the chatbot can speak from.

This is the "direct injection" approach (not RAG): the learning summary is small
and structured, so we put it straight into the prompt. The chatbot then only
states facts that appear here — it doesn't invent mastery numbers.
"""

from __future__ import annotations

from chat.config import ROLE_KID
from learning.models import DiagnoseResult

NO_DATA_NOTE = "아직 학습 데이터가 없습니다. 구체적인 강약점은 말하지 말고 일반적으로 도와주세요."


def build_learning_context(result: DiagnoseResult | None, role: str) -> str:
    """Build the learning-status text injected into the system prompt.

    Args:
        result: the student's diagnosis (from learning.diagnose). None if no data.
        role: kid vs parent — controls detail level and encouragement framing.

    Returns:
        A plain-text fact block. If there is no usable data, returns NO_DATA_NOTE
        so the bot stays general instead of fabricating.
    """
    if result is None or (not result.weak_concepts and not result.mastery):
        return NO_DATA_NOTE

    lines: list[str] = []

    # strengths first (esp. for the kid, to lead with encouragement)
    strengths = [
        cid for cid, m in result.mastery.items()
        if not m.diagnostic_needed and m.mastery >= 0.7
    ]
    if strengths:
        lines.append(f"잘하는 개념: {', '.join(strengths)}")

    if result.weak_concepts:
        lines.append("부족한 개념:")
        for w in result.weak_concepts:
            if role == ROLE_KID:
                # hide raw numbers from the child; keep it soft
                lines.append(f"- {w.concept_id} (조금 더 연습 필요)")
            else:
                lines.append(f"- {w.concept_id}: 숙련도 {w.mastery_score:.0%}, 이유: {w.reason}")

    if result.diagnostic_needed:
        lines.append(f"아직 판단하기 이른 개념: {', '.join(result.diagnostic_needed)}")

    return "\n".join(lines) if lines else NO_DATA_NOTE
