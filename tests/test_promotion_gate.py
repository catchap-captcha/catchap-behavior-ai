"""승격 게이트 — 특히 **미측정이 통과로 새지 않는지**를 지킨다 (2026-08-18).

`training/promotion_gate.py` 의 존재 이유가 두 가지고, 시험도 그 둘을 본다.

  1. 재지 않은 항목(None)은 불합격이어야 한다.
  2. 기준값이 도구마다 갈라지면 안 된다 — 갈라지는 순간 "어느 것이 진짜 기준인가"를
     사람이 그때그때 고르게 되고, 그러면 기준이 없는 것과 같다.
"""

from __future__ import annotations

import re
from pathlib import Path

from training.promotion_gate import (
    MAX_HUMAN_FRR,
    MAX_KNOWN_BOT_ASR,
    MAX_UNSEEN_BOT_ASR,
    evaluate_promotion,
)

PASSING = dict(
    worst_human_frr=0.02,
    known_bot_asr=0.01,
    worst_unseen_asr=0.05,
    sealed_holdout_verified=True,
)


def test_다_통과하면_승격_가능():
    assert evaluate_promotion(**PASSING).passed


def test_재지_않은_항목은_불합격이다():
    """★이 파일에서 제일 중요한 시험.

    `None` 을 통과로 처리하면 "안 재봤다" 가 보고서에 "문제 없다" 로 찍힌다.
    승격에서 그 방향의 실수는 봇을 들여보낸 뒤에야 드러난다.
    """
    for field in ("worst_human_frr", "known_bot_asr", "worst_unseen_asr"):
        args = dict(PASSING) | {field: None}
        verdict = evaluate_promotion(**args)
        assert not verdict.passed, f"{field} 를 안 쟀는데 통과했다"
        assert "재지 않음" in verdict.report()


def test_기준값과_같으면_통과한다():
    """경계는 포함이다 — `<=`. 기준을 3%로 정해 놓고 3.0%를 떨어뜨리면
    기준이 실제로는 3%가 아닌 것이 된다."""
    assert evaluate_promotion(
        worst_human_frr=MAX_HUMAN_FRR,
        known_bot_asr=MAX_KNOWN_BOT_ASR,
        worst_unseen_asr=MAX_UNSEEN_BOT_ASR,
        sealed_holdout_verified=True,
    ).passed


def test_기준을_한_톨이라도_넘으면_불합격():
    verdict = evaluate_promotion(**(dict(PASSING) | {"worst_unseen_asr": MAX_UNSEEN_BOT_ASR + 1e-9}))
    assert not verdict.passed


def test_봉인_해시가_대조되지_않으면_막는다():
    """점수가 아무리 좋아도, 봉인이 봉인인지 증명 못 하면 근거가 못 된다."""
    assert not evaluate_promotion(**(dict(PASSING) | {"sealed_holdout_verified": False})).passed
    # 확인 자체를 안 한 경우도 막는다
    args = dict(PASSING); args.pop("sealed_holdout_verified")
    assert not evaluate_promotion(**args).passed


def test_shadow_관측이_부족하면_막는다():
    """오프라인 점수만으로 켜면 실사용자 오탐을 한 번도 안 보고 켜는 셈이다."""
    assert not evaluate_promotion(
        **PASSING, shadow_outcomes_observed=120, min_shadow_outcomes=1000).passed
    assert evaluate_promotion(
        **PASSING, shadow_outcomes_observed=1200, min_shadow_outcomes=1000).passed
    # 기본값(0)이면 이 전제조건은 보지 않는다 — 오프라인 심사용
    assert evaluate_promotion(**PASSING).passed


def _numbers_in(path: Path, pattern: str) -> list[float]:
    return [float(m) for m in re.findall(pattern, path.read_text())]


def test_다른_도구의_기준값이_갈라지지_않았다():
    """★기준 표류 감시.

    이 파일을 만든 이유가 "같은 기준이 네 파일에 흩어져 있었다" 였다. 도구들을
    한꺼번에 고치는 것은 위험해서 값만 옮겨 왔으므로, 대신 **갈라지면 여기서
    걸리게** 해 둔다. 도구 쪽 기준을 바꾸려면 이 시험도 같이 고쳐야 하고,
    그 순간 "기준을 바꾸는 중" 임이 드러난다.
    """
    ext = Path("tools/evaluate_candidate_external_holdout.py")
    if ext.exists():
        bot = _numbers_in(ext, r"MAX_BOT_ASR\s*=\s*([0-9.]+)")
        frr = _numbers_in(ext, r"MAX_HUMAN_FRR\s*=\s*([0-9.]+)")
        assert bot == [MAX_KNOWN_BOT_ASR], f"아는 봇 기준이 갈라졌다: {bot}"
        assert frr == [MAX_HUMAN_FRR], f"사람 오탐 기준이 갈라졌다: {frr}"

    con = Path("tools/consolidate_formal_validation.py")
    if con.exists():
        text = con.read_text()
        assert f"<= {MAX_HUMAN_FRR}" in text, "consolidate 의 사람 오탐 기준이 갈라졌다"
        assert f"<= {MAX_KNOWN_BOT_ASR}" in text, "consolidate 의 아는 봇 기준이 갈라졌다"
        assert f"<= {MAX_UNSEEN_BOT_ASR}" in text, "consolidate 의 미지 계열 기준이 갈라졌다"
