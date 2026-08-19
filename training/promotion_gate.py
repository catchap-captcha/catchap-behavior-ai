"""승격 기준을 **한 곳에** 둔다 — 그리고 미측정을 통과로 읽지 않는다.

왜 이 파일이 있나
-----------------
2026-08-18 기준으로 같은 기준이 네 파일에 흩어져 있었다.

    tools/consolidate_formal_validation.py   FRR 3% · 아는 5% · 미지 10%
    tools/evaluate_candidate_external_holdout.py   MAX_BOT_ASR 0.05 · MAX_HUMAN_FRR 0.03
    tools/aim_lofo.py                        FRR_BUDGET 0.05 · 기준 ≤10%
    docs/PROMOTION_STATUS_20260810.md        표에 사람이 손으로 적은 값

흩어져 있으면 **어느 것이 진짜 기준인지 아무도 확정할 수 없다.** 2026-08-10 에 하루
동안 결론이 세 번 뒤집힌 일이 있었는데, 숫자가 좋아진 것이 아니라 재는 방법이
세 번 고쳐진 것이었다. 기준이 사람 머릿속에만 있으면 그 일이 또 일어난다.

★핵심 설계 — **재지 않은 항목은 불합격이다.**
    `None` 을 통과로 처리하면 "안 재봤다" 가 "문제 없다" 로 보고서에 찍힌다.
    승격에서 그 방향의 실수는 되돌릴 수 없다(봇을 들여보낸 뒤에 안다). 반대 방향은
    한 번 더 재면 된다. 그래서 모르는 것은 전부 불합격 쪽으로 떨어뜨린다.

이 파일은 기준을 **정의만** 한다. 측정은 각 도구가 하던 대로 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ──────────────────────────────────────────────────────────────────────────
# 기준값. 여기가 유일한 출처다.
#
# 출처: tools/consolidate_formal_validation.py 의 `gates_without_replay`,
#       tools/evaluate_candidate_external_holdout.py 의 MAX_* 상수.
#       두 곳이 같은 값을 쓰고 있었고, 그 값을 그대로 옮겼다 — 이 파일을 만들면서
#       기준을 새로 정하지 않았다. 기준을 바꾸는 것은 별도의 결정이어야 한다.
# ──────────────────────────────────────────────────────────────────────────

MAX_HUMAN_FRR = 0.03
"""참가자 한 명 기준 최악 오탐. 평균이 아니라 **최악**을 본다 — 평균은 한 사람이
겪는 일을 감춘다(2026-08-14: 최악 14.1%가 한 명, 나머지 아홉 명은 최악 2.2%)."""

MAX_KNOWN_BOT_ASR = 0.05
"""학습에서 본 계열의 통과율."""

MAX_UNSEEN_BOT_ASR = 0.10
"""그 계열을 빼고 학습했을 때의 통과율(LOFO). 처음 보는 상대에 대한 값이라
아는 계열보다 느슨하게 잡혀 있다."""


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    measured: float | None
    limit: float
    note: str = ""

    def line(self) -> str:
        if self.measured is None:
            return f"  불합격  {self.name}: 재지 않음 (기준 ≤{self.limit * 100:.0f}%)"
        mark = "합격  " if self.passed else "불합격"
        return (f"  {mark}  {self.name}: {self.measured * 100:.1f}% "
                f"(기준 ≤{self.limit * 100:.0f}%)")


@dataclass(frozen=True)
class PromotionVerdict:
    gates: list[GateResult] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(g.passed for g in self.gates) and not self.blockers

    def report(self) -> str:
        lines = [g.line() for g in self.gates]
        for b in self.blockers:
            lines.append(f"  불합격  {b}")
        lines.append("")
        lines.append("판정: 승격 가능" if self.passed else "판정: ★승격 불가")
        return "\n".join(lines)


def _gate(name: str, measured: float | None, limit: float) -> GateResult:
    # 재지 않았으면(None) 불합격. 위 ★ 참고.
    passed = measured is not None and measured <= limit
    return GateResult(name=name, passed=passed, measured=measured, limit=limit)


def evaluate_promotion(
    *,
    worst_human_frr: float | None,
    known_bot_asr: float | None,
    worst_unseen_asr: float | None,
    sealed_holdout_verified: bool | None = None,
    shadow_outcomes_observed: int | None = None,
    min_shadow_outcomes: int = 0,
) -> PromotionVerdict:
    """세 관문 + 두 전제조건으로 승격 여부를 판정한다.

    `sealed_holdout_verified` 는 `tools/lockbox_audit.py --strict` 가 봉인 해시를
    빠짐없이 대조했는지다. 대조되지 않은 봉인으로 낸 점수는 "후보가 본 적 없다"를
    뒷받침하지 못하므로, 숫자가 아무리 좋아도 근거가 되지 않는다.

    `shadow_outcomes_observed` 는 실사용자에게서 모은 shadow 관측 수다. 오프라인
    점수만으로 승격하면 실사용자 오탐을 한 번도 안 보고 켜는 셈이 된다.
    `min_shadow_outcomes=0` 이면 이 전제조건은 확인하지 않는다(오프라인 심사).
    """
    gates = [
        _gate("사람 오탐 (참가자 최악)", worst_human_frr, MAX_HUMAN_FRR),
        _gate("아는 봇 통과율", known_bot_asr, MAX_KNOWN_BOT_ASR),
        _gate("미지 계열 통과율 (최악)", worst_unseen_asr, MAX_UNSEEN_BOT_ASR),
    ]

    blockers: list[str] = []
    if sealed_holdout_verified is None:
        blockers.append("봉인 홀드아웃 해시 대조 여부를 확인하지 않았다")
    elif not sealed_holdout_verified:
        blockers.append("봉인 홀드아웃 해시가 대조되지 않았다 — 이 점수는 근거가 못 된다")

    if min_shadow_outcomes > 0:
        if shadow_outcomes_observed is None:
            blockers.append(f"shadow 관측 수를 확인하지 않았다 (필요 {min_shadow_outcomes}건)")
        elif shadow_outcomes_observed < min_shadow_outcomes:
            blockers.append(
                f"shadow 관측이 부족하다: {shadow_outcomes_observed}건 "
                f"(필요 {min_shadow_outcomes}건)")

    return PromotionVerdict(gates=gates, blockers=blockers)
