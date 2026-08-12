"""조준 구간을 판단에 쓸 수 있는 모양으로 자른다.

조준(`aimmove`)은 집기 전 포인터 이동이다. 캡처 버퍼는 사람이 문제를 읽는 동안에도
계속 쌓이므로, 한 챌린지의 조준에는 목표로 뻗은 한 번의 움직임뿐 아니라 그 앞에서
훑어보고 망설인 것까지 들어 있다. 그대로 채점에 넣으면 근거가 아니라 잡음이 된다 —
실측으로 미지 계열 통과율이 8.4% 에서 7.1% 로 내려갔고, 사람 조준 길이 중앙값이
18점에서 12점이 됐다(2026-08-12).

**학습과 채점이 같은 함수를 써야 한다.** 한쪽만 자르면 그 차이가 그대로 오차가 된다.
학습 도구(`tools/train_with_aim.py`)도 여기를 부른다.

경계값 400ms 는 `tools/aim_segments.GAP_MS` 와 같다 — 위젯의 40ms 캡처 주기 네 배로,
사람 조준 간격(중앙 50ms · p95 334ms)의 꼬리 바깥이다.
"""

from __future__ import annotations

from typing import Any, Sequence

AIM_EVENT_TYPE = "aimmove"
AIM_GAP_MS = 400.0


def trim_aim(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """마지막 뻗기 구간의 조준만 남긴다. 드래그 이벤트는 건드리지 않는다.

    조준이 하나뿐이거나 아예 없으면 그대로 돌려준다 — 자를 것이 없다.
    """
    aim = sorted((e for e in events if e.get("event_type") == AIM_EVENT_TYPE),
                 key=lambda e: float(e.get("t_ms") or 0.0))
    if len(aim) < 2:
        return list(events)

    rest = [e for e in events if e.get("event_type") != AIM_EVENT_TYPE]
    start = 0
    for i in range(1, len(aim)):
        gap = float(aim[i].get("t_ms") or 0.0) - float(aim[i - 1].get("t_ms") or 0.0)
        if gap > AIM_GAP_MS:
            start = i
    return sorted(aim[start:] + rest, key=lambda e: float(e.get("t_ms") or 0.0))


def without_aim(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """조준을 뺀다. 조준 없이 학습된 모델에 넘길 때 쓴다."""
    return [e for e in events if e.get("event_type") != AIM_EVENT_TYPE]


__all__ = ["AIM_EVENT_TYPE", "AIM_GAP_MS", "trim_aim", "without_aim"]
