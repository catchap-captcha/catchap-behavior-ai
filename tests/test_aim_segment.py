"""조준 구간을 자르는 규칙 — 학습과 채점이 같은 것을 봐야 한다.

한쪽만 자르면 그 차이가 그대로 오차가 되므로, 이 규칙은 `app/services/aim_segment.py`
하나에만 있고 학습 도구도 그것을 부른다. 여기서 지키는 것은 그 규칙의 경계다.
"""

from __future__ import annotations

from app.services.aim_segment import AIM_GAP_MS, trim_aim, without_aim


def aim(t: float) -> dict:
    return {"event_type": "aimmove", "t_ms": t, "x": t, "y": t}


def drag(t: float, kind: str = "pointermove") -> dict:
    return {"event_type": kind, "t_ms": t, "x": t, "y": t}


def test_마지막_뻗기만_남긴다():
    """400ms 넘게 끊기면 그 앞은 다른 움직임이다 — 문제를 읽는 동안의 배회다."""
    events = [aim(0), aim(50), aim(100),
              aim(100 + AIM_GAP_MS + 1), aim(100 + AIM_GAP_MS + 51),
              drag(2000, "pointerdown"), drag(2050), drag(2100, "pointerup")]
    out = trim_aim(events)
    kept = [e["t_ms"] for e in out if e["event_type"] == "aimmove"]
    assert kept == [501.0, 551.0]
    # 드래그는 하나도 잃지 않는다.
    assert [e["t_ms"] for e in out if e["event_type"] != "aimmove"] == [2000, 2050, 2100]


def test_끊김이_없으면_그대로_둔다():
    events = [aim(0), aim(50), aim(100), drag(200, "pointerdown"), drag(250)]
    assert len(trim_aim(events)) == len(events)


def test_경계값에서는_자르지_않는다():
    """정확히 400ms 는 아직 같은 움직임이다. 넘어야 나눈다."""
    events = [aim(0), aim(AIM_GAP_MS), drag(1000, "pointerdown"), drag(1050)]
    kept = [e["t_ms"] for e in trim_aim(events) if e["event_type"] == "aimmove"]
    assert kept == [0.0, AIM_GAP_MS]


def test_조준이_없거나_하나뿐이면_그대로():
    only_drag = [drag(0, "pointerdown"), drag(50), drag(100, "pointerup")]
    assert trim_aim(only_drag) == only_drag
    one = [aim(0), drag(50, "pointerdown"), drag(100)]
    assert len(trim_aim(one)) == 3


def test_시간순으로_돌려준다():
    """자른 결과는 다시 섞이면 안 된다 — 뒤 단계가 순서를 믿는다."""
    events = [drag(2050), drag(2000, "pointerdown"), aim(1000), aim(1050)]
    assert [e["t_ms"] for e in trim_aim(events)] == [1000, 1050, 2000, 2050]


def test_without_aim_은_조준만_뺀다():
    events = [aim(0), aim(50), drag(100, "pointerdown"), drag(150)]
    out = without_aim(events)
    assert all(e["event_type"] != "aimmove" for e in out)
    assert len(out) == 2


def test_원본을_바꾸지_않는다():
    """자른 결과는 같은 dict 객체를 가리킨다. 여기서 손대면 저장·재생 경로가 쓰는
    원본까지 바뀐다 — 그래서 호출부가 `seq` 를 다시 매기지 않는다."""
    events = [aim(0), aim(1000), drag(2000, "pointerdown"), drag(2050)]
    before = [dict(e) for e in events]
    trim_aim(events)
    assert events == before
