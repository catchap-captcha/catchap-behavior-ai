"""Segmentation has to survive the event vocabularies we actually receive."""

from __future__ import annotations

from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags


def press(kind: str, seq: int) -> dict[str, object]:
    return {"seq": seq, "event_type": kind, "t_ms": seq * 20, "x": seq, "y": seq}


def drag(start: int, moves: int, down: str = "pointerdown", up: str = "pointerup"):
    out = [press(down, start)]
    out += [press("pointermove", start + 1 + i) for i in range(moves)]
    out.append(press(up, start + 1 + moves))
    return out


def test_splits_each_press_to_release():
    events = drag(0, 3) + drag(10, 4) + drag(20, 2)
    drags = split_drags(events)
    assert [move_count(d) for d in drags] == [3, 4, 2]


def test_hover_between_drags_is_not_a_drag():
    # Moves with no button held belong to no segment. Counting them would let a
    # caller inflate a drag the same way session aggregates were inflated.
    events = drag(0, 3) + [press("pointermove", 50), press("pointermove", 51)] + drag(60, 2)
    assert len(split_drags(events)) == 2
    assert [move_count(d) for d in split_drags(events)] == [3, 2]


def test_tool_event_names_are_one_drag_not_two():
    # Our red-team tools emit pointer_down AND drag_start for a single press; a
    # real browser emits one pointerdown. Reading that as two drags is how the
    # 2026-07-31 "드래그 0회 17건" miscount happened.
    events = [press("pointer_down", 0), press("drag_start", 1)]
    events += [press("pointer_move", 2 + i) for i in range(4)]
    events.append(press("drop", 10))
    drags = split_drags(events)
    assert len(drags) == 1
    assert move_count(drags[0]) == 4


def test_missing_release_still_yields_the_drag():
    # The movement happened; dropping it would silently shrink the evidence.
    events = drag(0, 3)[:-1]
    drags = split_drags(events)
    assert len(drags) == 1
    assert move_count(drags[0]) == 3


def test_teleport_drag_falls_below_the_move_floor():
    # Every bot in the `teleport` family emits exactly one intermediate move,
    # while the smallest human drag measured has 2.
    teleport = drag(0, 1)
    assert move_count(teleport) < MIN_MOVES_PER_DRAG

    human = drag(0, 2)
    assert move_count(human) >= MIN_MOVES_PER_DRAG


def test_no_drag_at_all_returns_empty():
    # The caller must be able to tell "no drag" from "a bad drag" and fall back
    # to the session score rather than invent a verdict.
    assert split_drags([press("pointermove", i) for i in range(5)]) == []


def test_move_floor_applies_to_the_session_not_each_drag():
    """One short drag among several is a human; all of them short is a teleport.

    Scoring each starved drag as 0 instead moved human FRR from 1.5% to 5.2% on
    the main-captcha data — 15 of 166 human sessions contain one such drag, and
    0 of them contain only such drags.
    """
    mixed = drag(0, 1) + drag(10, 6) + drag(20, 5)          # human-shaped
    teleport = drag(0, 1) + drag(10, 1) + drag(20, 1)       # teleport-shaped

    def starved(events):
        segs = split_drags(events)
        return sum(1 for d in segs if move_count(d) < MIN_MOVES_PER_DRAG), len(segs)

    assert starved(mixed) == (1, 3)      # some starved -> those are skipped
    assert starved(teleport) == (3, 3)   # all starved -> session is a bot
