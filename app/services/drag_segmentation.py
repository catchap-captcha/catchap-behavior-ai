"""Split a pointer trajectory into drags, so the model can score one at a time.

The trained features are session-level sums — `event_count`, `duration_ms`,
`total_distance`. Drag five objects instead of one and every one of them grows
fivefold, which is why a ruler-straight path scored `human_probability 1.0000`
when repeated five times and 0.0000 when done once (measured 2026-07-31).
Interaction *scale* decided the verdict, and scale costs an attacker one `for`
loop.

Scoring each drag separately removes that lever, and it also removes an illusion:
of the 56 multi-drag human sessions in the main-captcha data, 53 passed on the
session score but only 4 passed on their own drags. Those sessions were not being
recognised as human — their aggregates were simply large.
"""

from __future__ import annotations

from typing import Any, Iterable

# The captcha's own event names, plus the ones our red-team tools emit. A tool
# that sends `pointer_down` + `drag_start` for one press must not be read as two.
_DOWN = frozenset({"pointerdown", "pointer_down", "drag_start"})
_UP = frozenset({"pointerup", "pointer_up", "drop", "drag_end"})

# Below this, a "drag" is a teleport: every bot in the `teleport` family emits
# exactly one intermediate move, while the smallest human drag we have measured
# has 2 and the median has 11. Kept as a floor, not a classifier — an attacker
# who emits two moves is back in the model's hands, which is the point.
MIN_MOVES_PER_DRAG = 2


def split_drags(events: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Return one segment per press-to-release.

    Moves made while no button is held are dropped: hovering is not a drag, and
    counting it would let the caller inflate a segment the same way session-level
    aggregates were inflated.
    """
    drags: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] | None = None

    for event in events:
        kind = event.get("event_type")
        if kind in _DOWN:
            # A press while already pressed means we missed the release. Keep what
            # we have rather than discarding it — the movement did happen.
            if current is not None and len(current) > 1:
                drags.append(current)
            current = [event]
        elif kind in _UP:
            if current is not None:
                current.append(event)
                drags.append(current)
                current = None
        elif current is not None:
            current.append(event)

    if current is not None and len(current) > 1:
        drags.append(current)
    return drags


def move_count(drag: Iterable[dict[str, Any]]) -> int:
    return sum(1 for e in drag if e.get("event_type") in ("pointermove", "pointer_move"))
