"""Join the aim segment to the drag it precedes, and decide what may be judged on it.

Why join them
-------------
The drag alone is ~12 points over ~0.65s, and at that resolution two *different
people's* paths reach similarity 1.0000. Nothing built on top can hold a line
that human variation already crosses — measured repeatedly on 2026-08-10, most
directly as AUC 0.516 for replayed human motion even with the family in training.

Detection of a warping replay attacker depends almost entirely on how many points
the path carries:

     8~12 points   caught  1.4%      19~30   caught 50.8%
    13~18                  3.8%      31+            96.8%

The aim segment — the pointer travelling toward the object it is about to grab —
carries a median of 21 points. Joined to the drag's 12 that is 33, and 57.3% of
joined paths clear the 31-point bar where the attack dies. The drag alone clears
it 0% of the time.

**This changes nothing the user does.** Same screen, same task, same duration.
The pointer already travels that path; it was simply not being recorded.

Why the join is sound
---------------------
Both halves come off the same capture path with the same 40ms throttle
(`main.jsx` records `aim_move` and `pointer_move` through separate handlers but
identical gating), and both carry stage-normalized coordinates. The aim segment
ends where `pointerdown` fires, which is where the drag begins, so the two are
spatially and temporally contiguous — concatenation is the real path, not a
splice of two unrelated things.

What must NOT be judged
-----------------------
5.3% of joined paths still fall at 18 points or below, where the fingerprint axis
catches 3.8% of warped replays. A verdict of "no match" there is not evidence of
innocence, it is absence of evidence, and reporting it as the former is how a
defence talks itself into false confidence. `judgeable()` exists to make that
distinction explicit rather than leaving it to whoever reads the score.

⚠️ Unvalidated on real pairs. Until 2026-08-10 the aim capture stored only aim
events, so no recorded attempt contains both halves; the numbers above come from
measuring each half separately. `main.jsx` now sends `drag_events` alongside, so
paired data starts accumulating from the next collection — the join itself must
be re-checked against it before any of this is quoted as a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from app.services.replay_detector import MIN_POINTS_FOR_WARP_RESISTANCE

# Below this a joined path is too short for the fingerprint axis to say anything.
# Set to the same bar the comparator documents, deliberately: two names for one
# number is how they drift apart.
MIN_POINTS_FOR_FINGERPRINT = MIN_POINTS_FOR_WARP_RESISTANCE


@dataclass(frozen=True)
class JoinedPath:
    points: np.ndarray            # (N, 2) stage-normalized
    aim_points: int
    drag_points: int

    @property
    def total_points(self) -> int:
        return int(self.points.shape[0])

    def judgeable(self) -> bool:
        """Whether a fingerprint verdict on this path means anything.

        False is not a pass. It means the path is outside this axis's competence
        and the caller must fall back to the per-trace model alone.
        """
        return self.total_points >= MIN_POINTS_FOR_FINGERPRINT


def _points(events: Sequence[dict[str, Any]]) -> np.ndarray:
    rows = [(e.get("x"), e.get("y")) for e in events
            if e.get("x") is not None and e.get("y") is not None]
    return np.array(rows, dtype=float) if rows else np.zeros((0, 2), dtype=float)


def join(aim_events: Sequence[dict[str, Any]],
         drag_events: Sequence[dict[str, Any]]) -> JoinedPath:
    """Concatenate aim then drag, dropping the seam duplicate if there is one.

    The last aim sample and the first drag sample can land on the same pixel:
    `pointerdown` fires where the pointer already was. Left in, that produces a
    zero-length step, which makes step-length variance and turn angle undefined
    at the seam — exactly the features the fingerprint rests on.
    """
    aim = _points(aim_events)
    drag = _points(drag_events)
    if aim.shape[0] and drag.shape[0]:
        if float(np.linalg.norm(aim[-1] - drag[0])) < 1e-9:
            drag = drag[1:]
    points = np.vstack([aim, drag]) if aim.shape[0] or drag.shape[0] else np.zeros((0, 2))
    return JoinedPath(points=points, aim_points=int(aim.shape[0]), drag_points=int(drag.shape[0]))


def join_record(record: dict[str, Any]) -> JoinedPath:
    """Join one row as the collector writes it (`aim_events` + `drag_events`)."""
    return join(record.get("aim_events") or [], record.get("drag_events") or [])


def summarize(paths: Sequence[JoinedPath]) -> dict[str, float]:
    """Length distribution, which is the quantity that decides everything here."""
    if not paths:
        return {}
    totals = np.array([p.total_points for p in paths], dtype=float)
    return {
        "count": float(len(paths)),
        "median_points": float(np.median(totals)),
        "median_aim_points": float(np.median([p.aim_points for p in paths])),
        "median_drag_points": float(np.median([p.drag_points for p in paths])),
        "judgeable_fraction": float(np.mean([p.judgeable() for p in paths])),
        "at_or_below_18": float(np.mean(totals <= 18)),
    }
