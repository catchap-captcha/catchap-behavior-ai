"""Raw-attempt quality validation.

Decides a `quality_status` of ``valid`` / ``pending`` / ``rejected`` for an
attempt and records a human-readable ``rejection_reason``. Raw data is never
deleted here — the caller only stores the status alongside the raw events.

Severity model:
  * hard failures  -> ``rejected`` (unusable / structurally broken / impossible)
  * soft concerns  -> ``pending``  (looks off, needs human review before use)
  * clean          -> ``valid``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

QUALITY_VALID = "valid"
QUALITY_PENDING = "pending"
QUALITY_REJECTED = "rejected"

# Coordinates may sit slightly outside the CAPTCHA box (a drag can end past an
# edge); reject only when they blow past this multiple of the box size.
_OUT_OF_BOUNDS_MARGIN = 0.5   # 50% of width/height beyond the edge
_MIN_EVENTS = 2
_MIN_DISTINCT_COORDS = 2
_SERVER_TIME_TOLERANCE_MS = 5_000


@dataclass
class QualityResult:
    status: str
    reason: str | None = None
    checks: dict[str, Any] = field(default_factory=dict)


def validate_attempt(
    events: list[dict[str, Any]],
    *,
    captcha_width: float | None = None,
    captcha_height: float | None = None,
    presented_at=None,
    submitted_at=None,
    enforce_server_time_window: bool = False,
) -> QualityResult:
    """Run all quality checks for one attempt.

    Returns a :class:`QualityResult`. ``checks`` carries the raw measurements so
    they can be logged for debugging without re-running validation.
    """
    checks: dict[str, Any] = {}
    hard_fail: list[str] = []
    soft_flag: list[str] = []

    n = len(events)
    checks["event_count"] = n

    # 1) at least 2 events
    if n < _MIN_EVENTS:
        return QualityResult(QUALITY_REJECTED, "event_count_lt_2", {"event_count": n})

    ordered = sorted(events, key=lambda e: e.get("seq", 0))
    seqs = [e.get("seq") for e in ordered]
    types = [e.get("event_type") for e in ordered]

    # 2) seq starts at 0 and increases by 1 (strictly ordered, no gaps)
    expected = list(range(n))
    checks["seq_ok"] = seqs == expected
    if seqs != expected:
        hard_fail.append("seq_not_sequential")

    # 3) t_ms non-decreasing
    t_vals = [e.get("t_ms") for e in ordered]
    non_decreasing = all(
        (a is not None and b is not None and b >= a)
        for a, b in zip(t_vals, t_vals[1:])
    )
    checks["t_ms_non_decreasing"] = non_decreasing
    if not non_decreasing:
        hard_fail.append("t_ms_decreasing")

    # 4) pointerdown present
    if "pointerdown" not in types:
        hard_fail.append("missing_pointerdown")
    # 5) pointerup OR pointercancel present
    if "pointerup" not in types and "pointercancel" not in types:
        hard_fail.append("missing_pointerup_or_cancel")

    # 6) NaN / Infinity in any numeric field
    if _has_non_finite(ordered):
        hard_fail.append("non_finite_value")

    # 7) coordinates not wildly outside the CAPTCHA area
    oob = _out_of_bounds_count(ordered, captcha_width, captcha_height)
    checks["out_of_bounds_count"] = oob
    if oob > 0:
        hard_fail.append("coordinates_out_of_bounds")

    # 8) normalized coordinates within [0, 1] (soft: some overshoot expected)
    norm_bad = _normalized_out_of_range_count(ordered)
    checks["normalized_out_of_range_count"] = norm_bad
    if norm_bad > 0:
        soft_flag.append("normalized_out_of_range")

    # 9) duration > 0
    duration = None
    if t_vals[0] is not None and t_vals[-1] is not None:
        duration = t_vals[-1] - t_vals[0]
    checks["duration_ms"] = duration
    if duration is None or duration <= 0:
        hard_fail.append("duration_not_positive")

    # 10) duplicate timestamps count (soft)
    dup_ts = _duplicate_count(t_vals)
    checks["duplicate_timestamp_count"] = dup_ts
    if dup_ts > 0:
        soft_flag.append("duplicate_timestamps")

    # 11) duplicate (identical) events count (soft)
    dup_events = _duplicate_event_count(ordered)
    checks["duplicate_event_count"] = dup_events
    if dup_events > 0:
        soft_flag.append("duplicate_events")

    # 12) enough distinct coordinates
    distinct = len({(e.get("x"), e.get("y")) for e in ordered})
    checks["distinct_coordinates"] = distinct
    if distinct < _MIN_DISTINCT_COORDS:
        hard_fail.append("too_few_distinct_coordinates")

    # 13) submitted_at not before presented_at. The strict event-window check
    # is reserved for online prediction, where both timestamps come from the
    # trusted CAPTCHA server. Collection imports may contain legacy relative
    # event timestamps and must remain readable.
    if enforce_server_time_window and (
        presented_at is None or submitted_at is None
    ):
        hard_fail.append("missing_server_timing")
    elif presented_at is not None and submitted_at is not None:
        presented_ms = _epoch_ms(presented_at)
        submitted_ms = _epoch_ms(submitted_at)
        if submitted_ms < presented_ms:
            hard_fail.append("submitted_before_presented")
        elif enforce_server_time_window:
            first_event_ms = min(t_vals)
            last_event_ms = max(t_vals)
            server_duration_ms = submitted_ms - presented_ms
            checks["server_duration_ms"] = server_duration_ms
            checks["event_first_t_ms"] = first_event_ms
            checks["event_last_t_ms"] = last_event_ms
            within_server_window = (
                first_event_ms >= presented_ms - _SERVER_TIME_TOLERANCE_MS
                and last_event_ms <= submitted_ms + _SERVER_TIME_TOLERANCE_MS
                and duration <= server_duration_ms + _SERVER_TIME_TOLERANCE_MS
            )
            checks["event_timestamps_within_server_window"] = within_server_window
            if not within_server_window:
                hard_fail.append("event_timestamps_outside_server_window")

    if hard_fail:
        return QualityResult(QUALITY_REJECTED, ";".join(hard_fail), checks)
    if soft_flag:
        return QualityResult(QUALITY_PENDING, ";".join(soft_flag), checks)
    return QualityResult(QUALITY_VALID, None, checks)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _epoch_ms(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return round(value.timestamp() * 1000)


def _has_non_finite(events: list[dict[str, Any]]) -> bool:
    for e in events:
        for key in ("t_ms", "x", "y", "x_normalized", "y_normalized"):
            v = e.get(key)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return True
            if not math.isfinite(fv):
                return True
    return False


def _out_of_bounds_count(events, w, h) -> int:
    if not w or not h:
        return 0
    mx, my = w * _OUT_OF_BOUNDS_MARGIN, h * _OUT_OF_BOUNDS_MARGIN
    count = 0
    for e in events:
        x, y = e.get("x"), e.get("y")
        if x is None or y is None:
            continue
        if x < -mx or x > w + mx or y < -my or y > h + my:
            count += 1
    return count


def _normalized_out_of_range_count(events) -> int:
    count = 0
    for e in events:
        for key in ("x_normalized", "y_normalized"):
            v = e.get(key)
            if v is None:
                continue
            if v < 0.0 or v > 1.0:
                count += 1
    return count


def _duplicate_count(values) -> int:
    seen, dup = set(), 0
    for v in values:
        if v in seen:
            dup += 1
        else:
            seen.add(v)
    return dup


def _duplicate_event_count(events) -> int:
    seen, dup = set(), 0
    for e in events:
        key = (e.get("event_type"), e.get("t_ms"), e.get("x"), e.get("y"))
        if key in seen:
            dup += 1
        else:
            seen.add(key)
    return dup
