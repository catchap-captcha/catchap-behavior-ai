"""Record what was scored, not just what it scored.

A prediction row currently stores the score and nothing about the input that
produced it. That gap is not theoretical: replaying 499 stored attempts through
the same model reproduced only 82.4% of the recorded scores, and the remaining
88 could not be diagnosed. Every candidate cause was ruled out — same event set
(499/499 counts match), same model version, one prediction per attempt,
timestamps shift-invariant, coordinates stored as `double`, interaction counters
absent from the model's features — and the decisive test failed for its own
reason: `/predict` requires `captcha.width/height`, those were never stored, and
a guessed pair scores differently.

So the fix is to stop guessing. Each prediction carries a digest of the exact
events that were scored, the feature vector that came out of them, and the
challenge geometry. Then "does the stored input still produce the stored score?"
is one query instead of an afternoon.

Floats are rounded before hashing for the same reason `app/db.py` does it in the
captcha: a 17-digit float that round-trips through a JSON column comes back
subtly different, the digest changes, and a matching input looks like a mismatch.
That bug already cost this project a day once.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

_COORD_DIGITS = 6
_SCORE_DIGITS = 6

# The only event fields the extractor reads. Digesting anything else would make
# the digest change when unrelated columns are added.
_SCORED_FIELDS = ("seq", "event_type", "t_ms", "x", "y")


def _round(value: Any, digits: int) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return round(float(value), digits)


def _canonical_events(events: Iterable[dict[str, Any]]) -> list[list[Any]]:
    return [
        [_round(event.get(field), _COORD_DIGITS) for field in _SCORED_FIELDS]
        for event in events
    ]


def _digest(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


def build_audit(
    *,
    events: list[dict[str, Any]],
    features: dict[str, float],
    captcha_width: int | None,
    captcha_height: int | None,
    scoring_unit: str,
    session_human_score: float,
    per_drag: dict[str, Any] | None,
) -> dict[str, Any]:
    """Everything needed to recompute this prediction later, and nothing else."""
    rounded_features = {
        name: _round(value, _SCORE_DIGITS) for name, value in sorted(features.items())
    }
    return {
        "scoring_unit": scoring_unit,
        "session_human_score": _round(session_human_score, _SCORE_DIGITS),
        "per_drag": per_drag,
        # Geometry is part of the input: /predict requires it, and without it a
        # replay cannot reconstruct the request at all.
        "captcha": {"width": captcha_width, "height": captcha_height},
        "event_count": len(events),
        "input_digest": _digest(_canonical_events(events)),
        "feature_digest": _digest(rounded_features),
        "features": rounded_features,
    }
