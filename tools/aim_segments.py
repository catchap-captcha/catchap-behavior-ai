"""Aim segments: the pointer travel *before* the grab, and what it is made of.

Why bother
----------
The drag itself is 12 points over ~650ms, and that turned out to be the ceiling:
human drags resemble each other so closely that path similarity between two
*different people* reaches 1.0000. No replay-derived bot can be separated on a
surface that coarse.

The aim segment is the same pointer, a moment earlier — travelling toward the
object it is about to grab. Measured here it carries 16 points over ~1049ms,
which is more of everything. Whether *more* also means *more separable* is the
question these tools exist to answer; it is not assumed.

Segmentation
------------
One recorded row is one drop, but a row is not one aiming motion. The capture
buffer keeps accumulating while the person reads the question, looks away, or
walks off — the longest gap observed inside a single row is 25 minutes. Bursts
are therefore split at `GAP_MS`, and only the ones with enough points are used.

`GAP_MS = 400` is four throttle periods. Chosen because human aiming intervals
sit at a 50ms median with p95 at 334ms: 400 is past the tail of continuous
motion, and far below the pauses that separate one aim from the next.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import numpy as np

GAP_MS = 400.0
MIN_POINTS = 8

# The capture is throttled at 40ms in the widget (`main.jsx:268`). Anything a bot
# emits below this could never have come through the same path, so a bot that
# ignores it is not being tested against the real surface — it is being handed a
# giveaway. Verified against the human data: 0.0% of intervals fall below it.
THROTTLE_MS = 40.0


# The aim capture lives in the CAPTCHA repo, because that is where the widget
# that records it lives. Checkouts do not have to be siblings, so the location
# is overridable; the default is the layout this was developed in.
AIM_ROOT = Path(os.environ.get(
    "CATCHAP_AIM_DIR",
    Path(__file__).resolve().parents[2] / "ai-service-ms-behavior" / "data" / "aim",
))
AIM_CAPTURE = AIM_ROOT / "aim_20260808.jsonl"
PARTICIPANTS = AIM_ROOT / "participants.local.json"


def person_of(code: str) -> str:
    """Resolve a participant code to the person behind it.

    Codes are not people. On 2026-08-10 one person collected under three codes
    (`sw-aim`, `qq`, `ww`), and grouping by code produced a cross-person
    false-pair rate of 0.19% from data that contained exactly one person. The
    number was wrong in the direction that flatters the defence, and nothing in
    the data revealed it — it took asking.

    So person-level grouping goes through the mapping file, never through
    `participant_id` directly. An unmapped code resolves to itself, which is
    right for a genuinely new participant and wrong for an unrecorded alias:
    keep the file current as codes are handed out.
    """
    try:
        table = json.loads(PARTICIPANTS.read_text())
    except (OSError, json.JSONDecodeError):
        return code
    value = table.get(code)
    return value if isinstance(value, str) else code


def load_bursts(path: Path, exclude_probe: bool = True) -> list[list[dict]]:
    """Split every recorded row into continuous aiming bursts."""
    bursts: list[list[dict]] = []
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            if exclude_probe and str(record.get("participant_id", "")).startswith("zzprobe"):
                continue
            events = record.get("aim_events") or []
            if not events:
                continue
            current = [events[0]]
            for prev, nxt in zip(events, events[1:]):
                if nxt["timestamp_ms"] - prev["timestamp_ms"] > GAP_MS:
                    bursts.append(current)
                    current = [nxt]
                else:
                    current.append(nxt)
            bursts.append(current)
    return [b for b in bursts if len(b) >= MIN_POINTS]


def to_arrays(burst: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(points Nx2 in stage-normalized units, times in ms from burst start)."""
    xy = np.array([[e["x"], e["y"]] for e in burst], dtype=float)
    t = np.array([e["timestamp_ms"] for e in burst], dtype=float)
    return xy, t - t[0]


def speed_profile(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Per-step speed in normalized units per second."""
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    dt = np.maximum(np.diff(t), 1e-6) / 1000.0
    return step / dt


def submovements(speed: np.ndarray) -> list[int]:
    """Indices of local speed maxima that look like separate submovements.

    Human aiming is not one motion. It is a fast ballistic throw covering most
    of the distance, followed by slower feedback-driven corrections as the eye
    closes the remaining gap — the standard two-phase account of aimed movement,
    and the reason aiming time scales with target distance and size.

    A peak counts only if it rises meaningfully above the valley before it;
    without that guard, sensor noise on a smooth curve reports a dozen
    "submovements" and the feature measures jitter rather than structure.
    """
    if speed.size < 3:
        return []
    peaks: list[int] = []
    threshold = 0.25 * float(speed.max())
    for i in range(1, speed.size - 1):
        if speed[i] >= speed[i - 1] and speed[i] > speed[i + 1] and speed[i] > threshold:
            valley = float(speed[max(0, i - 3):i].min()) if i else float(speed[i])
            if speed[i] - valley > 0.15 * float(speed.max()):
                peaks.append(i)
    return peaks


def describe(bursts: list[list[dict]]) -> dict[str, float]:
    """Population summary — what a bot has to reproduce to be plausible."""
    rows: dict[str, list[float]] = {}

    def add(key: str, value: float) -> None:
        if value is not None and np.isfinite(value):
            rows.setdefault(key, []).append(float(value))

    for burst in bursts:
        xy, t = to_arrays(burst)
        speed = speed_profile(xy, t)
        if speed.size < 2:
            continue
        path_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        straight = float(np.linalg.norm(xy[-1] - xy[0]))

        add("duration_ms", t[-1])
        add("points", len(burst))
        add("path_length", path_len)
        add("straightness", straight / path_len if path_len > 1e-9 else 0.0)
        add("peak_speed", float(speed.max()))
        add("speed_cv", float(np.std(speed) / (np.mean(speed) + 1e-9)))
        add("submovements", len(submovements(speed)))
        # Where the fastest moment falls. Human aiming accelerates hard and
        # decelerates long, so the peak sits early; a symmetric easing curve
        # puts it at the midpoint.
        add("peak_at", float(np.argmax(speed)) / max(speed.size - 1, 1))
        # How much of the trip is spent crawling. The endgame of a human aim is
        # slow; a bot that eases uniformly never crawls.
        slow = float((speed < 0.2 * speed.max()).mean())
        add("slow_fraction", slow)
        add("interval_median_ms", float(np.median(np.diff(t))))

    return {k: statistics.median(v) for k, v in sorted(rows.items())}


def main() -> None:
    bursts = load_bursts(AIM_CAPTURE)
    print(f"조준 구간 {len(bursts)}개 (≥{MIN_POINTS}점)\n")
    for key, value in describe(bursts).items():
        print(f"  {key:22s} {value:10.4f}")


if __name__ == "__main__":
    main()
