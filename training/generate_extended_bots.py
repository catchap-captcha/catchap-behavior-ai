"""Generate seven additional defensive bot-trajectory families.

The output combines the existing three baseline families with seven harder
families. It does not solve a CAPTCHA or interact with third-party services;
it only creates labelled trajectories for training and red-team evaluation of
our own Human/Bot detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from app.services.quality_validator import QUALITY_REJECTED, validate_attempt
from app.services.replay_detector import trace_fingerprint_from_events
from training.generate_rule_bots import FAMILIES as BASE_FAMILIES


NEW_FAMILIES = [
    "bezier_curve",
    "stop_go",
    "overshoot_correct",
    "waypoint",
    "random_timing",
    "frame_quantized",
    "replay_warp",
]
ALL_FAMILIES = [*BASE_FAMILIES, *NEW_FAMILIES]
GENERATOR_VERSION = "extended_rule_v2"


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _smoothstep(value: float) -> float:
    return value * value * (3.0 - 2.0 * value)


def _event_rows(
    points: list[tuple[float, float, float]], width: int, height: int
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    previous_t = -1
    for index, (t_ms, x, y) in enumerate(points):
        x = _clamp(x, 0.0, float(width))
        y = _clamp(y, 0.0, float(height))
        timestamp = 0 if index == 0 else max(int(round(t_ms)), previous_t + 1)
        previous_t = timestamp
        events.append(
            {
                "seq": index,
                "event_type": (
                    "pointerdown"
                    if index == 0
                    else "pointerup" if index == len(points) - 1 else "pointermove"
                ),
                "t_ms": timestamp,
                "x": round(x, 3),
                "y": round(y, 3),
                "x_normalized": round(x / width, 6),
                "y_normalized": round(y / height, 6),
                "target_role": "slider_handle",
            }
        )
    return events


def _start_target(rng: random.Random, width: int, height: int) -> tuple[float, ...]:
    x0 = rng.uniform(width * 0.03, width * 0.12)
    x1 = rng.uniform(width * 0.68, width * 0.90)
    y0 = rng.uniform(height * 0.25, height * 0.72)
    y1 = _clamp(y0 + rng.uniform(-height * 0.20, height * 0.20), 10.0, height - 10.0)
    return x0, y0, x1, y1


def _bezier_curve(rng: random.Random, width: int, height: int) -> list[dict[str, Any]]:
    x0, y0, x1, y1 = _start_target(rng, width, height)
    c1 = (x0 + (x1 - x0) * rng.uniform(0.15, 0.40), y0 + rng.uniform(-70, 70))
    c2 = (x0 + (x1 - x0) * rng.uniform(0.60, 0.85), y1 + rng.uniform(-70, 70))
    count = rng.randint(45, 100)
    t = 0.0
    points = []
    for index in range(count):
        u = index / (count - 1)
        v = 1.0 - u
        x = v**3 * x0 + 3 * v * v * u * c1[0] + 3 * v * u * u * c2[0] + u**3 * x1
        y = v**3 * y0 + 3 * v * v * u * c1[1] + 3 * v * u * u * c2[1] + u**3 * y1
        if index:
            t += _clamp(rng.gauss(18.0, 5.0), 6.0, 42.0)
        points.append((t, x, y))
    return _event_rows(points, width, height)


def _stop_go(rng: random.Random, width: int, height: int) -> list[dict[str, Any]]:
    x0, y0, x1, y1 = _start_target(rng, width, height)
    count = rng.randint(42, 75)
    stop_indices = sorted(rng.sample(range(10, count - 10), k=rng.randint(2, 4)))
    t = 0.0
    points: list[tuple[float, float, float]] = []
    for index in range(count):
        u = _smoothstep(index / (count - 1))
        x = x0 + (x1 - x0) * u
        y = y0 + (y1 - y0) * u + math.sin(u * math.pi) * rng.uniform(-12, 12)
        if index:
            t += rng.uniform(10, 28)
        points.append((t, x, y))
        if index in stop_indices:
            for _ in range(rng.randint(2, 5)):
                t += rng.uniform(90, 320)
                points.append((t, x + rng.uniform(-0.25, 0.25), y + rng.uniform(-0.25, 0.25)))
    return _event_rows(points, width, height)


def _overshoot_correct(rng: random.Random, width: int, height: int) -> list[dict[str, Any]]:
    x0, y0, x1, y1 = _start_target(rng, width, height)
    overshoot_x = min(width - 2.0, x1 + rng.uniform(14, min(65, width * 0.15)))
    overshoot_y = _clamp(y1 + rng.uniform(-24, 24), 2.0, height - 2.0)
    outbound = rng.randint(34, 65)
    correction = rng.randint(10, 28)
    points = []
    t = 0.0
    for index in range(outbound):
        u = _smoothstep(index / (outbound - 1))
        x = x0 + (overshoot_x - x0) * u
        y = y0 + (overshoot_y - y0) * u + math.sin(u * math.pi) * rng.uniform(-10, 10)
        if index:
            t += rng.uniform(10, 26)
        points.append((t, x, y))
    t += rng.uniform(35, 140)
    for index in range(1, correction + 1):
        u = _smoothstep(index / correction)
        x = overshoot_x + (x1 - overshoot_x) * u
        y = overshoot_y + (y1 - overshoot_y) * u
        t += rng.uniform(12, 32)
        points.append((t, x, y))
    return _event_rows(points, width, height)


def _waypoint(rng: random.Random, width: int, height: int) -> list[dict[str, Any]]:
    x0, y0, x1, y1 = _start_target(rng, width, height)
    controls = [(x0, y0)]
    for u in sorted(rng.uniform(0.15, 0.85) for _ in range(rng.randint(2, 4))):
        controls.append(
            (
                x0 + (x1 - x0) * u,
                _clamp(y0 + (y1 - y0) * u + rng.uniform(-65, 65), 2.0, height - 2.0),
            )
        )
    controls.append((x1, y1))
    points = []
    t = 0.0
    for segment, (left, right) in enumerate(zip(controls, controls[1:])):
        count = rng.randint(10, 24)
        for index in range(count):
            if segment and index == 0:
                continue
            u = _smoothstep(index / (count - 1))
            x = left[0] + (right[0] - left[0]) * u
            y = left[1] + (right[1] - left[1]) * u
            if points:
                t += rng.uniform(9, 30)
            points.append((t, x, y))
    return _event_rows(points, width, height)


def _random_timing(rng: random.Random, width: int, height: int) -> list[dict[str, Any]]:
    x0, y0, x1, y1 = _start_target(rng, width, height)
    count = rng.randint(35, 95)
    points = []
    t = 0.0
    y_walk = 0.0
    for index in range(count):
        u = _smoothstep(index / (count - 1))
        y_walk = _clamp(y_walk + rng.gauss(0, 0.65), -7.0, 7.0)
        x = x0 + (x1 - x0) * u + rng.gauss(0, 0.55)
        y = y0 + (y1 - y0) * u + y_walk
        if index:
            dt = _clamp(rng.lognormvariate(math.log(17), 0.65), 3.0, 130.0)
            if rng.random() < 0.035:
                dt += rng.uniform(110, 420)
            t += dt
        points.append((t, x, y))
    return _event_rows(points, width, height)


def _frame_quantized(rng: random.Random, width: int, height: int) -> list[dict[str, Any]]:
    x0, y0, x1, y1 = _start_target(rng, width, height)
    count = rng.randint(30, 78)
    points = []
    t = 0.0
    for index in range(count):
        u = _smoothstep(index / (count - 1))
        x = round((x0 + (x1 - x0) * u) * 2.0) / 2.0
        y = round((y0 + (y1 - y0) * u + math.sin(u * math.pi * 2) * 2.0) * 2.0) / 2.0
        if index:
            t += 16.667 * rng.choices([1, 2, 3], weights=[82, 15, 3])[0]
        points.append((t, x, y))
    return _event_rows(points, width, height)


def _replay_warp(source: dict[str, Any], rng: random.Random) -> tuple[list[dict[str, Any]], int, int]:
    width = int(source["captcha"]["width"])
    height = int(source["captcha"]["height"])
    source_events = sorted(source["events"], key=lambda event: event.get("seq", 0))
    time_scale = rng.uniform(0.72, 1.38)
    spatial_scale = rng.uniform(0.88, 1.08)
    dx = rng.uniform(-width * 0.025, width * 0.025)
    dy = rng.uniform(-height * 0.04, height * 0.04)
    t0 = float(source_events[0]["t_ms"])
    points = []
    for event in source_events:
        x = (float(event["x"]) - width / 2) * spatial_scale + width / 2 + dx
        y = (float(event["y"]) - height / 2) * spatial_scale + height / 2 + dy
        t = (float(event["t_ms"]) - t0) * time_scale
        points.append((t, x, y))
    return _event_rows(points, width, height), width, height


GENERATORS = {
    "bezier_curve": _bezier_curve,
    "stop_go": _stop_go,
    "overshoot_correct": _overshoot_correct,
    "waypoint": _waypoint,
    "random_timing": _random_timing,
    "frame_quantized": _frame_quantized,
}


def build_payload(
    attempt_id: str,
    family: str,
    events: list[dict[str, Any]],
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "challenge_id": "extended_rulebot_challenge",
        "session_id": f"extended_rulebot_{family}",
        "anonymous_participant_id": None,
        "captcha": {"width": width, "height": height},
        "timing": {"presented_at": None, "submitted_at": None},
        "events": events,
        "interaction": {
            "regrab_count": 0,
            "retry_count": 0,
            "pointercancel_count": 0,
            "empty_click_count": 0,
            "failed_drop_count": 0,
        },
        "collection": {
            "label": "bot",
            "label_source": "replay_bot" if family == "replay_warp" else "rule_bot",
            "bot_family": family,
            "generator_version": GENERATOR_VERSION,
            "age_group": "unknown",
        },
        "position_correct": True,
        "interaction_success": True,
        "final_drop_error": 0.0,
    }


def reservoir_sample(path: Path, count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    eligible_seen = 0
    for row in load_jsonl(path):
        if len(row.get("events", [])) < 4:
            continue
        eligible_seen += 1
        if len(sample) < count:
            sample.append(row)
        else:
            pick = rng.randrange(eligible_seen)
            if pick < count:
                sample[pick] = row
    if len(sample) < count:
        raise ValueError(f"need {count} Human replay sources, found {len(sample)}")
    rng.shuffle(sample)
    return sample


def generate_new_families(
    per_family: int, human_attempts: Path, seed: int
) -> list[dict[str, Any]]:
    replay_sources = reservoir_sample(human_attempts, per_family, seed + 7001)
    payloads: list[dict[str, Any]] = []
    for family_index, family in enumerate(NEW_FAMILIES):
        for index in range(per_family):
            rng = random.Random(seed * 1_000_003 + family_index * 10_007 + index)
            if family == "replay_warp":
                source = replay_sources[index]
                events, width, height = _replay_warp(source, rng)
            else:
                width, height = rng.choice([(360, 200), (420, 220), (540, 280)])
                events = GENERATORS[family](rng, width, height)
            quality = validate_attempt(events, captcha_width=width, captcha_height=height)
            if quality.status == QUALITY_REJECTED:
                raise ValueError(f"generated invalid {family} row {index}: {quality.reason}")
            attempt_id = f"rulebot_{GENERATOR_VERSION}_{family}_{index:06d}"
            payload = build_payload(attempt_id, family, events, width, height)
            if family == "replay_warp":
                payload["collection"]["replay_source_fingerprint"] = (
                    trace_fingerprint_from_events(source["events"])
                )
            payloads.append(payload)
    return payloads


def write_combined(
    base_bots: Path,
    human_attempts: Path,
    output: Path,
    per_family: int,
    seed: int,
) -> dict[str, Any]:
    base_rows = list(load_jsonl(base_bots))
    base_counts = Counter(row["collection"]["bot_family"] for row in base_rows)
    expected_base = {family: per_family for family in BASE_FAMILIES}
    if dict(base_counts) != expected_base:
        raise ValueError(f"base family counts must be {expected_base}, found {dict(base_counts)}")

    new_rows = generate_new_families(per_family, human_attempts, seed)
    rows = [*base_rows, *new_rows]
    attempt_ids = [row["attempt_id"] for row in rows]
    if len(attempt_ids) != len(set(attempt_ids)):
        raise ValueError("duplicate bot attempt_id")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    counts = Counter(row["collection"]["bot_family"] for row in rows)
    manifest = {
        "dataset_name": output.stem,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "total_rows": len(rows),
        "family_count": len(counts),
        "per_family": dict(sorted(counts.items())),
        "base_families": BASE_FAMILIES,
        "new_families": NEW_FAMILIES,
        "inputs": {
            "base_bots": {"path": str(base_bots), "sha256": sha256(base_bots)},
            "human_replay_sources": {
                "path": str(human_attempts),
                "sha256": sha256(human_attempts),
                "direct_source_ids_exported": False,
            },
        },
        "output": {"path": str(output), "bytes": output.stat().st_size, "sha256": sha256(output)},
        "notes": [
            "Defensive synthetic data for our own detector only.",
            "Replay-warp rows preserve Human-like motion and are intentionally difficult hard negatives.",
            "No production model is promoted by this generator.",
        ],
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bots", type=Path, required=True)
    parser.add_argument("--human-attempts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = write_combined(
        args.base_bots, args.human_attempts, args.out, args.per_family, args.seed
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
