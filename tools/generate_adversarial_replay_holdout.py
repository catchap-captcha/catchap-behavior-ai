"""Generate a defensive, unseen adversarial replay holdout for local testing.

Every generated trace combines rotation, geometry-preserving resampling,
non-linear local timing, and a mild scale/translation change. It never calls a
third-party CAPTCHA and is only used to evaluate this repository's detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from app.services.quality_validator import QUALITY_REJECTED, validate_attempt
from app.services.replay_detector import trace_fingerprint_from_events
from training.generate_extended_bots import build_payload, load_jsonl


GENERATOR_VERSION = "adversarial_replay_holdout_v1"
ATTACK_VARIANT = "rotation_resample_nonlinear_timing_local_speed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_points(source: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(source["events"], key=lambda event: event.get("seq", 0))
    points = np.asarray([[float(row["x"]), float(row["y"])] for row in rows], dtype=float)
    times = np.asarray([float(row["t_ms"]) for row in rows], dtype=float)
    return points, times


def _arc_resample(points: np.ndarray, count: int) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1e-9:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, cumulative[-1], count)
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, dimension]) for dimension in range(2)]
    )


def _local_nonlinear_times(
    source_times: np.ndarray,
    target_count: int,
    rng: random.Random,
) -> tuple[np.ndarray, dict[str, float]]:
    source_progress = np.linspace(0.0, 1.0, len(source_times))
    target_progress = np.linspace(0.0, 1.0, target_count)
    source_relative = np.maximum.accumulate(source_times - source_times[0])
    duration = max(float(source_relative[-1]), 1.0)
    base_dt = np.diff(np.interp(target_progress, source_progress, source_relative / duration))
    base_dt = np.maximum(base_dt, 1e-5)

    slow_center = rng.uniform(0.20, 0.45)
    fast_center = rng.uniform(0.55, 0.85)
    slow_strength = rng.uniform(1.8, 3.3)
    fast_strength = rng.uniform(0.35, 0.70)
    width = rng.uniform(0.06, 0.14)
    midpoints = (target_progress[:-1] + target_progress[1:]) / 2.0
    slow = 1.0 + (slow_strength - 1.0) * np.exp(-((midpoints - slow_center) / width) ** 2)
    fast = 1.0 - (1.0 - fast_strength) * np.exp(-((midpoints - fast_center) / width) ** 2)
    nonlinear_power = rng.uniform(0.72, 1.42)
    local_dt = base_dt * slow * fast
    local_dt = np.maximum(local_dt, 1e-6) ** nonlinear_power
    time_scale = rng.uniform(0.72, 1.38)
    local_dt = local_dt / local_dt.sum() * duration * time_scale
    times = np.concatenate(([0.0], np.cumsum(local_dt)))
    return times, {
        "time_scale": round(time_scale, 6),
        "nonlinear_power": round(nonlinear_power, 6),
        "slow_strength": round(slow_strength, 6),
        "fast_strength": round(fast_strength, 6),
    }


def adversarial_replay_warp(
    source: dict[str, Any],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int, int, dict[str, float | int]]:
    """Apply combined spatial, sampling, and time-profile replay transformations."""
    width = int(source["captcha"]["width"])
    height = int(source["captcha"]["height"])
    points, source_times = _source_points(source)
    target_count = max(4, int(round(len(points) * rng.uniform(0.68, 1.34))))
    resampled = _arc_resample(points, target_count)

    angle_degrees = rng.choice((-1.0, 1.0)) * rng.uniform(4.0, 13.0)
    angle = math.radians(angle_degrees)
    scale = rng.uniform(0.90, 1.06)
    translation = np.asarray(
        [rng.uniform(-width * 0.018, width * 0.018), rng.uniform(-height * 0.025, height * 0.025)]
    )
    center = np.asarray([width / 2.0, height / 2.0])
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )
    transformed = (resampled - center) @ rotation.T * scale + center + translation
    transformed[:, 0] = np.clip(transformed[:, 0], 0.0, float(width))
    transformed[:, 1] = np.clip(transformed[:, 1], 0.0, float(height))
    times, timing_metadata = _local_nonlinear_times(source_times, target_count, rng)

    events: list[dict[str, Any]] = []
    previous_time = -1
    for index, (point, raw_time) in enumerate(zip(transformed, times)):
        timestamp = 0 if index == 0 else max(int(round(raw_time)), previous_time + 1)
        previous_time = timestamp
        events.append(
            {
                "seq": index,
                "event_type": (
                    "pointerdown"
                    if index == 0
                    else "pointerup" if index == len(transformed) - 1 else "pointermove"
                ),
                "t_ms": timestamp,
                "x": round(float(point[0]), 3),
                "y": round(float(point[1]), 3),
                "x_normalized": round(float(point[0] / width), 6),
                "y_normalized": round(float(point[1] / height), 6),
                "target_role": "slider_handle",
            }
        )
    return events, width, height, {
        "rotation_degrees": round(angle_degrees, 6),
        "spatial_scale": round(scale, 6),
        "source_event_count": len(points),
        "target_event_count": target_count,
        **timing_metadata,
    }


def generate_holdout(
    human_attempts: Path,
    split_manifest: Path,
    output: Path,
    *,
    count: int = 1000,
    seed: int = 20260722,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    split_of = json.loads(split_manifest.read_text(encoding="utf-8"))["attempt_to_split"]
    eligible = [
        row
        for row in load_jsonl(human_attempts)
        if split_of.get(row["attempt_id"]) == "test"
        and row.get("anonymous_participant_id")
        and len(row.get("events", [])) >= 4
    ]
    if len(eligible) < count:
        raise ValueError(f"need {count} untouched Human sources, found {len(eligible)}")

    randomizer = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(randomizer.sample(eligible, count)):
        events, width, height, transform = adversarial_replay_warp(
            source,
            random.Random(seed * 1_000_003 + index),
        )
        quality = validate_attempt(events, captcha_width=width, captcha_height=height)
        if quality.status == QUALITY_REJECTED:
            raise ValueError(f"generated invalid replay row {index}: {quality.reason}")
        payload = build_payload(
            f"adversarial_replay_holdout_v1_{index:06d}",
            "replay_adversarial_combined",
            events,
            width,
            height,
        )
        payload["collection"].update(
            {
                "label_source": "external_adversarial_replay_holdout",
                "generator_version": GENERATOR_VERSION,
                "replay_source_fingerprint": trace_fingerprint_from_events(source["events"]),
                "source_split": "test",
                "attack_variant": ATTACK_VARIANT,
                "transform": transform,
            }
        )
        rows.append(payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "dataset_name": output.stem,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generator_version": GENERATOR_VERSION,
        "attack_variant": ATTACK_VARIANT,
        "seed": seed,
        "rows": len(rows),
        "source_policy": "unique Human attempts from untouched participant test split",
        "source_attempts_unique": len(
            {row["collection"]["replay_source_fingerprint"] for row in rows}
        ),
        "source_identifiers_exported": False,
        "inputs": {
            "human_attempts": {"path": str(human_attempts), "sha256": _sha256(human_attempts)},
            "split_manifest": {"path": str(split_manifest), "sha256": _sha256(split_manifest)},
        },
        "output": {"path": str(output), "sha256": _sha256(output), "bytes": output.stat().st_size},
        "production_written": False,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-attempts", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            generate_holdout(
                args.human_attempts,
                args.split_manifest,
                args.output,
                count=args.count,
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
