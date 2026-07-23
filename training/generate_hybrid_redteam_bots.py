"""Generate offline hybrid-motion red-team bots for detector weakness testing.

The generator starts from a PCA + GMM trajectory distribution, then applies
motion policies which combine curved paths, speed/turn coupling, event
coalescing, and display-frame timing.  It is deliberately offline: it only
reads local JSONL and writes synthetic JSONL; it never opens a CAPTCHA,
controls a browser, or makes network requests.

Every output is detector-forbidden.  ``calibration`` is for inspecting the
generator and attack diagnostics, while ``external_holdout`` is sealed for a
future one-time detector evaluation.  Neither may be used to fit a detector or
choose its threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.services.quality_validator import QUALITY_REJECTED, validate_attempt
from training.generate_ml_bots import (
    GeneratorConfig,
    _novelty_distances,
    fit_generator,
    load_jsonl,
    sample_payloads,
    sha256,
    vectorize_attempt,
)


GENERATOR_VERSION = "hybrid_pca_gmm_motion_v1"
BOT_FAMILY = "hybrid_motion_redteam"
HARD_BOT_FAMILY = "hybrid_motion_redteam_hard"
EXTERNAL_ROLES = ("external_holdout", "external_holdout_hard")
ROLES = ("calibration", *EXTERNAL_ROLES)


@dataclass(frozen=True)
class MotionPolicy:
    curvature: tuple[float, float]
    jitter: tuple[float, float]
    time_power: tuple[float, float]
    duration_scale: tuple[float, float]
    turn_slowdown: tuple[float, float]
    coalesce_fraction: tuple[float, float]
    frame_ms: tuple[int, ...]
    late_correction_probability: float


# The external profile is intentionally different from calibration.  It is not
# a stronger version to tune against; it is a separate future evaluation family.
POLICIES = {
    "calibration": MotionPolicy(
        curvature=(0.008, 0.030),
        jitter=(0.0008, 0.0030),
        time_power=(0.84, 1.22),
        duration_scale=(0.84, 1.22),
        turn_slowdown=(0.08, 0.32),
        coalesce_fraction=(0.05, 0.14),
        frame_ms=(8, 10),
        late_correction_probability=0.45,
    ),
    "external_holdout": MotionPolicy(
        curvature=(0.025, 0.065),
        jitter=(0.0020, 0.0065),
        time_power=(0.65, 1.45),
        duration_scale=(0.72, 1.38),
        turn_slowdown=(0.28, 0.70),
        coalesce_fraction=(0.16, 0.28),
        frame_ms=(12, 16),
        late_correction_probability=0.78,
    ),
    # This profile favors subtle, mixed event-delivery traces rather than only
    # extreme motion. It is a separate sealed evaluation family, not a target
    # for detector fitting or threshold adjustment.
    "external_holdout_hard": MotionPolicy(
        curvature=(0.010, 0.055),
        jitter=(0.0005, 0.0040),
        time_power=(0.78, 1.28),
        duration_scale=(0.82, 1.18),
        turn_slowdown=(0.14, 0.52),
        coalesce_fraction=(0.02, 0.16),
        frame_ms=(8, 10, 12, 16),
        late_correction_probability=0.72,
    ),
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(max(value, lower), upper)


def _choice_range(randomizer: random.Random, bounds: tuple[float, float]) -> float:
    return randomizer.uniform(*bounds)


def _allowed_source_rows(
    human_attempts_path: Path,
    source_human_features_path: Path,
) -> list[dict[str, Any]]:
    """Use only source IDs present in the explicitly supplied development set."""
    allowed_ids = {
        str(row["attempt_id"])
        for row in load_jsonl(source_human_features_path)
        if row.get("label") == "human" and row.get("attempt_id")
    }
    rows = [
        row
        for row in load_jsonl(human_attempts_path)
        if str(row.get("attempt_id")) in allowed_ids and len(row.get("events") or []) >= 4
    ]
    if len(rows) < 20:
        raise ValueError(f"need at least 20 eligible Human source attempts, found {len(rows)}")
    return rows


def _coalesced_indices(count: int, fraction: float, randomizer: random.Random) -> list[int]:
    """Drop a subset of middle pointermove events while keeping both endpoints."""
    middle = list(range(1, count - 1))
    drop_count = min(len(middle) - 2, int(round(len(middle) * fraction)))
    if drop_count <= 0:
        return list(range(count))
    removed = set(randomizer.sample(middle, drop_count))
    return [index for index in range(count) if index not in removed]


def _motion_events(
    base_events: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    policy: MotionPolicy,
    randomizer: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply a policy without retaining any original-source trace identifiers."""
    count = len(base_events)
    points = np.asarray(
        [
            [float(event["x_normalized"]), float(event["y_normalized"])]
            for event in base_events
        ],
        dtype=float,
    )
    old_times = np.asarray([float(event["t_ms"]) for event in base_events], dtype=float)
    old_duration = max(float(old_times[-1] - old_times[0]), float(count - 1))
    progress = np.linspace(0.0, 1.0, count)

    start, end = points[0].copy(), points[-1].copy()
    direction = end - start
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm < 1e-6:
        direction = np.asarray([1.0, 0.0])
        direction_norm = 1.0
    normal = np.asarray([-direction[1], direction[0]]) / direction_norm

    curvature = _choice_range(randomizer, policy.curvature) * randomizer.choice((-1.0, 1.0))
    phase = randomizer.uniform(-0.35, 0.35)
    envelope = np.sin(math.pi * progress)
    arc = curvature * envelope * np.sin(math.pi * (progress + phase))
    points = points + arc[:, None] * normal

    jitter = _choice_range(randomizer, policy.jitter)
    noise = np.asarray(
        [[randomizer.gauss(0.0, jitter), randomizer.gauss(0.0, jitter)] for _ in range(count)]
    )
    noise[0] = 0.0
    noise[-1] = 0.0
    points = points + noise

    correction_used = randomizer.random() < policy.late_correction_probability
    if correction_used:
        correction_start = randomizer.uniform(0.62, 0.78)
        correction_envelope = np.clip((progress - correction_start) / (1.0 - correction_start), 0.0, 1.0)
        correction = (
            curvature
            * randomizer.uniform(0.35, 0.80)
            * np.sin(math.pi * correction_envelope)
        )
        points = points - correction[:, None] * normal

    points[0] = start
    points[-1] = end
    points = np.clip(points, 0.0, 1.0)

    power = _choice_range(randomizer, policy.time_power)
    warped_progress = np.power(progress, power)
    segment_distance = np.linalg.norm(np.diff(points, axis=0), axis=1)
    segments = np.diff(points, axis=0)
    headings = np.arctan2(segments[:, 1], segments[:, 0])
    turns = np.zeros(count - 1)
    if len(headings) > 1:
        turns[1:] = np.abs(np.angle(np.exp(1j * np.diff(headings)))) / math.pi
    slowdown = _choice_range(randomizer, policy.turn_slowdown)
    timing_weight = np.diff(warped_progress) + slowdown * turns + 0.08 * segment_distance
    timing_weight = np.maximum(timing_weight, 1e-4)
    duration = old_duration * _choice_range(randomizer, policy.duration_scale)
    intervals = timing_weight / timing_weight.sum() * duration
    frame_ms = randomizer.choice(policy.frame_ms)
    cumulative = np.concatenate(([0.0], np.cumsum(intervals)))
    cumulative = np.round(cumulative / frame_ms) * frame_ms
    cumulative[0] = 0.0
    for index in range(1, count):
        cumulative[index] = max(cumulative[index], cumulative[index - 1] + 1.0)

    keep = _coalesced_indices(count, _choice_range(randomizer, policy.coalesce_fraction), randomizer)
    events: list[dict[str, Any]] = []
    for sequence, index in enumerate(keep):
        x_normalized, y_normalized = points[index]
        events.append(
            {
                "seq": sequence,
                "event_type": "pointerdown" if sequence == 0 else "pointerup" if sequence == len(keep) - 1 else "pointermove",
                "t_ms": int(round(float(cumulative[index]))),
                "x": round(float(x_normalized * width), 3),
                "y": round(float(y_normalized * height), 3),
                "x_normalized": round(float(x_normalized), 6),
                "y_normalized": round(float(y_normalized), 6),
                "target_role": "captcha_area",
            }
        )
    return events, {
        "curvature_amplitude": round(curvature, 6),
        "time_power": round(power, 6),
        "turn_slowdown": round(slowdown, 6),
        "frame_interval_ms": frame_ms,
        "correction_used": correction_used,
        "event_coalescing": round(1.0 - len(keep) / count, 6),
    }


def _payload(
    attempt_id: str,
    events: list[dict[str, Any]],
    *,
    role: str,
    width: int,
    height: int,
    novelty_distance: float,
    mutation: dict[str, Any],
) -> dict[str, Any]:
    is_external = role in EXTERNAL_ROLES
    training_usage = "external_holdout_only" if is_external else "redteam_only"
    evaluation_role = "external_holdout" if is_external else "redteam_calibration"
    bot_family = HARD_BOT_FAMILY if role == "external_holdout_hard" else BOT_FAMILY
    return {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "challenge_id": "hybrid_redteam_challenge",
        "session_id": f"hybrid_redteam_{role}",
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
            "label_source": "hybrid_redteam_generated",
            "bot_family": bot_family,
            "generator_version": GENERATOR_VERSION,
            "training_usage": training_usage,
            "evaluation_role": evaluation_role,
            "novelty_distance": round(novelty_distance, 6),
            "mutation": mutation,
            "age_group": "unknown",
        },
        "position_correct": True,
        "interaction_success": True,
        "final_drop_error": 0.0,
    }


def generate_payloads_from_fitted(
    generator: dict[str, Any],
    *,
    role: str,
    count: int,
    config: GeneratorConfig,
) -> tuple[list[dict[str, Any]], int, int]:
    """Generate hybrid payloads in bounded batches from one fitted PCA-GMM.

    Larger red-team runs must not materialize every pre-mutation candidate at
    once.  Each batch uses a distinct deterministic GMM seed, while all
    candidates still go through the post-mutation quality and novelty checks.
    """
    if role not in ROLES:
        raise ValueError(f"unsupported role: {role}")
    policy = POLICIES[role]
    accepted: list[dict[str, Any]] = []
    rejected_quality = 0
    rejected_novelty = 0
    batch_index = 0
    max_batches = max(20, math.ceil(count / 64) * 3)

    while len(accepted) < count and batch_index < max_batches:
        remaining = count - len(accepted)
        candidate_count = min(256, max(64, remaining * 2))
        batch_config = replace(config, seed=config.seed + 1009 * batch_index)
        # GaussianMixture.sample reuses its fitted random_state. Change that
        # state per batch so repeated calls do not return the same candidates.
        generator["gmm"].random_state = batch_config.seed
        base_candidates = sample_payloads(
            generator,
            count=candidate_count,
            role="external_holdout",
            config=batch_config,
        )
        randomizer = random.Random(batch_config.seed + (101 if role == "calibration" else 211))
        for candidate in base_candidates:
            captcha = candidate["captcha"]
            width, height = int(captcha["width"]), int(captcha["height"])
            events, mutation = _motion_events(
                candidate["events"],
                width=width,
                height=height,
                policy=policy,
                randomizer=randomizer,
            )
            quality = validate_attempt(events, captcha_width=width, captcha_height=height)
            if quality.status == QUALITY_REJECTED:
                rejected_quality += 1
                continue
            vector = vectorize_attempt({"captcha": captcha, "events": events}, config.point_count)
            novelty = float(_novelty_distances(generator["nearest"], vector.reshape(1, -1))[0])
            if novelty < config.min_novelty_distance:
                rejected_novelty += 1
                continue
            attempt_id = f"redteam_{GENERATOR_VERSION}_{role}_{len(accepted):06d}"
            accepted.append(
                _payload(
                    attempt_id,
                    events,
                    role=role,
                    width=width,
                    height=height,
                    novelty_distance=novelty,
                    mutation=mutation,
                )
            )
            if len(accepted) == count:
                break
        batch_index += 1

    if len(accepted) != count:
        raise RuntimeError(
            f"generated {len(accepted)}/{count} hybrid samples after {batch_index} batches; "
            f"quality={rejected_quality}, novelty={rejected_novelty}. "
            "Review generator settings before lowering novelty."
        )
    return accepted, rejected_quality, rejected_novelty


def generate_dataset(
    *,
    human_attempts_path: Path,
    source_human_features_path: Path,
    output_path: Path,
    role: str,
    count: int,
    config: GeneratorConfig,
    model_path: Path | None = None,
) -> dict[str, Any]:
    if role not in ROLES:
        raise ValueError(f"unsupported role: {role}")
    if count < 1:
        raise ValueError("count must be positive")

    source_rows = _allowed_source_rows(human_attempts_path, source_human_features_path)
    generator = fit_generator(source_rows, config)
    policy = POLICIES[role]
    accepted, rejected_quality, rejected_novelty = generate_payloads_from_fitted(
        generator,
        role=role,
        count=count,
        config=config,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for payload in accepted:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "generator_type": "PCA + diagonal GaussianMixture + hybrid motion policy",
                "generator_version": GENERATOR_VERSION,
                "bot_family": BOT_FAMILY,
                "source_role": "human_development_only",
                "pca": generator["pca"],
                "gmm": generator["gmm"],
                "point_count": config.point_count,
                "dimensions": generator["dimensions"],
                "event_count_choices": generator["event_count_choices"],
                "source_count": generator["source_count"],
                "policy": asdict(policy),
                "config": asdict(config),
            },
            model_path,
        )

    is_external = role in EXTERNAL_ROLES
    training_usage = "external_holdout_only" if is_external else "redteam_only"
    manifest = {
        "dataset_name": output_path.stem,
        "generator_type": "PCA + diagonal GaussianMixture + hybrid motion policy",
        "generator_version": GENERATOR_VERSION,
        "bot_family": HARD_BOT_FAMILY if role == "external_holdout_hard" else BOT_FAMILY,
        "role": role,
        "evaluation_role": "external_holdout" if is_external else "redteam_calibration",
        "training_usage": training_usage,
        "detector_training_forbidden": True,
        "threshold_tuning_forbidden": True,
        "source_attempt_count": generator["source_count"],
        "source_attempt_ids_exported": False,
        "policy": asdict(policy),
        "config": asdict(config),
        "count": len(accepted),
        "quality_rejected": rejected_quality,
        "novelty_rejected": rejected_novelty,
        "inputs": {
            "human_attempts": {"path": str(human_attempts_path), "sha256": sha256(human_attempts_path)},
            "human_development_features": {
                "path": str(source_human_features_path),
                "sha256": sha256(source_human_features_path),
            },
        },
        "model_path": str(model_path) if model_path is not None else None,
        "output": {"path": str(output_path), "sha256": sha256(output_path)},
        "notes": [
            "Offline defensive red-team data only; no browser or network interaction.",
            "Output contains no Human source attempt IDs and is novelty-filtered after mutation.",
            "Calibration and external output are both forbidden from detector fitting and threshold tuning.",
            "External-holdout output is reserved for a future one-time evaluation.",
        ],
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-attempts", type=Path, required=True)
    parser.add_argument("--source-human-features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--point-count", type=int, default=48)
    parser.add_argument("--pca-components", type=int, default=24)
    parser.add_argument("--gmm-components", type=int, default=8)
    parser.add_argument("--min-novelty-distance", type=float, default=0.015)
    parser.add_argument("--model-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate_dataset(
        human_attempts_path=args.human_attempts,
        source_human_features_path=args.source_human_features,
        output_path=args.out,
        role=args.role,
        count=args.count,
        config=GeneratorConfig(
            point_count=args.point_count,
            pca_components=args.pca_components,
            gmm_components=args.gmm_components,
            min_novelty_distance=args.min_novelty_distance,
            seed=args.seed,
        ),
        model_path=args.model_out,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
