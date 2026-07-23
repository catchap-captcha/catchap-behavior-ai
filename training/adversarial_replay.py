"""Defensive replay transformations with explicitly separated data profiles.

The development profile is allowed in offline model fitting.  The external
profile uses disjoint transform ranges and a different local-time curve, so it
must remain outside fitting and threshold selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ReplayTransformProfile:
    name: str
    generator_version: str
    rotation_abs_degrees: tuple[float, float]
    resample_ratio_ranges: tuple[tuple[float, float], ...]
    time_scale_ranges: tuple[tuple[float, float], ...]
    nonlinear_power_ranges: tuple[tuple[float, float], ...]
    local_curve: str
    slow_strength: tuple[float, float]
    fast_strength: tuple[float, float]
    curvature_amplitude_ratio: tuple[float, float]
    curvature_cycles: tuple[float, float]
    adaptive_resample_strength: tuple[float, float]
    micro_jitter_ratio: tuple[float, float]


DEVELOPMENT_PROFILE = ReplayTransformProfile(
    name="development_rotation_local_gaussian_v1",
    generator_version="adversarial_replay_development_v1",
    rotation_abs_degrees=(2.0, 6.0),
    resample_ratio_ranges=((0.80, 0.94), (1.06, 1.20)),
    time_scale_ranges=((0.82, 1.24),),
    nonlinear_power_ranges=((0.90, 1.16),),
    local_curve="two_gaussian_regions",
    slow_strength=(1.35, 2.10),
    fast_strength=(0.58, 0.80),
    curvature_amplitude_ratio=(0.0, 0.0),
    curvature_cycles=(0.0, 0.0),
    adaptive_resample_strength=(0.0, 0.0),
    micro_jitter_ratio=(0.0, 0.0),
)

DEVELOPMENT_BROAD_PROFILE = ReplayTransformProfile(
    name="development_broad_curvature_multiscale_v1",
    generator_version="adversarial_replay_development_broad_v1",
    rotation_abs_degrees=(0.5, 8.0),
    resample_ratio_ranges=((0.48, 0.76), (1.24, 1.60)),
    time_scale_ranges=((0.58, 0.86), (1.16, 1.56)),
    nonlinear_power_ranges=((0.70, 0.94), (1.10, 1.40)),
    local_curve="multi_pulse_regions",
    slow_strength=(1.60, 3.60),
    fast_strength=(0.30, 0.70),
    curvature_amplitude_ratio=(0.006, 0.028),
    curvature_cycles=(0.5, 2.5),
    adaptive_resample_strength=(0.0, 0.0),
    micro_jitter_ratio=(0.0, 0.0),
)

DEVELOPMENT_COMPOSITE_PROFILE = ReplayTransformProfile(
    name="development_composite_adaptive_replay_v1",
    generator_version="adversarial_replay_development_composite_v1",
    rotation_abs_degrees=(1.0, 9.0),
    resample_ratio_ranges=((0.50, 0.70), (1.30, 1.60)),
    time_scale_ranges=((0.58, 0.86), (1.16, 1.60)),
    nonlinear_power_ranges=((0.66, 0.92), (1.12, 1.46)),
    local_curve="multi_pulse_regions",
    slow_strength=(1.80, 4.00),
    fast_strength=(0.22, 0.65),
    curvature_amplitude_ratio=(0.015, 0.032),
    curvature_cycles=(0.75, 3.0),
    adaptive_resample_strength=(0.12, 0.30),
    micro_jitter_ratio=(0.003, 0.012),
)

EXTERNAL_HOLDOUT_PROFILE = ReplayTransformProfile(
    name="external_rotation_alternating_timewarp_v1",
    generator_version="adversarial_replay_external_holdout_v2",
    rotation_abs_degrees=(9.0, 15.0),
    resample_ratio_ranges=((0.56, 0.72), (1.30, 1.46)),
    time_scale_ranges=((0.62, 0.76), (1.30, 1.46)),
    nonlinear_power_ranges=((0.58, 0.82), (1.28, 1.58)),
    local_curve="alternating_sine_regions",
    slow_strength=(2.50, 4.00),
    fast_strength=(0.22, 0.48),
    curvature_amplitude_ratio=(0.0, 0.0),
    curvature_cycles=(0.0, 0.0),
    adaptive_resample_strength=(0.0, 0.0),
    micro_jitter_ratio=(0.0, 0.0),
)

FRESH_EXTERNAL_HOLDOUT_PROFILE = ReplayTransformProfile(
    name="external_fresh_participant_asymmetric_warp_v1",
    generator_version="adversarial_replay_fresh_external_v1",
    rotation_abs_degrees=(12.0, 20.0),
    resample_ratio_ranges=((0.30, 0.46), (1.70, 1.95)),
    time_scale_ranges=((0.34, 0.54), (1.70, 1.98)),
    nonlinear_power_ranges=((0.38, 0.62), (1.52, 1.82)),
    local_curve="asymmetric_triangle_regions",
    slow_strength=(3.80, 5.50),
    fast_strength=(0.12, 0.35),
    curvature_amplitude_ratio=(0.038, 0.060),
    curvature_cycles=(3.0, 5.0),
    adaptive_resample_strength=(0.36, 0.55),
    micro_jitter_ratio=(0.014, 0.025),
)

PROFILES = {
    DEVELOPMENT_PROFILE.name: DEVELOPMENT_PROFILE,
    DEVELOPMENT_BROAD_PROFILE.name: DEVELOPMENT_BROAD_PROFILE,
    DEVELOPMENT_COMPOSITE_PROFILE.name: DEVELOPMENT_COMPOSITE_PROFILE,
    EXTERNAL_HOLDOUT_PROFILE.name: EXTERNAL_HOLDOUT_PROFILE,
    FRESH_EXTERNAL_HOLDOUT_PROFILE.name: FRESH_EXTERNAL_HOLDOUT_PROFILE,
}
DEVELOPMENT_PROFILE_NAMES = frozenset(
    (
        DEVELOPMENT_PROFILE.name,
        DEVELOPMENT_BROAD_PROFILE.name,
        DEVELOPMENT_COMPOSITE_PROFILE.name,
    )
)


def get_profile(name: str) -> ReplayTransformProfile:
    try:
        return PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown replay transform profile: {name}") from error


def _sample_ranges(
    randomizer: random.Random,
    ranges: tuple[tuple[float, float], ...],
) -> float:
    lower, upper = randomizer.choice(ranges)
    return randomizer.uniform(lower, upper)


def _source_points(source: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rows = sorted(source["events"], key=lambda event: event.get("seq", 0))
    points = np.asarray([[float(row["x"]), float(row["y"])] for row in rows], dtype=float)
    times = np.asarray([float(row["t_ms"]) for row in rows], dtype=float)
    return points, times


def _arc_resample(
    points: np.ndarray,
    count: int,
    target_progress: np.ndarray | None = None,
) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] <= 1e-9:
        return np.repeat(points[:1], count, axis=0)
    progress = target_progress if target_progress is not None else np.linspace(0.0, 1.0, count)
    targets = progress * cumulative[-1]
    return np.column_stack(
        [np.interp(targets, cumulative, points[:, dimension]) for dimension in range(2)]
    )


def _adaptive_arc_resample(
    points: np.ndarray,
    count: int,
    randomizer: random.Random,
    profile: ReplayTransformProfile,
) -> tuple[np.ndarray, dict[str, float]]:
    low, high = profile.adaptive_resample_strength
    if high <= 0.0:
        return _arc_resample(points, count), {
            "adaptive_resample_strength": 0.0,
            "adaptive_resample_cycles": 0.0,
            "adaptive_resample_phase": 0.0,
        }

    strength = randomizer.uniform(low, high)
    cycles = randomizer.uniform(1.0, 4.0)
    phase = randomizer.uniform(0.0, 2.0 * math.pi)
    source_progress = np.linspace(0.0, 1.0, max(len(points), 4))
    density = np.clip(
        1.0 + strength * np.sin(2.0 * math.pi * cycles * source_progress + phase),
        0.1,
        None,
    )
    cumulative_density = np.concatenate(([0.0], np.cumsum((density[1:] + density[:-1]) / 2.0)))
    cumulative_density /= cumulative_density[-1]
    target_progress = np.interp(
        np.linspace(0.0, 1.0, count), cumulative_density, source_progress
    )
    return _arc_resample(points, count, target_progress), {
        "adaptive_resample_strength": round(strength, 6),
        "adaptive_resample_cycles": round(cycles, 6),
        "adaptive_resample_phase": round(phase, 6),
    }


def _local_time_profile(
    source_times: np.ndarray,
    target_count: int,
    randomizer: random.Random,
    profile: ReplayTransformProfile,
) -> tuple[np.ndarray, dict[str, float | str]]:
    source_progress = np.linspace(0.0, 1.0, len(source_times))
    target_progress = np.linspace(0.0, 1.0, target_count)
    source_relative = np.maximum.accumulate(source_times - source_times[0])
    duration = max(float(source_relative[-1]), 1.0)
    base_dt = np.diff(np.interp(target_progress, source_progress, source_relative / duration))
    base_dt = np.maximum(base_dt, 1e-5)
    midpoints = (target_progress[:-1] + target_progress[1:]) / 2.0
    slow_strength = randomizer.uniform(*profile.slow_strength)
    fast_strength = randomizer.uniform(*profile.fast_strength)

    if profile.local_curve == "two_gaussian_regions":
        slow_center = randomizer.uniform(0.22, 0.42)
        fast_center = randomizer.uniform(0.58, 0.78)
        width = randomizer.uniform(0.08, 0.15)
        slow = 1.0 + (slow_strength - 1.0) * np.exp(-((midpoints - slow_center) / width) ** 2)
        fast = 1.0 - (1.0 - fast_strength) * np.exp(-((midpoints - fast_center) / width) ** 2)
        local_multiplier = slow * fast
        curve_metadata = {
            "slow_center": slow_center,
            "fast_center": fast_center,
            "curve_width": width,
        }
    elif profile.local_curve == "alternating_sine_regions":
        cycles = randomizer.choice((2.0, 2.5, 3.0))
        phase = randomizer.uniform(0.15, 1.20)
        wave = np.sin(2.0 * math.pi * cycles * midpoints + phase)
        normalized = (wave + 1.0) / 2.0
        local_multiplier = fast_strength + (slow_strength - fast_strength) * normalized
        curve_metadata = {"wave_cycles": cycles, "wave_phase": phase}
    elif profile.local_curve == "multi_pulse_regions":
        cycles = randomizer.uniform(1.5, 3.5)
        phase = randomizer.uniform(0.0, math.pi)
        wave = np.sin(2.0 * math.pi * cycles * midpoints + phase)
        pulse = np.sin(2.0 * math.pi * (cycles / 2.0) * midpoints + phase / 2.0)
        normalized = np.clip(0.5 + 0.35 * wave + 0.15 * pulse, 0.0, 1.0)
        local_multiplier = fast_strength + (slow_strength - fast_strength) * normalized
        curve_metadata = {"pulse_cycles": cycles, "pulse_phase": phase}
    elif profile.local_curve == "asymmetric_triangle_regions":
        cycles = randomizer.choice((2.0, 3.0, 4.0))
        phase = randomizer.uniform(0.0, 1.0)
        skew = randomizer.uniform(0.24, 0.46)
        fractional = np.mod(cycles * midpoints + phase, 1.0)
        triangle = np.where(
            fractional < skew,
            fractional / skew,
            (1.0 - fractional) / (1.0 - skew),
        )
        local_multiplier = fast_strength + (slow_strength - fast_strength) * triangle
        curve_metadata = {
            "triangle_cycles": cycles,
            "triangle_phase": phase,
            "triangle_skew": skew,
        }
    else:  # Defensive future-proofing when a new profile is added.
        raise ValueError(f"unsupported local time curve: {profile.local_curve}")

    nonlinear_power = _sample_ranges(randomizer, profile.nonlinear_power_ranges)
    local_dt = np.maximum(base_dt * local_multiplier, 1e-6) ** nonlinear_power
    time_scale = _sample_ranges(randomizer, profile.time_scale_ranges)
    local_dt = local_dt / local_dt.sum() * duration * time_scale
    times = np.concatenate(([0.0], np.cumsum(local_dt)))
    return times, {
        "local_curve": profile.local_curve,
        "time_scale": round(time_scale, 6),
        "nonlinear_power": round(nonlinear_power, 6),
        "slow_strength": round(slow_strength, 6),
        "fast_strength": round(fast_strength, 6),
        **{name: round(value, 6) for name, value in curve_metadata.items()},
    }


def _apply_curvature_drift(
    points: np.ndarray,
    randomizer: random.Random,
    profile: ReplayTransformProfile,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, float]]:
    low, high = profile.curvature_amplitude_ratio
    if len(points) < 3 or high <= 0.0:
        return points, {
            "curvature_amplitude_px": 0.0,
            "curvature_cycles": 0.0,
            "curvature_phase": 0.0,
        }

    amplitude = randomizer.uniform(low, high) * min(width, height)
    cycles = randomizer.uniform(*profile.curvature_cycles)
    phase = randomizer.uniform(0.0, 2.0 * math.pi)
    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0])) / np.maximum(norms, 1e-9)
    progress = np.linspace(0.0, 1.0, len(points))
    envelope = np.sin(math.pi * progress)
    offsets = amplitude * envelope * np.sin(2.0 * math.pi * cycles * progress + phase)
    return points + normals * offsets[:, None], {
        "curvature_amplitude_px": round(amplitude, 6),
        "curvature_cycles": round(cycles, 6),
        "curvature_phase": round(phase, 6),
    }


def _apply_speed_correlated_micro_jitter(
    points: np.ndarray,
    times: np.ndarray,
    randomizer: random.Random,
    profile: ReplayTransformProfile,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, float]]:
    low, high = profile.micro_jitter_ratio
    if len(points) < 3 or high <= 0.0:
        return points, {"micro_jitter_amplitude_px": 0.0}

    amplitude = randomizer.uniform(low, high) * min(width, height)
    tangents = np.gradient(points, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0])) / np.maximum(norms, 1e-9)
    dt = np.maximum(np.gradient(times), 1.0)
    local_speed = norms[:, 0] / dt
    speed_scale = (local_speed - local_speed.min()) / max(float(np.ptp(local_speed)), 1e-9)
    noise = np.asarray([randomizer.gauss(0.0, 1.0) for _ in points], dtype=float)
    noise = np.convolve(noise, np.asarray((0.25, 0.5, 0.25)), mode="same")
    envelope = np.sin(math.pi * np.linspace(0.0, 1.0, len(points)))
    offsets = amplitude * envelope * (0.35 + 0.65 * speed_scale) * noise
    return points + normals * offsets[:, None], {
        "micro_jitter_amplitude_px": round(amplitude, 6),
    }


def adversarial_replay_warp(
    source: dict[str, Any],
    randomizer: random.Random,
    profile: ReplayTransformProfile,
) -> tuple[list[dict[str, Any]], int, int, dict[str, Any]]:
    """Return a transformed copy and auditable, non-identifying transform metadata."""
    width = int(source["captcha"]["width"])
    height = int(source["captcha"]["height"])
    points, source_times = _source_points(source)
    target_count = max(4, int(round(len(points) * _sample_ranges(randomizer, profile.resample_ratio_ranges))))
    resampled, resample_metadata = _adaptive_arc_resample(
        points,
        target_count,
        randomizer,
        profile,
    )
    resampled, curvature_metadata = _apply_curvature_drift(
        resampled,
        randomizer,
        profile,
        width,
        height,
    )

    angle_degrees = randomizer.choice((-1.0, 1.0)) * randomizer.uniform(*profile.rotation_abs_degrees)
    angle = math.radians(angle_degrees)
    scale = randomizer.uniform(0.90, 1.06)
    translation = np.asarray(
        [
            randomizer.uniform(-width * 0.018, width * 0.018),
            randomizer.uniform(-height * 0.025, height * 0.025),
        ]
    )
    center = np.asarray([width / 2.0, height / 2.0])
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=float,
    )
    transformed = (resampled - center) @ rotation.T * scale + center + translation
    times, timing_metadata = _local_time_profile(source_times, target_count, randomizer, profile)
    transformed, jitter_metadata = _apply_speed_correlated_micro_jitter(
        transformed,
        times,
        randomizer,
        profile,
        width,
        height,
    )
    transformed[:, 0] = np.clip(transformed[:, 0], 0.0, float(width))
    transformed[:, 1] = np.clip(transformed[:, 1], 0.0, float(height))

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
        "profile": profile.name,
        "profile_parameters": asdict(profile),
        "rotation_degrees": round(angle_degrees, 6),
        "spatial_scale": round(scale, 6),
        "source_event_count": len(points),
        "target_event_count": target_count,
        **resample_metadata,
        **curvature_metadata,
        **jitter_metadata,
        **timing_metadata,
    }


__all__ = [
    "DEVELOPMENT_PROFILE",
    "DEVELOPMENT_BROAD_PROFILE",
    "DEVELOPMENT_COMPOSITE_PROFILE",
    "DEVELOPMENT_PROFILE_NAMES",
    "EXTERNAL_HOLDOUT_PROFILE",
    "FRESH_EXTERNAL_HOLDOUT_PROFILE",
    "PROFILES",
    "ReplayTransformProfile",
    "adversarial_replay_warp",
    "get_profile",
]
