"""Named trajectory-only feature views for conservative score fusion.

The views use the same collected pointer trace but intentionally emphasize
different evidence.  They do not add any external or red-team holdout rows to
detector fitting.
"""

from __future__ import annotations

from app.services.feature_extractor_v23 import TRAJECTORY_ONLY_FEATURE_NAMES


_PHYSICS_FEATURES = {
    "speed_turn_abs_correlation",
    "turn_change_smoothness",
    "pause_position_entropy",
}

# Keeps the broad geometric and timing description while excluding the three
# new physics features.  This makes it a genuinely separate input view rather
# than a duplicate model with a different seed.
GENERAL_VIEW_NAMES = tuple(
    name for name in TRAJECTORY_ONLY_FEATURE_NAMES if name not in _PHYSICS_FEATURES
)

# Dynamics/physics view: it deliberately omits absolute route size and endpoint
# geometry so its decision relies on how the motion unfolded, not where it went.
DYNAMICS_PHYSICS_VIEW_NAMES = (
    "event_count",
    "duration_ms",
    "avg_speed",
    "max_speed",
    "speed_std",
    "avg_acceleration",
    "max_acceleration",
    "jerk_mean",
    "direction_changes",
    "pause_count",
    "pause_ratio",
    "interval_mean_ms",
    "interval_std_ms",
    "interval_cv",
    "duplicate_interval_ratio",
    "endpoint_adjustment_time",
    "final_segment_speed",
    "normalized_speed_p10",
    "normalized_speed_p50",
    "normalized_speed_p90",
    "normalized_acceleration_std",
    "normalized_jerk_std",
    "turn_angle_mean",
    "turn_angle_std",
    "turn_angle_p90",
    "turn_direction_change_ratio",
    "micro_move_ratio",
    "dwell_burst_count",
    "timing_entropy",
    "speed_peak_count",
    "speed_peak_position_ratio",
    "mid_to_edge_speed_ratio",
    "speed_burst_concentration",
    "peak_accel_decel_symmetry",
    "straight_burst_score",
    "interval_lag1_autocorrelation",
    "interval_delta_lag1_autocorrelation",
    "interval_second_difference_relative",
    "speed_turn_abs_correlation",
    "turn_change_smoothness",
    "pause_position_entropy",
)

FEATURE_VIEWS = {
    "general_without_physics": GENERAL_VIEW_NAMES,
    "dynamics_physics": DYNAMICS_PHYSICS_VIEW_NAMES,
}


def get_feature_view(name: str) -> tuple[str, ...]:
    """Return a fixed feature order for a named schema-2.3 trajectory view."""
    try:
        return FEATURE_VIEWS[name]
    except KeyError as error:
        raise ValueError(f"unknown trajectory feature view: {name}") from error


__all__ = [
    "DYNAMICS_PHYSICS_VIEW_NAMES",
    "FEATURE_VIEWS",
    "GENERAL_VIEW_NAMES",
    "get_feature_view",
]
