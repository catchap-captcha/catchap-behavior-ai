"""Feature-view boundaries used by formal score-fusion validation."""

from app.services.feature_extractor_v23 import TRAJECTORY_ONLY_FEATURE_NAMES
from app.services.trajectory_feature_views import (
    DYNAMICS_PHYSICS_VIEW_NAMES,
    GENERAL_VIEW_NAMES,
    get_feature_view,
)


def test_two_views_are_valid_distinct_schema_v23_subsets():
    all_names = set(TRAJECTORY_ONLY_FEATURE_NAMES)

    assert set(GENERAL_VIEW_NAMES) <= all_names
    assert set(DYNAMICS_PHYSICS_VIEW_NAMES) <= all_names
    assert set(GENERAL_VIEW_NAMES) != set(DYNAMICS_PHYSICS_VIEW_NAMES)
    assert "speed_turn_abs_correlation" not in GENERAL_VIEW_NAMES
    assert "speed_turn_abs_correlation" in DYNAMICS_PHYSICS_VIEW_NAMES
    assert get_feature_view("dynamics_physics") == DYNAMICS_PHYSICS_VIEW_NAMES
