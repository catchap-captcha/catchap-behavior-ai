"""Versioned feature profiles used by offline training tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class FeatureProfile:
    version: str
    names: tuple[str, ...]
    extractor: Callable[[Iterable[dict[str, Any]], dict[str, Any] | None], dict[str, float]]
    input_scope: str = "all_behavioral_features"


def get_feature_profile(version: str, *, trajectory_only: bool = False) -> FeatureProfile:
    if version == "1.0":
        from app.services.feature_extractor import (
            FEATURE_NAMES,
            TRAJECTORY_ONLY_FEATURE_NAMES,
            extract_features,
        )

        names = TRAJECTORY_ONLY_FEATURE_NAMES if trajectory_only else FEATURE_NAMES
        scope = "pointer_trajectory_only" if trajectory_only else "all_behavioral_features"
        return FeatureProfile(version, tuple(names), extract_features, scope)
    if version == "2.0":
        from app.services.feature_extractor_v2 import (
            FEATURE_NAMES,
            TRAJECTORY_ONLY_FEATURE_NAMES,
            extract_features,
        )

        names = TRAJECTORY_ONLY_FEATURE_NAMES if trajectory_only else FEATURE_NAMES
        scope = "pointer_trajectory_only" if trajectory_only else "all_behavioral_features"
        return FeatureProfile(version, tuple(names), extract_features, scope)
    if version == "2.1":
        from app.services.feature_extractor_v21 import (
            FEATURE_NAMES,
            TRAJECTORY_ONLY_FEATURE_NAMES,
            extract_features,
        )

        names = TRAJECTORY_ONLY_FEATURE_NAMES if trajectory_only else FEATURE_NAMES
        scope = "pointer_trajectory_only" if trajectory_only else "all_behavioral_features"
        return FeatureProfile(version, tuple(names), extract_features, scope)
    if version == "2.2":
        from app.services.feature_extractor_v22 import (
            FEATURE_NAMES,
            TRAJECTORY_ONLY_FEATURE_NAMES,
            extract_features,
        )

        names = TRAJECTORY_ONLY_FEATURE_NAMES if trajectory_only else FEATURE_NAMES
        scope = "pointer_trajectory_only" if trajectory_only else "all_behavioral_features"
        return FeatureProfile(version, tuple(names), extract_features, scope)
    if version == "2.3":
        from app.services.feature_extractor_v23 import (
            FEATURE_NAMES,
            TRAJECTORY_ONLY_FEATURE_NAMES,
            extract_features,
        )

        names = TRAJECTORY_ONLY_FEATURE_NAMES if trajectory_only else FEATURE_NAMES
        scope = "pointer_trajectory_only" if trajectory_only else "all_behavioral_features"
        return FeatureProfile(version, tuple(names), extract_features, scope)
    raise ValueError(f"unsupported feature schema version: {version}")


__all__ = ["FeatureProfile", "get_feature_profile"]
