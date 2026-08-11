"""Production model serving.

Loads the promoted model bundle from ``PRODUCTION_MODEL_DIR`` and scores feature
vectors. Convention: the positive class is Human (label == 1), so
``human_score = P(human)`` and ``bot_risk_score = 1 - human_score``.

If no bundle is present the service reports ``not ready`` and callers MUST return
HTTP 503 — a fake score is never produced.
"""

from __future__ import annotations

import glob
import os
import threading
from typing import Any

import numpy as np

from app.config import get_settings
from app.services.feature_extractor import FEATURE_NAMES

# Keys expected in a model bundle (written by the training pipeline).
BUNDLE_COMMON_KEYS = {
    "model_name",
    "model_version",
    "feature_schema_version",
    "threshold",
}
TWO_VIEW_MIN_FUSION = "min(P_human_general_without_physics, P_human_dynamics_physics)"


class ModelService:
    """Thread-safe holder for the current production model bundle."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bundle: dict[str, Any] | None = None

    # --- lifecycle ---
    def load(self) -> bool:
        """(Re)load the newest bundle from the production dir. Returns is_ready."""
        import joblib  # local import keeps startup light

        settings = get_settings()
        path = self._find_bundle(settings.production_model_dir)
        with self._lock:
            if path is None:
                self._bundle = None
                return False
            bundle = joblib.load(path)
            if not self._is_supported_bundle(bundle):
                # A malformed bundle must not be served.
                self._bundle = None
                return False
            self._bundle = bundle
            return True

    @staticmethod
    def _find_bundle(directory: str) -> str | None:
        files = sorted(glob.glob(os.path.join(directory, "*.joblib")))
        return files[-1] if files else None

    @staticmethod
    def _is_supported_bundle(bundle: Any) -> bool:
        if not isinstance(bundle, dict) or BUNDLE_COMMON_KEYS - set(bundle):
            return False
        if "model" in bundle and "feature_names" in bundle:
            return True
        if bundle.get("score_fusion") != TWO_VIEW_MIN_FUSION:
            return False
        models = bundle.get("models")
        feature_views = bundle.get("feature_views")
        return (
            isinstance(models, dict)
            and isinstance(feature_views, dict)
            and bool(models)
            and set(models) == set(feature_views)
            and all(feature_views[name] for name in models)
        )

    # --- introspection ---
    def is_ready(self) -> bool:
        return self._bundle is not None

    @property
    def model_name(self) -> str | None:
        return self._bundle["model_name"] if self._bundle else None

    @property
    def model_version(self) -> str | None:
        return self._bundle["model_version"] if self._bundle else None

    @property
    def feature_schema_version(self) -> str:
        if self._bundle:
            return str(self._bundle["feature_schema_version"])
        return get_settings().feature_schema_version

    @property
    def feature_input_scope(self) -> str:
        if self._bundle:
            return str(self._bundle.get("feature_input_scope", "all_behavioral_features"))
        return "all_behavioral_features"

    # --- scoring ---
    def score(self, features: dict[str, float]) -> dict[str, Any]:
        """Score one feature vector.

        Raises:
            RuntimeError: if no model is loaded (caller returns 503).
        """
        with self._lock:
            bundle = self._bundle
        if bundle is None:
            raise RuntimeError("model_not_ready")

        if "model" in bundle:
            order = bundle["feature_names"]
            vector = np.array([[float(features.get(name, 0.0)) for name in order]], dtype=float)
            human_score = self._positive_proba(bundle["model"], vector)
        else:
            view_scores = []
            for view_name, model in bundle["models"].items():
                order = bundle["feature_views"][view_name]
                vector = np.array([[float(features.get(name, 0.0)) for name in order]], dtype=float)
                view_scores.append(self._positive_proba(model, vector))
            human_score = min(view_scores)

        human_score = self._apply_density_veto(bundle, features, human_score)

        threshold = float(bundle["threshold"])
        prediction = "human" if human_score >= threshold else "bot"
        return {
            "human_score": round(human_score, 6),
            "bot_risk_score": round(1.0 - human_score, 6),
            "bot_decision": "low_risk" if prediction == "human" else "high_risk",
            "prediction": prediction,
            "threshold": threshold,
            "step_up_threshold": (
                float(bundle["step_up_threshold"])
                if bundle.get("step_up_threshold") is not None
                else None
            ),
            "model_name": bundle["model_name"],
            "model_version": bundle["model_version"],
            "feature_schema_version": bundle["feature_schema_version"],
        }

    @staticmethod
    def _apply_density_veto(bundle: dict[str, Any], features: dict[str, float],
                            human_score: float) -> float:
        """Force the score to 0 when the trace sits outside the human region.

        Gradient-boosted trees do not extrapolate. A region of feature space with
        no training points gets whatever leaf it happens to fall into, and the
        "perfectly uniform" corner lands on the human side — a straight constant
        speed bot, separable from humans at AUC 1.000 on a single feature, passed
        100% of the time when its family was held out of training. Adding features
        cannot fix an empty corner; only having an opinion about emptiness can.

        The density model supplies that opinion, and it is carried in the bundle
        rather than computed here so the veto and the threshold it was calibrated
        against can never drift apart.

        ⚠️ The threshold in a veto-equipped bundle assumes this runs. On the
        2026-08-10 candidate the veto rejects ~2.3% of human drags, and the
        operating point was re-read at that cost; scoring the same bundle without
        it drops the flag rate from 8.5% to 0.5% — which looks like a false-reject
        improvement and is really the model waving everything through, bots
        included. Bundles with no `density_veto` (everything deployed before
        2026-08-10) are unaffected and take the early return.
        """
        veto = bundle.get("density_veto")
        names = bundle.get("density_feature_names")
        floor = bundle.get("veto_below")
        if veto is None or not names or floor is None:
            return human_score
        vector = np.nan_to_num(
            np.array([[float(features.get(name, 0.0) or 0.0) for name in names]], dtype=float))
        try:
            density = float(veto.score(vector)[0])
        except Exception:                      # noqa: BLE001 - a broken veto must not
            return human_score                 # silently reject every human
        return 0.0 if density < float(floor) else human_score

    def score_per_drag(
        self,
        events: list[dict[str, Any]],
        extractor: Any,
        interaction: dict[str, Any] | None,
        threshold: float,
    ) -> dict[str, Any] | None:
        """Score each drag on its own and decide on the median.

        Returns None when the trajectory has no usable drag, so the caller can
        fall back to the session score rather than invent a verdict.

        The median, not the minimum: on the main-captcha data the minimum rejects
        52.2% of humans at the session threshold because one unlucky drag sinks the
        whole session, while the median rejects 1.5% and still blocks every bot.
        """
        from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags

        drags = split_drags(events)
        if not drags:
            return None

        # A drag too short to have a path carries no trajectory evidence, so it is
        # left out of the median rather than scored 0. Scoring it 0 costs real
        # users: 15 of 166 human sessions contain one such drag among several,
        # and forcing them to 0 moved human FRR from 1.5% to 5.2%.
        #
        # What separates a teleport is that *every* drag is starved — 5 of 5 in
        # the teleport family, 0 of 166 humans. So the floor applies to the
        # session, not to each drag.
        scores: list[float] = []
        starved = 0
        for drag in drags:
            if move_count(drag) < MIN_MOVES_PER_DRAG:
                starved += 1
                continue
            scores.append(float(self.score(extractor(drag, interaction))["human_score"]))

        if not scores:
            return {
                "human_score": 0.0,
                "drag_count": len(drags),
                "drag_scores": [],
                "starved_drags": starved,
                "prediction": "bot",
                "threshold": threshold,
                "reason": "every_drag_below_move_floor",
            }

        ordered = sorted(scores)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

        return {
            "human_score": round(median, 6),
            "drag_count": len(drags),
            "drag_scores": [round(s, 6) for s in scores],
            "starved_drags": starved,
            "prediction": "human" if median >= threshold else "bot",
            "threshold": threshold,
        }

    @staticmethod
    def _positive_proba(model: Any, vector: np.ndarray) -> float:
        """Return P(class == 1 == human) robustly across sklearn-style models."""
        proba = model.predict_proba(vector)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        try:
            idx = classes.index(1)
        except ValueError:
            idx = len(proba) - 1
        return float(proba[idx])


# Module-level singleton used by the API layer.
model_service = ModelService()


def feature_schema_version() -> str:
    """Current feature schema version (from the loaded bundle or config)."""
    return model_service.feature_schema_version


# Re-exported so callers can validate against the canonical list.
__all__ = ["model_service", "ModelService", "FEATURE_NAMES", "feature_schema_version"]
