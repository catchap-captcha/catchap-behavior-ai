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
BUNDLE_REQUIRED_KEYS = {
    "model",
    "model_name",
    "model_version",
    "feature_names",
    "feature_schema_version",
    "threshold",
}


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
            missing = BUNDLE_REQUIRED_KEYS - set(bundle)
            if missing:
                # A malformed bundle must not be served.
                self._bundle = None
                return False
            self._bundle = bundle
            return True

    @staticmethod
    def _find_bundle(directory: str) -> str | None:
        files = sorted(glob.glob(os.path.join(directory, "*.joblib")))
        return files[-1] if files else None

    # --- introspection ---
    def is_ready(self) -> bool:
        return self._bundle is not None

    @property
    def model_name(self) -> str | None:
        return self._bundle["model_name"] if self._bundle else None

    @property
    def model_version(self) -> str | None:
        return self._bundle["model_version"] if self._bundle else None

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

        order = bundle["feature_names"]
        vector = np.array([[float(features.get(name, 0.0)) for name in order]], dtype=float)
        human_score = self._positive_proba(bundle["model"], vector)

        threshold = float(bundle["threshold"])
        prediction = "human" if human_score >= threshold else "bot"
        return {
            "human_score": round(human_score, 6),
            "bot_risk_score": round(1.0 - human_score, 6),
            "bot_decision": "low_risk" if prediction == "human" else "high_risk",
            "prediction": prediction,
            "threshold": threshold,
            "model_name": bundle["model_name"],
            "model_version": bundle["model_version"],
            "feature_schema_version": bundle["feature_schema_version"],
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
    if model_service.is_ready():
        return model_service._bundle["feature_schema_version"]  # type: ignore[index]
    return get_settings().feature_schema_version


# Re-exported so callers can validate against the canonical list.
__all__ = ["model_service", "ModelService", "FEATURE_NAMES", "feature_schema_version"]
