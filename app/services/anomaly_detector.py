"""Human-only anomaly scoring for behavior features.

This component is deliberately separate from the supervised bot classifier.
It learns only the confirmed-human region and returns an empirical anomaly
percentile. The score is auxiliary evidence, not a calibrated bot probability.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class HumanIsolationForest:
    """Estimate how far a sample lies outside confirmed-human behavior."""

    def __init__(
        self,
        feature_names: Iterable[str],
        *,
        seed: int = 42,
        n_estimators: int = 300,
        min_training_samples: int = 20,
    ) -> None:
        self.feature_names = tuple(feature_names)
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")
        self.seed = seed
        self.n_estimators = n_estimators
        self.min_training_samples = min_training_samples
        self.model: IsolationForest | None = None
        self._reference_scores: np.ndarray | None = None

    def fit(self, human_features: pd.DataFrame | np.ndarray) -> HumanIsolationForest:
        """Fit on confirmed-human rows only."""
        matrix = self._as_matrix(human_features)
        if len(matrix) < self.min_training_samples:
            raise ValueError(
                "not enough confirmed-human rows for anomaly training: "
                f"need {self.min_training_samples}, found {len(matrix)}"
            )

        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination="auto",
            max_samples="auto",
            n_jobs=-1,
            random_state=self.seed,
        )
        model.fit(matrix)
        self.model = model
        self._reference_scores = np.sort(-model.score_samples(matrix))
        return self

    def anomaly_percentile(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return an empirical human-anomaly percentile in ``[0, 1]``.

        A value near 1 means that the sample is more unusual than nearly all
        confirmed-human training rows. It does not by itself prove automation.
        """
        model, reference = self._fitted_state()
        matrix = self._as_matrix(features)
        raw_scores = -model.score_samples(matrix)
        ranks = np.searchsorted(reference, raw_scores, side="right")
        return np.clip(ranks / len(reference), 0.0, 1.0)

    def score_one(self, features: Mapping[str, float]) -> float:
        """Score one feature mapping using the stored canonical feature order."""
        row = np.array(
            [[float(features.get(name, 0.0)) for name in self.feature_names]],
            dtype=float,
        )
        return float(self.anomaly_percentile(row)[0])

    def _fitted_state(self) -> tuple[IsolationForest, np.ndarray]:
        if self.model is None or self._reference_scores is None:
            raise RuntimeError("human anomaly detector is not fitted")
        return self.model, self._reference_scores

    def _as_matrix(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(features, pd.DataFrame):
            missing = set(self.feature_names) - set(features.columns)
            if missing:
                raise ValueError(f"missing anomaly features: {sorted(missing)}")
            matrix = features.loc[:, self.feature_names].to_numpy(dtype=float)
        else:
            matrix = np.asarray(features, dtype=float)

        if matrix.ndim != 2 or matrix.shape[1] != len(self.feature_names):
            raise ValueError(
                f"expected a 2D matrix with {len(self.feature_names)} features, "
                f"got shape {matrix.shape}"
            )
        if not np.isfinite(matrix).all():
            raise ValueError("anomaly features must all be finite")
        return matrix


def train_human_anomaly_detector(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    seed: int = 42,
) -> HumanIsolationForest:
    """Train the auxiliary detector using only label-1 human rows."""
    human_features = X_train.loc[y_train.to_numpy() == 1]
    detector = HumanIsolationForest(X_train.columns, seed=seed)
    return detector.fit(human_features)


__all__ = ["HumanIsolationForest", "train_human_anomaly_detector"]
