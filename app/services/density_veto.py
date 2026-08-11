"""The density model a veto-equipped bundle carries, and why it lives here.

This class is pickled *into* the model bundle. Where it is defined therefore
becomes part of the bundle's contract: joblib records the module path and looks
it up again at load time.

It used to be defined in `tools/train_density_veto.py`, which runs as a script,
so the bundle recorded it as `__main__.DensityVeto`. That resolves fine while
running the training script and nowhere else. Deploying such a bundle produced:

    AttributeError: Can't get attribute 'DensityVeto' on <module 'main'>

and the service came up with `model_loaded: false` while `/health` still
returned 200 — the pods looked healthy and scored nothing.

Every local check missed it because every local script did

    sys.modules["__main__"].DensityVeto = DensityVeto

before loading, which is exactly the condition production does not have. The
verification harness was patching away the failure it was meant to catch.

So the class lives under `app/` now: importable by the service, and recorded in
new bundles as `app.services.density_veto.DensityVeto`. Anything else pickled
into a bundle has to follow the same rule.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Absolute magnitudes an attacker changes for free by resizing the window, and
# lengths that differ between collection and legacy. The density model must not
# rest on either.
EXCLUDED = ("event_count", "duration_ms", "total_distance", "displacement",
            "y_deviation", "interval_mean_ms", "interval_std_ms")


class DensityVeto:
    """kNN distance plus isolation forest, fitted on humans only.

    Scores decay smoothly and are never clipped. Clipping to zero past a reference
    distance was silently fatal in the rebuild: real humans landed beyond it, all
    became exactly 0.0, and a veto calibrated to "further than any human" could
    never fire. Ordering has to survive past the reference — that is the region the
    veto exists to judge.
    """

    def __init__(self, X: np.ndarray, names: tuple[str, ...]) -> None:
        self.names = names
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.knn = NearestNeighbors(n_neighbors=min(15, len(Z))).fit(Z)
        distances, _ = self.knn.kneighbors(Z)
        self.reference = float(np.percentile(distances[:, 1:].mean(axis=1), 95)) or 1.0
        self.forest = IsolationForest(n_estimators=300, random_state=7).fit(Z)
        raw = self.forest.score_samples(Z)
        self.lo = float(np.percentile(raw, 5))
        self.hi = float(np.percentile(raw, 95))

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(np.nan_to_num(X))
        distances, _ = self.knn.kneighbors(Z)
        near = np.exp(-distances[:, 1:].mean(axis=1) / self.reference)
        raw = self.forest.score_samples(Z)
        span = max(self.hi - self.lo, 1e-9)
        iso = 1.0 / (1.0 + np.exp(-(raw - self.lo) / span * 4.0 + 2.0))
        return np.minimum(near, iso)
