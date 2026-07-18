"""Train the candidate tree models: RandomForest, ExtraTrees, XGBoost, LightGBM.

All models train on the SAME 29 features and the SAME train split, with a fixed
seed and class-imbalance handling. Hyperparameters are kept conservative and
reproducible; validation is used downstream to pick the decision threshold
(the primary tunable) — see :mod:`training.evaluate_models`.

Nothing here writes to ``models/`` — persistence and promotion are the
pipeline's job, and only after evaluation.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from training.split_dataset import SplitData

MODEL_NAMES = ["random_forest", "extra_trees", "xgboost", "lightgbm"]


def _scale_pos_weight(y: pd.Series) -> float:
    """XGBoost imbalance weight = negatives / positives (guarded)."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return (neg / pos) if pos > 0 else 1.0


def build_models(y_train: pd.Series, seed: int = 42) -> dict[str, Any]:
    """Construct the unfitted estimators with imbalance handling."""
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from xgboost import XGBClassifier

    return {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=_scale_pos_weight(y_train),
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
            tree_method="hist",
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            max_depth=-1,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        ),
    }


def train_all(split: SplitData, seed: int = 42) -> dict[str, Any]:
    """Fit all candidate models on the train split. Returns name -> model."""
    models = build_models(split.y_train, seed=seed)
    fitted: dict[str, Any] = {}
    for name, model in models.items():
        model.fit(split.X_train, split.y_train)
        fitted[name] = model
    return fitted
