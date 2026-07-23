"""Leakage-aware grouped cross-validation for decision-threshold calibration."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from training.build_dataset import Dataset
from training.evaluate_models import (
    Evaluation,
    evaluate_scores,
    positive_proba,
    select_threshold_from_scores,
)
from training.train_models import build_models


@dataclass
class GroupThresholdCalibration:
    model_name: str
    threshold: float
    requested_splits: int
    actual_splits: int
    max_human_frr: float
    eligible_rows: int
    excluded_anonymous_human_rows: int
    linked_human_groups: int
    bot_groups: int
    pooled_oof_metrics: Evaluation
    worst_fold_human_frr: float
    fold_threshold_min: float
    fold_threshold_median: float
    fold_threshold_max: float
    folds: list[dict[str, Any]]
    policy: str = "all-fold Human FRR constraint on out-of-fold scores"

    def to_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["pooled_oof_metrics"] = asdict(self.pooled_oof_metrics)
        return output


def row_group(row: dict[str, Any]) -> str | None:
    """Return the non-leaking group key used during threshold calibration."""
    if row.get("label") == "human":
        participant = row.get("anonymous_participant_id")
        return f"human::{participant}" if participant else None
    family = row.get("bot_family") or "unknown_family"
    generator = row.get("generator_version") or "unknown_generator"
    return f"bot::{family}::{generator}"


def calibrate_grouped_threshold(
    ds: Dataset,
    rows: list[dict[str, Any]],
    development_indices: list[int],
    model_name: str,
    *,
    seed: int = 42,
    n_splits: int = 5,
    max_human_frr: float = 0.03,
) -> GroupThresholdCalibration:
    """Calibrate one architecture using grouped out-of-fold predictions.

    Anonymous Human rows are omitted because their person-level group is
    unknown. They may still be used when fitting the final development model,
    but they cannot safely influence a participant-group threshold.
    """
    if len(ds) != len(rows):
        raise ValueError("dataset and source row counts differ")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    eligible: list[int] = []
    groups: list[str] = []
    excluded_anonymous = 0
    for index in development_indices:
        group = row_group(rows[index])
        if group is None:
            excluded_anonymous += 1
            continue
        eligible.append(index)
        groups.append(group)

    y = ds.y.iloc[eligible].reset_index(drop=True)
    X = ds.X.iloc[eligible].reset_index(drop=True)
    group_array = np.asarray(groups, dtype=object)
    labels = y.to_numpy(dtype=int)
    class_group_counts = [len(set(group_array[labels == label])) for label in (0, 1)]
    actual_splits = min(n_splits, *class_group_counts)
    if actual_splits < 2:
        raise ValueError(
            "grouped threshold calibration needs at least two Human and Bot groups"
        )

    splitter = StratifiedGroupKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=seed,
    )
    oof_scores = np.full(len(eligible), np.nan, dtype=float)
    fold_ids = np.full(len(eligible), -1, dtype=int)
    fold_timings: dict[int, float] = {}

    for fold, (train_rel, validation_rel) in enumerate(
        splitter.split(X, labels, groups=group_array)
    ):
        model = build_models(y.iloc[train_rel], seed=seed + fold)[model_name]
        model.fit(X.iloc[train_rel], y.iloc[train_rel])
        started = time.perf_counter()
        oof_scores[validation_rel] = positive_proba(model, X.iloc[validation_rel])
        fold_timings[fold] = (
            (time.perf_counter() - started) * 1000.0 / max(len(validation_rel), 1)
        )
        fold_ids[validation_rel] = fold

    if not np.isfinite(oof_scores).all() or (fold_ids < 0).any():
        raise RuntimeError("grouped calibration did not produce every OOF score")

    threshold = select_threshold_from_scores(
        oof_scores,
        labels,
        max_frr=max_human_frr,
        fold_ids=fold_ids,
    )
    pooled = evaluate_scores(
        oof_scores,
        labels,
        model_name=model_name,
        threshold=threshold,
        metrics_on="group_cv_oof",
        avg_inference_ms=float(np.mean(list(fold_timings.values()))),
    )

    folds: list[dict[str, Any]] = []
    individual_thresholds: list[float] = []
    for fold in range(actual_splits):
        mask = fold_ids == fold
        fold_threshold = select_threshold_from_scores(
            oof_scores[mask],
            labels[mask],
            max_frr=max_human_frr,
        )
        individual_thresholds.append(fold_threshold)
        metrics = evaluate_scores(
            oof_scores[mask],
            labels[mask],
            model_name=model_name,
            threshold=threshold,
            metrics_on=f"group_cv_fold_{fold}",
            avg_inference_ms=fold_timings[fold],
        )
        folds.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "human_rows": int((labels[mask] == 1).sum()),
                "bot_rows": int((labels[mask] == 0).sum()),
                "human_groups": len(set(group_array[mask & (labels == 1)])),
                "bot_groups": len(set(group_array[mask & (labels == 0)])),
                "standalone_threshold": fold_threshold,
                "metrics_at_shared_threshold": asdict(metrics),
            }
        )

    fold_frrs = [item["metrics_at_shared_threshold"]["human_frr"] for item in folds]
    return GroupThresholdCalibration(
        model_name=model_name,
        threshold=threshold,
        requested_splits=n_splits,
        actual_splits=actual_splits,
        max_human_frr=max_human_frr,
        eligible_rows=len(eligible),
        excluded_anonymous_human_rows=excluded_anonymous,
        linked_human_groups=len(set(group_array[labels == 1])),
        bot_groups=len(set(group_array[labels == 0])),
        pooled_oof_metrics=pooled,
        worst_fold_human_frr=max(fold_frrs),
        fold_threshold_min=float(np.min(individual_thresholds)),
        fold_threshold_median=float(np.median(individual_thresholds)),
        fold_threshold_max=float(np.max(individual_thresholds)),
        folds=folds,
    )


def fit_development_model(
    ds: Dataset,
    development_indices: list[int],
    model_name: str,
    *,
    seed: int = 42,
):
    """Fit one final candidate on all development rows after OOF calibration."""
    y = ds.y.iloc[development_indices].reset_index(drop=True)
    X = ds.X.iloc[development_indices].reset_index(drop=True)
    model = build_models(y, seed=seed)[model_name]
    model.fit(X, y)
    return model


__all__ = [
    "GroupThresholdCalibration",
    "calibrate_grouped_threshold",
    "fit_development_model",
    "row_group",
]
