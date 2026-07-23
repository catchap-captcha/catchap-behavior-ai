"""Decision-threshold selection and final model evaluation.

Positive class = Human (1). The base helper supports a validation split, while
the local security runner uses grouped out-of-fold scores so no single
participant allocation determines the threshold. The TEST split is scored only
after calibration.

Human False Rejection Rate (FRR) = fraction of true humans predicted as bot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

# Threshold is chosen to respect this FRR budget when possible.
DEFAULT_MAX_FRR = 0.03


@dataclass
class Evaluation:
    model_name: str
    threshold: float
    accuracy: float
    human_precision: float
    human_recall: float
    human_f1: float
    bot_recall: float
    human_frr: float
    roc_auc: float | None
    pr_auc: float | None
    confusion_matrix: list[list[int]]
    avg_inference_ms: float
    feature_importance: dict[str, float]
    metrics_on: str  # "validation" or "test"


def positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Return ``P(Human)`` for a sklearn-compatible binary classifier."""
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
    return proba[:, idx]


def select_threshold_from_scores(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    max_frr: float = DEFAULT_MAX_FRR,
    fold_ids: np.ndarray | None = None,
) -> float:
    """Select the strongest threshold that respects the Human FRR budget.

    When ``fold_ids`` is supplied, the FRR constraint must hold in every fold,
    not only in the pooled rows. This prevents a large participant group in one
    fold from hiding poor behavior in another fold.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if scores.ndim != 1 or y_true.ndim != 1 or len(scores) != len(y_true):
        raise ValueError("scores and y_true must be same-length 1D arrays")
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("scores must contain finite values")
    if not 0.0 <= max_frr <= 1.0:
        raise ValueError("max_frr must be between 0 and 1")

    human_scores = np.sort(scores[y_true == 1])
    bot_scores = np.sort(scores[y_true == 0])
    if not len(human_scores) or not len(bot_scores):
        raise ValueError("threshold calibration requires Human and Bot rows")

    candidates = np.unique(np.concatenate([scores, [0.5]]))
    pooled_frr = np.searchsorted(human_scores, candidates, side="left") / len(human_scores)
    valid = pooled_frr <= max_frr

    if fold_ids is not None:
        fold_ids = np.asarray(fold_ids)
        if fold_ids.ndim != 1 or len(fold_ids) != len(scores):
            raise ValueError("fold_ids must match scores")
        for fold in np.unique(fold_ids):
            fold_humans = np.sort(scores[(fold_ids == fold) & (y_true == 1)])
            fold_bots = scores[(fold_ids == fold) & (y_true == 0)]
            if not len(fold_humans) or not len(fold_bots):
                raise ValueError(f"fold {fold!r} must contain Human and Bot rows")
            fold_frr = np.searchsorted(fold_humans, candidates, side="left") / len(
                fold_humans
            )
            valid &= fold_frr <= max_frr

    if not valid.any():
        return float(candidates[0])

    bot_recall = np.searchsorted(bot_scores, candidates, side="left") / len(bot_scores)
    valid_indices = np.flatnonzero(valid)
    best_recall = bot_recall[valid_indices].max()
    best_indices = valid_indices[bot_recall[valid_indices] == best_recall]
    return float(candidates[best_indices[-1]])


def select_threshold_per_human_group(
    scores: np.ndarray,
    y_true: np.ndarray,
    human_group_ids: np.ndarray,
    *,
    max_frr: float = DEFAULT_MAX_FRR,
) -> float:
    """Select a threshold that respects the FRR budget for every Human group.

    Fold-level constraints can still hide one participant whose traces are
    consistently judged as risky. This stricter calibration uses only
    out-of-fold development scores and requires every identified Human group
    to satisfy the same FRR budget before Bot recall is optimized.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    human_group_ids = np.asarray(human_group_ids, dtype=object)
    if scores.ndim != 1 or y_true.ndim != 1 or len(scores) != len(y_true):
        raise ValueError("scores and y_true must be same-length 1D arrays")
    if human_group_ids.ndim != 1 or len(human_group_ids) != len(scores):
        raise ValueError("human_group_ids must match scores")
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("scores must contain finite values")
    if not 0.0 <= max_frr <= 1.0:
        raise ValueError("max_frr must be between 0 and 1")

    human_mask = y_true == 1
    bot_scores = np.sort(scores[y_true == 0])
    if not human_mask.any() or not len(bot_scores):
        raise ValueError("threshold calibration requires Human and Bot rows")
    if any(group in (None, "") for group in human_group_ids[human_mask]):
        raise ValueError("every Human score needs a non-empty group id")

    candidates = np.unique(np.concatenate([scores, [0.5]]))
    valid = np.ones(len(candidates), dtype=bool)
    for group in np.unique(human_group_ids[human_mask]):
        group_scores = np.sort(scores[human_mask & (human_group_ids == group)])
        group_frr = np.searchsorted(group_scores, candidates, side="left") / len(group_scores)
        valid &= group_frr <= max_frr

    if not valid.any():
        return float(candidates[0])

    bot_recall = np.searchsorted(bot_scores, candidates, side="left") / len(bot_scores)
    valid_indices = np.flatnonzero(valid)
    best_recall = bot_recall[valid_indices].max()
    best_indices = valid_indices[bot_recall[valid_indices] == best_recall]
    return float(candidates[best_indices[-1]])


def select_threshold(
    model: Any, X_val: pd.DataFrame, y_val: pd.Series, max_frr: float = DEFAULT_MAX_FRR
) -> float:
    """Pick the threshold on validation.

    Among thresholds that keep Human FRR <= ``max_frr``, choose the one with the
    highest Bot recall. If none satisfy the FRR budget, fall back to the
    threshold with the lowest FRR (safest for real users).
    """
    return select_threshold_from_scores(
        positive_proba(model, X_val),
        y_val.to_numpy(),
        max_frr=max_frr,
    )


def evaluate(
    model: Any,
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float,
    metrics_on: str,
) -> Evaluation:
    """Compute the full metric set at a fixed threshold."""
    t0 = time.perf_counter()
    scores = positive_proba(model, X)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    avg_ms = elapsed_ms / max(len(X), 1)

    return evaluate_scores(
        scores,
        y.to_numpy(),
        model_name=model_name,
        threshold=threshold,
        metrics_on=metrics_on,
        avg_inference_ms=avg_ms,
        feature_importance=_feature_importance(model, X.columns.tolist()),
    )


def evaluate_scores(
    scores: np.ndarray,
    y_true: np.ndarray,
    *,
    model_name: str,
    threshold: float,
    metrics_on: str,
    avg_inference_ms: float = 0.0,
    feature_importance: dict[str, float] | None = None,
) -> Evaluation:
    """Compute metrics from precomputed ``P(Human)`` scores."""
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    if scores.ndim != 1 or y_true.ndim != 1 or len(scores) != len(y_true):
        raise ValueError("scores and y_true must be same-length 1D arrays")

    pred = (scores >= threshold).astype(int)

    # human = positive class (1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=[1], average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()

    has_both_classes = len(np.unique(y_true)) == 2
    roc = _safe(lambda: roc_auc_score(y_true, scores)) if has_both_classes else None
    pr = _safe(lambda: average_precision_score(y_true, scores)) if has_both_classes else None

    return Evaluation(
        model_name=model_name,
        threshold=float(threshold),
        accuracy=float((pred == y_true).mean()),
        human_precision=float(prec),
        human_recall=float(rec),
        human_f1=float(f1),
        bot_recall=_bot_recall(y_true, pred),
        human_frr=_human_frr(y_true, pred),
        roc_auc=roc,
        pr_auc=pr,
        confusion_matrix=cm,
        avg_inference_ms=float(avg_inference_ms),
        feature_importance=feature_importance or {},
        metrics_on=metrics_on,
    )


def _human_frr(y_true: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of true humans (1) predicted as bot (0)."""
    humans = y_true == 1
    n = int(humans.sum())
    if n == 0:
        return 0.0
    return float((pred[humans] == 0).sum() / n)


def _bot_recall(y_true: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of true bots (0) correctly predicted as bot."""
    bots = y_true == 0
    n = int(bots.sum())
    if n == 0:
        return 0.0
    return float((pred[bots] == 0).sum() / n)


def _feature_importance(model: Any, names: list[str]) -> dict[str, float]:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return {}
    return {n: float(v) for n, v in zip(names, imp)}


def _safe(fn):
    try:
        return float(fn())
    except Exception:
        return None
