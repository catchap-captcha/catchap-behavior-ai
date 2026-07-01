"""Threshold selection (on validation) and final evaluation (on test).

Positive class = Human (1). The decision threshold is chosen on the VALIDATION
split so that the Human False Rejection Rate stays within budget while keeping
Bot recall high; the TEST split is then scored exactly once with that threshold.

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


def _positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else proba.shape[1] - 1
    return proba[:, idx]


def select_threshold(
    model: Any, X_val: pd.DataFrame, y_val: pd.Series, max_frr: float = DEFAULT_MAX_FRR
) -> float:
    """Pick the threshold on validation.

    Among thresholds that keep Human FRR <= ``max_frr``, choose the one with the
    highest Bot recall. If none satisfy the FRR budget, fall back to the
    threshold with the lowest FRR (safest for real users).
    """
    scores = _positive_proba(model, X_val)
    y = y_val.to_numpy()
    candidates = np.unique(np.concatenate([scores, [0.5]]))

    best_t, best_bot_recall = None, -1.0
    fallback_t, fallback_frr = 0.5, 1.0
    for t in candidates:
        pred = (scores >= t).astype(int)
        frr = _human_frr(y, pred)
        bot_recall = _bot_recall(y, pred)
        if frr < fallback_frr:
            fallback_frr, fallback_t = frr, float(t)
        if frr <= max_frr and bot_recall > best_bot_recall:
            best_bot_recall, best_t = bot_recall, float(t)
    return best_t if best_t is not None else fallback_t


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
    scores = _positive_proba(model, X)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    avg_ms = elapsed_ms / max(len(X), 1)

    y_true = y.to_numpy()
    pred = (scores >= threshold).astype(int)

    # human = positive class (1)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, pred, labels=[1], average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, pred, labels=[0, 1]).tolist()

    roc = _safe(lambda: roc_auc_score(y_true, scores))
    pr = _safe(lambda: average_precision_score(y_true, scores))

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
        avg_inference_ms=float(avg_ms),
        feature_importance=_feature_importance(model, X.columns.tolist()),
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
