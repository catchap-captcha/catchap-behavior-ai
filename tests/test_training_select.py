"""Tests for model selection rules and the end-to-end train/evaluate path."""

from __future__ import annotations

import pytest

from training.evaluate_models import Evaluation
from training.select_best_model import select_best


def _eval(name, frr, bot_recall, f1, ms):
    return Evaluation(
        model_name=name, threshold=0.5, accuracy=0.9,
        human_precision=0.9, human_recall=1 - frr, human_f1=f1,
        bot_recall=bot_recall, human_frr=frr, roc_auc=0.9, pr_auc=0.9,
        confusion_matrix=[[1, 0], [0, 1]], avg_inference_ms=ms,
        feature_importance={}, metrics_on="test",
    )


def test_only_models_within_frr_budget_are_candidates():
    evals = [
        _eval("a", frr=0.05, bot_recall=0.99, f1=0.9, ms=1.0),  # too high FRR
        _eval("b", frr=0.02, bot_recall=0.80, f1=0.9, ms=1.0),
    ]
    sel = select_best(evals, max_frr=0.03)
    assert sel.selected is not None
    assert sel.selected.model_name == "b"


def test_highest_bot_recall_wins_among_candidates():
    evals = [
        _eval("a", frr=0.01, bot_recall=0.90, f1=0.9, ms=1.0),
        _eval("b", frr=0.02, bot_recall=0.95, f1=0.8, ms=1.0),
    ]
    sel = select_best(evals, max_frr=0.03)
    assert sel.selected.model_name == "b"


def test_f1_breaks_bot_recall_tie():
    evals = [
        _eval("a", frr=0.01, bot_recall=0.95, f1=0.85, ms=1.0),
        _eval("b", frr=0.02, bot_recall=0.95, f1=0.92, ms=1.0),
    ]
    sel = select_best(evals, max_frr=0.03)
    assert sel.selected.model_name == "b"


def test_speed_breaks_full_tie():
    evals = [
        _eval("a", frr=0.01, bot_recall=0.95, f1=0.9, ms=5.0),
        _eval("b", frr=0.02, bot_recall=0.95, f1=0.9, ms=1.0),
    ]
    sel = select_best(evals, max_frr=0.03)
    assert sel.selected.model_name == "b"


def test_no_candidate_means_no_promotion_and_warning():
    evals = [
        _eval("a", frr=0.10, bot_recall=0.99, f1=0.9, ms=1.0),
        _eval("b", frr=0.20, bot_recall=0.99, f1=0.9, ms=1.0),
    ]
    sel = select_best(evals, max_frr=0.03)
    assert sel.selected is None
    assert sel.reason == "no_model_meets_frr_budget"
    assert sel.warning is not None


def test_full_train_evaluate_select_smoke(training_rows):
    """Exercise train -> threshold -> evaluate -> select on fixtures.

    Skips if xgboost / lightgbm are not installed. Uses fixtures only; nothing
    is written to models/production and the result is NOT a real performance
    figure.
    """
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")

    from training.build_dataset import build_dataset
    from training.evaluate_models import evaluate, select_threshold
    from training.split_dataset import split_dataset
    from training.train_models import train_all

    ds = build_dataset(training_rows)
    split = split_dataset(ds, seed=42)
    models = train_all(split, seed=42)
    assert set(models) == {
        "random_forest",
        "extra_trees",
        "xgboost",
        "lightgbm",
    }

    evals = []
    for name, model in models.items():
        t = select_threshold(model, split.X_val, split.y_val)
        evals.append(evaluate(model, name, split.X_test, split.y_test, t, "test"))
    # all metrics finite and in range
    for e in evals:
        assert 0.0 <= e.human_frr <= 1.0
        assert 0.0 <= e.bot_recall <= 1.0
    # selection returns a decision object (selected or a warning)
    sel = select_best(evals)
    assert (sel.selected is not None) or (sel.warning is not None)
