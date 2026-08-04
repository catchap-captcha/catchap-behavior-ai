"""Does model diversity break a score-guided weakset?

The weakset is, by construction, the set of trajectories that *this* model scores
as human — 16,000 candidates were searched to find 100. So no threshold can
exclude it, and adversarial training on it already failed once: on 2026-07-22 the
hard-negative candidate pushed known-bot ASR from 8.17% to 1.58% and pushed
unseen-human FRR from 2.39% to 9.06%, which is worse than the disease.

What has never been tried is diversity. A weakset optimised against one decision
surface has no reason to sit inside a different one. If K models trained on the
same data but with different seeds and different feature subsets disagree about
these points, then requiring all of them to say "human" costs the attacker a
search per model rather than one search.

This measures that, and it measures the price: an ensemble that rejects the
weakset by also rejecting humans has bought nothing.

The 7-participant human holdout is deliberately NOT used here. It has been scored
once against the current candidate and must stay sealed for the final decision;
selecting an ensemble against it would consume it silently.

    .venv/bin/python tools/ensemble_vs_weakset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.feature_extractor_v23 import extract_features  # noqa: E402
from app.services.trajectory_feature_views import get_feature_view  # noqa: E402

LOCKBOX = Path("data/interim/revalidation_human_lockbox_h20066_v23corr_20260722")
BOTS = Path("data/interim/bot_features_v23corr_20260722.jsonl")
WEAKSET = Path("data/interim/hybrid_motion_score_guided_weakset_100_20260722.jsonl")
EXTERNAL = Path("data/interim/hybrid_motion_redteam_external_holdout_500_20260722.jsonl")

VIEWS = ("general_without_physics", "dynamics_physics")


def load_features(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_events_as_features(path: Path) -> list[dict]:
    """Red-team sets ship raw events; the training sets ship features."""
    out = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            events = rec.get("events") or []
            if len(events) >= 3:
                out.append(extract_features(events, None))
    return out


def vectors(rows: list[dict], names: tuple[str, ...]) -> np.ndarray:
    return np.array([[float(r.get(n) or 0.0) for n in names] for r in rows])


def train_member(X: np.ndarray, y: np.ndarray, seed: int, feature_fraction: float,
                 family: str = "lgbm"):
    """Seed changes move the boundary a little; a different algorithm moves it a lot.

    Boosted trees, bagged trees and randomized-split trees carve the space in
    genuinely different ways, so a point tuned to sit inside one has no reason to
    sit inside the others.
    """
    if family == "rf":
        model = RandomForestClassifier(
            n_estimators=300, max_features=feature_fraction, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1)
    elif family == "et":
        model = ExtraTreesClassifier(
            n_estimators=300, max_features=feature_fraction, min_samples_leaf=2,
            class_weight="balanced", random_state=seed, n_jobs=-1)
    else:
        model = LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            subsample=0.8, subsample_freq=1, colsample_bytree=feature_fraction,
            class_weight="balanced", random_state=seed, verbose=-1)
    model.fit(X, y)
    return model


def human_prob(model, X: np.ndarray) -> np.ndarray:
    idx = list(model.classes_).index(1)
    return model.predict_proba(X)[:, idx]


def main() -> None:
    humans = load_features(LOCKBOX / "human_development.jsonl")
    bots = load_features(BOTS)
    weak = load_events_as_features(WEAKSET)
    ext = load_events_as_features(EXTERNAL)
    print(f"학습 사람 {len(humans)} · 봇 {len(bots)} | 약점셋 {len(weak)} · 외부 holdout {len(ext)}\n")

    # Hold out a slice of development humans to set each ensemble's threshold, so
    # the operating point is never chosen on the sets we then report.
    rng = np.random.default_rng(20260804)
    idx = rng.permutation(len(humans))
    cal = [humans[i] for i in idx[: len(humans) // 5]]
    fit = [humans[i] for i in idx[len(humans) // 5:]]

    def build(families: tuple[str, ...], fraction: float):
        members = {}
        for view in VIEWS:
            names = get_feature_view(view)
            X = np.vstack([vectors(fit, names), vectors(bots, names)])
            y = np.array([1] * len(fit) + [0] * len(bots))
            members[view] = [train_member(X, y, 20260804 + i, fraction, fam)
                             for i, fam in enumerate(families)]
        return members

    def score(members, rows: list[dict]) -> np.ndarray:
        """min over every member of every view — any model may raise risk."""
        if not rows:
            return np.array([])
        per = []
        for view, models in members.items():
            names = get_feature_view(view)
            X = vectors(rows, names)
            per.extend(human_prob(m, X) for m in models)
        return np.min(np.vstack(per), axis=0)

    print(f"  {'구성':22s}{'임계값':>12s}{'사람 FRR':>10s}{'약점셋 ASR':>12s}{'외부 ASR':>10s}")
    configs = (
        ("단일 lgbm (기준선)", ("lgbm",), 0.9),
        ("lgbm 3중 (시드만)", ("lgbm", "lgbm", "lgbm"), 0.6),
        ("lgbm+rf+et", ("lgbm", "rf", "et"), 0.6),
        ("lgbm+rf+et 2배", ("lgbm", "rf", "et", "lgbm", "rf", "et"), 0.5),
    )
    for label, families, fraction in configs:
        members = build(families, fraction)
        cal_scores = np.sort(score(members, cal))
        # Threshold at the calibration humans' 1st percentile: FRR 1% by construction.
        threshold = float(cal_scores[max(0, int(len(cal_scores) * 0.01) - 1)])
        frr = float((score(members, cal) < threshold).mean() * 100)
        weak_asr = float((score(members, weak) >= threshold).mean() * 100)
        ext_asr = float((score(members, ext) >= threshold).mean() * 100)
        print(f"  {label:22s}{threshold:>12.6f}{frr:>9.2f}%{weak_asr:>11.1f}%{ext_asr:>9.1f}%")


if __name__ == "__main__":
    main()
