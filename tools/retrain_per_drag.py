"""Train on drags, score drags, and hold out a whole person to see if it travels.

The problem this attacks
------------------------
The current model was trained on EDU-captcha point streams — trajectories with no
press/release boundaries at all — and is scored on main-captcha drags. It shows:
a handful of *human* sessions land near 0, and they belong to one person. Keeping
that person's FRR under 2% forces the threshold down to 1.6e-5, at which 13.4% of
bots also pass. Calibrate on the other person instead and bots stop passing, but
4.5% of the first person's sessions are rejected. No single threshold serves both.

That is a training-distribution problem, not a threshold problem, so this trains
on the surface we actually score: one example per drag, from the main captcha.

Leave-one-person-out, not leave-one-code-out: `sw-mouse` and `sw-mouse-v2` are the
same human on two days, and splitting on the code would put that person on both
sides and quietly manufacture the generalisation we are trying to measure.

    .venv/bin/python tools/retrain_per_drag.py data/interim/main_captcha_raw_20260803b.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags  # noqa: E402
from app.services.feature_extractor_v23 import extract_features  # noqa: E402
from app.services.trajectory_feature_views import get_feature_view  # noqa: E402
from tools.frr_operating_point import SELF_COLLECTED_HUMAN, person_of  # noqa: E402
from tools.per_drag_scoring import MODEL_PATH, Scorer, to_extractor_events  # noqa: E402

VIEWS = ("general_without_physics", "dynamics_physics")
TARGET_FRR = 0.02


def build(path: str) -> list[dict]:
    """One row per session, carrying its drags' feature vectors."""
    scorer = Scorer(MODEL_PATH)
    out = []
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        prod = rec.get("prod_human_probability")
        if prod is None:
            continue
        events = to_extractor_events(rec["events"])
        session = scorer.score(events)
        if session is None or abs(float(prod) - session) >= 0.01:
            continue                                  # not reproducible -> not evidence
        participant = rec.get("participant_id") or ""
        label = rec.get("label")
        if label is None and participant in SELF_COLLECTED_HUMAN:
            label = "human"
        if label not in ("human", "bot"):
            continue
        drags = [d for d in split_drags(events) if move_count(d) >= MIN_MOVES_PER_DRAG]
        if not drags:
            continue
        out.append({
            "person": person_of(participant) if label == "human" else f"bot:{participant[:12]}",
            "label": label,
            "baseline": statistics.median(
                [s for s in (scorer.score(d) for d in drags) if s is not None] or [0.0]),
            "drags": [extract_features(d, None) for d in drags],
        })
    return out


def matrix(rows: list[dict], view: str) -> tuple[np.ndarray, np.ndarray]:
    names = get_feature_view(view)
    X, y = [], []
    for row in rows:
        for feats in row["drags"]:
            X.append([float(feats.get(n) or 0.0) for n in names])
            y.append(1 if row["label"] == "human" else 0)
    return np.array(X), np.array(y)


def fused_session_scores(models: dict, rows: list[dict]) -> list[float]:
    """Median over drags of min(view scores) — the same rule the service uses."""
    out = []
    for row in rows:
        per_drag = []
        for feats in row["drags"]:
            views = []
            for view, model in models.items():
                names = get_feature_view(view)
                vec = np.array([[float(feats.get(n) or 0.0) for n in names]])
                views.append(float(model.predict_proba(vec)[0][list(model.classes_).index(1)]))
            per_drag.append(min(views))
        out.append(statistics.median(per_drag))
    return out


def rate(scores: list[float], threshold: float, above: bool) -> float:
    if not scores:
        return 0.0
    hit = sum(1 for s in scores if (s >= threshold) == above)
    return hit / len(scores)


def fit_threshold(human: list[float]) -> float:
    best = 0.0
    for candidate in sorted(set(human)):
        if rate(human, candidate, above=False) <= TARGET_FRR:
            best = candidate
    return best


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "data/interim/main_captcha_raw_20260803b.jsonl"
    rows = build(src)
    people = sorted({r["person"] for r in rows if r["label"] == "human"})
    bots = [r for r in rows if r["label"] == "bot"]
    drags = sum(len(r["drags"]) for r in rows)
    print(f"세션 {len(rows)}건 · 드래그 {drags}개 · 사람 {len(people)}명 {people} · 봇 {len(bots)}건\n")

    print(f"  {'빼놓은 사람':10s}{'':2s}{'모델':10s}{'임계값':>12s}"
          f"{'처음 보는 사람 FRR':>18s}{'봇 ASR':>9s}")
    for held in people:
        train = [r for r in rows if r["label"] == "bot" or r["person"] != held]
        test_h = [r for r in rows if r["label"] == "human" and r["person"] == held]
        train_h = [r for r in train if r["label"] == "human"]
        if not train_h or not test_h:
            continue

        models = {}
        for view in VIEWS:
            X, y = matrix(train, view)
            m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                               min_child_samples=10, class_weight="balanced",
                               random_state=20260803, verbose=-1)
            m.fit(X, y)
            models[view] = m

        # Threshold from the training people only — never from the held-out person.
        th = fit_threshold(fused_session_scores(models, train_h))
        frr = rate(fused_session_scores(models, test_h), th, above=False)
        asr = rate(fused_session_scores(models, bots), th, above=True)
        print(f"  {held:10s}{'':2s}{'재학습':10s}{th:>12.6f}{frr*100:>17.1f}%{asr*100:>8.1f}%")

        # Same split, current model, so the comparison is like for like.
        th0 = fit_threshold([r["baseline"] for r in train_h])
        frr0 = rate([r["baseline"] for r in test_h], th0, above=False)
        asr0 = rate([r["baseline"] for r in bots], th0, above=True)
        print(f"  {'':10s}{'':2s}{'현재':10s}{th0:>12.6f}{frr0*100:>17.1f}%{asr0*100:>8.1f}%")


if __name__ == "__main__":
    main()
