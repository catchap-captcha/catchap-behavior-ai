"""Does patching the weaknesses we found also close the ones we did not?

The obvious next move is to red-team harder: search for more weak points, train on
them, repeat. That only pays if patching a *found* weakness closes *unfound* ones
too. If it does not, each round buys the exact points it was given and the search
finds fresh ones next time — whack-a-mole with a human-FRR bill each round
(2026-07-22: hard-negative training took unseen-human FRR from 2.39% to 9.06%).

So this splits the weakset in half by generator seed, trains on half A only, and
asks what happened to half B. Half B was found by the same search against the same
model, so it is the friendliest possible test of transfer: if patching does not
generalise even here, it will not generalise to a search we have not run.

Reported at a fixed human FRR, because a patch that closes weaknesses by
rejecting people has closed nothing.

    .venv/bin/python tools/redteam_patch_transfer.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.feature_extractor_v23 import extract_features  # noqa: E402
from app.services.trajectory_feature_views import get_feature_view  # noqa: E402

LOCKBOX = Path("data/interim/revalidation_human_lockbox_h20066_v23corr_20260722")
HOLDOUT = Path("data/interim/human_holdout_prevtest_h2219_20260804/human_holdout.jsonl")
BOTS = Path("data/interim/bot_features_v23corr_20260722.jsonl")
WEAKSET = Path("data/interim/redteam_weakness_search_iter1_20260722/"
               "redteam_weakset_iter1_20260722.jsonl")
EXTERNAL = Path("data/interim/hybrid_motion_redteam_external_holdout_500_20260722.jsonl")
# A weakset built by a DIFFERENT search (hybrid_motion, score-guided) against the
# same model. Transfer to the unseen half of one search only shows that search
# produced a homogeneous set; transfer to a different search is the real question.
OTHER_WEAKSET = Path("data/interim/hybrid_motion_score_guided_weakset_100_20260722.jsonl")

VIEWS = ("general_without_physics", "dynamics_physics")
TARGET_FRR = 0.01


def features_from_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def features_from_events(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            events = json.loads(line).get("events") or []
            if len(events) >= 3:
                out.append(extract_features(events, None))
    return out


def vec(rows: list[dict], names) -> np.ndarray:
    X = np.array([[float(r.get(n) or 0.0) for n in names] for r in rows], dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def fit(humans, bots, weak_patch, names, weight):
    X = np.vstack([vec(humans, names), vec(bots, names)]
                  + ([vec(weak_patch, names)] if weak_patch else []))
    y = np.array([1] * len(humans) + [0] * len(bots) + [0] * len(weak_patch))
    w = np.array([1.0] * len(humans) + [1.0] * len(bots) + [weight] * len(weak_patch))
    m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                       class_weight="balanced", random_state=20260804, verbose=-1)
    m.fit(X, y, sample_weight=w)
    return m


def main() -> None:
    humans = features_from_jsonl(LOCKBOX / "human_development.jsonl")
    holdout = features_from_jsonl(HOLDOUT)
    bots = features_from_jsonl(BOTS)
    weak = features_from_events(WEAKSET)
    ext = features_from_events(EXTERNAL)
    other = features_from_events(OTHER_WEAKSET)

    rng = np.random.default_rng(20260804)
    order = rng.permutation(len(weak))
    half = len(order) // 2
    weak_a = [weak[i] for i in order[:half]]      # patched
    weak_b = [weak[i] for i in order[half:]]      # never shown to the model
    print(f"사람 {len(humans)} · 봇 {len(bots)} · 약점셋 {len(weak)} "
          f"(패치 {len(weak_a)} / 미공개 {len(weak_b)}) · 외부 {len(ext)}\n")

    cal_idx = rng.permutation(len(humans))[: len(humans) // 5]
    cal = [humans[i] for i in cal_idx]
    fit_h = [humans[i] for i in range(len(humans)) if i not in set(cal_idx)]

    print(f"  {'구성':26s}{'사람 FRR':>9s}{'패치':>8s}{'미공개':>9s}"
          f"{'다른 탐색':>11s}{'외부':>8s}{'홀드아웃':>10s}")
    for label, patch, weight in (
        ("패치 없음 (기준선)", [], 0.0),
        ("절반 패치 · 가중 1", weak_a, 1.0),
        ("절반 패치 · 가중 3", weak_a, 3.0),
    ):
        models = {v: fit(fit_h, bots, patch, get_feature_view(v), weight) for v in VIEWS}

        def score(rows):
            if not rows:
                return np.array([])
            per = []
            for v, m in models.items():
                X = vec(rows, get_feature_view(v))
                per.append(m.predict_proba(X)[:, list(m.classes_).index(1)])
            return np.min(np.vstack(per), axis=0)

        cal_scores = np.sort(score(cal))
        th = float(cal_scores[max(0, int(len(cal_scores) * TARGET_FRR) - 1)])
        pct = lambda arr, above=True: (float((arr >= th).mean() * 100) if above
                                       else float((arr < th).mean() * 100))
        print(f"  {label:26s}{pct(score(cal), False):>8.2f}%"
              f"{pct(score(weak_a)):>7.1f}%{pct(score(weak_b)):>8.1f}%"
              f"{pct(score(other)):>10.1f}%{pct(score(ext)):>7.1f}%"
              f"{pct(score(holdout), False):>9.2f}%")


if __name__ == "__main__":
    main()
