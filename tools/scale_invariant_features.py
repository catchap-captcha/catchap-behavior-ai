"""Does dropping the pixel scale cost anything?

17 of the model's features are computed in pixels — `avg_speed`, `max_speed`,
`displacement`, `jerk_mean` and friends. Pixels come from
`x_normalized * stage_width`, so the same hand movement produces different
numbers on a 500px stage and a 375px one. Our data already contains both
(500: 422 sessions, 640: 43, 375: 34, 333: 28, 1024: 12), and the captcha is
about to be embedded in a container whose width nobody has decided yet.

The cheap fix is to ask for a fixed width. The durable fix is to compute the
features from the normalized coordinates the captcha already sends, so the width
stops mattering — including when the widget turns out to be responsive and every
user has a different one.

That only works if the absolute scale was not carrying real signal. This measures
that: same rows, same model family, same human FRR, features computed both ways.

    .venv/bin/python tools/scale_invariant_features.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.feature_extractor_v23 import extract_features  # noqa: E402
from app.services.trajectory_feature_views import get_feature_view  # noqa: E402

HUMANS = Path("data/raw/human_db_snapshot_20260721T012239Z/human_attempts.jsonl")
BOTS = Path("data/interim/extended_bots_10000_20260721.jsonl")
WEAKSET = Path("data/interim/redteam_weakness_search_iter1_20260722/"
               "redteam_weakset_iter1_20260722.jsonl")
EXTERNAL = Path("data/interim/hybrid_motion_redteam_external_holdout_500_20260722.jsonl")

VIEWS = ("general_without_physics", "dynamics_physics")
TARGET_FRR = 0.02


def load(path: Path, normalized: bool, limit: int | None = None) -> list[dict]:
    """Featurize raw events either in pixels (as today) or in 0..1 coordinates.

    Normalized keeps time in ms, so speeds become fraction-of-stage per ms. The
    unit changes; what the feature means does not.
    """
    rows = []
    with path.open() as f:
        for line in f:
            rec = json.loads(line)
            events = rec.get("events") or []
            if len(events) < 3:
                continue
            if normalized:
                events = [dict(e, x=e.get("x_normalized"), y=e.get("y_normalized"))
                          for e in events
                          if e.get("x_normalized") is not None]
                if len(events) < 3:
                    continue
            rows.append(extract_features(events, None))
            if limit and len(rows) >= limit:
                break
    return rows


def vec(rows, names) -> np.ndarray:
    X = np.array([[float(r.get(n) or 0.0) for n in names] for r in rows], dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def build(humans, bots, patch, seed=20260804):
    models, views = {}, {}
    for view in VIEWS:
        names = list(get_feature_view(view))
        X = np.vstack([vec(humans, names), vec(bots, names)]
                      + ([vec(patch, names)] if patch else []))
        y = np.array([1] * len(humans) + [0] * len(bots) + [0] * len(patch))
        m = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                           class_weight="balanced", random_state=seed, verbose=-1)
        m.fit(X, y)
        models[view], views[view] = m, names
    return models, views


def fuse(models, views, rows) -> np.ndarray:
    if not rows:
        return np.array([])
    per = []
    for name, model in models.items():
        per.append(model.predict_proba(vec(rows, views[name]))[:, list(model.classes_).index(1)])
    return np.min(np.vstack(per), axis=0)


def main() -> None:
    rng = random.Random(20260804)
    for normalized in (False, True):
        label = "정규화 (크기 무관)" if normalized else "픽셀 (현재)"
        humans = load(HUMANS, normalized, limit=6000)
        bots = load(BOTS, normalized, limit=6000)
        weak = load(WEAKSET, normalized)
        ext = load(EXTERNAL, normalized)

        idx = list(range(len(humans)))
        rng.shuffle(idx)
        cut = int(len(idx) * 0.75)
        fit_h = [humans[i] for i in idx[:cut]]
        cal_h = [humans[i] for i in idx[cut:]]

        models, views = build(fit_h, bots, weak)
        cal = np.sort(fuse(models, views, cal_h))
        th = float(cal[max(0, int(len(cal) * TARGET_FRR) - 1)])

        frr = float((fuse(models, views, cal_h) < th).mean() * 100)
        known = float((fuse(models, views, bots) >= th).mean() * 100)
        weak_asr = float((fuse(models, views, weak) >= th).mean() * 100)
        ext_asr = float((fuse(models, views, ext) >= th).mean() * 100)

        if not normalized:
            print(f"  {'구성':22s}{'사람 FRR':>10s}{'알려진 봇':>11s}"
                  f"{'약점셋':>9s}{'외부':>8s}")
        print(f"  {label:22s}{frr:>9.2f}%{known:>10.2f}%{weak_asr:>8.2f}%{ext_asr:>7.2f}%")


if __name__ == "__main__":
    main()
