"""Red-team, patch, red-team again — and see whether it converges or plays whack-a-mole.

One round of patching closed everything we had: the unseen half of a weakset, a
weakset from a different search, the external holdout, and a fresh search found
zero evaders. But one round proves little. If each patch just moves the boundary
and the next search finds new gaps, the loop never ends and every round costs
human FRR. If the gaps run out, the criterion "미지 family 최악 ASR <= 10%" is
reachable head-on and does not need redefining.

So this runs the loop properly:

    search (fresh seed) -> weakset -> retrain including it -> search again

Each round searches with a *different* seed. Reusing one would re-generate the
same candidates the model was just taught, and finding nothing would mean
nothing.

Every model is recalibrated to the same human FRR as the baseline, so rounds are
compared at a fixed cost to real users rather than by quietly tightening.

    .venv/bin/python tools/redteam_defense_loop.py --rounds 3 --count 4000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.trajectory_feature_views import get_feature_view  # noqa: E402
from tools.redteam_patch_transfer import (BOTS, HOLDOUT, LOCKBOX, VIEWS,  # noqa: E402
                                          features_from_events, features_from_jsonl, vec)

BASE_MODEL = Path("models/candidate/revalidation_two_view_baseline_20260722/two_view_fusion.joblib")
HUMAN_ATTEMPTS = Path("data/raw/human_db_snapshot_20260721T012239Z/human_attempts.jsonl")
SOURCE_FEATURES = Path("data/interim/human_features_v23corr_20260722.jsonl")
WORK = Path("/tmp/defense_loop")


def fuse(models: dict, views: dict, rows: list[dict]) -> np.ndarray:
    per = []
    for name, model in models.items():
        X = vec(rows, views[name])
        per.append(model.predict_proba(X)[:, list(model.classes_).index(1)])
    return np.min(np.vstack(per), axis=0)


def train(humans, bots, patches, target_frr, base_bundle):
    """Fit on base data plus every weakset found so far, then calibrate.

    Weight 1, not 3. On 2026-07-22 a 3x weight on boundary bots pushed unseen-human
    FRR from 2.39% to 9.06%; the transfer test on 08-04 showed weight 1 keeps FRR
    flat and closes just as much.
    """
    models, views = {}, {}
    for view in VIEWS:
        names = list(get_feature_view(view))
        blocks = [vec(humans, names), vec(bots, names)] + [vec(p, names) for p in patches if p]
        labels = [1] * len(humans) + [0] * len(bots) + [0] * sum(len(p) for p in patches)
        model = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                               class_weight="balanced", random_state=20260804, verbose=-1)
        model.fit(np.vstack(blocks), np.array(labels))
        models[view], views[view] = model, names

    scores = np.sort(fuse(models, views, humans))
    threshold = float(scores[max(0, int(len(scores) * target_frr) - 1)])

    bundle = dict(base_bundle)
    bundle.update({"models": models, "feature_views": views, "threshold": threshold,
                   "step_up_threshold": None})
    return bundle


def search(model_path: Path, tag: str, count: int, seed: int) -> dict:
    out_dir, report = WORK / f"out_{tag}", WORK / f"{tag}.json"
    subprocess.run(
        [sys.executable, "-W", "ignore", "tools/redteam_weakness_search.py",
         "--model", str(model_path), "--human-attempts", str(HUMAN_ATTEMPTS),
         "--source-human-features", str(SOURCE_FEATURES), "--out-dir", str(out_dir),
         "--report", str(report), "--run-tag", tag, "--count", str(count), "--seed", str(seed)],
        check=True, capture_output=True, env={"PYTHONPATH": ".", "PATH": "/usr/bin:/bin"},
    )
    doc = json.loads(report.read_text())
    weak_path = Path(doc["outputs"]["weak_set_path"])
    doc["_weak_rows"] = features_from_events(weak_path) if weak_path.exists() else []
    return doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--count", type=int, default=4000)
    args = ap.parse_args()

    WORK.mkdir(exist_ok=True)
    base = joblib.load(BASE_MODEL)
    humans = features_from_jsonl(LOCKBOX / "human_development.jsonl")
    holdout = features_from_jsonl(HOLDOUT)
    bots = features_from_jsonl(BOTS)

    target_frr = float((fuse(base["models"], base["feature_views"], humans)
                        < base["threshold"]).mean())
    print(f"사람 {len(humans)} · 봇 {len(bots)} · 기준 사람 FRR {target_frr*100:.3f}% 고정\n")

    model_path, patches = BASE_MODEL, []
    print(f"  {'라운드':>6s}{'탐색 시드':>10s}{'회피':>7s}{'근접':>7s}"
          f"{'누적 패치':>10s}{'홀드아웃 FRR':>13s}")

    for r in range(1, args.rounds + 1):
        seed = 20260804 + r * 1000          # a fresh region each round
        doc = search(model_path, f"r{r}", args.count, seed)
        counts = doc["counts"]

        bundle = joblib.load(model_path)
        hold_frr = float((fuse(bundle["models"], bundle["feature_views"], holdout)
                          < bundle["threshold"]).mean() * 100)
        print(f"  {r:>6d}{seed:>10d}{counts['evaders']:>7d}{counts['near_miss']:>7d}"
              f"{sum(len(p) for p in patches):>10d}{hold_frr:>12.2f}%")

        if not doc["_weak_rows"]:
            print(f"         → 이 라운드에서 약점이 나오지 않았다")
            continue

        patches.append(doc["_weak_rows"])
        new_bundle = train(humans, bots, patches, target_frr, base)
        model_path = WORK / f"model_r{r}.joblib"
        joblib.dump(new_bundle, model_path)

    print("\n  마지막 모델로 홀드아웃 재확인 (참고값 — 이 홀드아웃은 이미 소진됐다)")
    final = joblib.load(model_path)
    print(f"    처음 보는 사람 FRR {float((fuse(final['models'], final['feature_views'], holdout) < final['threshold']).mean()*100):.2f}%")


if __name__ == "__main__":
    main()
