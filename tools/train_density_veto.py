"""Add a density veto to the existing two-view detector, and record how.

What this fixes
---------------
Every candidate so far fails the same gate: unseen bot families. Leave-one-family-out
on 08-06 put the deployed model at 25.2% on `ml_pca_gmm`, and the reason turned out
not to be missing features.

Measured in the parallel rebuild (`/Users/apple/Documents/new`): a family of perfectly
uniform, straight-line traces separates from humans on a *single* feature with AUC
1.000 — and a gradient-boosted model trained without that family still passes 45.5%
of it. Trees do not extrapolate. A region of feature space with no training points
gets whatever leaf it happens to fall into, and the "perfectly uniform" corner landed
on the human side. Adding features cannot fix an empty corner; only having an opinion
about emptiness can.

A density model has the opposite failure: weak wherever humans are varied, but it
never mistakes an empty corner for a person. Combining them took that family from
45.5% to 0.1% with no change in false rejections.

The veto is calibrated so it cannot cost FRR that the detector was not already going
to cost: it fires only when a trace sits further from the human cloud than any human
the density model was fitted on.

One thing that had to be got right: the density model is fitted on collection humans
only. Legacy traces carry ~132 events against the main captcha's ~13, and pooling them
collapses the human region onto the legacy cluster — 66/131 real main-captcha drags
scored 0 when legacy was included, even after dropping length-dependent features. The
discriminative half still trains on everything; only the density half is surface-bound.

    .venv/bin/python tools/train_density_veto.py \
        --base models/candidate/revalidation_two_view_participant_safe_20260722 \
        --version density_veto_20260808
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags  # noqa: E402
from app.services.feature_extractor_v23 import extract_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLLECTION = ROOT / "data" / "interim" / "collection_20260806.jsonl"
SPLIT = ROOT / "data" / "metadata" / "collection_split_20260806.json"

# Absolute magnitudes an attacker changes for free by resizing the window, and
# lengths that differ between collection and legacy. The density model must not
# rest on either.
EXCLUDED = ("event_count", "duration_ms", "total_distance", "displacement",
            "y_deviation", "interval_mean_ms", "interval_std_ms")


class DensityVeto:
    """kNN distance plus isolation forest, fitted on humans only.

    Scores decay smoothly and are never clipped. Clipping to zero past a reference
    distance was silently fatal in the rebuild: real humans landed beyond it, all
    became exactly 0.0, and a veto calibrated to "further than any human" could
    never fire. Ordering has to survive past the reference — that is the region the
    veto exists to judge.
    """

    def __init__(self, X: np.ndarray, names: tuple[str, ...]) -> None:
        self.names = names
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.knn = NearestNeighbors(n_neighbors=min(15, len(Z))).fit(Z)
        distances, _ = self.knn.kneighbors(Z)
        self.reference = float(np.percentile(distances[:, 1:].mean(axis=1), 95)) or 1.0
        self.forest = IsolationForest(n_estimators=300, random_state=7).fit(Z)
        raw = self.forest.score_samples(Z)
        self.lo = float(np.percentile(raw, 5))
        self.hi = float(np.percentile(raw, 95))

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = self.scaler.transform(np.nan_to_num(X))
        distances, _ = self.knn.kneighbors(Z)
        near = np.exp(-distances[:, 1:].mean(axis=1) / self.reference)
        raw = self.forest.score_samples(Z)
        span = max(self.hi - self.lo, 1e-9)
        iso = 1.0 / (1.0 + np.exp(-(raw - self.lo) / span * 4.0 + 2.0))
        return np.minimum(near, iso)


def collection_drags(people: set[str]) -> list[list[dict]]:
    out = []
    with COLLECTION.open() as f:
        for line in f:
            record = json.loads(line)
            code = record.get("participant_id") or ""
            if code.split("-")[0] not in people:
                continue
            if record.get("quality_status") != "valid":
                continue
            rows = record.get("events") or []
            if not rows:
                continue
            base = rows[0].get("client_timestamp_ms") or 0
            events = [{
                "seq": r.get("seq"), "event_type": r.get("event_type"),
                "t_ms": float((r.get("client_timestamp_ms") or base) - base),
                "x": float(r["x_pixel"]), "y": float(r["y_pixel"]),
            } for r in rows if r.get("x_pixel") is not None]
            out.extend(d for d in split_drags(events)
                       if move_count(d) >= MIN_MOVES_PER_DRAG)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--version", required=True)
    args = ap.parse_args()

    bundle = joblib.load(args.base / "two_view_fusion.joblib")
    split = json.loads(SPLIT.read_text())
    sealed = set(split["holdout_people"])
    train_people = set(split["training_people"])
    print(f"봉인 {sorted(sealed)} 제외 · 밀도 학습 {sorted(train_people)}")

    drags = collection_drags(train_people)
    if not drags:
        raise SystemExit("밀도 모델을 학습할 드래그가 없다")

    # Union of the two views is the detector's own feature space; drop the
    # magnitude-bound names so a window resize cannot move the veto.
    view_names: list[str] = []
    for names in bundle["feature_views"].values():
        for n in names:
            if n not in view_names:
                view_names.append(n)
    names = tuple(n for n in view_names if n not in EXCLUDED)
    print(f"밀도 특징 {len(names)}개 (원 {len(view_names)}개에서 크기 의존 "
          f"{len(view_names)-len(names)}개 제외)")

    X = np.nan_to_num(np.asarray(
        [[extract_features(d, None).get(n) or 0.0 for n in names] for d in drags],
        dtype=float))
    density = DensityVeto(X, names)
    floor = float(np.min(density.score(X)))
    print(f"드래그 {len(X)}개 · 거부권 문턱 {floor:.6f} "
          "(밀도 모델이 본 가장 먼 사람)")

    out = dict(bundle)
    out["model_version"] = args.version
    out["density_veto"] = density
    out["density_feature_names"] = names
    out["veto_below"] = floor
    out["veto_note"] = (
        "사람 영역 밖일 때만 거부한다. 문턱은 밀도 모델이 학습에서 본 가장 먼 사람이라 "
        "새 오탐을 만들지 않는다. 밀도 모델은 수집분 사람만으로 학습 — 레거시를 섞으면 "
        "사람 영역이 옛 화면 모양으로 잡혀 메인 캡차 사람 66/131 이 0점이 된다."
    )
    out["density_fit"] = {
        "drags": len(X), "people": sorted(train_people),
        "excluded_features": list(EXCLUDED),
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": str(args.base),
    }

    dest = ROOT / "models" / "candidate" / args.version
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, dest / "two_view_fusion.joblib")

    report = ROOT / "reports" / args.version
    report.mkdir(parents=True, exist_ok=True)
    (report / "fit_density_veto.json").write_text(json.dumps(
        {k: v for k, v in out.items()
         if k not in ("models", "density_veto")},
        ensure_ascii=False, indent=1, default=str) + "\n")
    print(f"  -> {dest/'two_view_fusion.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
