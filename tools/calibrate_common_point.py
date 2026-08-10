"""Put every candidate on the same operating point before comparing them.

The mistake this exists to stop
-------------------------------
Twice on 2026-08-06 a candidate was declared best or hopeless because the two
axes were read at two different thresholds. `redteam_evolution_search` uses the
threshold stored in the bundle; `frr_candidate_compare` refits its own. So:

    deployed    threshold 0.99995   evasion 31.1%   FRR 7.3%
    scale_aug   threshold 0.00148   evasion 84.0%   FRR 2.4%

Neither row says which model is better. They are points on different curves.
Raising a threshold moves both numbers together, so a comparison is only a
comparison when every model is held to the same promise.

The promise used here is the one the project already committed to on 07-30:
pooled human FRR <= 2% on the main-captcha collection, scored per drag with the
session median — the same rule the service applies. Sealed people (sw, ms) are
refused, so the operating point is never fitted on the evaluation set.

Writes a copy of each bundle with the recalibrated threshold, leaving the
original untouched, so the red-team tool can then be pointed at the copy.

    .venv/bin/python tools/calibrate_common_point.py --target 0.02
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags  # noqa: E402
from app.services.feature_extractor_v23 import extract_features  # noqa: E402
from tools.collection_split import person_of  # noqa: E402

COLLECTION = Path("data/interim/collection_20260806.jsonl")
SPLIT = Path("data/metadata/collection_split_20260806.json")
BOT_MARKERS = ("pwbot", "rtbot", "botprobe", "probe", "signalcheck", "zzprobe")

CANDIDATES = [
    "revalidation_two_view_participant_safe_20260722",
    "scale_pixel_20260804",
    "drag_unit_frr5_20260806",
    "scale_aug_20260806",
]


def human_sessions(only: set[str] | None = None) -> list[list[dict]]:
    sealed = set(json.loads(SPLIT.read_text())["holdout_people"])
    out = []
    with COLLECTION.open() as f:
        for line in f:
            record = json.loads(line)
            code = record.get("participant_id") or ""
            if not code or any(m in code.lower() for m in BOT_MARKERS):
                continue
            person = person_of(code)
            if person in sealed:
                continue                      # never fit on the evaluation set
            if only and person not in only:
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
            drags = [d for d in split_drags(events) if move_count(d) >= MIN_MOVES_PER_DRAG]
            if drags:
                out.append(drags)
    return out


def session_score(bundle: dict, drags: list[list[dict]]) -> float:
    models, views = bundle["models"], bundle["feature_views"]
    scores = []
    for drag in drags:
        feats = extract_features(drag, None)
        per = []
        for view, names in views.items():
            row = np.array([[float(feats.get(n) or 0.0) for n in names]])
            per.append(float(models[view].predict_proba(np.nan_to_num(row))[0][1]))
        scores.append(min(per))
    return float(np.median(scores)) if scores else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=0.02)
    ap.add_argument("--suffix", default="_cal")
    ap.add_argument("--only-person", nargs="*", default=[])
    ap.add_argument("--candidates", nargs="*", default=None)
    args = ap.parse_args()

    sessions = human_sessions(set(args.only_person))
    print(f"보정용 사람 세션 {len(sessions)} (봉인 제외) · 목표 오탐 {args.target:.0%}\n")

    print(f"  {'모델':46s}{'원래 임계':>14s}{'보정 임계':>14s}{'실제 오탐':>10s}")
    for name in (args.candidates or CANDIDATES):
        path = Path("models/candidate") / name / "two_view_fusion.joblib"
        if not path.exists():
            print(f"  {name:46s}  없음")
            continue
        bundle = joblib.load(path)
        scores = np.array([session_score(bundle, d) for d in sessions])
        ordered = np.sort(scores)
        # Largest threshold that still rejects no more than `target` of humans.
        # At threshold ordered[k] exactly k humans fall below it, so the budget
        # allows k = floor(target * n). Taking ordered[k-1] instead — as this did
        # until 08-06 — spends one fewer rejection than the budget permits, which
        # on 58 sessions means the threshold sits at the very lowest human score
        # and every bot above it counts as a pass. That inflated family ASR.
        index = min(int(len(ordered) * args.target), len(ordered) - 1)
        point = float(ordered[index])
        frr = float((scores < point).mean())

        out = dict(bundle)
        out["threshold"] = point
        out["model_version"] = f"{bundle.get('model_version', name)}{args.suffix}"
        out["threshold_calibration"] = {
            "policy": "pooled human FRR on main-captcha collection, per-drag median",
            "target_frr": args.target,
            "sessions": len(sessions),
            "sealed_excluded": sorted(json.loads(SPLIT.read_text())["holdout_people"]),
            "note": "common operating point so evasion and FRR are read off one curve",
        }
        dest = Path("models/candidate") / f"{name}{args.suffix}"
        dest.mkdir(parents=True, exist_ok=True)
        joblib.dump(out, dest / "two_view_fusion.joblib")
        print(f"  {name:46s}{float(bundle['threshold']):>14.6f}{point:>14.6f}{frr:>9.1%}")

    print(f"\n  보정본 -> models/candidate/*{args.suffix}/two_view_fusion.joblib")
    print("  이제 redteam_evolution_search 를 보정본에 걸면 두 축이 같은 점이 된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
