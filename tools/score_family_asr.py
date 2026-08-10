"""Score one bot family's ASR against a model that never saw it.

This is the written criterion, not a stress test
------------------------------------------------
`SECURITY_ACCEPTANCE_CRITERIA.md:29` says:

    P1 | 미지 Bot 최악 ASR | 10% 이하 | leave-one-family-out

and §3.1 adds that a seed split inside one generator does not count. So the
measurement is: hold a whole family out of training, then score that family.

For most of 2026-08-06 the gate was checked with `redteam_evolution_search`
instead. That tool optimises motion policies directly against the model under
test — it is an upper bound on a determined adversary, and a useful one, but it
is a bar nobody set. Judging a candidate by it and refusing promotion is moving
the goalposts just as surely as lowering them would be.

Both numbers belong in the report. This one decides the gate; the evolutionary
one goes under 잔존 위험 (§4.7).

Scored per drag with the session median, at the operating point the bundle
carries — so it must be a bundle calibrated by `calibrate_common_point.py`,
otherwise the ASR is read off a threshold that no human budget justifies.

    .venv/bin/python tools/score_family_asr.py \
        --model models/candidate/lofo_vae_cal/two_view_fusion.joblib \
        --family data/interim/vae_bots_development_1000_20260721.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags  # noqa: E402
from app.services.feature_extractor_v23 import extract_features  # noqa: E402


def session_scores(bundle: dict, path: Path, limit: int) -> list[float]:
    models, views = bundle["models"], bundle["feature_views"]
    out = []
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            drags = [d for d in split_drags(record.get("events") or [])
                     if move_count(d) >= MIN_MOVES_PER_DRAG]
            if not drags:
                continue
            per = []
            for drag in drags:
                feats = extract_features(drag, None)
                views_scores = []
                for view, names in views.items():
                    row = np.array([[float(feats.get(n) or 0.0) for n in names]])
                    views_scores.append(
                        float(models[view].predict_proba(np.nan_to_num(row))[0][1]))
                per.append(min(views_scores))
            out.append(statistics.median(per))
            if len(out) >= limit:
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--family", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    threshold = float(bundle["threshold"])
    scores = np.array(session_scores(bundle, args.family, args.limit))
    if scores.size == 0:
        print("드래그가 나오는 행이 없다")
        return 1

    asr = float((scores >= threshold).mean())
    verdict = "통과" if asr <= 0.10 else "미달"
    print(f"  {args.family.name:56s} {scores.size:5d}건  ASR {asr*100:6.2f}%  "
          f"(임계 {threshold:.6f})  기준 ≤10% {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
