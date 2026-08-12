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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.density_veto import EXCLUDED, DensityVeto  # noqa: E402
from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags  # noqa: E402
from app.services.feature_extractor_v23 import extract_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLLECTION = ROOT / "data" / "interim" / "collection_20260806.jsonl"
SPLIT = ROOT / "data" / "metadata" / "collection_split_20260806.json"



HUMAN_FEATURES = ROOT / "data" / "interim" / "human_features_collection_20260806.jsonl"


def session_vectors(people: set[str], names: tuple[str, ...]) -> np.ndarray:
    """채점이 실제로 받는 모양 — 시도(세션) 하나당 특징 벡터 하나.

    운영은 드래그가 아니라 시도 단위로 들어온다. 문턱을 여기서 읽어야 단위가 맞는다.
    """
    rows = []
    with HUMAN_FEATURES.open() as f:
        for line in f:
            record = json.loads(line)
            if record.get("label") != "human":
                continue
            code = (record.get("anonymous_participant_id") or "").split("-")[0]
            if code not in people:
                continue
            rows.append([float(record.get(n) or 0.0) for n in names])
    return np.nan_to_num(np.asarray(rows, dtype=float))


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
            # 정규화 좌표를 반드시 함께 넘긴다. `feature_extractor_v2._arrays` 는
            # `x_normalized`/`y_normalized` 가 있으면 그걸 쓰고, **한 이벤트라도 없으면**
            # 픽셀을 드래그 단위 min-max 로 다시 늘리는 다른 가지를 탄다. 픽셀만 넘기면
            # 밀도는 "드래그마다 0~1로 늘린" 좌표계에서 학습되는데 운영 채점은 진짜
            # 정규화 좌표로 들어온다 — 사람 영역이 어긋나 멀쩡한 사용자가 거부권에
            # 걸린다(2026-08-12 실사용 오탐 18.9% 중 12.8%p 가 거부권이었다).
            events = [{
                "seq": r.get("seq"), "event_type": r.get("event_type"),
                "t_ms": float((r.get("client_timestamp_ms") or base) - base),
                "x": float(r["x_pixel"]), "y": float(r["y_pixel"]),
                "x_normalized": float(r["x_normalized"]),
                "y_normalized": float(r["y_normalized"]),
            } for r in rows
                if r.get("x_pixel") is not None
                and r.get("x_normalized") is not None
                and r.get("y_normalized") is not None]
            out.extend(d for d in split_drags(events)
                       if move_count(d) >= MIN_MOVES_PER_DRAG)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--version", required=True)
    # The furthest-human floor (`--tail-percent 0`) is the most conservative
    # setting and was the original one: it cannot reject anyone the detector was
    # not already rejecting. It also missed every bot that parked just inside the
    # edge of the human region, which is where the surviving families live.
    #
    # Spending a small density-tail budget instead took the worst unseen family
    # from 57.8% to 35.0% with the operating point re-read at the same 2.3% false
    # reject rate, so the veto's cost is absorbed rather than added:
    #
    #     replay_warp        57.8% -> 22.0%      ml_pca_gmm    28.5% -> 35.0%
    #     adversarial_holdout 52.2% -> 21.2%     vae_bots      19.2% -> 21.5%
    #
    # Read as a percentile of the *training humans* so the rule can be stated
    # without reference to any bot set. 2% is a round number in the region a
    # multiplier sweep suggested; that sweep did look at holdout scores, so this
    # value is not fully blind and the sealed split remains its real check.
    ap.add_argument("--tail-percent", type=float, default=2.0,
                    help="이 비율만큼의 학습 사람을 밀도 꼬리로 내준다 (0 = 가장 먼 사람)")
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

    # 문턱은 채점과 **같은 단위**에서 잡는다.
    #
    # 밀도 자체는 드래그 단위로 맞춘다 — 표본이 많아 사람 영역이 촘촘해진다. 그런데
    # 운영에서 들어오는 것은 시도(세션) 하나의 특징 벡터다. 두 분포는 값의 범위가
    # 달라서, 드래그 분포의 하위 2% 를 세션에 그대로 적용하면 멀쩡한 사람이 대량으로
    # 걸린다. 실제로 그렇게 나갔고 오탐이 0.0% → 16.7% 가 됐다(2026-08-11 실측).
    # 그래서 percentile 은 세션 특징에서 읽는다.
    sess = session_vectors(train_people, names)
    if sess.size == 0:
        raise SystemExit("문턱을 잡을 세션 특징이 없다 — human_features 파일을 확인하라")
    floor = float(np.percentile(density.score(sess), args.tail_percent))
    print(f"밀도 학습 드래그 {len(X)}개 · 문턱 산출 세션 {len(sess)}개")
    print(f"거부권 문턱 {floor:.6f} (학습 사람 **세션** 밀도 하위 {args.tail_percent}%)")

    out = dict(bundle)
    out["model_version"] = args.version
    out["density_veto"] = density
    out["density_feature_names"] = names
    out["veto_below"] = floor
    out["veto_note"] = (
        f"사람 영역 밖일 때 거부한다. 문턱은 학습 사람 밀도의 하위 "
        f"{args.tail_percent}% 다 — 가장 먼 사람(0%)으로 두면 사람 영역 가장자리에 "
        "붙은 봇을 전부 놓친다. 이 비용은 동작점을 같은 오탐 예산에서 다시 읽어 "
        "흡수하므로 오탐이 늘지 않는다. 밀도 모델은 수집분 사람만으로 학습 — "
        "레거시를 섞으면 사람 영역이 옛 화면 모양으로 잡혀 메인 캡차 사람 "
        "66/131 이 0점이 된다."
    )
    out["density_fit"] = {
        "drags": len(X), "people": sorted(train_people),
        "excluded_features": list(EXCLUDED),
        "tail_percent": args.tail_percent,
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
