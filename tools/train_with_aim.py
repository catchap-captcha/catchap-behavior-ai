"""조준을 넣어 두 뷰 분류기를 다시 맞춘다 — 그리고 오탐 3% 에서 무엇이 잡히는지 본다.

왜
--
분류기 오탐 6.2%p 는 짧고 곧은 드래그에 몰려 있다(실사용 1,167건: 튕긴 시도의 드래그
10.9점 vs 통과 15.2점). 그 시도들의 조준 구간은 멀쩡하다(19.2점 vs 20.1점) — 근거가
없는 게 아니라 분류기가 그 구간을 못 보고 있다. 조준을 이으면 근거가 10.9 -> 30.1 점이
된다.

사람은 실사용 궤적을 쓴다. 옛 수집분(20,066건)에는 조준이 없어서 못 쓴다 — 사람만
조준이 없고 봇만 있으면 모델은 "조준 없으면 사람"을 배운다.

읽을 때 주의
------------
A·B·C 계열은 부드러움 하나로 사람과 갈린다(jerk AUC 0.01~0.03). 학습에는 쓰되
**승격 판단에는 쓰지 않는다** — 이들을 잡는 것은 방어력이 아니라 공격자가 약하다는
사실을 재는 것이다. 판단 근거는 D_replay 뿐이다. 진짜 사람 조준을 그대로 쓰고, 조준
본문 어떤 특징으로도 사람과 안 갈린다(최고 AUC 0.448, `join_aim_to_bots --verify`).

    .venv/bin/python tools/train_with_aim.py --version aim_two_view_20260812
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.services.feature_extractor_v23 import extract_features  # noqa: E402
from app.services.trajectory_feature_views import get_feature_view  # noqa: E402
from app.services.aim_segment import AIM_GAP_MS, trim_aim  # noqa: E402
from tools.export_aim_from_production import is_sealed  # noqa: E402
from training.evaluate_models import select_threshold_per_human_group  # noqa: E402

HUMAN_EXPORT = ROOT / "data" / "interim" / "aim_production_20260812.jsonl"
# 멈칫 계열. 빼는 편이 낫다 — 봇 최악 8.4% -> 7.1%. 다만 이걸 뺀다고 특정 참가자가
# 풀리지는 않는다(jy 14.6% -> 14.1%). 빠름·적은 멈칫·큰 회전이 함께 움직여서,
# 하나를 빼면 모델이 나머지로 같은 판단을 한다.
PAUSE_FEATURES = frozenset({"pause_count", "pause_ratio", "pause_position_entropy",
                            "dwell_burst_count", "idle_duration_ms"})
GEOMETRY = Path("/private/tmp/claude-501/-Users-apple-Documents----"
                "/5d3fc21e-53e5-49ae-9182-8aeaed0b6968/scratchpad/chal_geom.tsv")
JOINED = ROOT / "data" / "interim" / "joined"
VIEWS = ("general_without_physics", "dynamics_physics")
# 봉인 참가자. **정확히 일치로 보면 안 된다** — 운영 값이 목적별로 갈라져 있다
# (ms · ms-captcha · sw-mouse · sw-captcha · sw-mouse-v2 · sw-aim). 일치로만 걸면
# 46건만 빠지고 537건이 학습에 남는다. `is_sealed` 가 접두어로 거른다.
SEALED = {"ms", "sw"}
# 승격 판단에 쓰는 계열. 나머지는 학습에만 쓴다(위 주석 참고).
JUDGED = "D_replay"


def human_rows() -> list[dict]:
    """챌린지 단위로 되모은다 — 채점이 받는 단위가 시도 하나이기 때문이다.

    내보내기는 드래그마다 한 행이라 그대로 쓰면 한 시도가 여러 표본이 되고, 같은
    사람의 같은 시도가 학습과 검증에 나뉘어 들어간다.
    """
    geometry = {}
    for line in GEOMETRY.read_text().splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 3:
            geometry[f[0]] = (float(f[1]), float(f[2]))

    grouped: dict[str, dict] = {}
    for line in HUMAN_EXPORT.read_text().splitlines():
        row = json.loads(line)
        person = row.get("participant_id")
        if not person or is_sealed(person, SEALED):
            continue
        slot = grouped.setdefault(row["challenge_id"],
                                  {"person": person, "events": []})
        for kind, key in (("aimmove", "aim_events"), ("pointermove", "drag_events")):
            for e in row.get(key) or []:
                if e.get("x") is None:
                    continue
                slot["events"].append({
                    "event_type": kind, "t_ms": float(e.get("timestamp_ms") or 0.0),
                    "x_normalized": float(e["x"]), "y_normalized": float(e["y"]),
                })

    out = []
    for cid, slot in grouped.items():
        events = sorted(slot["events"], key=lambda e: e["t_ms"])
        if len(events) < 8:
            continue
        w, h = geometry.get(cid, (500.0, 375.0))
        base = events[0]["t_ms"]
        for i, e in enumerate(events):
            e["seq"] = i
            e["t_ms"] -= base
            e["x"] = e["x_normalized"] * w
            e["y"] = e["y_normalized"] * h
        out.append({"group": f"person::{slot['person']}", "person": slot["person"],
                    "events": events, "label": 1})
    return out


def bot_rows() -> list[dict]:
    """계열은 **봇 생성기** 다 — 조준 종류가 아니라.

    승격 기준의 "미지 Bot 최악 ASR" 은 처음 보는 생성기를 뜻한다. 조준은 가장 정직한
    D_replay 하나로 고정하고(진짜 사람 조준 그대로, 조준 본문 최고 AUC 0.448),
    생성기별로 하나씩 빼면서 잰다.
    """
    out = []
    for path in sorted(JOINED.glob("*.jsonl")):
        family = path.stem
        for line in path.read_text().splitlines():
            row = json.loads(line)
            events = row.get("events") or []
            if len(events) < 8:
                continue
            # 봇은 진짜 사람 조준을 그대로 쓴다(D_replay). 조준을 빌려준 사람과 **같은
            # 그룹**에 둬야, 그 사람이 학습에 있는 폴드에서 같은 조준이 사람·봇 양쪽에
            # 걸치지 않는다. 안 묶으면 모델은 조준으로 가르는 법을 배울 수 없다 —
            # 같은 궤적이 한쪽은 사람, 한쪽은 봇이라고 가르쳐지기 때문이다.
            # 계열(생성기) 단위 홀드아웃은 이것과 별개로 `--lofo` 가 처리한다.
            donor = row.get("aim_person")
            group = f"person::{donor}" if donor else f"bot::{family}"
            out.append({"group": group, "family": family,
                        "events": events, "label": 0})
    return out


def view_names(view: str) -> tuple[str, ...]:
    return tuple(n for n in get_feature_view(view) if n not in PAUSE_FEATURES)


def features(rows: list[dict], names: tuple[str, ...]) -> np.ndarray:
    return np.nan_to_num(np.asarray(
        [[float(r["feats"].get(n) or 0.0) for n in names] for r in rows], dtype=float))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--max-frr", type=float, default=0.03)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20260812)
    # 승격 기준은 **모르는 공격**이다. 생성기를 하나씩 통째로 빼고, 그것을 한 번도
    # 못 본 모델로 잰다. 폴드는 사람 기준으로 나뉘므로 조준을 빌려준 사람도 함께 빠진다.
    ap.add_argument("--lofo", action="store_true",
                    help="생성기를 하나씩 빼면서 잰다 (미지 공격 기준)")
    # 원래 기준은 `participant` — 참가자 **한 명도** 예산을 넘지 않는 문턱이다.
    # 지금 데이터로는 도달할 수 없다: 사람 10명 중 안 멈칫하는 분이 1명뿐이라
    # 모델이 그 움직임을 배울 표본이 없고, 그 한 명을 3% 아래로 맞추면 문턱이
    # 0 근처까지 내려가 미지 계열이 78% 통과한다. `overall` 은 그걸 알고 내리는
    # 선택이다 — 전체 오탐만 예산 안에 두고, 특정 참가자의 초과는 문서에 남긴다.
    ap.add_argument("--frr-mode", choices=("participant", "overall"),
                    default="participant")
    args = ap.parse_args()

    humans, bots = human_rows(), bot_rows()
    rows = humans + bots
    for r in rows:
        events = trim_aim(r["events"])
        for i, e in enumerate(events):
            e["seq"] = i
        r["feats"] = extract_features(events, None)
    y = np.array([r["label"] for r in rows])
    groups = np.array([r["group"] for r in rows], dtype=object)

    people = sorted({r["person"] for r in humans})
    print(f"  사람 {len(humans)}건 · {len(people)}명 {people}")
    fam = collections.Counter(r["family"] for r in bots)
    print(f"  봇 {len(bots)}건 · {dict(fam)}")

    # 사람 표본이 봇의 1/6 이라 가중치를 준다. 안 주면 모델이 사람을 통째로 버리는
    # 쪽으로 기울고, 오탐이 곧 성능인 이 문제에서 그건 정답이 아니다.
    weight = {0: 1.0, 1: float(len(bots)) / max(len(humans), 1)}

    family_of = np.array([r.get("family") or "" for r in rows], dtype=object)
    person_of = np.array([r.get("person", "") if r["label"] == 1 else "" for r in rows],
                         dtype=object)

    def run(keep: np.ndarray) -> tuple[np.ndarray, dict]:
        """`keep` 인 행만으로 학습하고, 사람 점수는 폴드 밖에서 얻는다."""
        scores = {v: np.full(len(rows), np.nan) for v in VIEWS}
        fitted = {}
        idx = np.flatnonzero(keep)
        for view in VIEWS:
            X = features(rows, view_names(view))
            splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True,
                                            random_state=args.seed)
            for tr, te in splitter.split(X[idx], y[idx], groups[idx]):
                clf = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                     class_weight=weight, random_state=args.seed, verbose=-1)
                clf.fit(X[idx][tr], y[idx][tr])
                scores[view][idx[te]] = clf.predict_proba(X[idx][te])[:, 1]
            final = LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                                   class_weight=weight, random_state=args.seed, verbose=-1)
            final.fit(X[idx], y[idx])
            fitted[view] = final
            # 학습에서 뺀 행(=미지 계열)은 그 계열을 못 본 모델로 점수를 매긴다.
            out = np.flatnonzero(~keep)
            if out.size:
                scores[view][out] = final.predict_proba(X[out])[:, 1]
        return np.minimum(scores[VIEWS[0]], scores[VIEWS[1]]), fitted

    def report_humans(fused: np.ndarray, threshold: float) -> float:
        worst = 0.0
        for who in people:
            m = (person_of == who) & (y == 1)
            frr = float(np.mean(fused[m] < threshold)) if m.any() else 0.0
            worst = max(worst, frr)
            mark = "" if frr <= args.max_frr else "  ← 초과"
            print(f"    {who[:12]:>12}  {int(m.sum()):>4}건   {frr*100:>5.1f}%{mark}")
        print(f"    {'전체':>12}  {len(humans):>4}건   "
              f"{float(np.mean(fused[y == 1] < threshold))*100:>5.1f}%   최악 {worst*100:.1f}%")
        return worst

    def pick(scores: np.ndarray) -> float:
        if args.frr_mode == "overall":
            return float(np.quantile(scores[y == 1], args.max_frr))
        return select_threshold_per_human_group(scores, y, person_of, max_frr=args.max_frr)

    fused, models = run(np.ones(len(rows), dtype=bool))
    threshold = pick(fused)
    print(f"\n  문턱 {threshold:.6f} (사람 참가자마다 오탐 {args.max_frr*100:.0f}% 이하)")
    print(f"\n  사람별 오탐 (out-of-fold)")
    report_humans(fused, threshold)

    print(f"\n  봇 통과율 — 아는 공격 (그 계열도 학습에 있음)")
    for family in sorted(fam):
        m = (family_of == family) & (y == 0)
        print(f"    {family[:34]:>34}  {int(m.sum()):>5}건   "
              f"{float(np.mean(fused[m] >= threshold))*100:>5.1f}%")

    if args.lofo:
        print(f"\n  봇 통과율 — 미지 공격 (그 계열을 빼고 학습)")
        worst_asr, worst_name = 0.0, ""
        for family in sorted(fam):
            keep = ~((family_of == family) & (y == 0))
            f_scores, _ = run(keep)
            th = pick(f_scores)
            m = (family_of == family) & (y == 0)
            asr = float(np.mean(f_scores[m] >= th))
            if asr > worst_asr:
                worst_asr, worst_name = asr, family
            print(f"    {family[:34]:>34}  {int(m.sum()):>5}건   {asr*100:>5.1f}%")
        print(f"\n  최악 미지 계열: {worst_name} {worst_asr*100:.1f}%  (기준 10% 이하)")

    dest = ROOT / "models" / "candidate" / args.version
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "models": models, "feature_views": {v: list(view_names(v)) for v in VIEWS},
        "model_name": "lightgbm_general_dynamics_min_fusion_with_aim",
        "model_version": args.version, "threshold": float(threshold),
        "feature_schema_version": "2.3", "feature_input_scope": "pointer_trajectory_only",
        "score_fusion": "min(P_human_general_without_physics, P_human_dynamics_physics)",
        "uses_aim": True,
        "fit_scope": "실사용 사람(조준 포함) + 조준을 이은 봇 4계열",
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "max_frr": args.max_frr, "frr_mode": args.frr_mode,
        "aim_gap_ms": AIM_GAP_MS,
        "dropped_features": sorted(PAUSE_FEATURES),
    }, dest / "two_view_fusion.joblib")
    print(f"\n  -> {dest/'two_view_fusion.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
