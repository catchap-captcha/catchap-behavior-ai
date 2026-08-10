"""Train the two-view detector on drags instead of whole sessions, with a record.

Why this exists
---------------
Two things went wrong on 2026-08-06 and this fixes both.

`scale_pixel_20260804` is the only model that clears every gate (evasion 7.1%,
pooled FRR 1.7%, worst participant 4.9%) and it cannot be promoted, because
nothing in the repo builds it. It was made by a throwaway script that was never
saved. A model you cannot rebuild cannot be retrained, defended, or repaired.

`retrain_per_drag.py` was handed only main-captcha rows, which left two people in
training and produced 9.9/25.0/22.7% unseen-person FRR — exactly the failure
`collection_split.py` had written down in advance. The fix is not more people; it
is to stop discarding the 20,066 legacy sessions. An earlier note claimed those
carry no press/release boundaries and cannot be segmented. That was wrong:
800/800 sampled legacy sessions segment cleanly, one drag each. So drag-unit
training gets 50 participant groups, not 3.

Scoring matches the service exactly: per drag, fuse the two views with min(), and
take the session's median across its drags. The move floor applies to the
SESSION — a session whose drags are all too short is a bot, but one short drag
among several is not, which is what pushed FRR 1.5% -> 5.2% when applied per drag.

Sealed people are refused, not filtered, and every external holdout and the
red-team weakset are excluded by name. The weakset's own manifest says
`detector_training_forbidden: true`; training on it is what disqualified
`patched_weakset_iter1_20260804` regardless of its score.

    .venv/bin/python tools/train_drag_unit_candidate.py \
        --version drag_unit_20260806 --max-human-frr 0.01
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags  # noqa: E402
from app.services.feature_extractor_v23 import FEATURE_SCHEMA_VERSION, extract_features  # noqa: E402
from app.services.trajectory_feature_views import get_feature_view  # noqa: E402
from tools.collection_split import person_of  # noqa: E402

VIEWS = ("general_without_physics", "dynamics_physics")

LEGACY_HUMANS = Path("data/raw/human_db_snapshot_20260721T012239Z/human_attempts.jsonl")
COLLECTION = Path("data/interim/collection_20260806.jsonl")
SPLIT = Path("data/metadata/collection_split_20260806.json")

# Development bots only. Anything named *external*, *holdout* or *weakset* is a
# measuring stick — training on it destroys the measurement, which is the whole
# point of the lockbox discipline.
BOT_SOURCES = [
    # Main-captcha-shaped. Without these every bot in training is legacy-shaped
    # (~340 events/session vs ~12 for a real drag), so `event_count` separates
    # them with AUC 1.000 and the detector never learns the hard signals —
    # pause_count is human median 3, generated 0, and it already ships unused.
    "data/interim/main_captcha_bots_nop5_20260806.jsonl",
    "data/interim/extended_bots_10000_20260721.jsonl",
    "data/interim/rule_bots_3000.jsonl",
    "data/interim/adversarial_replay_broad_development_3000_20260721.jsonl",
    "data/interim/adversarial_replay_composite_development_3000_20260721.jsonl",
    "data/interim/vae_bots_development_1000_20260721.jsonl",
    "data/interim/ml_pca_gmm_development_1000_20260721.jsonl",
    "data/interim/playwright_ease_burst_development_300_20260722.jsonl",
]
FORBIDDEN = ("external", "holdout", "weakset", "lockbox")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


def drags_of(events: list[dict]) -> list[list[dict]]:
    return [d for d in split_drags(events) if move_count(d) >= MIN_MOVES_PER_DRAG]


# Observed captcha stage widths, from ai_behavior_attempts: 500 dominates but
# 333/375/640/1024 all occur. Sampling from these is not data augmentation for
# its own sake — it is how the scale-dependent features get taken away.
STAGE_WIDTHS = (333.0, 375.0, 500.0, 500.0, 500.0, 640.0, 1024.0)


def rescale(drag: list[dict], factor: float) -> list[dict]:
    """Same gesture, different stage. Absolute-magnitude features move, shape
    and timing features do not.

    Why this is the whole experiment: `scale_pixel` — the only candidate that
    clears every gate — leans on turn_angle_mean, micro_move_ratio, interval_cv,
    turn_direction_change_ratio and pause_count. The deployed model leans on
    total_distance, y_deviation and normalized_speed_p10. Those are absolute
    magnitudes, and an attacker changes them by resizing the window. Training
    across stage sizes makes them unlearnable, which forces the model onto the
    shape and rhythm features `scale_pixel` actually uses.
    """
    return [dict(e, x=float(e["x"]) * factor, y=float(e["y"]) * factor) for e in drag]


def legacy_events(record: dict) -> list[dict]:
    """Legacy traces already carry seq/event_type/t_ms/x/y in pixels."""
    return record.get("events") or []


def collection_events(record: dict) -> list[dict]:
    """DB rows. Pixels, matching production — measured in per_drag_scoring."""
    rows = record.get("events") or []
    if not rows:
        return []
    base = rows[0].get("client_timestamp_ms") or 0
    out = []
    for r in rows:
        x, y = r.get("x_pixel"), r.get("y_pixel")
        if x is None or y is None:
            continue
        out.append({
            "seq": r.get("seq"), "event_type": r.get("event_type"),
            "t_ms": float((r.get("client_timestamp_ms") or base) - base),
            "x": float(x), "y": float(y),
        })
    return out


def load_sessions(limit_legacy: int | None, augment: int = 0, seed: int = 0,
                  exclude: set[str] | None = None,
                  exclude_family: str | None = None) -> list[dict]:
    """One entry per session: label, group, and its drags' feature vectors."""
    rng = random.Random(seed)
    exclude = exclude or set()
    sealed = set(json.loads(SPLIT.read_text())["holdout_people"])
    training = set(json.loads(SPLIT.read_text())["training_people"])
    print(f"봉인 {sorted(sealed)} 제외 · 수집 학습 {sorted(training)}")

    sessions: list[dict] = []

    kept = 0
    with LEGACY_HUMANS.open() as f:
        for line in f:
            if limit_legacy and kept >= limit_legacy:
                break
            record = json.loads(line)
            ds = drags_of(legacy_events(record))
            if not ds:
                continue
            participant = record.get("anonymous_participant_id") or "legacy_unknown"
            sessions.append({
                "label": "human", "group": f"human::{participant}", "source": "legacy",
                "drags": featurise(ds, augment, rng),
            })
            kept += 1
    print(f"  레거시 사람 {kept}세션")

    if COLLECTION.exists():
        added = 0
        with COLLECTION.open() as f:
            for line in f:
                record = json.loads(line)
                code = record.get("participant_id") or ""
                person = person_of(code)
                if person in sealed or person in exclude:
                    # sealed: refused, never scored. exclude: held out so the
                    # operating point and the attack substrate can both be
                    # fitted on someone this model has never seen.
                    continue
                if person not in training:
                    continue
                if record.get("quality_status") != "valid":
                    continue
                ds = drags_of(collection_events(record))
                if not ds:
                    continue
                sessions.append({
                    "label": "human", "group": f"human::{person}", "source": "collection",
                    "drags": featurise(ds, augment, rng),
                })
                added += 1
        print(f"  수집 사람 {added}세션 (메인 캡차 표면)")

    for name in BOT_SOURCES:
        path = Path(name)
        if exclude_family and exclude_family in path.name:
            # leave-one-family-out: the written criterion for "미지 Bot 최악 ASR"
            # (SECURITY_ACCEPTANCE_CRITERIA.md:29). A seed split inside one
            # generator does not count — §3.1 says so explicitly.
            print(f"  [LOFO] 학습 제외 {path.name}")
            continue
        low = path.name.lower()
        if any(bad in low for bad in FORBIDDEN):
            raise SystemExit(f"학습 금지 파일이 목록에 있다: {path.name}")
        if not path.exists():
            print(f"  ! 없음 {path.name}")
            continue
        added = 0
        with path.open() as f:
            for line in f:
                record = json.loads(line)
                collection = record.get("collection") or {}
                # Allow-list, not deny-list: an unknown usage string is refused.
                # `redteam_only` is the one that disqualified patched_weakset.
                if collection.get("training_usage") not in (
                    None, "development", "development_only", "detector_training"
                ):
                    continue
                ds = drags_of(record.get("events") or [])
                if not ds:
                    continue
                family = collection.get("bot_family") or "unknown"
                generator = collection.get("generator_version") or "unknown"
                sessions.append({
                    "label": "bot", "group": f"bot::{family}::{generator}", "source": path.name,
                    "drags": featurise(ds, augment, rng),
                })
                added += 1
        print(f"  봇 {added}세션  {path.name}")

    return sessions


def featurise(drags: list[list[dict]], augment: int, rng: random.Random) -> list[dict]:
    """One feature row per drag, plus `augment` copies of the session at other
    stage sizes. Copies stay inside the same session so they never split across
    a CV fold — an augmented twin on the other side would be leakage."""
    rows = [extract_features(d, None) for d in drags]
    for _ in range(augment):
        factor = rng.choice(STAGE_WIDTHS) / 500.0
        rows.extend(extract_features(rescale(d, factor), None) for d in drags)
    return rows


def matrices(sessions: list[dict], view: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = get_feature_view(view)
    X, y, g = [], [], []
    for index, s in enumerate(sessions):
        for feats in s["drags"]:
            X.append([float(feats.get(n) or 0.0) for n in names])
            y.append(1 if s["label"] == "human" else 0)
            g.append(index)
    return np.array(X, dtype=float), np.array(y), np.array(g)


def session_scores(models: dict, sessions: list[dict]) -> list[float]:
    """min() across views per drag, median across drags — the service's rule."""
    per_view = {}
    for view, model in models.items():
        names = get_feature_view(view)
        rows, owner = [], []
        for index, s in enumerate(sessions):
            for feats in s["drags"]:
                rows.append([float(feats.get(n) or 0.0) for n in names])
                owner.append(index)
        X = np.nan_to_num(np.array(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        proba = model.predict_proba(X)[:, list(model.classes_).index(1)]
        per_view[view] = (proba, owner)

    a, owner = per_view[VIEWS[0]]
    b, _ = per_view[VIEWS[1]]
    fused = np.minimum(a, b)
    grouped: dict[int, list[float]] = defaultdict(list)
    for value, index in zip(fused, owner):
        grouped[index].append(float(value))
    return [statistics.median(grouped[i]) for i in range(len(sessions))]


def fit_views(sessions: list[dict], seed: int) -> dict:
    models = {}
    for view in VIEWS:
        X, y, _ = matrices(sessions, view)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        model = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                               random_state=seed, verbose=-1)
        model.fit(X, y)
        models[view] = model
    return models


def calibrate(sessions: list[dict], max_human_frr: float, seed: int) -> tuple[float, dict]:
    """Out-of-fold, grouped, then the strictest threshold every human group survives."""
    labels = np.array([1 if s["label"] == "human" else 0 for s in sessions])
    groups = np.array([s["group"] for s in sessions])
    oof = np.zeros(len(sessions))

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(sessions, labels, groups)):
        models = fit_views([sessions[i] for i in train_idx], seed + fold)
        scores = session_scores(models, [sessions[i] for i in test_idx])
        oof[test_idx] = scores
        print(f"  fold {fold} · 학습 {len(train_idx)} · 평가 {len(test_idx)}")

    human_groups = defaultdict(list)
    for score, group, label in zip(oof, groups, labels):
        if label == 1:
            human_groups[group].append(float(score))

    candidates = sorted({round(float(s), 9) for s in oof})
    threshold, per_group = 0.0, {}
    for candidate in candidates:
        rates = {g: sum(1 for s in v if s < candidate) / len(v) for g, v in human_groups.items()}
        if max(rates.values()) <= max_human_frr:
            threshold, per_group = candidate, rates
        else:
            break
    return threshold, {
        "human_groups": len(human_groups),
        "worst_group_frr": max(per_group.values()) if per_group else None,
        "pooled_human_frr": float(np.mean(oof[labels == 1] < threshold)),
        "pooled_bot_asr": float(np.mean(oof[labels == 0] >= threshold)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="drag_unit_20260806")
    ap.add_argument("--max-human-frr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--limit-legacy", type=int)
    ap.add_argument("--exclude-family", help="이 봇 파일을 학습에서 뺀다 (leave-one-family-out)")
    ap.add_argument("--exclude-person", nargs="*", default=[],
                    help="이 사람을 학습에서 뺀다 (정직한 미지 평가용)")
    ap.add_argument("--scale-augment", type=int, default=0,
                    help="세션마다 다른 스테이지 크기 사본을 N개 추가한다")
    args = ap.parse_args()

    sessions = load_sessions(args.limit_legacy, args.scale_augment, args.seed,
                             set(args.exclude_person), args.exclude_family)
    humans = sum(1 for s in sessions if s["label"] == "human")
    drags = sum(len(s["drags"]) for s in sessions)
    groups = len({s["group"] for s in sessions})
    print(f"\n세션 {len(sessions)} (사람 {humans} · 봇 {len(sessions)-humans})"
          f" · 드래그 {drags} · 그룹 {groups}\n")

    print("임계 보정 (참여자 그룹 단위 OOF)")
    threshold, stats = calibrate(sessions, args.max_human_frr, args.seed)
    print(f"\n  임계 {threshold:.9f}  사람그룹 {stats['human_groups']}개"
          f"  최악그룹 FRR {stats['worst_group_frr']:.4f}"
          f"  전체 사람 FRR {stats['pooled_human_frr']:.4f}"
          f"  봇 ASR {stats['pooled_bot_asr']:.4f}\n")

    print("최종 학습 (전체)")
    models = fit_views(sessions, args.seed)

    out_dir = Path("models/candidate") / args.version
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_name": "lightgbm_general_dynamics_min_fusion",
        "model_version": args.version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_input_scope": "pointer_trajectory_only",
        "feature_views": {v: list(get_feature_view(v)) for v in VIEWS},
        "models": models,
        "score_fusion": "min(P_human_general_without_physics, P_human_dynamics_physics)",
        "scoring_unit": "per_drag_median",
        "threshold": threshold,
        "hard_negative": None,
        "fit_scope": "drag-unit; legacy humans + non-sealed collection humans + development bots",
        "threshold_calibration": {
            "max_human_frr": args.max_human_frr,
            "human_frr_policy": "per_participant",
            "policy": "every human participant group must satisfy the FRR constraint on grouped OOF session scores",
            **stats,
        },
    }
    joblib.dump(bundle, out_dir / "two_view_fusion.joblib")

    report_dir = Path("reports") / args.version
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "fit_candidate.json").write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": args.version,
        "seed": args.seed,
        "sessions": len(sessions), "humans": humans, "drags": drags, "groups": groups,
        "threshold": threshold,
        "calibration": stats,
        "inputs": {
            "legacy_humans": {"path": str(LEGACY_HUMANS), "sha256": sha256(LEGACY_HUMANS)},
            "collection": {"path": str(COLLECTION), "sha256": sha256(COLLECTION)},
            "split": json.loads(SPLIT.read_text()),
            "bots": [{"path": p, "sha256": sha256(Path(p))} for p in BOT_SOURCES if Path(p).exists()],
        },
        "excluded": "sealed people, every *external*/*holdout*/*weakset*/*lockbox* file",
    }, ensure_ascii=False, indent=2) + "\n")

    print(f"\n모델 -> {out_dir/'two_view_fusion.joblib'}")
    print(f"기록 -> {report_dir/'fit_candidate.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
