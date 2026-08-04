"""Reject what is far from real humans, not just what looks bot-like.

Every defence tried so far is discriminative: a boundary between human and bot.
The weakset is built by searching for points *inside* the human side of that
boundary — 16,000 candidates to find 100 — so no threshold on it can help, and
adversarial training on it blew up unseen-human FRR (2.39% → 9.06%, 07-22).

But a discriminative boundary encloses far more space than humans actually
occupy. A hill-climbed point can sit deep inside "human" while being nowhere
near any real human trajectory. That distance is a signal nothing here measures.

So this adds a second, generative question: how far is this trajectory from the
humans we have seen? Distance to the k nearest training humans in standardized
feature space. It costs the attacker something different — not "look less like a
bot" but "look like an actual human", which is the thing that was supposed to be
expensive all along.

Reported at a fixed human FRR so the comparison is honest: a gate that rejects
the weakset by also rejecting humans has bought nothing.

    .venv/bin/python tools/human_density_gate.py
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.feature_extractor_v23 import extract_features  # noqa: E402
from app.services.trajectory_feature_views import get_feature_view  # noqa: E402

LOCKBOX = Path("data/interim/revalidation_human_lockbox_h20066_v23corr_20260722")
HOLDOUT = Path("data/interim/human_holdout_prevtest_h2219_20260804/human_holdout.jsonl")
WEAKSET = Path("data/interim/hybrid_motion_score_guided_weakset_100_20260722.jsonl")
EXTERNAL = Path("data/interim/hybrid_motion_redteam_external_holdout_500_20260722.jsonl")
BOTS = Path("data/interim/bot_features_v23corr_20260722.jsonl")

K = 5


def rows_from_features(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def rows_from_events(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        events = json.loads(line).get("events") or []
        if len(events) >= 3:
            out.append(extract_features(events, None))
    return out


def matrix(rows: list[dict], names: tuple[str, ...]) -> np.ndarray:
    X = np.array([[float(r.get(n) or 0.0) for n in names] for r in rows], dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def participant_sweep(k, seed, by_person, people, names, Xw, Xe, Xh):
    """Threshold from k participants under the real criteria, then evaluate.

    Leave-one-participant-out inside the k, so no participant sets the threshold
    that protects them. The threshold is the strictest one that still keeps the
    pooled FRR under 3% and every single participant under 5%.
    """
    rng = np.random.default_rng(seed)
    order = list(people)
    rng.shuffle(order)
    selected = order[:k]

    per = {}
    for p in selected:
        others = [r for q in selected if q != p for r in by_person[q]]
        X = matrix(others, names)
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        nn = NearestNeighbors(n_neighbors=K).fit(np.clip((X - mu) / sd, -8, 8))
        Xp = np.clip((matrix(by_person[p], names) - mu) / sd, -8, 8)
        per[p] = nn.kneighbors(Xp)[0].mean(1)

    pooled = np.concatenate(list(per.values()))
    cut = float(np.quantile(pooled, 0.97))
    for q in np.linspace(0.80, 0.999, 120):
        c = float(np.quantile(pooled, q))
        if (pooled > c).mean() <= 0.03 and max((v > c).mean() for v in per.values()) <= 0.05:
            cut = c
            break

    rows = [r for p in selected for r in by_person[p]]
    X = matrix(rows, names)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    nn = NearestNeighbors(n_neighbors=K).fit(np.clip((X - mu) / sd, -8, 8))
    d = lambda A: nn.kneighbors(np.clip((A - mu) / sd, -8, 8))[0].mean(1)
    return ((d(Xw) > cut).mean() * 100, (d(Xe) > cut).mean() * 100,
            (d(Xh) > cut).mean() * 100, len(rows))


def main() -> None:
    names = tuple(dict.fromkeys(
        get_feature_view("general_without_physics") + get_feature_view("dynamics_physics")))

    train_h = rows_from_features(LOCKBOX / "human_development.jsonl")
    holdout_h = rows_from_features(HOLDOUT)
    weak = rows_from_events(WEAKSET)
    ext = rows_from_events(EXTERNAL)
    bots = rows_from_features(BOTS)
    print(f"학습 사람 {len(train_h)} · 홀드아웃 사람 {len(holdout_h)} | "
          f"약점셋 {len(weak)} · 외부 {len(ext)} · 학습봇 {len(bots)}\n")

    Xtr = matrix(train_h, names)
    # Rank-based standardization: several features are heavy-tailed, and a single
    # extreme value would otherwise dominate every distance.
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    z = lambda X: np.clip((X - mu) / sd, -8, 8)

    # Fit on a subsample: 15k x 15k neighbour search is not needed to estimate
    # where the human mass is, and the holdout stays untouched either way.
    rng = np.random.default_rng(20260804)
    idx = rng.permutation(len(Xtr))
    fit_idx, cal_idx = idx[: int(len(idx) * 0.8)], idx[int(len(idx) * 0.8):]
    nn = NearestNeighbors(n_neighbors=K).fit(z(Xtr[fit_idx]))

    def distance(rows: list[dict]) -> np.ndarray:
        if not rows:
            return np.array([])
        d, _ = nn.kneighbors(z(matrix(rows, names)))
        return d.mean(axis=1)          # mean distance to the k nearest humans

    Xw, Xe, Xh = matrix(weak, names), matrix(ext, names), matrix(holdout_h, names)

    by_person = collections.defaultdict(list)
    for r in train_h:
        if r.get("anonymous_participant_id"):
            by_person[r["anonymous_participant_id"]].append(r)
    people = list(by_person)

    cal_d = distance([train_h[i] for i in cal_idx])
    sets = {
        "학습 사람 (보정)": cal_d,
        "홀드아웃 사람 (처음 보는 7명)": distance(holdout_h),
        "약점셋": distance(weak),
        "외부 holdout 봇": distance(ext),
        "학습 봇": distance(bots),
    }
    print(f"  {'집합':30s}{'n':>6s}{'중앙 거리':>12s}{'90퍼센타일':>12s}")
    for label, d in sets.items():
        if len(d):
            print(f"  {label:30s}{len(d):>6d}{np.median(d):>12.3f}{np.quantile(d, 0.9):>12.3f}")

    # A pooled percentile is NOT this project's policy. The model calibrates
    # per participant ("every Human participant group must satisfy the FRR
    # constraint"), and that difference is not cosmetic: pooled 3% blocks 60% of
    # the weakset, per-participant 3% blocks 5%. The promotion criteria are
    # pooled <=3% AND worst participant <=5%, so that is what gets reported.
    print(f"\n  참여자 수에 따른 성능 (기준: 전체 FRR<=3% · 최악 참여자<=5%)")
    print(f"  {'참여자':>7s}{'세션':>8s}{'약점셋 차단':>13s}{'외부 차단':>12s}"
          f"{'처음보는사람 오탐':>17s}")
    for k in (4, 8, 12, 16, 20, len(people)):
        runs = [participant_sweep(k, seed, by_person, people, names, Xw, Xe, Xh)
                for seed in (1, 2, 3)]
        w, e, h, n = (float(np.mean([r[i] for r in runs])) for i in range(4))
        print(f"  {k:>7d}{int(n):>8d}{w:>12.1f}%{e:>11.1f}%{h:>16.1f}%")

    # --- combine with the shipped discriminative model -----------------------
    import joblib
    bundle = joblib.load(
        "models/candidate/revalidation_two_view_participant_safe_20260722/two_view_fusion.joblib")

    def model_score(rows):
        if not rows:
            return np.array([])
        per = []
        for view, model in bundle["models"].items():
            X = matrix(rows, tuple(bundle["feature_views"][view]))
            per.append(model.predict_proba(X)[:, list(model.classes_).index(1)])
        return np.min(np.vstack(per), axis=0)

    model_scores = {
        "홀드아웃 사람 (처음 보는 7명)": model_score(holdout_h),
        "약점셋": model_score(weak),
        "외부 holdout 봇": model_score(ext),
    }
    combined(sets, cal_d, model_scores, float(bundle["threshold"]))


def combined(sets, cal_d, model_scores, model_threshold):
    """Pass only if the model says human AND the point is near real humans.

    Two independent questions — "does it look unlike a bot" and "does it look
    like a human we have actually seen" — so an attacker has to satisfy both.
    The weakset satisfies the first by construction; it was never searched
    against the second.
    """
    print(f"\n  판별 모델과 결합 — 주의: 아래 거리 임계는 풀링 백분위수다.")
    print(f"  이 프로젝트의 실제 정책(참여자별)에서는 위 표를 봐야 한다.")
    print(f"  (모델 임계 {model_threshold:.7f})")
    print(f"  {'거리 FRR 목표':>12s}{'거리 임계':>11s}{'약점셋 ASR':>12s}"
          f"{'외부 ASR':>11s}{'홀드아웃 FRR':>13s}")
    base_weak = float((model_scores["약점셋"] >= model_threshold).mean() * 100)
    base_ext = float((model_scores["외부 holdout 봇"] >= model_threshold).mean() * 100)
    base_hold = float((model_scores["홀드아웃 사람 (처음 보는 7명)"] < model_threshold).mean() * 100)
    print(f"  {'거리 안 씀':>12s}{'-':>11s}{base_weak:>11.1f}%{base_ext:>10.1f}%{base_hold:>12.1f}%")
    for target in (0.01, 0.02, 0.03, 0.05):
        cut = float(np.quantile(cal_d, 1 - target))
        def asr(name):
            passes = (model_scores[name] >= model_threshold) & (sets[name] <= cut)
            return float(passes.mean() * 100)
        hold_pass = (model_scores["홀드아웃 사람 (처음 보는 7명)"] >= model_threshold) & \
                    (sets["홀드아웃 사람 (처음 보는 7명)"] <= cut)
        print(f"  {target*100:>11.0f}%{cut:>11.3f}{asr('약점셋'):>11.1f}%"
              f"{asr('외부 holdout 봇'):>10.1f}%{(1 - hold_pass.mean()) * 100:>12.1f}%")


if __name__ == "__main__":
    main()
