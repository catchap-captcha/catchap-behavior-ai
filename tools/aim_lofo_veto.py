"""Same leave-one-family-out, with the density veto that fixed the drag surface.

Why this run exists
-------------------
The plain run produced a result that only looks absurd until you recognise it:
`A_linear` — constant speed along a straight line, the most trivially detectable
bot in the set, separable from humans at AUC 1.000 on a single feature — evaded
100% of the time when it was the held-out family.

That is the empty corner, and it is the same failure already diagnosed on the
drag surface. Gradient-boosted trees do not extrapolate. A region of feature
space containing no training points gets whatever leaf it happens to fall into,
and "perfectly uniform" is a corner no other family visits. Adding features
cannot fix it; only having an opinion about emptiness can.

The density model supplies that opinion. It is fitted on humans alone and asks
one question — is this anywhere near a person? — with a threshold set at the
furthest training human, so it cannot reject anyone the detector was not already
going to reject. On the drag surface this took the worst unseen family from
100.0% to 58.6% with false rejections moving 2.3% -> 2.4%.

Whether it rescues `D_replay` is a different matter, and the one that decides
whether the aim segment is worth collecting at scale. A replay of a real human
aim sits *inside* the human cloud by construction, which is exactly where a
density model has nothing to say.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from tools.aim_features import FEATURE_NAMES, MAGNITUDE_BOUND, matrix
from tools.aim_lofo import BOT_DIR, FRR_BUDGET, SEED, human_groups, load_bot
from tools.train_density_veto import DensityVeto


def run(names: tuple[str, ...], label: str) -> dict[str, float]:
    humans, groups = human_groups()
    H = matrix(humans, names)
    families = {p.stem: matrix(load_bot(p), names) for p in sorted(BOT_DIR.glob("*.jsonl"))}

    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    rng.shuffle(unique)
    test_groups = set(unique[: max(1, len(unique) // 3)])
    is_test = np.array([g in test_groups for g in groups])
    H_train, H_test = H[~is_test], H[is_test]

    # Fitted on training humans only. Bots never touch it, and neither do the
    # humans the operating point is read from.
    density = DensityVeto(H_train, names)
    veto_floor = float(np.min(density.score(H_train)))

    print(f"\n=== {label} ===")
    print(f"특징 {len(names)}개 · 밀도 문턱 {veto_floor:.6f} (학습 사람 중 가장 먼 사람)")
    header = f"  {'뺀 계열':14s}{'거부권 전':>10s}{'거부권 후':>10s}{'사람 오탐':>10s}"
    print(header)

    results = {}
    for held in families:
        seen = [f for f in families if f != held]
        X = np.vstack([H_train] + [families[f] for f in seen])
        y = np.concatenate([np.zeros(len(H_train))] + [np.ones(len(families[f])) for f in seen])
        model = GradientBoostingClassifier(random_state=SEED, n_estimators=200,
                                           max_depth=3, learning_rate=0.06).fit(X, y)

        def combined(M: np.ndarray) -> np.ndarray:
            """Bot probability, forced to 1.0 outside the human region."""
            p = model.predict_proba(M)[:, 1]
            return np.where(density.score(M) < veto_floor, 1.0, p)

        # One operating point, calibrated on the combined score so the veto's
        # cost is inside the budget rather than spent silently beside it.
        human_scores = combined(H_test)
        threshold = float(np.quantile(human_scores, 1.0 - FRR_BUDGET))
        frr = float((human_scores >= threshold).mean())

        plain_t = float(np.quantile(model.predict_proba(H_test)[:, 1], 1.0 - FRR_BUDGET))
        before = float((model.predict_proba(families[held])[:, 1] < plain_t).mean())
        after = float((combined(families[held]) < threshold).mean())

        results[held] = after
        arrow = "→" if abs(after - before) > 0.005 else " "
        print(f"  {held:14s}{before*100:9.1f}%{after*100:9.1f}%{frr*100:9.1f}%  {arrow}")

    worst = max(results.values())
    print(f"  최악 미지 계열 {worst*100:.1f}%")
    return results


def main() -> int:
    shape_only = tuple(n for n in FEATURE_NAMES if n not in MAGNITUDE_BOUND)
    full = run(FEATURE_NAMES, "전체 특징 + 밀도 거부권")
    shape = run(shape_only, "크기 제외 + 밀도 거부권")
    print("\n  요약 — 최악 미지 계열")
    print(f"    전체 특징   {max(full.values())*100:5.1f}%")
    print(f"    크기 제외   {max(shape.values())*100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
