"""Leave-one-family-out on the aim segment, at one operating point.

The question
------------
The 12-point drag surface is unwinnable against replay-derived bots, and the
reason is resolution: at 12 points two *different humans* reach path similarity
1.0000, so no model can hold a line that human variation already crosses. The
aim segment carries 16 points over 1049ms instead. This asks whether the extra
resolution buys separability, or just more of the same.

The measurement follows the rules that earlier mistakes forced into place:

  one operating point   the threshold is fixed by a human false-reject budget,
                        once, and every family is read at that same threshold.
                        Calibrating per family was how a 48.9% result turned out
                        to really be 61.5%.
  unseen family         the held-out family is absent from training entirely.
                        A seed split inside one generator does not count.
  no giveaways          `aim_giveaways.py` runs first; every family's best
                        single feature sits below 0.99 by construction, so no
                        result here rests on a generator artefact.
  humans split by       bursts from one challenge never straddle train and test.
  challenge

What this cannot do
-------------------
The numbers below were taken when every human aim segment belonged to one
person. Three more people collected afterwards, and this was deliberately not
re-run: the result is negative — the strongest family evades regardless — and
a negative does not improve with more people, because the attack does not get
weaker as the human pool grows.

The positive finding on this surface is elsewhere (trajectory fingerprinting,
`docs/AIM_SEGMENT_BOT_STUDY_20260810.md` §④) and *was* re-earned across four
people, because a positive proves nothing until it is.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from tools.aim_features import FEATURE_NAMES, MAGNITUDE_BOUND, matrix
from tools.aim_segments import AIM_CAPTURE as HUMAN, load_bursts

BOT_DIR = Path("data/interim/aim_bots")
FRR_BUDGET = 0.05
SEED = 20260810


def human_groups() -> tuple[list[list[dict]], np.ndarray]:
    """Bursts plus the challenge each came from, so splits can respect it."""
    bursts, groups = [], []
    with HUMAN.open() as f:
        for line in f:
            record = json.loads(line)
            if str(record.get("participant_id", "")).startswith("zzprobe"):
                continue
            challenge = record.get("challenge_id") or "?"
            row_bursts = load_bursts_from(record)
            bursts.extend(row_bursts)
            groups.extend([challenge] * len(row_bursts))
    return bursts, np.array(groups)


def load_bursts_from(record: dict) -> list[list[dict]]:
    from tools.aim_segments import GAP_MS, MIN_POINTS
    events = record.get("aim_events") or []
    if not events:
        return []
    out, current = [], [events[0]]
    for prev, nxt in zip(events, events[1:]):
        if nxt["timestamp_ms"] - prev["timestamp_ms"] > GAP_MS:
            out.append(current)
            current = [nxt]
        else:
            current.append(nxt)
    out.append(current)
    return [b for b in out if len(b) >= MIN_POINTS]


def load_bot(path: Path) -> list[list[dict]]:
    return [json.loads(line)["aim_events"] for line in path.open()]


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

    print(f"\n=== {label} ===")
    print(f"특징 {len(names)}개 · 사람 학습 {len(H_train)} / 시험 {len(H_test)} "
          f"(챌린지 {len(unique)}개를 갈라서)")
    print(f"  {'뺀 계열':14s}{'그 계열 ASR':>12s}{'사람 오탐':>11s}{'본 계열 평균 ASR':>17s}")

    results = {}
    for held in families:
        seen = [f for f in families if f != held]
        X = np.vstack([H_train] + [families[f] for f in seen])
        y = np.concatenate([np.zeros(len(H_train))] + [np.ones(len(families[f])) for f in seen])

        model = GradientBoostingClassifier(random_state=SEED, n_estimators=200,
                                           max_depth=3, learning_rate=0.06).fit(X, y)

        # One threshold, fixed by the human budget on held-out humans only —
        # never by anything the bots do.
        human_scores = model.predict_proba(H_test)[:, 1]
        threshold = float(np.quantile(human_scores, 1.0 - FRR_BUDGET))
        frr = float((human_scores >= threshold).mean())

        asr = float((model.predict_proba(families[held])[:, 1] < threshold).mean())
        seen_asr = float(np.mean([
            (model.predict_proba(families[f])[:, 1] < threshold).mean() for f in seen]))
        results[held] = asr
        mark = "" if asr <= 0.10 else ("  ⚠️" if asr > 0.30 else "  △")
        print(f"  {held:14s}{asr*100:11.1f}%{frr*100:10.1f}%{seen_asr*100:16.1f}%{mark}")

    worst = max(results.values())
    print(f"  최악 미지 계열 ASR {worst*100:.1f}%  (기준 ≤10% "
          f"{'통과' if worst <= 0.10 else '미달'})")
    return results


def main() -> int:
    shape_only = tuple(n for n in FEATURE_NAMES if n not in MAGNITUDE_BOUND)
    full = run(FEATURE_NAMES, "전체 특징")
    shape = run(shape_only, "크기 의존 특징 제외 (창 크기로 못 피하게)")

    print("\n  요약 — 최악 미지 계열")
    print(f"    전체 특징      {max(full.values())*100:5.1f}%")
    print(f"    크기 제외      {max(shape.values())*100:5.1f}%")
    print("\n  ⚠️ 이 숫자는 사람 1명일 때 잰 상한이다. 부정적 결과라 사람이 늘어도\n     바뀌지 않지만, 성적표로 인용하면 안 된다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
