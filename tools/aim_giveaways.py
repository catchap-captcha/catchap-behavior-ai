"""Per-feature AUC of every bot family against humans, before any model exists.

This runs first, on purpose. A single feature that separates at AUC 1.000 means
the family is being caught on an artefact of how it was generated rather than on
how it moves, and any model trained afterwards inherits that flattery.

It has already happened twice in this project. Bots emitted at an unthrottled
cadence separated on interval alone; a synthetic family with uniform step
lengths separated on step-length variance. Both produced beautiful numbers that
described nothing.

The rule applied here: any feature above 0.98 is treated as a generator bug to
fix, not as a defensive win to report.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.aim_features import FEATURE_NAMES, matrix
from tools.aim_segments import AIM_CAPTURE as HUMAN, load_bursts

BOT_DIR = Path("data/interim/aim_bots")


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based AUC, symmetric — 0.5 is useless, 1.0 is a giveaway."""
    both = np.concatenate([pos, neg])
    order = both.argsort().argsort().astype(float) + 1
    r = order[: pos.size].sum()
    a = (r - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    return max(a, 1 - a)


def load_bot(path: Path) -> list[list[dict]]:
    return [json.loads(line)["aim_events"] for line in path.open()]


def main() -> int:
    humans = load_bursts(HUMAN)
    H = matrix(humans)
    print(f"사람 {H.shape[0]}건 · 특징 {H.shape[1]}개\n")

    worst: list[tuple[float, str, str]] = []
    for path in sorted(BOT_DIR.glob("*.jsonl")):
        B = matrix(load_bot(path))
        scores = [(auc(B[:, i], H[:, i]), FEATURE_NAMES[i]) for i in range(H.shape[1])]
        scores.sort(reverse=True)
        print(f"  {path.stem}  ({B.shape[0]}건)")
        for a, name in scores[:5]:
            flag = "  ⚠️거저 잡힘" if a >= 0.98 else ""
            print(f"      {name:22s} AUC {a:.3f}{flag}")
        worst.append((scores[0][0], path.stem, scores[0][1]))
        print()

    print("  요약 — 계열별 최고 단일 특징")
    for a, family, name in sorted(worst, reverse=True):
        verdict = "⚠️ 생성 버그" if a >= 0.98 else ("높음" if a >= 0.90 else "정상")
        print(f"    {family:12s} {name:22s} {a:.3f}  {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
