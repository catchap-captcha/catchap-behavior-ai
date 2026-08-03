"""Pick the per-drag operating point on one participant, test it on the other.

Why not just sweep the threshold
--------------------------------
The 1.5% FRR reported on 2026-08-03 came from choosing 0.01 on the same 134
sessions it was measured on. That is the exact shape of the failure that killed
the hard-negative candidate on 07-22: OOF said 2.39%, untouched test said 9.06%.
A number chosen and measured on one set says nothing about the next user.

With two participants we cannot prove generalisation, but we can stop pretending.
Fit the threshold on participant A, report it on participant B, then swap. The
gap between the two is the honest error bar on "1.5%".

It also compares aggregations. The median was picked because the minimum
rejected 52% of humans, but nothing checked whether some other summary does
better on a held-out person.

    .venv/bin/python tools/frr_operating_point.py data/interim/main_captcha_raw_20260803b.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags  # noqa: E402
from app.services.feature_extractor_v23 import extract_features  # noqa: E402
from app.services.model_service import model_service  # noqa: E402
from tools.per_drag_scoring import MODEL_PATH, Scorer, to_extractor_events  # noqa: E402

TARGET_FRR = 0.02          # the promotion criterion, fixed 2026-07-30


def aggregations() -> dict[str, callable]:
    def trimmed(scores: list[float]) -> float:
        """Drop the worst drag, then take the median of the rest.

        One bad drag out of five should not sink a session — that is what made
        the minimum useless — but neither should a single good drag rescue one.
        """
        if len(scores) <= 2:
            return statistics.median(scores)
        return statistics.median(sorted(scores)[1:])

    return {
        "min": min,
        "median": statistics.median,
        "trimmed": trimmed,
        "p25": lambda s: statistics.quantiles(s, n=4)[0] if len(s) >= 4 else min(s),
        "mean": statistics.fmean,
    }


# Sessions collected by us but not yet labelled in the DB. Naming them here
# rather than defaulting to "unlabelled means human" keeps an unrelated bot run
# from silently entering the human set.
SELF_COLLECTED_HUMAN = ("sw-mouse-v2", "sw-trackpad", "sw-touch")

# The same human appears under several participant codes — sw-mouse and
# sw-mouse-v2 are one person on two days, and a device split adds more. Folding
# by code instead of by person would put the same person on both sides of a
# leave-one-out split, which is precisely the leakage the split exists to stop.
def person_of(participant: str) -> str:
    return participant.split("-")[0] or participant


def load(path: str) -> list[dict]:
    scorer = Scorer(MODEL_PATH)
    rows = []
    for line in Path(path).read_text().splitlines():
        rec = json.loads(line)
        prod = rec.get("prod_human_probability")
        if prod is None:
            continue
        events = to_extractor_events(rec["events"])
        session = scorer.score(events)
        # Only rows where our pipeline reproduces production. A row we cannot
        # reproduce is one where we and production disagree about the input.
        if session is None or abs(float(prod) - session) >= 0.01:
            continue
        drags = split_drags(events)
        usable = [d for d in drags if move_count(d) >= MIN_MOVES_PER_DRAG]
        if not usable:
            scores = [0.0]                      # every drag starved -> teleport
        else:
            scores = [scorer.score(d) for d in usable]
            scores = [s for s in scores if s is not None] or [0.0]
        participant = rec.get("participant_id") or ""
        label = rec.get("label")
        if label is None and participant in SELF_COLLECTED_HUMAN:
            label = "human"
        rows.append({
            "participant": participant,
            "label": label,
            "session": session,
            "drag_scores": scores,
        })
    return rows


def frr_at(scores: list[float], threshold: float) -> float:
    return sum(1 for s in scores if s < threshold) / len(scores) if scores else 0.0


def asr_at(scores: list[float], threshold: float) -> float:
    return sum(1 for s in scores if s >= threshold) / len(scores) if scores else 0.0


def fit_threshold(human: list[float], bot: list[float]) -> float:
    """Lowest threshold whose human FRR still meets target, i.e. most bots blocked.

    Chosen only from the fitting participant's scores, never from the held-out
    one — that separation is the entire point of this file.
    """
    best = 0.0
    for candidate in sorted(set(human)):
        if frr_at(human, candidate) <= TARGET_FRR:
            best = candidate
    return best


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "data/interim/main_captcha_raw_20260803b.jsonl"
    model_service._bundle = joblib.load(MODEL_PATH)
    rows = load(src)

    humans = defaultdict(list)
    for r in rows:
        if r["label"] == "human":
            humans[person_of(r["participant"])].append(r)
    bots = [r for r in rows if r["label"] == "bot"]
    print(f"재현되는 행 · 사람 {sum(len(v) for v in humans.values())} "
          f"({len(humans)}명) · 봇 {len(bots)}\n")
    for p, v in sorted(humans.items()):
        print(f"  {p:16s} {len(v):>4d}건")

    for name, agg in aggregations().items():
        print(f"\n=== 집계: {name} ===")
        by_p = {p: [agg(r["drag_scores"]) for r in v] for p, v in humans.items()}
        bot_scores = [agg(r["drag_scores"]) for r in bots]

        print(f"  {'임계값 고른 사람':14s}{'임계값':>12s}{'그 사람 FRR':>12s}"
              f"{'처음 보는 사람 FRR':>18s}{'봇 ASR':>9s}")
        for fit_p in sorted(by_p):
            held = [s for p, v in by_p.items() if p != fit_p for s in v]
            if not held:
                continue
            th = fit_threshold(by_p[fit_p], bot_scores)
            print(f"  {fit_p:16s}{th:>12.6f}{frr_at(by_p[fit_p], th)*100:>11.1f}%"
                  f"{frr_at(held, th)*100:>17.1f}%{asr_at(bot_scores, th)*100:>8.1f}%")


if __name__ == "__main__":
    main()
