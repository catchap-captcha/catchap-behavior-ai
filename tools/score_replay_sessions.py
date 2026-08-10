"""Score replay attacks the way they actually arrive: as sessions, not single drags.

Why the existing holdouts cannot answer this
--------------------------------------------
Every replay holdout in `data/interim` stores one drag per row. Replay detection
compares an attempt against *other attempts*, so on those files there is nothing
to compare against and the measured detection is 0% — not because the defence
fails, but because the test is not a test. That zero was read as a failure for
most of 2026-08-10.

A real replay attacker holds a library of captured human traces and reuses them
across the attempts in a session. This builds that shape from real human drags
and measures what the comparator sees.

What it shows
-------------
At a 3% human false-pair budget, with four drags per session (500 trials):

    attack                     rotation-invariant    DTW (deployed)
    one trace reused                    100.0%               0.0%
    warped 0.01                          99.8%               0.0%
    warped 0.02                          99.6%               0.0%
    library of 5                         80.2%               0.0%

Three things follow.

The deployed DTW comparator is not merely weak here, it is *inverted*: innocent
human sessions sit at 0.917 while replay sessions sit at 0.834-0.874. Its 99.9th
percentile on innocent sessions is already 1.0000, so no budget produces a usable
threshold at all — every column above is a floor, not a measurement.

The rotation-invariant comparator needs the 3% budget. At 1% its threshold also
pins to 1.0000 and only byte-identical reuse survives; warped replays all escape.
That is a real cost to weigh, and the escalation-captcha design is what makes it
affordable — a false hit there means solving one more captcha, not being blocked.

And the library number is not a defence property. 79.5% is very close to 80.8%,
the chance that four draws from five sources contain a duplicate — the detector
only fires when the attacker happens to reuse a source *within one session*. With
a library of 100 the ceiling is 5.9%. Widening the comparison beyond the session
is the fix; a better comparator is not.

    .venv/bin/python tools/score_replay_sessions.py
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from math import prod
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags  # noqa: E402
from app.services.replay_detector import (DynamicTimeWarpingComparator,  # noqa: E402
                                          ProcrustesPathComparator)

ROOT = Path(__file__).resolve().parent.parent
COLLECTION = ROOT / "data" / "interim" / "collection_20260806.jsonl"
SPLIT = ROOT / "data" / "metadata" / "collection_split_20260806.json"


def human_paths() -> list[np.ndarray]:
    """Real drags from the training people. Sealed people are never touched."""
    training = set(json.loads(SPLIT.read_text())["training_people"])
    out: list[np.ndarray] = []
    with COLLECTION.open() as f:
        for line in f:
            record = json.loads(line)
            if (record.get("participant_id") or "?").split("-")[0] not in training:
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
            out.extend(np.array([[e["x"], e["y"]] for e in d], dtype=float)
                       for d in split_drags(events) if move_count(d) >= MIN_MOVES_PER_DRAG)
    return out


def warp(path: np.ndarray, amplitude: float, rng: random.Random) -> np.ndarray:
    """Rigid transform plus a smooth bend — what reuse against a new target allows."""
    n = len(path)
    u = np.linspace(0, 1, n)
    offset = np.zeros((n, 2))
    for k in (1, 2, 3):
        for axis in (0, 1):
            offset[:, axis] += rng.gauss(0, amplitude / k) * np.sin(np.pi * k * u)
    theta = rng.uniform(0, 2 * np.pi)
    scale = rng.uniform(0.7, 1.4)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return (path + offset - path[0]) @ rot.T * scale + path[0]


def max_pair(paths: list[np.ndarray], comparator) -> float:
    """Most similar pair in the session — one reuse is enough to give the attacker away."""
    return max((comparator.similarity(paths[i], paths[j])
                for i in range(len(paths)) for j in range(i + 1, len(paths))), default=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drags-per-session", type=int, default=4)
    ap.add_argument("--trials", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    humans = human_paths()
    rng = random.Random(args.seed)
    n = args.drags_per_session
    print(f"사람 드래그 {len(humans)}개 · 세션당 {n}개\n")

    attacks = [("궤적 1개 재사용", 0.0, 1), ("변형 0.01", 0.01, 1),
               ("변형 0.02", 0.02, 1), ("라이브러리 5개", 0.01, 5)]

    for name, comparator in (("회전 불변", ProcrustesPathComparator()),
                             ("DTW (현재 배포)", DynamicTimeWarpingComparator())):
        innocent = np.array([max_pair([humans[rng.randrange(len(humans))] for _ in range(n)],
                                      comparator) for _ in range(args.trials + 100)])
        print(f"  {name}   (무고한 세션 중앙 {np.median(innocent):.4f})")
        print(f"    {'사람 오탐':>9s}{'문턱':>9s}" + "".join(f"{a[0]:>16s}" for a in attacks))
        for budget in (0.001, 0.01, 0.03):
            threshold = float(np.quantile(innocent, 1 - budget))
            cells = ""
            for _, amplitude, library in attacks:
                hits = []
                for _ in range(args.trials):
                    sources = [humans[rng.randrange(len(humans))] for _ in range(library)]
                    session = [warp(sources[rng.randrange(library)], amplitude, rng)
                               for _ in range(n)]
                    hits.append(max_pair(session, comparator) >= threshold)
                cells += f"{np.mean(hits) * 100:>15.1f}%"
            print(f"    {budget * 100:>8.1f}%{threshold:>9.4f}{cells}")
        print()

    print("  라이브러리 공격의 검출 상한 = 세션 안에서 같은 원본이 겹칠 확률")
    print(f"    {'라이브러리':>10s}{'상한':>9s}")
    for library in (1, 5, 10, 20, 50, 100):
        ceiling = 1.0 if library < n else 1 - prod((library - i) / library for i in range(n))
        print(f"    {library:>10d}{ceiling * 100:>8.1f}%")
    print("  → 방어의 성질이 아니라 우연의 확률이다. 세션을 넘는 이력이 있어야 올라간다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
