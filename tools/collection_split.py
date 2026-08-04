"""Decide who is training data and who is the holdout — before anyone sees a score.

Why this is a script and not a decision
---------------------------------------
The legacy human data is fully spent: 15,802 rows went to training, and both
sealed sets (10 participants, then 7) have been scored. There is no human data
left to evaluate any future candidate on. The next collection is the only chance
to build one, and a holdout chosen after looking at results is not a holdout.

So the rule is fixed here, in advance, and it has no free parameters at decision
time: hash each participant code with a salt published before the participant
list exists, sort ascending, and the first N are sealed. Whoever runs it gets the
same answer, and nobody — including me — can steer it.

The salt is the commit date of this file. It is written down so that changing it
later shows up as a diff.

    .venv/bin/python tools/collection_split.py jy-mouse ms-mouse my-mouse ...
    .venv/bin/python tools/collection_split.py --holdout 3 jy ms my th sw ...
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Fixed 2026-08-04, before any participant list existed. Changing it changes
# every assignment, which is why it lives in version control rather than a flag.
SALT = "catchap-collection-split-20260804"

MIN_PEOPLE = 8
MIN_SESSIONS_PER_PERSON_PER_DEVICE = 40


def person_of(code: str) -> str:
    """`jy-mouse` and `jy-trackpad` are one person and must not be split apart."""
    return code.split("-")[0]


def rank(person: str) -> str:
    return hashlib.sha256(f"{SALT}:{person}".encode()).hexdigest()


def split(codes: list[str], holdout_people: int) -> dict:
    people = sorted({person_of(c) for c in codes})
    ordered = sorted(people, key=rank)
    holdout = set(ordered[:holdout_people])
    return {
        "salt": SALT,
        "people": len(people),
        "holdout_people": sorted(holdout),
        "training_people": [p for p in ordered if p not in holdout],
        "assignment": {
            c: ("holdout" if person_of(c) in holdout else "training") for c in sorted(codes)
        },
        "ranks": {p: rank(p)[:12] for p in ordered},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="+", help="participant codes, e.g. jy-mouse jy-trackpad")
    ap.add_argument("--holdout", type=int, default=3, help="people to seal (default 3)")
    ap.add_argument("--write", type=Path, help="write the assignment to this path")
    args = ap.parse_args()

    result = split(args.codes, args.holdout)
    people = result["people"]

    print(f"참여자 {people}명 · 봉인 {len(result['holdout_people'])}명 "
          f"· 학습 {len(result['training_people'])}명\n")
    print(f"  {'참여자 코드':22s}{'배정':10s}")
    for code, side in result["assignment"].items():
        print(f"  {code:22s}{'봉인 (평가용)' if side == 'holdout' else '학습'}")

    print(f"\n  해시 순서 (salt={SALT})")
    for p, h in result["ranks"].items():
        mark = "봉인" if p in result["holdout_people"] else ""
        print(f"    {h}  {p:12s}{mark}")

    if people < MIN_PEOPLE:
        print(f"\n  ⚠️  {people}명은 부족하다. 최소 {MIN_PEOPLE}명이어야 학습 5~6 / 봉인 2~3 이 된다.")
        print(f"      8/03 측정: 사람 2명으로 재학습하면 처음 보는 사람 오탐이 10~22% 로 나빠진다.")

    if args.write:
        args.write.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(f"\n  기록 -> {args.write}")


if __name__ == "__main__":
    sys.exit(main())
