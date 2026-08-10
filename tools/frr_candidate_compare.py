"""Per-participant FRR for every candidate, each in its own coordinate space.

Why this exists
---------------
2026-08-06: the red-team re-measurement changed which candidate looks best
(patched_weakset 6.6% < scale_pixel 7.1% < scale_normalized 10.8%), but the
human side was still a 120-row legacy slice at the stored threshold with no
per-participant calibration. A defence is two numbers, not one; choosing on
evasion alone is how the 07-22 hard-negative candidate got as far as it did.

Deliberately does NOT touch the sealed holdout. `lockbox_audit` marks
`human_holdout` (h2219) consumed by revalidation_two_view_participant_safe, and
rule 1 says a holdout is not opened while a retrain is planned. Development
comparison, not promotion evidence.

Two scoring units, because they need different operating points
---------------------------------------------------------------
* session — features over the whole attempt, compared to the model's own
  threshold. This is what production does today.
* per-drag — median over drags, compared to a per-drag operating point. The
  model threshold does NOT transfer here: drag medians sit far below session
  scores, and reusing 0.99995 rejects ~60% of real humans.

The per-drag point is fitted on one participant and reported on the other, then
swapped. Choosing and measuring on the same set is exactly what made the 08-03
"1.5%" meaningless and what killed the 07-22 candidate (OOF 2.39%, test 9.06%).

⚠️  Do not read these numbers next to `redteam_evolution_search` output
------------------------------------------------------------------------
This tool REFITS an operating point; the red-team tool uses the model's STORED
threshold. Two axes measured at two different thresholds do not form a trade-off
curve, and on 2026-08-06 that produced a table where `drag_unit` looked excellent
on FRR (1.6%) and hopeless on evasion (92.5%) purely because its stored threshold
was 0.0122 while `scale_pixel`'s was 0.99979. Raising a threshold moves both
numbers at once. Before comparing candidates, calibrate them to a common budget
and re-measure both axes there.

    .venv/bin/python tools/frr_candidate_compare.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.feature_extractor_v23 import extract_features  # noqa: E402
from tools.per_drag_scoring import split_drags  # noqa: E402

HUMANS = Path("data/interim/main_captcha_raw_20260803b.jsonl")
MIN_MOVES_PER_DRAG = 2
TARGET_FRR = 0.02              # promotion criterion, fixed 2026-07-30
# Not people. `zzprobe` is 하지영's embed check (2026-08-06 11:46 KST, lecture
# EMBED-CHECK) — the iframe loaded but nobody solved anything, so it is a row
# with no trajectory behind it. Named so it sorts last and reads as non-human.
BOT_CODE_MARKERS = ("pwbot", "rtbot", "botprobe", "probe", "signalcheck", "zzprobe")

# (directory, coordinate space) — space established empirically 2026-08-06 by
# scoring human traces both ways; see RESULT_REDTEAM_REMEASURE_20260806.md.
CANDIDATES = [
    ("revalidation_two_view_participant_safe_20260722", "pixel"),
    ("scale_pixel_20260804", "pixel"),
    ("patched_weakset_iter1_20260804", "pixel"),
    ("scale_normalized_20260804", "normalized"),
    ("surface_aware_20260806", "pixel"),
    ("drag_unit_20260806", "pixel"),
    ("drag_unit_frr5_20260806", "pixel"),
    ("scale_aug_20260806", "pixel"),
]


def person_of(code: str) -> str:
    """sw-mouse and sw-mouse-v2 are one person. Splitting on the code inflated
    generalisation on 08-03; the criteria are per person, not per code."""
    return code.split("-")[0]


def to_events(rows: list[dict], space: str) -> list[dict]:
    if not rows:
        return []
    base = rows[0].get("client_timestamp_ms") or 0
    out = []
    for r in rows:
        if space == "normalized":
            x, y = r.get("x_normalized"), r.get("y_normalized")
        else:
            x, y = r.get("x_pixel"), r.get("y_pixel")
        if x is None or y is None:
            continue
        out.append({
            "seq": r.get("seq"),
            "event_type": r.get("event_type"),
            "t_ms": float((r.get("client_timestamp_ms") or base) - base),
            "x": float(x),
            "y": float(y),
        })
    return out


class Scorer:
    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.models = bundle["models"]
        self.views = bundle["feature_views"]
        self.threshold = float(bundle["threshold"])

    def _score(self, events: list[dict]) -> float | None:
        if len(events) < 3:
            return None
        feats = extract_features(events, None)
        probs = []
        for view, names in self.views.items():
            row = np.array([[float(feats.get(n) or 0.0) for n in names]])
            probs.append(float(self.models[view].predict_proba(row)[0][1]))
        return min(probs)

    def session(self, events: list[dict]) -> float:
        value = self._score(events)
        return 0.0 if value is None else value

    def per_drag(self, events: list[dict]) -> float:
        """Median over drags. The move floor applies to the SESSION, not each drag:
        per-drag application pushed FRR 1.5% -> 5.2% because 15/166 human sessions
        contain one short drag while none consist only of short drags."""
        scores = []
        for drag in split_drags(events):
            moves = sum(1 for e in drag if e["event_type"] in ("pointermove", "pointer_move"))
            if moves < MIN_MOVES_PER_DRAG:
                continue
            value = self._score(drag)
            if value is not None:
                scores.append(value)
        if not scores:
            return 0.0                 # every drag starved -> treated as bot
        return float(np.median(scores))


def load_humans(path: Path, keep: set[str] | None) -> dict[str, list[dict]]:
    by_person: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            code = record.get("participant_id")
            if not code or record.get("label") == "bot":
                continue
            if any(marker in code.lower() for marker in BOT_CODE_MARKERS):
                continue
            person = person_of(code)
            if keep is not None and person not in keep:
                continue          # sealed people never get scored, not even by accident
            by_person[person].append(record)
    return by_person


def fitted_point(scores: list[float]) -> float:
    """Largest threshold that keeps FRR at or under target on THIS participant."""
    if not scores:
        return 0.0
    ordered = sorted(scores)
    index = int(len(ordered) * TARGET_FRR)
    return ordered[max(index - 1, 0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=HUMANS)
    ap.add_argument("--people", nargs="*", default=None,
                    help="score only these people; sealed people must be left out")
    args = ap.parse_args()

    keep = set(args.people) if args.people else None
    by_person = load_humans(args.data, keep)
    print(f"사람 데이터 {args.data.name}")
    for person, records in sorted(by_person.items()):
        print(f"  {person:10s} {len(records):4d}세션")
    people = sorted(by_person)
    if keep:
        print(f"  (봉인 제외: {', '.join(sorted(keep))} 만 채점)")
    print()

    for name, space in CANDIDATES:
        path = Path("models/candidate") / name / "two_view_fusion.joblib"
        if not path.exists():
            print(f"{name} — 모델 없음")
            continue
        scorer = Scorer(path)

        session_scores, drag_scores = {}, {}
        for person, records in by_person.items():
            events = [to_events(r.get("events") or [], space) for r in records]
            session_scores[person] = [scorer.session(e) for e in events]
            drag_scores[person] = [scorer.per_drag(e) for e in events]

        print(f"{name}  ({space})")

        # session unit — production today, model's own threshold
        total = sum(len(v) for v in session_scores.values())
        bad = sum(sum(1 for s in v if s < scorer.threshold) for v in session_scores.values())
        detail = "  ".join(
            f"{p} {sum(1 for s in session_scores[p] if s < scorer.threshold)/len(session_scores[p])*100:.1f}%"
            for p in people)
        print(f"  세션 단위 (임계 {scorer.threshold:.5f})   전체 {bad/max(total,1)*100:5.1f}%   {detail}")

        # per-drag unit — leave one person out. The operating point is fitted on
        # everyone else pooled and reported on the person it never saw, which is
        # the only shape that answers "what happens to the next user".
        if len(people) >= 3:
            worst = 0.0
            rejected = total = 0
            for held_out in people:
                pool = [s for p in people if p != held_out for s in drag_scores[p]]
                point = fitted_point(pool)
                values = drag_scores[held_out]
                bad = sum(1 for s in values if s < point)
                frr = bad / max(len(values), 1)
                worst = max(worst, frr)
                rejected += bad
                total += len(values)
                print(f"  드래그 단위   {held_out} 를 빼고 맞춘 뒤 {held_out} 에서 재면"
                      f" {frr*100:5.1f}%  ({bad}/{len(values)}, 점 {point:.4f})")
            overall = rejected / max(total, 1)
            gates = [("전체 ≤2%", overall <= 0.02), ("최악 참여자 ≤5%", worst <= 0.05)]
            verdict = "  ".join(f"{n} {'통과' if ok else '미달'}" for n, ok in gates)
            print(f"  → 전체 {overall*100:.1f}% ({rejected}/{total}) · 최악 참여자 "
                  f"{worst*100:.1f}%   {verdict}")
        elif len(people) == 2:
            for fit_on in people:
                other = [p for p in people if p != fit_on][0]
                point = fitted_point(drag_scores[fit_on])
                values = drag_scores[other]
                frr = sum(1 for s in values if s < point) / max(len(values), 1)
                print(f"  드래그 단위   {fit_on} 에서 맞추고 {other} 에서 재면 {frr*100:5.1f}%")
        print()

    print("  승급 기준  실험 후보 ≤3% · Production ≤1% · 최악 참여자 ≤5%")
    print("  ⚠️  개발용 비교. 실질 참여자 2명이고 봉인 홀드아웃은 열지 않았다.")


if __name__ == "__main__":
    main()
