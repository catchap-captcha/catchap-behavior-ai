"""Score a veto-equipped candidate on the real holdout families, at one point.

The rebuild's numbers came from four generators written in a day, and one of them
was wrong in a way that mattered: an emulated "browser automation" family scored
0.0% while the actual Playwright captures in this repo scored 100.0%. Emulating
Chromium's event loop is not the same as running it. So the verdict has to come
from the holdouts that were captured rather than invented.

Only sets whose manifest says `external_holdout_only` are eligible, and each one
is checked for overlap with the training rows before it is scored — three files
in this directory were labelled holdouts while sitting 100% inside training, and
`lockbox_audit` also found three genuine holdouts that had been demoted because
their `training_usage` was simply absent.

Consumed holdouts are refused. Scoring a spent set produces a number that looks
like evidence and is not.

    .venv/bin/python tools/score_veto_holdouts.py \
        --model models/candidate/density_veto_20260808 --target-frr 0.03
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags  # noqa: E402
from app.services.feature_extractor_v23 import extract_features  # noqa: E402
from tools.train_density_veto import DensityVeto  # noqa: E402,F401

# joblib stored the class as __main__.DensityVeto because train_density_veto ran
# as a script. Bind it here so the bundle loads from any entry point.
sys.modules["__main__"].DensityVeto = DensityVeto

ROOT = Path(__file__).resolve().parent.parent
COLLECTION = ROOT / "data" / "interim" / "collection_20260806.jsonl"
SPLIT = ROOT / "data" / "metadata" / "collection_split_20260806.json"
TRAINING_SETS = ("data/interim/bot_features_v23corr_20260722.jsonl",
                 "data/interim/human_features_v23corr_20260722.jsonl")


def row_ids(path: Path) -> set[str]:
    out = set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = r.get("attempt_id") or r.get("session_id")
            if key:
                out.add(key)
    return out


class Scorer:
    def __init__(self, path: Path) -> None:
        self.b = joblib.load(path / "two_view_fusion.joblib")
        self.veto = self.b.get("density_veto")
        self.veto_names = self.b.get("density_feature_names")
        self.veto_below = self.b.get("veto_below")

    def _one(self, events: list[dict]) -> float:
        feats = extract_features(events, None)
        probs = []
        for view, names in self.b["feature_views"].items():
            row = np.array([[float(feats.get(n) or 0.0) for n in names]])
            probs.append(float(self.b["models"][view].predict_proba(
                np.nan_to_num(row))[0][1]))
        score = min(probs)
        if self.veto is not None:
            vec = np.array([[float(feats.get(n) or 0.0) for n in self.veto_names]])
            if float(self.veto.score(np.nan_to_num(vec))[0]) < self.veto_below:
                return 0.0
        return score

    def session(self, events: list[dict]) -> float:
        drags = [d for d in split_drags(events) if move_count(d) >= MIN_MOVES_PER_DRAG]
        if not drags:
            return 0.0
        return statistics.median(self._one(d) for d in drags)


def tune_sessions(people: set[str]) -> list[list[dict]]:
    out = []
    with COLLECTION.open() as f:
        for line in f:
            record = json.loads(line)
            code = record.get("participant_id") or ""
            if code.split("-")[0] not in people:
                continue
            if record.get("quality_status") != "valid":
                continue
            rows = record.get("events") or []
            if not rows:
                continue
            base = rows[0].get("client_timestamp_ms") or 0
            out.append([{
                "seq": r.get("seq"), "event_type": r.get("event_type"),
                "t_ms": float((r.get("client_timestamp_ms") or base) - base),
                "x": float(r["x_pixel"]), "y": float(r["y_pixel"]),
            } for r in rows if r.get("x_pixel") is not None])
    return out


def eligible_holdouts(trained: set[str]) -> list[tuple[str, Path]]:
    out = []
    for manifest in sorted((ROOT / "data").rglob("*.manifest.json")):
        try:
            doc = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            continue
        if doc.get("training_usage") != "external_holdout_only":
            continue
        if doc.get("evaluation_consumed"):
            continue                       # spent: not evidence any more
        data = manifest.parent / manifest.name.replace(".manifest.json", "")
        if not data.exists():
            continue
        own = row_ids(data)
        if own and trained and len(own & trained) / len(own) > 0:
            continue                       # labelled holdout, actually training
        out.append((manifest.name.replace(".jsonl.manifest.json", ""), data))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--target-frr", type=float, default=0.03)
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    split = json.loads(SPLIT.read_text())
    # The deployed model trained on legacy humans; the collection people were
    # never in its training set, so any of them can set the point. Sealed stay out.
    tune_people = {"jy"}
    if tune_people & set(split["holdout_people"]):
        raise SystemExit("동작점을 봉인 사람에게서 잡으려 한다")

    scorer = Scorer(args.model)
    human = [scorer.session(e) for e in tune_sessions(tune_people)]
    ordered = sorted(human)
    index = min(int(len(ordered) * args.target_frr), len(ordered) - 1)
    point = float(ordered[index])
    frr = float(np.mean(np.asarray(human) < point))
    print(f"{args.model.name} · 거부권 {'있음' if scorer.veto is not None else '없음'}")
    print(f"동작점 {point:.8f} · tune={sorted(tune_people)} {len(human)}세션 "
          f"· 오탐 {frr:.1%}\n")

    trained: set[str] = set()
    for p in TRAINING_SETS:
        path = ROOT / p
        if path.exists():
            trained |= row_ids(path)

    print(f"  {'봇 홀드아웃 계열':46s}{'행':>6s}{'통과율':>9s}")
    worst = 0.0
    for name, path in eligible_holdouts(trained):
        scores = []
        with path.open() as f:
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                record = json.loads(line)
                events = record.get("events") or []
                if events:
                    scores.append(scorer.session(events))
        if not scores:
            print(f"  {name[:46]:46s}     0   드래그 없음")
            continue
        asr = float(np.mean(np.asarray(scores) >= point))
        worst = max(worst, asr)
        print(f"  {name[:46]:46s}{len(scores):>6d}{asr*100:8.1f}%")

    print(f"\n  최악 계열 {worst*100:.1f}%")
    for checkpoints in (5, 10, 20):
        attempts = float("inf") if worst <= 0 else 1.0 / (worst ** checkpoints)
        cell = f"{attempts:,.0f}" if attempts < 1e13 else "사실상 불가"
        print(f"    체크포인트 {checkpoints:2d}개 → 강의 완주 기대 시도수 {cell}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
