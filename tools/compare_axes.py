"""Measure three defence axes on one window, then compare every combination.

The axes
--------
    veto         density: is this trace outside where humans live
    consistency  are these attempts the same hand, or a pool of borrowed ones
    replay       do the attempts in this window repeat each other

They fail differently, which is the only reason combining them is interesting.
The veto took the worst holdout family from 100% to 58.6% and left `replay_warp`
and `adversarial_replay` behind — both are warped real human motion, so they sit
*inside* the human region and a density model has nothing to say about them.
Consistency and replay both claim to speak to exactly that case.

Why every combination gets its own operating point
--------------------------------------------------
Each axis spends false rejections. On 08-06 two axes were calibrated separately
at 3% each, the combination spent 6.3%, and the extra budget looked like a 48.9%
result that was really 61.5%. So the combined score is calibrated as one score,
once, on `tune` humans the models never trained on.

Judging is per window, not per attempt: consistency and replay do not exist for a
single attempt, and farming a lecture is many attempts by construction.

    .venv/bin/python tools/compare_axes.py --window 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags  # noqa: E402
from app.services.feature_extractor_v23 import extract_features  # noqa: E402
from app.services.replay_detector import DynamicTimeWarpingComparator, path_from_events  # noqa: E402
from tools.train_density_veto import DensityVeto  # noqa: E402,F401
from tools.score_veto_holdouts import (Scorer, eligible_holdouts, row_ids,  # noqa: E402
                                       tune_sessions)

sys.modules["__main__"].DensityVeto = DensityVeto

ROOT = Path(__file__).resolve().parent.parent
SPLIT = ROOT / "data" / "metadata" / "collection_split_20260806.json"
TRAINING_SETS = ("data/interim/bot_features_v23corr_20260722.jsonl",
                 "data/interim/human_features_v23corr_20260722.jsonl")


def feature_vector(events: list[dict], names: tuple[str, ...]) -> np.ndarray:
    drags = [d for d in split_drags(events) if move_count(d) >= MIN_MOVES_PER_DRAG]
    if not drags:
        return np.zeros(len(names))
    rows = [[float(extract_features(d, None).get(n) or 0.0) for n in names]
            for d in drags]
    return np.nan_to_num(np.mean(np.asarray(rows, dtype=float), axis=0))


class Axes:
    """Per-window scores for the three axes. Each returns 1.0 = looks human."""

    def __init__(self, names: tuple[str, ...], windows: list,
                 vectors: np.ndarray) -> None:
        self.names = names
        self.mean = vectors.mean(axis=0)
        std = vectors.std(axis=0)
        self.std = np.where(std > 0, std, 1.0)
        spreads = [self._spread(vectors[list(w)]) for w in windows]
        # Two-sided: replaying ONE trace collapses the spread, which no person
        # does either. A one-sided test hands that counter over for free.
        self.low = float(np.percentile(spreads, 1.5))
        self.high = float(np.percentile(spreads, 98.5))
        self.comparator = DynamicTimeWarpingComparator()

    def _spread(self, vectors: np.ndarray) -> float:
        z = (vectors - self.mean) / self.std
        return float(np.mean(np.std(z, axis=0)))

    def consistency(self, vectors: np.ndarray) -> float:
        s = self._spread(vectors)
        if self.low <= s <= self.high:
            return 1.0
        return float(max(0.0, s / self.low if s < self.low else self.high / s))

    def replay(self, paths: list) -> float:
        """1.0 when the window's traces are as distinct as a person's are."""
        best = 0.0
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                if paths[i].shape[0] < 3 or paths[j].shape[0] < 3:
                    continue
                best = max(best, float(self.comparator.similarity(paths[i], paths[j])))
        return 1.0 - best


def collect(events_list: list, scorer: Scorer, names: tuple[str, ...]):
    scores, vectors, paths = [], [], []
    for events in events_list:
        scores.append(scorer.session(events))
        vectors.append(feature_vector(events, names))
        paths.append(path_from_events(events))
    return scores, np.asarray(vectors), paths


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/candidate/density_veto_20260808")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--target-frr", type=float, default=0.03)
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--trials", type=int, default=800)
    args = ap.parse_args()

    split = json.loads(SPLIT.read_text())
    tune_people = {"jy"}
    if tune_people & set(split["holdout_people"]):
        raise SystemExit("동작점을 봉인 사람에게서 잡으려 한다")

    scorer = Scorer(Path(args.model))
    names = tuple(scorer.veto_names)
    rng = np.random.default_rng(20260808)

    h_scores, h_vectors, h_paths = collect(tune_sessions(tune_people), scorer, names)
    print(f"{Path(args.model).name} · 창 {args.window} · tune {len(h_scores)}세션",
          flush=True)

    windows = [rng.choice(len(h_scores), args.window, replace=False)
               for _ in range(args.trials)]
    axes = Axes(names, windows, h_vectors)
    print(f"일관성 사람 띠 [{axes.low:.3f}, {axes.high:.3f}]\n", flush=True)

    trained: set[str] = set()
    for p in TRAINING_SETS:
        path = ROOT / p
        if path.exists():
            trained |= row_ids(path)

    families = {}
    for name, path in eligible_holdouts(trained):
        events_list = []
        with path.open() as f:
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                ev = json.loads(line).get("events") or []
                if ev:
                    events_list.append(ev)
        if len(events_list) >= args.window:
            families[name] = collect(events_list, scorer, names)
            print(f"  준비 {name[:44]:46s}{len(events_list):>5d}", flush=True)

    combos = {"veto": ("v",), "veto+일관성": ("v", "c"),
              "veto+재생": ("v", "r"), "veto+일관성+재생": ("v", "c", "r")}

    def window_scores(scores, vectors, paths, use, n):
        out = []
        for _ in range(n):
            idx = rng.choice(len(scores), args.window, replace=False)
            parts = [float(np.median([scores[i] for i in idx]))]
            if "c" in use:
                parts.append(axes.consistency(vectors[idx]))
            if "r" in use:
                parts.append(axes.replay([paths[i] for i in idx]))
            out.append(min(parts))
        return np.asarray(out)

    short = {k: k.split("_")[0][:11] for k in families}
    print(f"\n{'결합':18s}" + "".join(f"{short[k]:>12s}" for k in families) +
          f"{'최악':>8s}{'오탐':>7s}", flush=True)
    best = None
    for label, use in combos.items():
        h = window_scores(h_scores, h_vectors, h_paths, use, args.trials)
        ordered = np.sort(h)
        point = float(ordered[min(int(len(ordered) * args.target_frr),
                                  len(ordered) - 1)])
        frr = float((h < point).mean())
        cells, worst = [], 0.0
        for key, (s, v, p) in families.items():
            asr = float((window_scores(s, v, p, use, args.trials) >= point).mean())
            worst = max(worst, asr)
            cells.append(asr)
        print(f"{label:18s}" + "".join(f"{c*100:11.1f}%" for c in cells) +
              f"{worst*100:7.1f}%{frr*100:6.1f}%", flush=True)
        if best is None or worst < best[1]:
            best = (label, worst)

    label, worst = best
    print(f"\n  최선 {label} · 최악 계열 {worst*100:.1f}%")
    for checkpoints in (5, 10, 20):
        attempts = float("inf") if worst <= 0 else 1.0 / (worst ** checkpoints)
        cell = f"{attempts:,.0f}" if attempts < 1e13 else "사실상 불가"
        print(f"    체크포인트 {checkpoints:2d}개 → 강의 완주 기대 시도수 {cell}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
