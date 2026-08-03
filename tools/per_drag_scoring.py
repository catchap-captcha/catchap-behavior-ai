"""Score a session per drag instead of as one blob, and see what changes.

Why
---
The model reads session-level aggregates — `event_count`, `duration_ms`,
`total_distance`. Drag an object five times and every one of them grows fivefold,
so on 2026-07-31 a ruler-straight path scored `human_probability 1.0000` while the
same path dragged once scored 0.0000. Interaction *scale* decides the verdict, and
scale costs an attacker one `for` loop.

Scoring each drag on its own removes that lever: five copies of a straight drag are
still five straight drags. This measures whether that actually holds on the real
main-captcha data, and what it costs on the human side.

The first thing it prints is a reproduction check — our session-level score against
the one production recorded. If those don't line up, nothing below means anything.

    .venv/bin/python tools/per_drag_scoring.py data/interim/main_captcha_raw_20260803.jsonl
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.feature_extractor_v23 import extract_features  # noqa: E402

MODEL_PATH = "models/candidate/revalidation_two_view_participant_safe_20260722/two_view_fusion.joblib"


def to_extractor_events(rows: list[dict]) -> list[dict]:
    """DB rows -> what the extractor reads: seq, event_type, t_ms, x, y."""
    if not rows:
        return []
    base = rows[0].get("client_timestamp_ms") or 0
    out = []
    for r in rows:
        # Pixels. The captcha puts 0..1 on the wire and the legacy training traces
        # are 0..1, so normalized looks like the obvious choice — but it reproduces
        # production far worse (61% of rows within 0.01, median error 2.4e-3) than
        # pixels does (82%, median 1.4e-5). Production scores pixels; measured, not
        # assumed.
        x, y = r.get("x_pixel"), r.get("y_pixel")
        if x is None or y is None:
            x, y = r.get("x_normalized") or 0, r.get("y_normalized") or 0
        out.append({
            "seq": r.get("seq"),
            "event_type": r.get("event_type"),
            "t_ms": float((r.get("client_timestamp_ms") or base) - base),
            "x": float(x),
            "y": float(y),
        })
    return out


def split_drags(events: list[dict]) -> list[list[dict]]:
    """One segment per pointerdown..pointerup. Moves outside a press are dropped —
    hovering is not a drag, and including it would smuggle the scale back in."""
    drags, current = [], None
    for e in events:
        kind = e["event_type"]
        if kind in ("pointerdown", "pointer_down", "drag_start"):
            current = [e]
        elif kind in ("pointerup", "pointer_up", "drag_end"):
            if current is not None:
                current.append(e)
                drags.append(current)
                current = None
        elif current is not None:
            current.append(e)
    if current and len(current) > 2:
        drags.append(current)          # unterminated drag: keep it, it still happened
    return drags


class Scorer:
    def __init__(self, path: str):
        bundle = joblib.load(path)
        self.models = bundle["models"]
        self.views = bundle["feature_views"]
        self.threshold = bundle["threshold"]

    def score(self, events: list[dict]) -> float | None:
        if len(events) < 3:
            return None
        feats = extract_features(events, None)
        probs = []
        for view, names in self.views.items():
            row = np.array([[float(feats.get(n) or 0.0) for n in names]])
            probs.append(float(self.models[view].predict_proba(row)[0][1]))
        return min(probs)               # the fusion the model was calibrated with


def main() -> None:
    src = sys.argv[1] if len(sys.argv) > 1 else "data/interim/main_captcha_raw_20260803.jsonl"
    scorer = Scorer(MODEL_PATH)
    rows = []

    for line in Path(src).read_text().splitlines():
        rec = json.loads(line)
        events = to_extractor_events(rec["events"])
        drags = split_drags(events)
        drag_scores = [s for s in (scorer.score(d) for d in drags) if s is not None]
        rows.append({
            "participant": rec.get("participant_id") or "",
            "label": rec.get("label"),
            "prod": rec.get("prod_human_probability"),
            "session": scorer.score(events),
            "n_drags": len(drags),
            "drag_min": min(drag_scores) if drag_scores else None,
            "drag_median": statistics.median(drag_scores) if drag_scores else None,
        })

    # --- reproduction check -------------------------------------------------
    pairs = [(r["prod"], r["session"]) for r in rows
             if r["prod"] is not None and r["session"] is not None]
    print(f"재현 검증 — 운영 점수와 비교 가능한 {len(pairs)}건")
    if pairs:
        diffs = [abs(float(p) - s) for p, s in pairs]
        close = sum(1 for d in diffs if d < 0.01)
        print(f"  차이 0.01 미만 {close}/{len(pairs)}건 ({close/len(pairs)*100:.1f}%) · "
              f"중앙 {statistics.median(diffs):.6f} · 최대 {max(diffs):.6f}")
    # Everything below is computed only on rows we can reproduce. A row we cannot
    # reproduce is one where our pipeline and production disagree about the input,
    # so its per-drag score would be answering a different question than its
    # session score — the comparison would be meaningless, not merely noisy.
    before = len(rows)
    rows = [r for r in rows
            if r["prod"] is not None and r["session"] is not None
            and abs(float(r["prod"]) - r["session"]) < 0.01]
    print(f"  재현되는 {len(rows)}건만 사용 ({before - len(rows)}건 제외)")

    th = scorer.threshold
    def summarize(title: str, subset: list[dict]) -> None:
        if not subset:
            return
        print(f"\n{title}  n={len(subset)}")
        print(f"  {'드래그':>5s}{'n':>5s}{'세션점수':>11s}{'통과':>6s}"
              f"{'드래그min':>11s}{'통과':>6s}{'드래그중앙':>11s}{'통과':>6s}")
        by_drags = defaultdict(list)
        for r in subset:
            by_drags[min(r["n_drags"], 6)].append(r)
        for k in sorted(by_drags):
            g = [r for r in by_drags[k] if r["session"] is not None and r["drag_min"] is not None]
            if not g:
                continue
            se = [r["session"] for r in g]
            dm = [r["drag_min"] for r in g]
            dd = [r["drag_median"] for r in g]
            print(f"  {k:>5d}{len(g):>5d}{statistics.mean(se):>11.4f}"
                  f"{sum(1 for v in se if v >= th):>6d}"
                  f"{statistics.mean(dm):>11.4f}{sum(1 for v in dm if v >= th):>6d}"
                  f"{statistics.mean(dd):>11.4f}{sum(1 for v in dd if v >= th):>6d}")

    summarize("사람 (label=human)", [r for r in rows if r["label"] == "human"])
    summarize("봇 (label=bot)", [r for r in rows if r["label"] == "bot"])
    summarize("Playwright 봇", [r for r in rows if r["participant"].startswith("pwbot-")])
    summarize("라벨 없음", [r for r in rows if r["label"] not in ("human", "bot")])

    out = Path("reports/per_drag_scoring_20260803.jsonl")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(f"\n행 {len(rows)}건 -> {out}")


if __name__ == "__main__":
    main()
