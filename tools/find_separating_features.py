"""Find what the 55 features miss about evolved attacks.

The dead end this starts from
-----------------------------
The evolutionary red team lands traces scoring 0.9965 on a model whose humans sit
between 0.5 and 1.0. That is not a threshold problem — the two populations
overlap, so every operating point either passes those bots or blocks real people.
Moving the line cannot fix a line that has attackers on both sides of it.

So the question is not "where is the line" but "what does the attacker fail to
copy". Anything the generator does not vary is a candidate.

    BOUNDS = curvature · jitter · time_power · duration_scale · turn_slowdown
             coalesce_fraction · frame_ms · late_correction_probability

Every one of those shapes the PATH or scales time smoothly. None of them touches
the fine structure of the clock: `frame_ms` is drawn from {8,10,12,16,20} and then
used as a fixed step, so synthetic inter-event gaps collapse onto a few values.
A real mouse polls against a display that is not in lockstep with the hand, so
its gaps smear. That asymmetry costs the attacker nothing to fix once they know,
which is exactly why it must be measured before it is trusted.

Method: score every candidate statistic by AUC between real human drags and
generated attack drags, on the same drag segmentation the service uses. AUC 0.5
is useless, 1.0 is perfect separation. Reported next to the 55 shipped features
so a new signal is only interesting if it beats what we already have.

    .venv/bin/python tools/find_separating_features.py --attacks 600
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags  # noqa: E402
from app.services.feature_extractor_v23 import extract_features  # noqa: E402
from training.generate_hybrid_redteam_bots import MotionPolicy, _motion_events  # noqa: E402
from tools.redteam_evolution_search import BOUNDS, FRAME_CHOICES, load_bases, random_policy  # noqa: E402

COLLECTION = Path("data/interim/collection_20260806.jsonl")
SEALED = {"sw", "ms"}


def collection_drags(limit: int) -> list[list[dict]]:
    out = []
    with COLLECTION.open() as f:
        for line in f:
            record = json.loads(line)
            code = record.get("participant_id") or ""
            if code.split("-")[0] in SEALED:
                continue                      # sealed people are not a lab bench
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
            } for r in rows if r.get("x_pixel") is not None and r.get("y_pixel") is not None]
            for drag in split_drags(events):
                if move_count(drag) >= MIN_MOVES_PER_DRAG:
                    out.append(drag)
            if len(out) >= limit:
                break
    return out[:limit]


def main_captcha_bases(limit: int) -> list[dict]:
    """Attack substrate shaped like the surface we actually deploy on.

    The evolution search warps LEGACY traces — ~340 events per session. Main
    captcha drags are ~12. Warping the wrong substrate produces attacks that a
    single feature (`event_count`, AUC 1.000) separates perfectly, which makes
    the defence look strong for a reason no real attacker would hand us.
    """
    out = []
    with COLLECTION.open() as f:
        for line in f:
            record = json.loads(line)
            code = record.get("participant_id") or ""
            if code.split("-")[0] in SEALED or record.get("quality_status") != "valid":
                continue
            rows = record.get("events") or []
            events = [{
                "seq": r.get("seq"), "event_type": r.get("event_type"),
                "t_ms": float(r.get("client_timestamp_ms") or 0),
                "x": float(r["x_pixel"]), "y": float(r["y_pixel"]),
                "x_normalized": r.get("x_normalized"), "y_normalized": r.get("y_normalized"),
            } for r in rows if r.get("x_pixel") is not None]
            if len(events) >= 8:
                out.append({"events": events,
                            "captcha": {"width": record.get("stage_width") or 500,
                                        "height": record.get("stage_height") or 375}})
            if len(out) >= limit:
                break
    return out


def attack_drags(count: int, seed: int, surface: str) -> list[list[dict]]:
    """Traces from the same generator the evolution search optimises over."""
    rng = random.Random(seed)
    bases = load_bases(400) if surface == "legacy" else main_captcha_bases(300)
    if not bases:
        raise SystemExit("공격 기반 궤적이 없다")
    out = []
    guard = 0
    while len(out) < count and guard < count * 50:
        guard += 1
        policy = random_policy(rng)
        base = rng.choice(bases)
        captcha = base.get("captcha") or {}
        events, _meta = _motion_events(
            base["events"], width=int(captcha.get("width") or 500),
            height=int(captcha.get("height") or 320), policy=policy, randomizer=rng)
        for drag in split_drags(events):
            if move_count(drag) >= MIN_MOVES_PER_DRAG:
                out.append(drag)
    return out[:count]


# ---- candidate statistics the generator does not vary -----------------------

def _dt(drag: list[dict]) -> np.ndarray:
    t = np.array([float(e["t_ms"]) for e in drag])
    return np.diff(np.sort(t))


def dt_distinct_ratio(drag) -> float:
    """How many distinct gap values, per event. A fixed frame step collapses this."""
    d = _dt(drag)
    if d.size == 0:
        return 0.0
    return len(set(np.round(d, 3).tolist())) / d.size


def dt_entropy(drag) -> float:
    d = _dt(drag)
    if d.size < 2:
        return 0.0
    counts = np.unique(np.round(d, 1), return_counts=True)[1]
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(p))) if len(p) > 1 else 0.0


def dt_cv(drag) -> float:
    """Coefficient of variation of the gaps — a clock's jitter, not the hand's."""
    d = _dt(drag)
    if d.size < 2 or d.mean() == 0:
        return 0.0
    return float(d.std() / d.mean())


def dt_mode_share(drag) -> float:
    d = np.round(_dt(drag), 1)
    if d.size == 0:
        return 0.0
    values, counts = np.unique(d, return_counts=True)
    return float(counts.max() / d.size)


def speed_autocorr(drag) -> float:
    """Lag-1 correlation of speed. A smooth eased path is far more predictable."""
    t = np.array([float(e["t_ms"]) for e in drag])
    p = np.array([[float(e["x"]), float(e["y"])] for e in drag])
    dt = np.diff(t)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1)
    v = np.divide(step, dt, out=np.zeros_like(step), where=dt > 0)
    if v.size < 3 or v.std() == 0:
        return 0.0
    a, b = v[:-1], v[1:]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def step_quantisation(drag) -> float:
    """Fraction of moves whose pixel step repeats an earlier one exactly."""
    p = np.array([[float(e["x"]), float(e["y"])] for e in drag])
    steps = np.round(np.diff(p, axis=0), 3)
    if steps.shape[0] == 0:
        return 0.0
    seen = {tuple(s) for s in steps.tolist()}
    return 1.0 - len(seen) / steps.shape[0]


CANDIDATES = {
    "dt_distinct_ratio": dt_distinct_ratio,
    "dt_entropy": dt_entropy,
    "dt_cv": dt_cv,
    "dt_mode_share": dt_mode_share,
    "speed_autocorr": speed_autocorr,
    "step_quantisation": step_quantisation,
}


def auc(human: np.ndarray, bot: np.ndarray) -> float:
    """Rank AUC, folded so 0.5 is useless and 1.0 is perfect either direction."""
    values = np.concatenate([human, bot])
    order = values.argsort().argsort().astype(float) + 1
    n_h, n_b = len(human), len(bot)
    rank_sum = order[:n_h].sum()
    a = (rank_sum - n_h * (n_h + 1) / 2) / (n_h * n_b)
    return max(a, 1 - a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", type=int, default=600)
    ap.add_argument("--humans", type=int, default=600)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--surface", choices=("legacy","main"), default="main")
    args = ap.parse_args()

    humans = collection_drags(args.humans)
    bots = attack_drags(args.attacks, args.seed, args.surface)
    print(f"사람 드래그 {len(humans)}개 · 공격 드래그 {len(bots)}개 (기반 {args.surface})\n")

    rows = []
    for name, fn in CANDIDATES.items():
        h = np.array([fn(d) for d in humans], dtype=float)
        b = np.array([fn(d) for d in bots], dtype=float)
        rows.append((name, auc(h, b), float(np.median(h)), float(np.median(b)), "새 후보"))

    shipped_h = [extract_features(d, None) for d in humans]
    shipped_b = [extract_features(d, None) for d in bots]
    for name in shipped_h[0]:
        h = np.array([float(f.get(name) or 0.0) for f in shipped_h])
        b = np.array([float(f.get(name) or 0.0) for f in shipped_b])
        if h.std() == 0 and b.std() == 0:
            continue
        rows.append((name, auc(h, b), float(np.median(h)), float(np.median(b)), "현재 55개"))

    rows.sort(key=lambda r: -r[1])
    print(f"  {'특징':28s}{'AUC':>7s}{'사람 중앙':>12s}{'봇 중앙':>12s}   출처")
    for name, score, mh, mb, source in rows[:22]:
        print(f"  {name:28s}{score:7.3f}{mh:12.4f}{mb:12.4f}   {source}")

    best_new = max((r for r in rows if r[4] == "새 후보"), key=lambda r: r[1])
    best_old = max((r for r in rows if r[4] == "현재 55개"), key=lambda r: r[1])
    print(f"\n  새 후보 최고   {best_new[0]} AUC {best_new[1]:.3f}")
    print(f"  현재 최고      {best_old[0]} AUC {best_old[1]:.3f}")
    if best_new[1] <= best_old[1]:
        print("  → 새 축이 기존을 못 이긴다. 다른 방향을 찾아야 한다.")


if __name__ == "__main__":
    main()
