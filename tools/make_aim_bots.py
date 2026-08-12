"""Four aim-segment attackers, weakest to strongest.

The premise
-----------
The attacker knows where the objects are. That is not a concession, it is the
situation: object positions come down in the challenge payload, so any bot
worth defending against reads them rather than looking for them. Aiming is
therefore a point-to-point move with a known destination, and the only thing
left to fake is the motion in between.

Every family is handed the same advantages, deliberately:

  * the same start and target as a real human aim, so no family can be caught
    on geometry it never had to reproduce;
  * a duration drawn from the human distribution, so none can be caught on
    taking an implausible amount of time;
  * the widget's 40ms capture throttle, so none emits a cadence that could not
    have survived the real capture path.

Giving the attacker these for free is the point. A defence that only works
because the bot picked a silly duration has not been tested. In the previous
build, bots that skipped the throttle separated at AUC 1.000 on interval alone
and the whole evaluation was worthless.

The families
------------
A_linear    constant speed along a straight line. The floor.
B_eased     Bezier path, minimum-jerk easing, gaussian jitter. This is what
            off-the-shelf humanised-cursor libraries actually do.
C_ballistic explicit ballistic throw plus corrective submovements — an attacker
            who read the motor-control literature and implemented the two-phase
            structure the human data shows.
D_replay    real human aim tracks, affine-mapped onto the new start/target.
            The strongest, and the one that matters: replay-derived bots are
            what defeated the 12-point drag surface. If D wins here too, the
            aim segment buys nothing and there is no reason to collect more.

Nothing here is fitted against a detector. These are generators, and they are
written once, before any model exists to evade.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from tools.aim_segments import AIM_CAPTURE, THROTTLE_MS, load_bursts, to_arrays

FAMILIES = ("A_linear", "B_eased", "C_ballistic", "D_replay")


def _emit(points: np.ndarray, times: np.ndarray) -> list[dict]:
    """Turn a continuous path into events the capture path could have produced.

    The widget drops any move that arrives within 40ms of the last one kept, so
    the bot's own samples are filtered the same way rather than assumed clean.
    """
    events: list[dict] = []
    last = -1e9
    for (x, y), t in zip(points, times):
        if t - last < THROTTLE_MS:
            continue
        last = t
        events.append({
            "seq": len(events),
            "type": "aim_move",
            "x": float(np.clip(x, 0.0, 1.0)),
            "y": float(np.clip(y, 0.0, 1.0)),
            "timestamp_ms": float(t),
            "is_trusted": True,
            "pointer_type": "mouse",
            "coalesced_count": 1,
        })
    return events


def human_intervals(bursts: list[list[dict]]) -> np.ndarray:
    """Every gap observed between consecutive human aim samples.

    Not a fitted distribution — the actual pool, resampled from. A gaussian
    around the 50ms median looked reasonable and was wrong: human bursts pause
    (p95 is 334ms), and a bot with tidy 50ms spacing separated on `interval_cv`
    at AUC 0.966 before any model was involved. That is a giveaway manufactured
    by the generator, and it would have flattered every number downstream.
    """
    out = []
    for burst in bursts:
        _, t = to_arrays(burst)
        out.extend(np.diff(t))
    return np.array([x for x in out if THROTTLE_MS <= x <= 400.0])


def _sample_times(duration: float, rng: random.Random,
                  pool: np.ndarray | None = None) -> np.ndarray:
    """Cadence bootstrapped from real human gaps, stretched to `duration`."""
    times, t = [0.0], 0.0
    while t < duration:
        gap = (float(pool[rng.randrange(pool.size)]) if pool is not None and pool.size
               else max(THROTTLE_MS, rng.gauss(50.0, 6.0)))
        t += gap
        times.append(min(t, duration))
    return np.array(times)


_POOL: np.ndarray | None = None


def a_linear(start, target, duration, rng, _bursts):
    t = _sample_times(duration, rng, _POOL)
    u = (t / max(duration, 1e-6))[:, None]
    return _emit(start + (target - start) * u, t)


def _bezier(start, target, rng):
    """Cubic control points offset perpendicular to the straight line."""
    direct = target - start
    normal = np.array([-direct[1], direct[0]])
    span = np.linalg.norm(direct) or 1e-9
    c1 = start + direct * 0.3 + normal * rng.gauss(0, 0.12)
    c2 = start + direct * 0.7 + normal * rng.gauss(0, 0.12)
    return lambda u: (
        (1 - u) ** 3 * start[None] + 3 * (1 - u) ** 2 * u * c1[None]
        + 3 * (1 - u) * u ** 2 * c2[None] + u ** 3 * target[None]
    ), span


def b_eased(start, target, duration, rng, _bursts):
    t = _sample_times(duration, rng, _POOL)
    u = (t / max(duration, 1e-6))
    # Minimum-jerk easing: the standard smooth reach, symmetric about the middle.
    eased = (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5)[:, None]
    curve, _ = _bezier(start, target, rng)
    pts = curve(eased)
    pts += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.004, pts.shape)
    pts[-1] = target
    return _emit(pts, t)


def c_ballistic(start, target, duration, rng, _bursts):
    """Ballistic throw, then corrections — the structure the human data shows.

    An attacker reproducing `peak_at≈0` and a long `slow_fraction` would build
    exactly this: cover most of the distance fast, then close the remainder in
    a few slower feedback-style submovements.
    """
    t = _sample_times(duration, rng, _POOL)
    u = t / max(duration, 1e-6)
    ballistic_end = rng.uniform(0.25, 0.45)
    reach = rng.uniform(0.80, 0.94)

    # Fraction of the total distance covered by each moment.
    progress = np.empty_like(u)
    fast = u <= ballistic_end
    # Decelerating arc for the throw: fast at once, easing off.
    v = np.clip(u[fast] / ballistic_end, 0, 1)
    progress[fast] = reach * (1 - (1 - v) ** 2)

    remaining_u = u[~fast]
    if remaining_u.size:
        w = (remaining_u - ballistic_end) / max(1 - ballistic_end, 1e-6)
        corrections = rng.randint(1, 3)
        closed = np.zeros_like(w)
        left = 1.0 - reach
        for k in range(corrections):
            lo, hi = k / corrections, (k + 1) / corrections
            share = left * (0.65 if k < corrections - 1 else 1.0)
            seg = np.clip((w - lo) / max(hi - lo, 1e-6), 0, 1)
            closed += share * (1 - (1 - seg) ** 2)
            left -= share
        progress[~fast] = reach + closed
    progress = np.clip(progress, 0, 1)[:, None]

    curve, _ = _bezier(start, target, rng)
    pts = curve(progress)
    pts += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.005, pts.shape)
    pts[-1] = target
    return _emit(pts, t)


def d_replay(start, target, duration, rng, bursts):
    """Replay a real human aim, mapped onto the new start and target.

    Similarity transform only — translate, rotate, scale — so the *shape* of a
    genuine human aim survives intact. This is the attack that made the 12-point
    drag surface unwinnable, because at that resolution one human's path is
    indistinguishable from another's.
    """
    dst_vec = target - start
    dst_len = float(np.linalg.norm(dst_vec))

    # Pick a recording that actually fits the move, instead of stretching an
    # arbitrary one onto it. Taking whatever came first meant scales from 0.15x
    # to 4.9x, and 13% of the resulting points landed outside the stage and were
    # clipped flat against its border — manufacturing sharp corners that showed
    # up as `turn_abs_mean` AUC 0.878. An attacker holding a library of captured
    # aims chooses a near-length one; only my generator was careless.
    # 방향도 같은 이유로 맞춘다. 아무 방향으로나 돌리면 옆으로 크게 휘는 조준일수록
    # 화면을 벗어나 버려지고, 곧은 것만 살아남는다 — 그 편향이 `linearity` AUC 0.69 로
    # 나왔다(2026-08-12). 길이를 맞추는 것과 같은 논리다: 캡처 라이브러리를 든
    # 공격자는 방향이 비슷한 것을 고르지, 아무 것이나 90도 돌려 쓰지 않는다.
    pts, src_t = None, None
    for attempt in range(40):
        candidate, cand_t = to_arrays(rng.choice(bursts))
        src_vec = candidate[-1] - candidate[0]
        src_len = float(np.linalg.norm(src_vec)) or 1e-9
        scale = dst_len / src_len
        if not (0.6 <= scale <= 1.7) and attempt < 30:
            continue
        theta = float(np.arctan2(dst_vec[1], dst_vec[0]) - np.arctan2(src_vec[1], src_vec[0]))
        turn = abs((theta + np.pi) % (2 * np.pi) - np.pi)
        if turn > np.pi / 4 and attempt < 30:
            continue
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        moved = (candidate - candidate[0]) @ rot.T * scale + start
        if moved.min() >= -0.02 and moved.max() <= 1.02:
            pts, src_t = moved, cand_t
            break
    if pts is None:
        return []
    # Jitter has to scale with the track. A fixed 0.003 was relatively huge on a
    # shrunk trace and negligible on a stretched one, which put `turn_abs_mean`
    # at AUC 0.877 — the defence would have been reading my scaling mistake.
    # The last point is left alone: under a similarity transform it already lands
    # on the target, and snapping it produced one anomalous final step.
    # Just enough noise to break exact-hash matching, and no more. The
    # fingerprint rounds at 1e-6, so 5e-4 defeats it with room to spare while
    # staying invisible in the shape statistics.
    #
    # 0.003 was three orders too generous for the wrong reason: 62% of a human
    # aim is spent crawling, so most steps are short, and 1.5px of noise on a
    # 12px step swings the direction hard. Measured against the human median
    # turn of 0.3124 —
    #
    #     jitter 0.0000 -> 0.2672     0.0005 -> 0.3386
    #     jitter 0.0010 -> 0.4228     0.0030 -> 0.6950
    #
    # so my own generator was the thing being detected at AUC 0.878. An attacker
    # tuning against the same statistic lands on 0.0005, which is the stronger
    # attack and therefore the one that belongs here.
    pts += np.random.default_rng(rng.randrange(1 << 30)).normal(0, 0.0005 * scale, pts.shape)

    # The source's OWN timestamps, rescaled. Falling back to `linspace` on a
    # length mismatch was the bug that made this family uniformly spaced and
    # separable on `interval_cv` alone: a replay attacker holds the original
    # cadence, gaps and all, and has no reason to throw it away.
    # The source's own timestamps, unstretched. Rescaling them to some other
    # sampled duration pushed samples back under the 40ms throttle, which
    # decimated the track to 12 points at 63ms against the human 16 at 50ms and
    # more than doubled per-step turning (0.31 -> 0.73). The defence would have
    # been detecting my resampling, not the attack. A replay attacker has the
    # original cadence and every reason to keep it.
    return _emit(pts, src_t)


GENERATORS = {"A_linear": a_linear, "B_eased": b_eased,
              "C_ballistic": c_ballistic, "D_replay": d_replay}


def generate(family: str, humans: list[list[dict]], count: int, seed: int) -> list[list[dict]]:
    global _POOL
    _POOL = human_intervals(humans)
    rng = random.Random(seed)
    out = []
    while len(out) < count:
        template = rng.choice(humans)
        xy, t = to_arrays(template)
        start, target = xy[0].copy(), xy[-1].copy()
        # Duration from the human pool, not from the template it is imitating,
        # so the bot is not handed a per-sample match it would not have.
        duration = float(to_arrays(rng.choice(humans))[1][-1])
        if duration < 2 * THROTTLE_MS:
            continue
        events = GENERATORS[family](start, target, duration, rng, humans)
        if len(events) >= 8:
            out.append(events)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--human", type=Path, default=AIM_CAPTURE)
    ap.add_argument("--out", type=Path, default=Path("data/interim/aim_bots"))
    ap.add_argument("--count", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args()

    humans = load_bursts(args.human)
    print(f"사람 조준 구간 {len(humans)}개를 본보기로 쓴다\n")
    args.out.mkdir(parents=True, exist_ok=True)

    for i, family in enumerate(FAMILIES):
        bursts = generate(family, humans, args.count, args.seed + i)
        path = args.out / f"{family}.jsonl"
        with path.open("w") as f:
            for b in bursts:
                f.write(json.dumps({"family": family, "aim_events": b}) + "\n")
        lengths = [len(b) for b in bursts]
        print(f"  {family:12s} {len(bursts):4d}건 · 점 중앙값 {int(np.median(lengths)):3d} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
