"""Drive the deployed main CAPTCHA with scripted bot trajectories.

Every bot family we have was generated against the *previous* collection tool.
The main CAPTCHA (drag/drop, server-stored batches, receipt chain) has a
different interaction shape, so the measured ASR does not transfer: a naive
straight-line drag scored ``human_probability 0.999999`` against the live
service during integration testing.  This tool exists to measure ASR on the
distribution we actually serve.

It speaks the real protocol end to end — challenge, proof-of-work, batched
pointer events with the receipt chain, then verify.  Nothing is faked, so a
run here is directly comparable to a human session collected the same day.

Only the session ids and per-attempt outcomes are written locally.  Whether an
attempt reached the model, and what the model said, is read afterwards from
``catchap_ai.ai_behavior_attempts`` by joining on ``session_id`` — the tool
deliberately has no database access.

Usage (nothing runs without an explicit target and --confirm):

    python tools/redteam_main_captcha.py --base http://localhost:18000 \
        --style linear --count 30 --confirm

    python tools/redteam_main_captcha.py --base http://localhost:18000 \
        --style replay --replay-source traces.jsonl --count 30 --confirm

Styles
    linear   straight interpolation, uniform intervals.  The cheapest possible
             bot.  If this passes, nothing else matters.
    bezier   cubic Bezier with per-point jitter, eased timing, occasional
             overshoot-and-correct.  Mimics a hurried human.
    replay   replays recorded human pointer traces, rescaled onto the current
             challenge geometry.  Defeats shape-based features by construction;
             the receipt chain and DTW replay detection are what should catch it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Iterable

# The captcha is not publicly reachable; a run must be aimed at the tunnel or
# at loopback on the captcha host itself.  This is a red-team tool against our
# own service — refusing unknown targets keeps it from being pointed elsewhere.
ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"}


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

def _request(base: str, path: str, body: dict | None = None,
             headers: dict | None = None, timeout: float = 20.0) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data,
                                     method="POST" if data is not None else "GET")
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        raw = error.read().decode()
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, raw


def _leading_zero_bits(digest: bytes) -> int:
    count = 0
    for byte in digest:
        if byte == 0:
            count += 8
            continue
        count += 8 - byte.bit_length()
        break
    return count


def solve_pow(seed: str, bits: int, ceiling: int = 1 << 26) -> str | None:
    """Return a nonce meeting the difficulty, or None if the ceiling is hit.

    A real client burns this in a worker while the user reads the question, so
    paying it here is part of measuring the true cost of the attack.
    """
    for candidate in range(ceiling):
        nonce = str(candidate)
        digest = hashlib.sha256(f"{seed}:{nonce}".encode()).digest()
        if _leading_zero_bits(digest) >= bits:
            return nonce
    return None


# --------------------------------------------------------------------------
# trajectory styles
# --------------------------------------------------------------------------

def _linear(start: tuple[float, float], end: tuple[float, float],
            rng: random.Random, steps: int = 6) -> list[tuple[float, float, float]]:
    """Straight interpolation, uniform 45 ms intervals.

    Deliberately the dumbest generator we can write.  It is the control: any
    detector that lets this through is not detecting automation at all.
    """
    points = []
    for index in range(1, steps + 1):
        ratio = index / (steps + 1)
        points.append((
            start[0] + (end[0] - start[0]) * ratio,
            start[1] + (end[1] - start[1]) * ratio,
            45.0,
        ))
    return points


def _bezier(start: tuple[float, float], end: tuple[float, float],
            rng: random.Random, steps: int = 22) -> list[tuple[float, float, float]]:
    """Cubic Bezier with jitter, eased timing and an occasional correction.

    Control points are pushed off the straight line by a random fraction of the
    travel distance, so the path curves the way a wrist does.  Intervals follow
    an ease-in-out profile because humans accelerate away from the grab point
    and decelerate into the drop.
    """
    span = math.dist(start, end) or 1e-6
    normal = ((end[1] - start[1]) / span, -(end[0] - start[0]) / span)

    def control(ratio: float, bow: float) -> tuple[float, float]:
        base = (start[0] + (end[0] - start[0]) * ratio,
                start[1] + (end[1] - start[1]) * ratio)
        offset = span * bow
        return (base[0] + normal[0] * offset, base[1] + normal[1] * offset)

    c1 = control(0.30, rng.uniform(-0.22, 0.22))
    c2 = control(0.70, rng.uniform(-0.22, 0.22))

    points: list[tuple[float, float, float]] = []
    for index in range(1, steps + 1):
        t = index / (steps + 1)
        inv = 1 - t
        x = (inv ** 3 * start[0] + 3 * inv ** 2 * t * c1[0]
             + 3 * inv * t ** 2 * c2[0] + t ** 3 * end[0])
        y = (inv ** 3 * start[1] + 3 * inv ** 2 * t * c1[1]
             + 3 * inv * t ** 2 * c2[1] + t ** 3 * end[1])
        # per-point jitter — a real pointer never lands exactly on the curve
        x += rng.gauss(0, 0.0015)
        y += rng.gauss(0, 0.0015)
        # ease-in-out: slow at both ends, fast through the middle
        speed = math.sin(math.pi * t) + 0.25
        points.append((x, y, max(8.0, rng.gauss(34.0 / speed, 6.0))))

    # roughly a third of drags overshoot and come back
    if rng.random() < 0.35:
        over = (end[0] + (end[0] - c2[0]) * 0.05, end[1] + (end[1] - c2[1]) * 0.05)
        points.append((over[0], over[1], max(8.0, rng.gauss(30.0, 6.0))))
        points.append((end[0], end[1], max(8.0, rng.gauss(48.0, 10.0))))
    return points


def _replay(start: tuple[float, float], end: tuple[float, float],
            rng: random.Random, trace: list[dict[str, Any]]) -> list[tuple[float, float, float]]:
    """Rescale a recorded human trace onto this challenge's grab/drop points.

    The shape and the timing are a real person's; only the endpoints move.  Any
    feature that looks at curvature or interval statistics alone is blind to
    this, which is the point of measuring it separately.
    """
    moves = [event for event in trace
             if event.get("x") is not None and event.get("y") is not None]
    if len(moves) < 3:
        raise ValueError("replay trace has too few coordinate events")

    src_start = (float(moves[0]["x"]), float(moves[0]["y"]))
    src_end = (float(moves[-1]["x"]), float(moves[-1]["y"]))
    src_dx, src_dy = src_end[0] - src_start[0], src_end[1] - src_start[1]
    dst_dx, dst_dy = end[0] - start[0], end[1] - start[1]
    scale_x = dst_dx / src_dx if abs(src_dx) > 1e-9 else 1.0
    scale_y = dst_dy / src_dy if abs(src_dy) > 1e-9 else 1.0

    points: list[tuple[float, float, float]] = []
    previous_ts = float(moves[0].get("timestamp_ms", 0))
    for event in moves[1:]:
        x = start[0] + (float(event["x"]) - src_start[0]) * scale_x
        y = start[1] + (float(event["y"]) - src_start[1]) * scale_y
        ts = float(event.get("timestamp_ms", previous_ts))
        gap = max(4.0, min(400.0, ts - previous_ts))
        previous_ts = ts
        points.append((min(1.0, max(0.0, x)), min(1.0, max(0.0, y)), gap))
    return points


STYLES = {"linear": _linear, "bezier": _bezier, "replay": _replay}


# --------------------------------------------------------------------------
# one attempt
# --------------------------------------------------------------------------

def _events_for(challenge: dict[str, Any], style: str, rng: random.Random,
                trace: list[dict[str, Any]] | None,
                timing: str = "synthetic") -> tuple[list[list[dict]], list[str]]:
    """Build the batched event stream for one solve.

    Batches mirror what the browser sends: the load event, one batch per object
    drag, then the submit.  Event ``seq`` must be globally contiguous or the
    server rejects the batch.
    """
    zone = challenge["drop_zone"]
    drop = (zone["x"] + zone["width"] / 2, zone["y"] + zone["height"] / 2)

    seq = 0
    clock = int(time.time() * 1000)

    def event(kind: str, object_id: str | None = None,
              x: float | None = None, y: float | None = None,
              gap: float = 40.0) -> dict[str, Any]:
        """Stamp one event.

        ``timing`` is the variable that matters most and is easiest to get
        wrong.  A generator that computes ``clock += 40`` produces intervals
        with zero variance, which the interval-statistics features flag on
        sight — it measures "can we spot synthetic timestamps", not "can we
        spot automation".  Real automation (Playwright, Selenium, or a plain
        sleep loop) gets scheduler jitter for free because the timestamps come
        from the event loop.  ``real`` sleeps and reads the wall clock so the
        run reflects that.
        """
        nonlocal seq, clock
        if timing == "real":
            time.sleep(max(0.0, gap) / 1000.0)
            clock = int(time.time() * 1000)
        else:
            clock += int(gap)
        # The schema enforces 0 <= x,y <= 1 and rejects the whole batch with 422
        # otherwise.  Bezier curvature plus jitter overshoots the edge on a few
        # percent of drags, so clamp once here rather than in every generator.
        # Six decimals matches the server-side canonicalisation used for the
        # batch payload hash — sending more precision than that is pointless.
        row = {"seq": seq, "type": kind, "object_id": object_id,
               "x": None if x is None else round(min(1.0, max(0.0, float(x))), 6),
               "y": None if y is None else round(min(1.0, max(0.0, float(y))), 6),
               "timestamp_ms": clock}
        seq += 1
        return row

    batches: list[list[dict]] = [[event("challenge_loaded")]]
    selected: list[str] = []
    for obj in challenge["objects"]:
        object_id = obj["object_id"]
        hx, hy, hw, hh = obj["hit_region"]
        grab = (hx + hw / 2, hy + hh / 2)

        if style == "replay":
            path = _replay(grab, drop, rng, trace or [])
        else:
            path = STYLES[style](grab, drop, rng)

        batch = [event("pointer_down", object_id, grab[0], grab[1], gap=rng.uniform(120, 400)),
                 event("drag_start", object_id, grab[0], grab[1], gap=rng.uniform(8, 20))]
        for x, y, gap in path:
            batch.append(event("pointer_move", object_id, x, y, gap=gap))
        batch.append(event("drop", object_id, drop[0], drop[1], gap=rng.uniform(20, 60)))
        batch.append(event("selection_add", object_id, gap=rng.uniform(5, 15)))

        # the server caps a batch at behavior_batch_max_events
        cap = int(challenge.get("behavior_batch_max_events") or 32)
        for start in range(0, len(batch), cap):
            batches.append(batch[start:start + cap])
        selected.append(object_id)

    batches.append([event("submit", gap=rng.uniform(150, 600))])
    return batches, selected


def run_attempt(base: str, site_key: str, style: str, rng: random.Random,
                trace: list[dict[str, Any]] | None, pace: float,
                timing: str = "synthetic", select: str = "one") -> dict[str, Any]:
    """Run one full solve and return what happened.

    ``select`` decides what gets submitted, and it changes what the run
    measures:

    ``all``   submit every object.  An enumerating attacker cannot tell a
              honeypot from a real object, so this measures how often the
              honeypot alone ends the attempt.  It rarely reaches the model.
    ``one``   submit a single object.  Wrong most of the time, but the answer
              is not what we are measuring — the behaviour model scores the
              trajectory either way.  This is the mode that yields ASR.
    ``none``  submit nothing.  Cannot trip the honeypot and the coordinate
              binding has nothing to check, so it always reaches the model.
              Useful as a floor, but the empty selection is itself a signal.
    """
    headers = {"X-Captcha-Site-Key": site_key}
    session_id = f"botprobe-{style}-{timing}-{int(time.time() * 1000)}"

    status, challenge = _request(base, "/api/captcha/challenges", {
        "purpose": "signup", "risk_level": "high", "session_id": session_id}, headers)
    if status != 201:
        return {"session_id": session_id, "style": style,
                "outcome": "challenge_failed", "status": status, "detail": challenge}

    nonce = challenge.get("behavior_nonce")
    if not nonce:
        return {"session_id": session_id, "style": style,
                "outcome": "no_behavior_nonce", "status": status}

    batches, selected = _events_for(challenge, style, rng, trace, timing)

    previous = None
    for index, batch in enumerate(batches):
        time.sleep(pace)
        status, result = _request(
            base, f"/api/captcha/challenges/{challenge['challenge_id']}/behavior-batches",
            {"session_id": session_id, "nonce": nonce, "batch_seq": index,
             "previous_receipt": previous, "events": batch}, headers)
        if status != 200:
            return {"session_id": session_id, "style": style, "outcome": "batch_rejected",
                    "batch_seq": index, "status": status, "detail": result}
        previous = result["receipt"]

    if select == "one":
        selected = [rng.choice(selected)] if selected else []
    elif select == "none":
        selected = []
    body: dict[str, Any] = {"selected_object_ids": selected, "session_id": session_id,
                            "duration_ms": 4200, "client_signals": {}}
    pow_spec = challenge.get("pow")
    if pow_spec:
        solved = solve_pow(pow_spec["seed"], int(pow_spec["bits"]))
        if solved is None:
            return {"session_id": session_id, "style": style, "outcome": "pow_exhausted"}
        body["pow_nonce"] = solved

    status, result = _request(
        base, f"/api/captcha/challenges/{challenge['challenge_id']}/verify", body, headers)
    row = {"session_id": session_id, "style": style, "timing": timing,
           "challenge_id": challenge["challenge_id"],
           "object_count": len(challenge["objects"]),
           "batch_count": len(batches), "verify_status": status}
    if isinstance(result, dict):
        row["success"] = result.get("success")
        row["blocked"] = result.get("blocked")
        row["step_up"] = result.get("step_up")
        row["reason"] = result.get("reason") or result.get("pow_failed")
        if result.get("behavior_debug"):
            row["telemetry_reason"] = result["behavior_debug"].get("telemetry_reason")
    row["outcome"] = ("honeypot" if row.get("reason") == "honeypot"
                      else "verified" if status == 200 else "verify_failed")
    return row


# --------------------------------------------------------------------------

def _load_traces(path: Path) -> list[list[dict[str, Any]]]:
    """Load recorded pointer traces for replay.

    One JSON array of events per line, ordered, each with x/y/timestamp_ms.
    Export these from ``catchap_ai.ai_pointer_events`` grouped by attempt.
    """
    traces = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            events = payload if isinstance(payload, list) else payload.get("events", [])
            if events:
                traces.append(events)
    if not traces:
        raise SystemExit(f"no usable traces in {path}")
    return traces


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True,
                        help="captcha base URL (tunnel or loopback only)")
    parser.add_argument("--style", required=True, choices=sorted(STYLES))
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--select", choices=("all", "one", "none"), default="one",
                        help="what to submit. 'all' trips the honeypot and rarely reaches the "
                             "model; 'one' is the mode that yields ASR (see run_attempt).")
    parser.add_argument("--timing", choices=("synthetic", "real"), default="real",
                        help="synthetic computes timestamps arithmetically (zero interval "
                             "variance); real sleeps and reads the wall clock. Default real, "
                             "because that is what an actual bot produces.")
    parser.add_argument("--seed", type=int, default=None,
                        help="fix for a reproducible run; default is time-seeded")
    parser.add_argument("--pace", type=float, default=0.2,
                        help="seconds between batches (default 0.2, matches the browser)")
    parser.add_argument("--replay-source", type=Path,
                        help="JSONL of recorded human traces, required for --style replay")
    parser.add_argument("--out", type=Path, default=Path("reports/redteam_main_captcha.jsonl"))
    parser.add_argument("--confirm", action="store_true",
                        help="required; this drives the live captcha")
    args = parser.parse_args(list(argv) if argv is not None else None)

    host = urlparse(args.base).hostname
    if host not in ALLOWED_HOSTS:
        raise SystemExit(
            f"refusing target {host!r}. This drives a live CAPTCHA and is meant for our own "
            f"service over the tunnel or loopback. Allowed: {sorted(ALLOWED_HOSTS)}")
    if not args.confirm:
        raise SystemExit("pass --confirm; this sends real solve attempts to the live captcha")

    if args.style == "replay" and not args.replay_source:
        raise SystemExit("--style replay needs --replay-source")
    trace_pool = _load_traces(args.replay_source) if args.style == "replay" else None

    status, config = _request(args.base, "/api/config")
    if status != 200:
        raise SystemExit(f"cannot read site key from {args.base}: {status} {config}")
    site_key = config["siteKey"]

    rng = random.Random(args.seed)
    rows = []
    for index in range(args.count):
        trace = rng.choice(trace_pool) if trace_pool else None
        row = run_attempt(args.base, site_key, args.style, rng, trace, args.pace, args.timing, args.select)
        rows.append(row)
        print(f"[{index + 1:>3}/{args.count}] {row['outcome']:<16} "
              f"success={row.get('success')} reason={row.get('reason')} {row['session_id']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["outcome"]] = counts.get(row["outcome"], 0) + 1
    print("\noutcomes:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print(f"appended {len(rows)} rows to {args.out}")
    print("\nASR is computed from the model side, not from here. Join these session_ids "
          "against catchap_ai.ai_behavior_attempts / ai_model_predictions:")
    print("  SELECT a.session_id, p.predicted_label, p.human_probability, p.recommended_action")
    print("  FROM ai_behavior_attempts a JOIN ai_model_predictions p ON p.attempt_id = a.id")
    print("  WHERE a.session_id LIKE 'botprobe-%';")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
