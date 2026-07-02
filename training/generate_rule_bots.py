"""Rule-based bot trajectory generator (defensive research use only).

Purpose: create labelled Bot drag samples for training our OWN CAPTCHA's
Human/Bot detector. This is the only data source we can produce before any real
users exist, and it seeds the Bot class + multiple bot families for the baseline
model.

⚠️ This tool does NOT solve, bypass or attack any CAPTCHA. It only synthesizes
mechanically-imperfect drag trajectories and labels them as bots so our detector
can learn to recognize them. It generates data ABOUT bots; it is not a bot.

Bot families (naive, easy tier — GAN bots come later as the hard tier):
  * straight — perfect linear path, constant speed, fixed 16ms cadence
  * accel    — smooth ease-in acceleration, no jitter, zero direction changes
  * jitter   — straight-ish with small mechanical noise (naive "humanization")

Output modes:
  * --out FILE   append collect-shaped payloads as JSONL (default; no DB needed)
  * --post URL   POST each payload to the collect API (needs COLLECT_API_KEY)

Each payload carries label='bot', label_source='rule_bot', bot_family=<family>,
generator_version, so it flows through the SAME collect pipeline (quality check +
feature extraction) as real data.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Any

from app.config import get_settings

FAMILIES = ["straight", "accel", "jitter"]
GENERATOR_VERSION = "rule_v1"

_DEFAULT_WIDTH = 420
_DEFAULT_HEIGHT = 220
_FRAME_MS = 16  # ~60fps cadence a naive script would emit


def _smoothstep(f: float) -> float:
    """Ease-in/ease-out in [0,1] -> [0,1] (smooth, mechanical acceleration)."""
    return f * f * (3.0 - 2.0 * f)


def generate_events(
    family: str,
    *,
    rng: random.Random,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
) -> list[dict[str, Any]]:
    """Generate one bot drag as pointer events (collect-compatible).

    The path always starts near the left, ends near a target x on the right,
    begins with pointerdown and ends with pointerup — so it passes quality
    validation — while carrying the mechanical signature of its family.

    Args:
        family: one of FAMILIES.
        rng: seeded RNG (reproducible per attempt).
        width, height: CAPTCHA pixel size.

    Returns:
        A list of pointer-event dicts (seq, event_type, t_ms, x, y, normalized).
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")

    n = rng.randint(20, 55)
    x0 = rng.uniform(5, 20)
    x1 = rng.uniform(250, min(380, width - 20))
    y_base = rng.uniform(height * 0.3, height * 0.6)

    events: list[dict[str, Any]] = []
    t = 0.0
    prev_t_ms = -1  # enforce strictly increasing t_ms (avoid duplicate timestamps)
    for i in range(n):
        frac = i / (n - 1)

        if family == "straight":
            x = x0 + (x1 - x0) * frac
            y = y_base
            dt = _FRAME_MS  # perfectly fixed cadence
        elif family == "accel":
            x = x0 + (x1 - x0) * _smoothstep(frac)  # speed ramps up then down
            y = y_base
            dt = _FRAME_MS
        else:  # jitter — naive humanization: small noise on position + timing
            x = x0 + (x1 - x0) * frac + rng.uniform(-2.0, 2.0)
            y = y_base + rng.uniform(-2.5, 2.5)
            dt = _FRAME_MS + rng.uniform(-3.0, 3.0)

        if i == 0:
            etype, t = "pointerdown", 0.0
        else:
            # advance time for every point after the first, including pointerup,
            # so the final segment has a normal dt (no artificial speed spike)
            etype = "pointerup" if i == n - 1 else "pointermove"
            t += max(0.0, dt)

        x = min(max(x, 0.0), width)
        y = min(max(y, 0.0), height)
        t_ms = 0 if i == 0 else max(int(round(t)), prev_t_ms + 1)
        prev_t_ms = t_ms
        events.append({
            "seq": i,
            "event_type": etype,
            "t_ms": t_ms,
            "x": round(x, 3),
            "y": round(y, 3),
            "x_normalized": round(x / width, 6),
            "y_normalized": round(y / height, 6),
            "target_role": "slider_handle",
        })
    return events


def build_collect_payload(
    attempt_id: str, events: list[dict[str, Any]], family: str,
    width: int = _DEFAULT_WIDTH, height: int = _DEFAULT_HEIGHT,
) -> dict[str, Any]:
    """Wrap generated events in the collect API request shape (labelled as bot)."""
    last_t = events[-1]["t_ms"] if events else 0
    return {
        "schema_version": get_settings().api_schema_version,
        "attempt_id": attempt_id,
        "challenge_id": "rulebot_challenge",
        "session_id": f"rulebot_{family}",
        "anonymous_participant_id": None,
        "captcha": {"width": width, "height": height},
        "timing": {
            "presented_at": "2026-01-01T00:00:00Z",
            "submitted_at": "2026-01-01T00:00:00Z",
        },
        "events": events,
        "interaction": {
            "regrab_count": 0, "retry_count": 0, "pointercancel_count": 0,
            "empty_click_count": 0, "failed_drop_count": 0,
        },
        "collection": {
            "label": "bot",
            "label_source": "rule_bot",
            "bot_family": family,
            "generator_version": GENERATOR_VERSION,
            "age_group": "unknown",
        },
    }


def generate_batch(count: int, families: list[str], seed: int = 42) -> list[dict[str, Any]]:
    """Generate `count` payloads spread evenly across the requested families."""
    payloads: list[dict[str, Any]] = []
    for i in range(count):
        family = families[i % len(families)]
        rng = random.Random(seed * 100003 + i)  # reproducible per index
        events = generate_events(family, rng=rng)
        attempt_id = f"rulebot_{GENERATOR_VERSION}_{family}_{i:06d}"
        payloads.append(build_collect_payload(attempt_id, events, family))
    return payloads


def write_jsonl(payloads: list[dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for p in payloads:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")


def post_to_collect(payloads: list[dict[str, Any]], url: str, api_key: str) -> tuple[int, int]:
    """POST payloads to the collect API. Returns (ok_count, fail_count)."""
    import urllib.error
    import urllib.request

    ok = fail = 0
    for p in payloads:
        data = json.dumps(p).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json", "X-API-Key": api_key},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                ok += 1 if resp.status == 200 else 0
                fail += 0 if resp.status == 200 else 1
        except urllib.error.URLError:
            fail += 1
    return ok, fail


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Generate rule-based bot samples (defensive research).")
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--families", nargs="+", default=FAMILIES, choices=FAMILIES)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/interim/rule_bots.jsonl",
                   help="JSONL output path (default mode)")
    p.add_argument("--post", type=str, default=None,
                   help="collect API URL to POST to instead of writing a file")
    args = p.parse_args(argv)

    print("규칙 기반 봇 생성기 — 방어 연구용(우리 CAPTCHA 탐지 학습). 우회/공격 도구 아님.")
    payloads = generate_batch(args.count, args.families, seed=args.seed)

    if args.post:
        api_key = get_settings().collect_api_key
        if not api_key:
            print("COLLECT_API_KEY 가 설정되지 않았습니다.", file=sys.stderr)
            return 2
        ok, fail = post_to_collect(payloads, args.post, api_key)
        print(f"collect API 전송 완료: 성공 {ok}, 실패 {fail}")
        return 0 if fail == 0 else 1

    write_jsonl(payloads, args.out)
    by_family: dict[str, int] = {}
    for p_ in payloads:
        fam = p_["collection"]["bot_family"]
        by_family[fam] = by_family.get(fam, 0) + 1
    print(f"봇 {len(payloads)}개 생성 → {args.out}")
    print(f"  family별: {by_family}")
    print("  다음: --post <collect_url> 로 보내면 품질검사·Feature 계산을 거쳐 DB에 쌓입니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
