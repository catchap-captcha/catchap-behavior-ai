"""Generate attack traces shaped like the surface we deploy on.

Why this exists
---------------
Every one of the 21,300 development bot traces is warped from the legacy
collection screen: ~340 events per session, one long drag. Main captcha drags are
~12 events. So `event_count` alone separates our bots from our humans with
AUC 1.000 — the detector has never had to learn anything harder, and a real
attacker aiming at the main captcha would never hand us that.

Re-running the same generator over main-captcha traces drops the best single
feature to AUC 0.947 and leaves five signals above 0.9, the strongest being
`pause_count` (human median 3, generated 0) — humans stop mid-drag and the
generator has no knob that produces a stop. Those features already ship. The
model does not use them because it has never seen a bot that needed them.

Sealed people (sw, ms) are refused as attack substrate too. Warping their traces
would put their movement into training through the back door and quietly destroy
the only unseen-person evaluation set we have.

    .venv/bin/python tools/generate_main_captcha_bots.py --count 6000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.generate_hybrid_redteam_bots import _motion_events  # noqa: E402
from tools.collection_split import person_of  # noqa: E402
from tools.redteam_evolution_search import random_policy  # noqa: E402

COLLECTION = Path("data/interim/collection_20260806.jsonl")
SPLIT = Path("data/metadata/collection_split_20260806.json")


def bases() -> list[dict]:
    sealed = set(json.loads(SPLIT.read_text())["holdout_people"])
    out = []
    with COLLECTION.open() as f:
        for line in f:
            record = json.loads(line)
            code = record.get("participant_id") or ""
            if person_of(code) in sealed:
                continue
            if record.get("quality_status") != "valid":
                continue
            rows = record.get("events") or []
            events = [{
                "seq": r.get("seq"), "event_type": r.get("event_type"),
                "t_ms": float(r.get("client_timestamp_ms") or 0),
                "x": float(r["x_pixel"]), "y": float(r["y_pixel"]),
                "x_normalized": r.get("x_normalized"), "y_normalized": r.get("y_normalized"),
            } for r in rows if r.get("x_pixel") is not None]
            if len(events) >= 8:
                out.append({
                    "events": events,
                    "width": int(record.get("stage_width") or 500),
                    "height": int(record.get("stage_height") or 375),
                })
    print(f"기반 궤적 {len(out)}개 (봉인 {sorted(sealed)} 제외)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--out", type=Path,
                    default=Path("data/interim/main_captcha_bots_development_20260806.jsonl"))
    args = ap.parse_args()

    substrate = bases()
    if not substrate:
        raise SystemExit("기반 궤적이 없다")

    rng = random.Random(args.seed)
    written = 0
    with args.out.open("w") as dst:
        while written < args.count:
            policy = random_policy(rng)
            base = rng.choice(substrate)
            events, meta = _motion_events(base["events"], width=base["width"],
                                          height=base["height"], policy=policy,
                                          randomizer=rng)
            if len(events) < 3:
                continue
            dst.write(json.dumps({
                "schema_version": "1.0",
                "attempt_id": f"mcbot-{args.seed}-{written:06d}",
                "challenge_id": "main_captcha_shaped_redteam",
                "session_id": f"mcbot-{args.seed}-{written:06d}",
                "anonymous_participant_id": None,
                "captcha": {"width": base["width"], "height": base["height"]},
                "timing": {"presented_at": None, "submitted_at": None},
                "events": events,
                "interaction": {},
                "collection": {
                    "label": "bot",
                    "label_source": "main_captcha_shaped_generated",
                    "bot_family": "main_captcha_motion",
                    # Batch suffix keeps leakage-aware splitting able to separate
                    # generations; the base name preserves provenance.
                    "generator_version": f"hybrid_motion_main_v1_b{written // 1000}",
                    "training_usage": "development_only",
                    "age_group": "unknown",
                    "policy": {k: getattr(policy, k) for k in
                               ("curvature", "jitter", "time_power", "duration_scale",
                                "turn_slowdown", "coalesce_fraction", "frame_ms",
                                "late_correction_probability")},
                    "meta": meta if isinstance(meta, dict) else None,
                },
                "position_correct": True,
                "interaction_success": True,
                "final_drop_error": 0.0,
            }, ensure_ascii=False, default=str) + "\n")
            written += 1

    print(f"생성 {written}건 -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
