"""Turn the 2026-08-06 main-captcha collection into schema-2.3 training rows.

Why this matters more than the row count suggests
-------------------------------------------------
The deployed detector has never seen a human on the main captcha surface. Every
human row it trained on came from the legacy collection screen, and that is the
whole explanation for 07-31: FRR 0.11% on the screen it knew, 33.3% on the one
it did not. 177 sessions is nothing next to 20,066 legacy rows, but they are the
only rows from the surface the model is actually deployed against.

Sealed people are refused, not filtered. `collection_split.py` fixed sw and ms
as the holdout using a salt committed before any participant existed; letting
them leak into training through a flag would quietly destroy the only unseen-
person evaluation set we have, and nothing downstream would notice.

    .venv/bin/python tools/build_collection_features.py \
        --in data/interim/collection_20260806.jsonl \
        --split data/metadata/collection_split_20260806.json \
        --out data/interim/human_features_collection_20260806.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.feature_extractor_v23 import (  # noqa: E402
    FEATURE_SCHEMA_VERSION,
    extract_features,
)
from tools.collection_split import person_of  # noqa: E402

BOT_CODE_MARKERS = ("pwbot", "rtbot", "botprobe", "probe", "signalcheck", "zzprobe")


def to_events(rows: list[dict]) -> list[dict]:
    """Pixels, matching production. `per_drag_scoring` measured this: normalized
    reproduces production far worse (61% of rows within 0.01) than pixels (82%)."""
    if not rows:
        return []
    base = rows[0].get("client_timestamp_ms") or 0
    out = []
    for r in rows:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="source", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    split = json.loads(args.split.read_text())
    sealed = set(split["holdout_people"])
    training = set(split["training_people"])
    print(f"봉인 {sorted(sealed)} · 학습 {sorted(training)}  (salt={split['salt']})")

    written = 0
    skipped: dict[str, int] = {}
    with args.source.open() as src, args.out.open("w") as dst:
        for line in src:
            record = json.loads(line)
            code = record.get("participant_id") or ""
            person = person_of(code)

            if not code or any(m in code.lower() for m in BOT_CODE_MARKERS):
                skipped["봇/확인용"] = skipped.get("봇/확인용", 0) + 1
                continue
            if person in sealed:
                skipped["봉인"] = skipped.get("봉인", 0) + 1
                continue
            if person not in training:
                skipped["분할에 없음"] = skipped.get("분할에 없음", 0) + 1
                continue
            if record.get("quality_status") != "valid":
                skipped["품질 미달"] = skipped.get("품질 미달", 0) + 1
                continue

            events = to_events(record.get("events") or [])
            if len(events) < 3:
                skipped["이벤트 부족"] = skipped.get("이벤트 부족", 0) + 1
                continue

            row = {
                "attempt_id": record["attempt_id"],
                "challenge_id": "main_captcha_collection_20260806",
                "session_id": record["attempt_id"],
                # the grouping key. person, not code — jy-captcha and a future
                # jy-bank are one person and must never land on both sides.
                "anonymous_participant_id": person,
                "label": "human",
                "label_source": "controlled_collection",
                "owner_confirmed_human": True,
                "bot_family": None,
                "generator_version": None,
                "schema_version": "1.0",
                "feature_schema_version": FEATURE_SCHEMA_VERSION,
                "quality_status": "valid",
                "previous_quality_status": "valid",
                "previous_quality_reasons": [],
                "age_group": "adult",
                "consent_version": None,
                "input_type": "mouse",
                "position_correct": True,
                "interaction_success": True,
                "final_drop_error": 0.0,
                "collection_surface": code.rsplit("-", 1)[-1],
                "stage_width": record.get("stage_width"),
                "stage_height": record.get("stage_height"),
            }
            row.update(extract_features(events, None))
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n기록 {written}행 -> {args.out}")
    for reason, count in sorted(skipped.items(), key=lambda kv: -kv[1]):
        print(f"  건너뜀 {reason}: {count}")
    if not written:
        print("한 행도 안 나왔다 — 분할 파일과 참여자 코드를 대조하라.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
