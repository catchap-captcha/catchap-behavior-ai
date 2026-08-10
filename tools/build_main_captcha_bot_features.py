"""Turn the main-captcha-shaped bots into feature rows the formal trainer accepts.

Why this was missing
--------------------
`tools/generate_main_captcha_bots.py` produced 12,000 attack traces on 2026-08-06
precisely because every other bot in the corpus is warped from the legacy
collection screen — ~340 events per session against the main captcha's ~12, so
`event_count` alone separated our bots from our humans at AUC 1.000 and the
detector never had to learn anything harder.

They were then never wired into training. The deployed model's base was fitted
on 2026-07-22, before they existed, and nothing converted them into the feature
rows `run_formal_two_view_fusion.py` reads. Scored against that model they passed
67.8%, which made them the worst family by a wide margin — worse than anything in
the holdout set, and invisible because they were never scored.

Adding them to training took that family to 1.4% with the worst *unseen* family
also falling (69.5% -> 38.3% on leave-one-person-out).

Grouping
--------
The generator already stamps `bot_family` and a per-batch `generator_version`
(`hybrid_motion_main_v1_b0` .. `_b5`). Those are carried through verbatim, because
`build_local_split` splits within a family by `generator_version`: inventing a
single flat version would collapse six batches into one group, and the trainer
would refuse (`need at least 3 groups`) or — worse, if forced — let near-identical
traces span train and test.

    .venv/bin/python tools/build_main_captcha_bot_features.py
    cat data/interim/bot_features_v23corr_20260722.jsonl \
        data/interim/bot_features_main_captcha_20260810.jsonl \
        > data/interim/bot_features_plus_surface_20260810.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.services.drag_segmentation import MIN_MOVES_PER_DRAG, move_count, split_drags  # noqa: E402
from app.services.feature_extractor_v23 import extract_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SRC = ROOT / "data" / "interim" / "main_captcha_bots_development_20260806.jsonl"
DEFAULT_OUT = ROOT / "data" / "interim" / "bot_features_main_captcha_20260810.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    written = 0
    groups: dict[str, int] = {}
    with args.out.open("w") as sink:
        for index, line in enumerate(args.src.open()):
            record = json.loads(line)
            meta = record.get("collection") or {}
            for drag_index, drag in enumerate(split_drags(record.get("events") or [])):
                if move_count(drag) < MIN_MOVES_PER_DRAG:
                    continue
                version = meta.get("generator_version")
                groups[version] = groups.get(version, 0) + 1
                row = {
                    "attempt_id": f"{record.get('attempt_id')}_{drag_index}",
                    "challenge_id": record.get("challenge_id") or "maincap_challenge",
                    "session_id": record.get("session_id"),
                    "anonymous_participant_id": None,
                    "label": "bot",
                    "label_source": meta.get("label_source", "main_captcha_shaped_generated"),
                    "bot_family": meta.get("bot_family", "main_captcha_motion"),
                    "generator_version": version,
                    "generator_version_base": meta.get("generator_version_base")
                    or "hybrid_motion_main_v1",
                    "evaluation_role": None,
                    "schema_version": "1.0",
                    "feature_schema_version": "2.3",
                    "quality_status": "valid",
                    "position_correct": None,
                    "interaction_success": None,
                    "final_drop_error": None,
                }
                row.update(extract_features(drag, None))
                sink.write(json.dumps(row) + "\n")
                written += 1

    print(f"{written}행 -> {args.out}")
    print(f"  그룹 {len(groups)}개: {groups}")
    if len(groups) < 3:
        print("  ⚠️ 그룹이 3개 미만이면 학습 도구가 계열 안에서 분할하지 못한다")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
