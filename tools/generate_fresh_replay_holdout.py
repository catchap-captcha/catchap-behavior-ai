"""Generate a sealed replay holdout from participants absent from model data."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.services.quality_validator import QUALITY_REJECTED, validate_attempt
from app.services.replay_detector import trace_fingerprint_from_events
from training.adversarial_replay import FRESH_EXTERNAL_HOLDOUT_PROFILE, adversarial_replay_warp
from training.generate_extended_bots import build_payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _participants(path: Path) -> set[str]:
    return {
        str(row["anonymous_participant_id"])
        for row in _load_jsonl(path)
        if row.get("anonymous_participant_id")
    }


def assess_readiness(
    human_attempts: Path,
    known_human_features: Path,
    *,
    min_fresh_participants: int,
    count: int,
) -> dict[str, int | bool]:
    known = _participants(known_human_features)
    fresh_rows = [
        row
        for row in _load_jsonl(human_attempts)
        if row.get("anonymous_participant_id")
        and row.get("anonymous_participant_id") not in known
        and len(row.get("events", [])) >= 4
    ]
    fresh_participants = {
        str(row["anonymous_participant_id"])
        for row in fresh_rows
        if row.get("anonymous_participant_id")
    }
    return {
        "known_participants": len(known),
        "fresh_participants": len(fresh_participants),
        "fresh_attempts": len(fresh_rows),
        "minimum_fresh_participants": min_fresh_participants,
        "required_attempts": count,
        "ready": bool(
            len(fresh_participants) >= min_fresh_participants and len(fresh_rows) >= count
        ),
    }


def _balanced_sources(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_participant[str(row["anonymous_participant_id"])].append(row)

    randomizer = random.Random(seed)
    for candidates in by_participant.values():
        randomizer.shuffle(candidates)
    participant_ids = sorted(by_participant)
    randomizer.shuffle(participant_ids)

    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for participant in participant_ids:
            if by_participant[participant]:
                selected.append(by_participant[participant].pop())
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"need {count} fresh Human sources, found {len(selected)}")
    return selected


def generate_holdout(
    human_attempts: Path,
    known_human_features: Path,
    output: Path,
    *,
    count: int,
    seed: int,
    min_fresh_participants: int,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    readiness = assess_readiness(
        human_attempts,
        known_human_features,
        min_fresh_participants=min_fresh_participants,
        count=count,
    )
    if not readiness["ready"]:
        raise ValueError(f"fresh participant holdout is not ready: {readiness}")

    known = _participants(known_human_features)
    eligible = [
        row
        for row in _load_jsonl(human_attempts)
        if row.get("anonymous_participant_id")
        and row.get("anonymous_participant_id") not in known
        and len(row.get("events", [])) >= 4
    ]
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(_balanced_sources(eligible, count, seed)):
        events, width, height, transform = adversarial_replay_warp(
            source,
            random.Random(seed * 1_000_003 + index),
            FRESH_EXTERNAL_HOLDOUT_PROFILE,
        )
        quality = validate_attempt(events, captcha_width=width, captcha_height=height)
        if quality.status == QUALITY_REJECTED:
            raise ValueError(f"generated invalid replay row {index}: {quality.reason}")
        payload = build_payload(
            f"fresh_external_replay_holdout_v1_{index:06d}",
            "replay_fresh_external",
            events,
            width,
            height,
        )
        payload["collection"].update(
            {
                "label_source": "fresh_participant_external_holdout",
                "generator_version": FRESH_EXTERNAL_HOLDOUT_PROFILE.generator_version,
                "replay_source_fingerprint": trace_fingerprint_from_events(source["events"]),
                "evaluation_role": "fresh_participant_external_holdout",
                "training_usage": "external_holdout_only",
                "attack_variant": FRESH_EXTERNAL_HOLDOUT_PROFILE.name,
                "transform": transform,
            }
        )
        rows.append(payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "dataset_name": output.stem,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_usage": "external_holdout_only",
        "evaluation_role": "fresh_participant_external_holdout",
        "profile": FRESH_EXTERNAL_HOLDOUT_PROFILE.name,
        "generator_version": FRESH_EXTERNAL_HOLDOUT_PROFILE.generator_version,
        "seed": seed,
        "rows": len(rows),
        "fresh_participant_count": readiness["fresh_participants"],
        "source_policy": "participants absent from the model's known Human feature set",
        "source_attempts_unique": len(
            {row["collection"]["replay_source_fingerprint"] for row in rows}
        ),
        "source_identifiers_exported": False,
        "inputs": {
            "human_attempts": {"path": str(human_attempts), "sha256": _sha256(human_attempts)},
            "known_human_features": {
                "path": str(known_human_features),
                "sha256": _sha256(known_human_features),
            },
        },
        "output": {"path": str(output), "sha256": _sha256(output), "bytes": output.stat().st_size},
        "production_written": False,
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-attempts", type=Path, required=True)
    parser.add_argument("--known-human-features", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--min-fresh-participants", type=int, default=54)
    parser.add_argument("--readiness-report", type=Path)
    parser.add_argument("--readiness-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    readiness = assess_readiness(
        args.human_attempts,
        args.known_human_features,
        min_fresh_participants=args.min_fresh_participants,
        count=args.count,
    )
    if args.readiness_report:
        args.readiness_report.parent.mkdir(parents=True, exist_ok=True)
        args.readiness_report.write_text(
            json.dumps(readiness, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if args.readiness_only:
        print(json.dumps(readiness, ensure_ascii=False, indent=2))
        return 0
    if args.output is None:
        raise ValueError("--output is required unless --readiness-only is used")
    print(
        json.dumps(
            generate_holdout(
                args.human_attempts,
                args.known_human_features,
                args.output,
                count=args.count,
                seed=args.seed,
                min_fresh_participants=args.min_fresh_participants,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
