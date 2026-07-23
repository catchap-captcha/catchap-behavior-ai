"""Generate isolated development or external adversarial replay datasets.

This is a defensive local-data generator for this repository's offline
detector.  It refuses to generate a development dataset from test sources, or
an external holdout from development sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.quality_validator import QUALITY_REJECTED, validate_attempt
from app.services.replay_detector import trace_fingerprint_from_events
from training.adversarial_replay import (
    DEVELOPMENT_PROFILE_NAMES,
    EXTERNAL_HOLDOUT_PROFILE,
    adversarial_replay_warp,
    get_profile,
)
from training.generate_extended_bots import build_payload, load_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_dataset(
    human_attempts: Path,
    split_manifest: Path,
    output: Path,
    *,
    source_role: str,
    profile_name: str,
    count: int,
    seed: int,
) -> dict[str, Any]:
    if source_role not in {"development", "external_holdout"}:
        raise ValueError("source_role must be development or external_holdout")
    allowed_profiles = (
        DEVELOPMENT_PROFILE_NAMES
        if source_role == "development"
        else frozenset((EXTERNAL_HOLDOUT_PROFILE.name,))
    )
    if profile_name not in allowed_profiles:
        raise ValueError(
            f"{source_role} must use an isolated profile {sorted(allowed_profiles)}; got {profile_name}"
        )
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    split_of = json.loads(split_manifest.read_text(encoding="utf-8"))["attempt_to_split"]
    profile = get_profile(profile_name)
    eligible = [
        row
        for row in load_jsonl(human_attempts)
        if row.get("anonymous_participant_id")
        and len(row.get("events", [])) >= 4
        and ((split_of.get(row["attempt_id"]) != "test") if source_role == "development" else (split_of.get(row["attempt_id"]) == "test"))
    ]
    if len(eligible) < count:
        raise ValueError(f"need {count} isolated Human sources, found {len(eligible)}")

    rows: list[dict[str, Any]] = []
    randomizer = random.Random(seed)
    for index, source in enumerate(randomizer.sample(eligible, count)):
        events, width, height, transform = adversarial_replay_warp(
            source,
            random.Random(seed * 1_000_003 + index),
            profile,
        )
        quality = validate_attempt(events, captcha_width=width, captcha_height=height)
        if quality.status == QUALITY_REJECTED:
            raise ValueError(f"generated invalid replay row {index}: {quality.reason}")
        payload = build_payload(
            f"adversarial_replay_{source_role}_v2_{index:06d}",
            "replay_adversarial",
            events,
            width,
            height,
        )
        payload["collection"].update(
            {
                "label_source": f"adversarial_replay_{source_role}",
                "generator_version": profile.generator_version,
                "replay_source_fingerprint": trace_fingerprint_from_events(source["events"]),
                "source_role": source_role,
                "source_split": "non_test" if source_role == "development" else "test",
                "attack_variant": profile.name,
                "transform": transform,
                "training_usage": (
                    "development_only" if source_role == "development" else "external_holdout_only"
                ),
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
        "source_role": source_role,
        "profile": profile_name,
        "generator_version": profile.generator_version,
        "seed": seed,
        "rows": len(rows),
        "source_policy": (
            "unique Human attempts from non-test participants"
            if source_role == "development"
            else "unique Human attempts from untouched test participants"
        ),
        "source_attempts_unique": len(
            {row["collection"]["replay_source_fingerprint"] for row in rows}
        ),
        "source_identifiers_exported": False,
        "profile_isolation": {
            "development_profiles": sorted(DEVELOPMENT_PROFILE_NAMES),
            "external_holdout_profile": EXTERNAL_HOLDOUT_PROFILE.name,
            "ranges_overlap": False,
            "time_curve_overlap": False,
        },
        "inputs": {
            "human_attempts": {"path": str(human_attempts), "sha256": _sha256(human_attempts)},
            "split_manifest": {"path": str(split_manifest), "sha256": _sha256(split_manifest)},
        },
        "output": {"path": str(output), "sha256": _sha256(output), "bytes": output.stat().st_size},
        "training_usage": (
            "development_only" if source_role == "development" else "external_holdout_only"
        ),
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
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-role", choices=("development", "external_holdout"), required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260722)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            generate_dataset(
                args.human_attempts,
                args.split_manifest,
                args.output,
                source_role=args.source_role,
                profile_name=args.profile,
                count=args.count,
                seed=args.seed,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
