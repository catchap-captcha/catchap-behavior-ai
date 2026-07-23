"""Generate an offline replay_warp holdout from untouched Human participants."""

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
from training.generate_extended_bots import _replay_warp, build_payload, load_jsonl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate_holdout(
    human_attempts: Path,
    split_manifest: Path,
    output: Path,
    *,
    count: int = 1000,
    seed: int = 20260721,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    split = json.loads(split_manifest.read_text(encoding="utf-8"))["attempt_to_split"]
    eligible = [
        row
        for row in load_jsonl(human_attempts)
        if split.get(row["attempt_id"]) == "test"
        and row.get("anonymous_participant_id")
        and len(row.get("events", [])) >= 4
    ]
    if len(eligible) < count:
        raise ValueError(f"need {count} untouched Human sources, found {len(eligible)}")

    randomizer = random.Random(seed)
    sources = randomizer.sample(eligible, count)
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        events, width, height = _replay_warp(
            source,
            random.Random(seed * 1_000_003 + index),
        )
        quality = validate_attempt(events, captcha_width=width, captcha_height=height)
        if quality.status == QUALITY_REJECTED:
            raise ValueError(f"generated invalid replay row {index}: {quality.reason}")
        payload = build_payload(
            f"external_replay_holdout_v1_{index:06d}",
            "replay_warp",
            events,
            width,
            height,
        )
        payload["collection"].update(
            {
                "label_source": "external_replay_holdout",
                "generator_version": "external_replay_holdout_v1",
                "replay_source_fingerprint": trace_fingerprint_from_events(source["events"]),
                "source_split": "test",
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
        "generator_version": "external_replay_holdout_v1",
        "seed": seed,
        "rows": len(rows),
        "source_policy": "unique Human attempts from untouched participant test split",
        "source_attempts_unique": len(
            {row["collection"]["replay_source_fingerprint"] for row in rows}
        ),
        "source_identifiers_exported": False,
        "inputs": {
            "human_attempts": {"path": str(human_attempts), "sha256": _sha256(human_attempts)},
            "split_manifest": {"path": str(split_manifest), "sha256": _sha256(split_manifest)},
        },
        "output": {"path": str(output), "sha256": _sha256(output), "bytes": output.stat().st_size},
        "production_written": False,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-attempts", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260721)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate_holdout(
        args.human_attempts,
        args.split_manifest,
        args.output,
        count=args.count,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
