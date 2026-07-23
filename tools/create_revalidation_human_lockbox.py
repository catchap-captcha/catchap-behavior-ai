"""Reserve a new participant-group Human lockbox from a local snapshot.

This is a prospective internal re-reservation, not newly collected Human
data.  It prevents further tuning against the previously consumed test split:

* previous test participants stay out of the next development set;
* a deterministic subset of the previous training participants becomes a new
  Human FRR lockbox;
* only the remaining rows may be used for the next candidate's fitting and
  threshold calibration.

The lockbox file receives a companion manifest that marks it as
``external_holdout_only`` so training entry points can reject it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _previous_human_splits(split_manifest: dict[str, Any]) -> dict[str, str]:
    output: dict[str, str] = {}
    for group, split_name in split_manifest["group_to_split"].items():
        if group.startswith("human::"):
            output[group.removeprefix("human::")] = split_name
    return output


def reserve_lockbox(
    rows: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    *,
    seed: str,
    target_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return development rows, lockbox rows, and an auditable manifest."""
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_fraction must be between 0 and 1")
    if any(row.get("label") != "human" for row in rows):
        raise ValueError("Human lockbox source must contain Human rows only")

    previous = _previous_human_splits(split_manifest)
    counts = Counter(row.get("anonymous_participant_id") for row in rows)
    train_participants = sorted(
        participant
        for participant, split_name in previous.items()
        if split_name == "train" and participant in counts
    )
    if len(train_participants) < 3:
        raise ValueError("need at least three previous training participants")
    ranked = sorted(
        train_participants,
        key=lambda participant: hashlib.sha256(f"{seed}:{participant}".encode()).digest(),
    )
    target_rows = round(sum(counts[participant] for participant in train_participants) * target_fraction)
    reserved: list[str] = []
    reserved_rows = 0
    for participant in ranked:
        reserved.append(participant)
        reserved_rows += counts[participant]
        if reserved_rows >= target_rows:
            break

    previous_test = {
        participant for participant, split_name in previous.items() if split_name == "test"
    }
    lockbox_participants = set(reserved)
    lockbox_rows = [
        row for row in rows if row.get("anonymous_participant_id") in lockbox_participants
    ]
    development_rows = [
        row
        for row in rows
        if row.get("anonymous_participant_id") not in lockbox_participants
        and row.get("anonymous_participant_id") not in previous_test
    ]
    if not lockbox_rows or not development_rows:
        raise ValueError("lockbox reservation produced an empty partition")

    manifest = {
        "kind": "prospective_internal_human_lockbox",
        "training_usage": "external_holdout_only",
        "source_role": "revalidation_human_lockbox",
        "limitations": [
            "re-reserved from an existing local Human snapshot; not newly collected data",
            "participant groups may have appeared in historic experiments before this reservation",
            "not valid as an independent post-deployment Human evaluation",
        ],
        "selection": {
            "previous_test_participants": sorted(previous_test),
            "previous_test_policy": "excluded from next development and lockbox",
            "eligible_population": "previous train participant groups only",
            "hash_seed": seed,
            "target_fraction_of_previous_train_rows": target_fraction,
            "target_rows": target_rows,
            "selection_rule": "ascending SHA-256 rank until target row count is reached",
            "lockbox_participants": sorted(lockbox_participants),
        },
        "counts": {
            "source_human_rows": len(rows),
            "lockbox_human_rows": len(lockbox_rows),
            "lockbox_participants": len(lockbox_participants),
            "previous_test_human_rows_excluded": sum(counts[participant] for participant in previous_test),
            "previous_test_participants_excluded": len(previous_test),
            "development_human_rows": len(development_rows),
            "anonymous_human_rows_kept_development_only": counts[None],
        },
    }
    return development_rows, lockbox_rows, manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.human_features)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    rows = _load_jsonl(source)
    split_manifest = json.loads(Path(args.previous_split_manifest).read_text(encoding="utf-8"))
    development, lockbox, manifest = reserve_lockbox(
        rows,
        split_manifest,
        seed=args.seed,
        target_fraction=args.target_fraction,
    )
    output_dir.mkdir(parents=True)
    development_path = output_dir / "human_development.jsonl"
    lockbox_path = output_dir / "human_lockbox.jsonl"
    development_count = _write_jsonl(development_path, development)
    lockbox_count = _write_jsonl(lockbox_path, lockbox)
    manifest["source"] = {
        "human_features": str(source),
        "human_features_sha256": _sha256(source),
        "previous_split_manifest": str(Path(args.previous_split_manifest)),
        "previous_split_manifest_sha256": _sha256(Path(args.previous_split_manifest)),
    }
    manifest["files"] = {
        development_path.name: {"rows": development_count, "sha256": _sha256(development_path)},
        lockbox_path.name: {"rows": lockbox_count, "sha256": _sha256(lockbox_path)},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lockbox_path.with_suffix(lockbox_path.suffix + ".manifest.json").write_text(
        json.dumps(
            {
                "training_usage": "external_holdout_only",
                "source_role": "revalidation_human_lockbox",
                "lockbox_manifest": str(output_dir / "manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-features", required=True)
    parser.add_argument("--previous-split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", default="catChap-lockbox-v1")
    parser.add_argument("--target-fraction", type=float, default=0.15)
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
