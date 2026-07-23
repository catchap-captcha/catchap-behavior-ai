"""Build a confirmed-Human training dataset from an anonymized DB snapshot.

The source snapshot must already contain pseudonymized IDs. This tool never
connects to the database and requires an explicit confirmation flag before it
turns the controlled collection into supervised Human labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.feature_extractor import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, extract_features
from app.services.feature_profiles import get_feature_profile


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path, rows: int) -> dict[str, Any]:
    return {"rows": rows, "bytes": path.stat().st_size, "sha256": sha256(path)}


def run(
    snapshot_dir: Path,
    output_dir: Path,
    confirmed_human: bool,
    feature_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> dict[str, Any]:
    if not confirmed_human:
        raise ValueError("--confirm-controlled-human is required before assigning Human labels")
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")

    profile = get_feature_profile(feature_schema_version)
    source_manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    curation = {
        row["attempt_id"]: row for row in load_jsonl(snapshot_dir / "curation_index.jsonl")
    }
    attempts_path = snapshot_dir / "human_attempts.jsonl"
    source_attempt_count = sum(1 for _ in load_jsonl(attempts_path))

    rejected = [row for row in curation.values() if row["quality_status"] == "rejected"]
    source_excluded = [
        row
        for row in curation.values()
        if row["quality_status"] != "rejected"
        and row.get("source_dataset_status") == "excluded"
    ]
    excluded = [dict(row) for row in rejected]
    for row in source_excluded:
        item = dict(row)
        item["reasons"] = [*item.get("reasons", []), "source_dataset_status_excluded"]
        excluded.append(item)
    rejection_reasons = Counter(
        reason for row in excluded for reason in row.get("reasons", [])
    )

    output_dir.mkdir(parents=True)
    features_path = output_dir / "human_features.jsonl"
    labels_path = output_dir / "human_labels.jsonl"
    excluded_path = output_dir / "excluded_from_training.jsonl"
    output_attempts_path = output_dir / "human_attempts.jsonl"

    included_attempt_count = write_jsonl(
        output_attempts_path,
        (
            payload
            for payload in load_jsonl(attempts_path)
            if curation[payload["attempt_id"]]["quality_status"] != "rejected"
            and curation[payload["attempt_id"]].get("source_dataset_status") != "excluded"
        ),
    )

    linked_rows = 0
    anonymous_rows = 0
    linked_participants: set[str] = set()
    included_ids: set[str] = set()

    def feature_rows() -> Iterable[dict[str, Any]]:
        nonlocal linked_rows, anonymous_rows
        for payload in load_jsonl(output_attempts_path):
            attempt_id = payload["attempt_id"]
            decision = curation.get(attempt_id)
            if decision is None or decision["quality_status"] == "rejected":
                raise ValueError(f"missing or rejected curation for included attempt: {attempt_id}")
            if attempt_id in included_ids:
                raise ValueError(f"duplicate attempt_id: {attempt_id}")
            included_ids.add(attempt_id)

            participant = payload.get("anonymous_participant_id")
            if participant:
                linked_rows += 1
                linked_participants.add(participant)
            else:
                anonymous_rows += 1

            interaction = payload.get("interaction") or {}
            features = profile.extractor(payload.get("events", []), interaction)
            if set(features) != set(profile.names):
                raise ValueError(f"feature schema mismatch for {attempt_id}")
            if any(not math.isfinite(float(value)) for value in features.values()):
                raise ValueError(f"non-finite feature for {attempt_id}")

            row = {
                "attempt_id": attempt_id,
                "challenge_id": payload.get("challenge_id"),
                "session_id": payload.get("session_id"),
                "anonymous_participant_id": participant,
                "label": "human",
                "label_source": "controlled_collection",
                "owner_confirmed_human": True,
                "bot_family": None,
                "generator_version": None,
                "schema_version": payload.get("schema_version", "1.0"),
                "feature_schema_version": profile.version,
                "quality_status": "valid",
                "previous_quality_status": decision["quality_status"],
                "previous_quality_reasons": decision.get("reasons", []),
                "age_group": (payload.get("collection") or {}).get("age_group", "unknown"),
                "consent_version": (payload.get("collection") or {}).get("consent_version"),
                "input_type": decision.get("input_type", "unknown"),
                "position_correct": payload.get("position_correct"),
                "interaction_success": payload.get("interaction_success"),
                "final_drop_error": payload.get("final_drop_error"),
            }
            row.update(features)
            yield row

    feature_count = write_jsonl(features_path, feature_rows())
    if feature_count != included_attempt_count:
        raise ValueError(
            f"attempt/feature count mismatch: attempts={included_attempt_count}, features={feature_count}"
        )

    label_count = write_jsonl(
        labels_path,
        (
            {
                "attempt_id": row["attempt_id"],
                "anonymous_participant_id": row.get("anonymous_participant_id"),
                "label": "human",
                "label_source": "controlled_collection",
                "collection_ownership_confirmed": True,
                "training_inclusion": "included",
            }
            for row in load_jsonl(output_attempts_path)
        ),
    )
    excluded_count = write_jsonl(excluded_path, excluded)

    dataset_name = output_dir.name
    manifest = {
        "dataset_schema_version": "1.0",
        "dataset_name": dataset_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_snapshot": str(snapshot_dir.resolve()),
        "label": "human",
        "label_source": "controlled_collection",
        "collection_ownership_confirmed": True,
        "selection": {
            "included": "all source rows with a usable pointer trace",
            "anonymous_rows": "included because the collection owner confirmed controlled Human attempts",
            "excluded": "rows with unusable traces or explicit source dataset_status=excluded",
        },
        "counts": {
            "source_rows": int(source_manifest["counts"]["total"]),
            "source_usable_attempt_rows": source_attempt_count,
            "included_human_rows": feature_count,
            "feature_rows": feature_count,
            "raw_attempt_rows": included_attempt_count,
            "linked_participant_rows": linked_rows,
            "anonymous_confirmed_human_rows": anonymous_rows,
            "distinct_linked_participants_with_traces": len(linked_participants),
            "excluded_from_training_rows": excluded_count,
            "excluded_unusable_trace_rows": len(rejected),
            "source_dataset_excluded_rows": len(source_excluded),
            "excluded_by_reason": dict(sorted(rejection_reasons.items())),
        },
        "feature_schema_version": profile.version,
        "feature_count": len(profile.names),
        "notes": {
            "unusable_trace_policy": "excluded; no zero-filled or fabricated features",
            "source_dataset_status_policy": "explicit excluded rows are never used for training",
            "anonymous_grouping_limit": "participant identity is unavailable, so person-level split cannot be guaranteed",
            "remote_database_modified": False,
            "attempts_storage": "filtered pseudonymized copy",
        },
        "files": {
            labels_path.name: file_meta(labels_path, label_count),
            features_path.name: file_meta(features_path, feature_count),
            output_attempts_path.name: file_meta(output_attempts_path, included_attempt_count),
            excluded_path.name: file_meta(excluded_path, excluded_count),
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.txt").write_text(
        f"""Catchap confirmed Human training dataset

- Source rows: {manifest['counts']['source_rows']}
- Included Human rows: {feature_count}
- Linked participant rows: {linked_rows}
- Anonymous confirmed Human rows: {anonymous_rows}
- Distinct linked participants: {len(linked_participants)}
- Excluded from training: {excluded_count}

The source database was read only. Direct database identifiers are not present.
""",
        encoding="ascii",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confirm-controlled-human", action="store_true")
    parser.add_argument("--feature-schema-version", choices=("1.0", "2.0", "2.1", "2.2", "2.3"), default="1.0")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run(
        args.snapshot_dir,
        args.output_dir,
        args.confirm_controlled_human,
        args.feature_schema_version,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
