"""Export a pseudonymized, read-only Human behavior snapshot from MySQL.

The exporter opens a repeatable-read, read-only consistent transaction. Direct
database identifiers and the snapshot-local HMAC key are never written.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
import secrets
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.services.feature_extractor import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, extract_features


MAX_NORMALIZED_SPEED_PER_MS = 0.02
MIN_TRACE_POINTS = 4


def _json_default(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    raise TypeError(f"cannot serialize {type(value)!r}")


def _dump_line(handle, row: dict[str, Any]) -> None:
    handle.write(
        json.dumps(
            row,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
        + "\n"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pseudonym(kind: str, value: str | None, key: bytes) -> str | None:
    if not value:
        return None
    digest = hmac.new(key, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()[:24]
    return f"{kind}_{digest}"


def _load_points(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    return value if isinstance(value, list) else None


def _parse_points(value: Any, width: Any, height: Any) -> tuple[list[dict[str, Any]], list[str]]:
    points = _load_points(value)
    if not points:
        return [], ["missing_or_invalid_trace"]
    try:
        box_w = float(width)
        box_h = float(height)
    except (TypeError, ValueError):
        return [], ["invalid_trace_dimensions"]
    if not (math.isfinite(box_w) and math.isfinite(box_h) and box_w > 0 and box_h > 0):
        return [], ["invalid_trace_dimensions"]
    if len(points) < MIN_TRACE_POINTS:
        return [], ["too_few_points"]

    parsed: list[tuple[float, float, float]] = []
    for point in points:
        if isinstance(point, dict):
            values = (point.get("t_ms"), point.get("x"), point.get("y"))
        elif isinstance(point, (list, tuple)) and len(point) >= 3:
            values = point[:3]
        else:
            return [], ["missing_or_invalid_trace"]
        try:
            t_ms, x_norm, y_norm = (float(item) for item in values)
        except (TypeError, ValueError):
            return [], ["missing_or_invalid_trace"]
        if not all(math.isfinite(item) for item in (t_ms, x_norm, y_norm)):
            return [], ["missing_or_invalid_trace"]
        parsed.append((t_ms, x_norm, y_norm))

    if any(right[0] < left[0] for left, right in zip(parsed, parsed[1:])):
        return [], ["non_monotonic_time"]

    events: list[dict[str, Any]] = []
    for index, (t_ms, x_norm, y_norm) in enumerate(parsed):
        events.append(
            {
                "seq": index,
                "event_type": (
                    "pointerdown"
                    if index == 0
                    else "pointerup" if index == len(parsed) - 1 else "pointermove"
                ),
                "t_ms": t_ms,
                "x": x_norm * box_w,
                "y": y_norm * box_h,
                "x_normalized": x_norm,
                "y_normalized": y_norm,
                "target_role": "captcha_area",
            }
        )
    return events, []


def _max_normalized_speed(events: list[dict[str, Any]]) -> float:
    maximum = 0.0
    for left, right in zip(events, events[1:]):
        dt = float(right["t_ms"]) - float(left["t_ms"])
        if dt <= 0:
            continue
        distance = math.hypot(
            float(right["x_normalized"]) - float(left["x_normalized"]),
            float(right["y_normalized"]) - float(left["y_normalized"]),
        )
        maximum = max(maximum, distance / dt)
    return maximum


def _quality_decision(
    *,
    events: list[dict[str, Any]],
    parse_reasons: list[str],
    participant: str | None,
) -> tuple[str, list[str]]:
    if parse_reasons:
        return "rejected", parse_reasons
    reasons: list[str] = []
    if participant is None:
        reasons.append("participant_group_unknown")
    if _max_normalized_speed(events) > MAX_NORMALIZED_SPEED_PER_MS:
        reasons.append("extreme_normalized_speed_review")
    return ("pending", reasons) if reasons else ("valid", [])


def _attempt_payload(
    row: dict[str, Any],
    *,
    record_id: str,
    participant: str | None,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    correct = row.get("interaction_result") in {"correct", "success", "pass"}
    return {
        "schema_version": "1.0",
        "attempt_id": record_id,
        "challenge_id": "legacy_behavior_capture",
        "session_id": record_id,
        "anonymous_participant_id": participant,
        "captcha": {"width": row.get("box_w"), "height": row.get("box_h")},
        "timing": {"presented_at": None, "submitted_at": None},
        "events": events,
        "interaction": {
            "regrab_count": 0,
            "retry_count": int(row.get("retry_count") or 0),
            "pointercancel_count": 0,
            "empty_click_count": 0,
            "failed_drop_count": 0 if correct else 1,
        },
        "collection": {
            "label": "human",
            "label_source": "controlled_collection",
            "bot_family": None,
            "generator_version": None,
            "age_group": row.get("actor_band") or "unknown",
            "consent_version": None,
        },
        "position_correct": correct,
        "interaction_success": correct,
        "final_drop_error": float(row.get("drop_distance_norm") or 0.0),
    }


def _summary_record(
    row: dict[str, Any],
    *,
    record_id: str,
    organization: str | None,
    participant: str | None,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "snapshot_schema_version": "1.0",
        "record_id": record_id,
        "anonymous_organization_id": organization,
        "anonymous_participant_id": participant,
        "summary": {
            "source_type": row.get("source_type"),
            "solve_time_ms": row.get("solve_time_ms"),
            "path_length": row.get("path_length"),
            "avg_speed": row.get("avg_speed"),
            "pause_count": row.get("pause_count"),
            "retry_count": row.get("retry_count"),
            "drop_distance_norm": row.get("drop_distance_norm"),
            "interaction_result": row.get("interaction_result"),
            "risk_level": row.get("risk_level"),
            "input_type": row.get("input_type") or "unknown",
            "source_sample_label": row.get("sample_label"),
            "source_dataset_status": row.get("dataset_status"),
            "occurred_at": row.get("occurred_at"),
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
        },
        "trace": {
            "points": [
                [event["t_ms"], event["x_normalized"], event["y_normalized"]]
                for event in events
            ],
            "point_count": row.get("point_count"),
            "duration_ms": row.get("trace_duration_ms"),
            "box_w": row.get("box_w"),
            "box_h": row.get("box_h"),
        }
        if events
        else None,
    }


def _feature_row(payload: dict[str, Any], input_type: str) -> dict[str, Any]:
    features = extract_features(payload["events"], payload["interaction"])
    return {
        "attempt_id": payload["attempt_id"],
        "challenge_id": payload["challenge_id"],
        "session_id": payload["session_id"],
        "anonymous_participant_id": payload["anonymous_participant_id"],
        "label": "human",
        "label_source": "controlled_collection",
        "bot_family": None,
        "generator_version": None,
        "schema_version": "1.0",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "quality_status": "valid",
        "age_group": payload["collection"]["age_group"],
        "consent_version": None,
        "input_type": input_type,
        "position_correct": payload["position_correct"],
        "interaction_success": payload["interaction_success"],
        "final_drop_error": payload["final_drop_error"],
        **features,
    }


def _source_rows(cursor) -> Iterable[dict[str, Any]]:
    cursor.execute(
        """
        SELECT
            s.id, s.organization_id, s.student_id, s.source_type,
            s.solve_time_ms, s.path_length, s.avg_speed, s.pause_count,
            s.retry_count, s.drop_distance_norm, s.interaction_result,
            s.risk_level, s.occurred_at, s.created_at, s.updated_at,
            s.dataset_status, s.input_type, s.sample_label, s.actor_band,
            t.points, t.point_count, t.duration_ms AS trace_duration_ms,
            t.box_w, t.box_h
        FROM behavior_summaries AS s
        LEFT JOIN behavior_traces AS t ON t.behavior_id = s.id
        ORDER BY s.created_at, s.id
        """
    )
    yield from cursor


def export_snapshot(connection, output_dir: Path, *, hmac_key: bytes | None = None) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    key = hmac_key or secrets.token_bytes(32)

    paths = {
        name: output_dir / name
        for name in (
            "behavior_snapshot.jsonl",
            "human_attempts.jsonl",
            "human_features_valid.jsonl",
            "curation_index.jsonl",
        )
    }
    counts = Counter()
    by_input = Counter()
    by_result = Counter()
    by_label = Counter()
    by_status = Counter()
    participants: set[str] = set()

    cursor = connection.cursor()
    cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ")
    cursor.execute("SET TRANSACTION READ ONLY")
    cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT")
    try:
        with (
            paths["behavior_snapshot.jsonl"].open("w", encoding="utf-8") as snapshot_handle,
            paths["human_attempts.jsonl"].open("w", encoding="utf-8") as attempts_handle,
            paths["human_features_valid.jsonl"].open("w", encoding="utf-8") as features_handle,
            paths["curation_index.jsonl"].open("w", encoding="utf-8") as curation_handle,
        ):
            for row in _source_rows(cursor):
                counts["total"] += 1
                record_id = _pseudonym("human_db", row["id"], key)
                organization = _pseudonym("organization", row.get("organization_id"), key)
                participant = _pseudonym("participant", row.get("student_id"), key)
                if participant:
                    participants.add(participant)
                    counts["linked"] += 1
                else:
                    counts["anonymous"] += 1

                events, parse_reasons = _parse_points(
                    row.get("points"), row.get("box_w"), row.get("box_h")
                )
                status, reasons = _quality_decision(
                    events=events, parse_reasons=parse_reasons, participant=participant
                )
                counts[f"quality_{status}"] += 1
                counts["with_trace" if events else "without_trace"] += 1
                by_input[row.get("input_type") or "unknown"] += 1
                by_result[row.get("interaction_result") or "unknown"] += 1
                by_label[row.get("sample_label") or "unknown"] += 1
                by_status[row.get("dataset_status") or "unknown"] += 1

                _dump_line(
                    snapshot_handle,
                    _summary_record(
                        row,
                        record_id=record_id,
                        organization=organization,
                        participant=participant,
                        events=events,
                    ),
                )
                _dump_line(
                    curation_handle,
                    {
                        "attempt_id": record_id,
                        "anonymous_participant_id": participant,
                        "quality_status": status,
                        "reasons": reasons,
                        "source_sample_label": row.get("sample_label"),
                        "source_dataset_status": row.get("dataset_status"),
                        "input_type": row.get("input_type") or "unknown",
                        "risk_level": row.get("risk_level"),
                    },
                )
                if status == "rejected":
                    continue

                payload = _attempt_payload(
                    row, record_id=record_id, participant=participant, events=events
                )
                _dump_line(attempts_handle, payload)
                counts["attempt_payloads"] += 1
                if status == "valid":
                    _dump_line(
                        features_handle,
                        _feature_row(payload, row.get("input_type") or "unknown"),
                    )
                    counts["valid_feature_rows"] += 1
    finally:
        connection.rollback()
        cursor.close()

    manifest = {
        "snapshot_schema_version": "1.0",
        "snapshot_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "database": connection.db.decode() if isinstance(connection.db, bytes) else connection.db,
            "tables": ["behavior_summaries", "behavior_traces"],
            "mode": "read_only_consistent_snapshot",
        },
        "privacy": {
            "direct_database_ids_exported": False,
            "student_and_organization_ids": "one-way snapshot-local HMAC pseudonyms",
            "pseudonym_key_persisted": False,
            "age_group": "source actor_band",
            "consent_status": "not_present_in_source_schema; verify before production training",
        },
        "labeling": {
            "database_values_preserved_as": "source_sample_label",
            "local_label": "human",
            "local_label_source": "controlled_collection",
            "remote_database_modified": False,
        },
        "quality_policy": {
            "valid": "linked participant, >=4 valid trace points, valid dimensions, monotonic time, no extreme normalized speed",
            "pending": "unknown participant group and/or extreme normalized speed; retained for explicit owner confirmation",
            "rejected": "missing/invalid trace, <4 points, invalid dimensions, or non-monotonic timestamps",
            "extreme_normalized_speed_per_ms": MAX_NORMALIZED_SPEED_PER_MS,
        },
        "counts": dict(sorted(counts.items())),
        "distinct_linked_participants": len(participants),
        "by_input_type": dict(sorted(by_input.items())),
        "by_interaction_result": dict(sorted(by_result.items())),
        "by_source_sample_label": dict(sorted(by_label.items())),
        "by_source_dataset_status": dict(sorted(by_status.items())),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_count": len(FEATURE_NAMES),
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
            for path in paths.values()
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "README.txt").write_text(
        """Catchap human behavior database snapshot

The source database was read only. Direct database identifiers and the
snapshot-local HMAC key are not present. See manifest.json for counts, quality
policy, provenance, and file hashes.
""",
        encoding="ascii",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--user", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--password-env", default="MYSQL_PASSWORD")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    password = os.getenv(args.password_env)
    if not password:
        raise SystemExit(f"{args.password_env} must be set")
    import pymysql

    connection = pymysql.connect(
        host=args.host,
        port=args.port,
        user=args.user,
        password=password,
        database=args.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.SSDictCursor,
        connect_timeout=10,
        read_timeout=120,
        write_timeout=30,
        autocommit=False,
    )
    try:
        manifest = export_snapshot(connection, args.output_dir)
    finally:
        connection.close()
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
