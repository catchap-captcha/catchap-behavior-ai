"""Data readiness gate.

Reads the ``ai_training_dataset`` view and decides whether there is enough
valid, labelled data to train. If not, it writes ``reports/data_readiness.json``
and exits with code 2 WITHOUT training or overwriting any model.

The readiness thresholds are project defaults to prevent premature training —
NOT absolute research thresholds. Override them via env or CLI.

The core :func:`compute_readiness` works on a list of row dicts so it is unit
testable without a database.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from app.config import get_settings
from app.services.feature_extractor import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

REPORT_PATH = os.path.join("reports", "data_readiness.json")


@dataclass
class Thresholds:
    min_human_samples: int
    min_bot_samples: int
    min_human_participants: int
    min_bot_families: int

    @classmethod
    def from_settings(cls) -> "Thresholds":
        s = get_settings()
        return cls(
            s.min_human_samples,
            s.min_bot_samples,
            s.min_human_participants,
            s.min_bot_families,
        )


@dataclass
class ReadinessReport:
    ready: bool
    reason: str
    human_samples: int
    required_human_samples: int
    bot_samples: int
    required_bot_samples: int
    human_participants: int
    required_human_participants: int
    bot_families: int
    required_bot_families: int
    label_source_missing: int
    feature_schema_mismatch: int
    feature_null_or_nonfinite: int
    class_imbalance_ratio: float | None
    participant_split_possible: bool
    generator_split_possible: bool
    missing: list[str] = field(default_factory=list)


def compute_readiness(
    rows: list[dict[str, Any]],
    thr: Thresholds,
    *,
    feature_names=FEATURE_NAMES,
    expected_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> ReadinessReport:
    """Assess readiness from training-view rows."""
    human = [r for r in rows if r.get("label") == "human"]
    bot = [r for r in rows if r.get("label") == "bot"]

    human_participants = len({r.get("anonymous_participant_id") for r in human if r.get("anonymous_participant_id")})
    bot_families = len({r.get("bot_family") for r in bot if r.get("bot_family")})
    label_source_missing = sum(1 for r in rows if not r.get("label_source"))

    schema_mismatch = sum(
        1 for r in rows
        if r.get("feature_schema_version") not in (None, expected_schema_version)
    )
    null_nonfinite = _count_bad_features(rows, feature_names)

    imbalance = None
    if human and bot:
        imbalance = round(max(len(human), len(bot)) / min(len(human), len(bot)), 4)

    missing: list[str] = []
    if len(human) < thr.min_human_samples:
        missing.append(f"Human 데이터 {thr.min_human_samples - len(human)}개 부족")
    if len(bot) < thr.min_bot_samples:
        missing.append(f"Bot 데이터 {thr.min_bot_samples - len(bot)}개 부족")
    if human_participants < thr.min_human_participants:
        missing.append(f"Human 참여자 {thr.min_human_participants - human_participants}명 부족")
    if bot_families < thr.min_bot_families:
        missing.append(f"Bot family {thr.min_bot_families - bot_families}종 부족")
    if schema_mismatch:
        missing.append(f"Feature 스키마 버전 불일치 {schema_mismatch}건")
    if null_nonfinite:
        missing.append(f"Feature NULL/NaN/Infinity {null_nonfinite}건")

    ready = len(missing) == 0
    return ReadinessReport(
        ready=ready,
        reason="ready" if ready else "data_not_ready",
        human_samples=len(human),
        required_human_samples=thr.min_human_samples,
        bot_samples=len(bot),
        required_bot_samples=thr.min_bot_samples,
        human_participants=human_participants,
        required_human_participants=thr.min_human_participants,
        bot_families=bot_families,
        required_bot_families=thr.min_bot_families,
        label_source_missing=label_source_missing,
        feature_schema_mismatch=schema_mismatch,
        feature_null_or_nonfinite=null_nonfinite,
        class_imbalance_ratio=imbalance,
        # a grouped split needs at least 2 distinct groups per class dimension
        participant_split_possible=human_participants >= 2,
        generator_split_possible=bot_families >= 2,
        missing=missing,
    )


def _count_bad_features(rows: list[dict[str, Any]], feature_names=FEATURE_NAMES) -> int:
    bad = 0
    for r in rows:
        for name in feature_names:
            if name not in r:
                continue
            v = r[name]
            if v is None:
                bad += 1
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                bad += 1
                continue
            if not math.isfinite(fv):
                bad += 1
    return bad


def write_report(report: ReadinessReport, path: str = REPORT_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(asdict(report), fh, ensure_ascii=False, indent=2)


def _fetch_rows_from_db() -> list[dict[str, Any]]:
    """Read the training view. Returns [] and prints a hint if DB/view absent."""
    from app.database.connection import check_connection, get_sessionmaker
    from app.database.repositories import TrainingDatasetRepository

    if not check_connection():
        print("MySQL 연결 실패: 연결 정보를 확인하세요.", file=sys.stderr)
        return []
    session = get_sessionmaker()()
    try:
        repo = TrainingDatasetRepository(session)
        if not repo.view_exists():
            print("ai_training_dataset 뷰가 없습니다. DB 담당자에게 DDL 적용을 요청하세요.", file=sys.stderr)
            return []
        return repo.fetch_all()
    finally:
        session.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check training data readiness.")
    parser.add_argument("--min-human-samples", type=int, default=None)
    parser.add_argument("--min-bot-samples", type=int, default=None)
    parser.add_argument("--min-human-participants", type=int, default=None)
    parser.add_argument("--min-bot-families", type=int, default=None)
    args = parser.parse_args(argv)

    thr = Thresholds.from_settings()
    if args.min_human_samples is not None:
        thr.min_human_samples = args.min_human_samples
    if args.min_bot_samples is not None:
        thr.min_bot_samples = args.min_bot_samples
    if args.min_human_participants is not None:
        thr.min_human_participants = args.min_human_participants
    if args.min_bot_families is not None:
        thr.min_bot_families = args.min_bot_families

    rows = _fetch_rows_from_db()
    report = compute_readiness(rows, thr)
    write_report(report)

    if report.ready:
        print("데이터 준비 완료: 학습을 진행할 수 있습니다.")
        return 0
    print("데이터가 준비되지 않았습니다 (data_not_ready):")
    for item in report.missing:
        print(f"  - {item}")
    print(f"보고서: {REPORT_PATH}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
