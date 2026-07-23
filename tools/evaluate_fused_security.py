"""Evaluate the final ML + replay + session-rate security decision."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd

from app.services.replay_detector import (
    DynamicTimeWarpingComparator,
    HistoricalAttempt,
    ReplayFeatures,
    compute_replay_features,
    path_from_events,
    trace_fingerprint,
    trace_fingerprint_from_events,
)
from app.services.risk_fusion import RiskFusionPolicy, fuse_behavior_risk
from app.services.feature_profiles import get_feature_profile
from tools.evaluate_replay_detection import select_similarity_threshold
from training.evaluate_models import positive_proba
from training.run_local_training import build_bot_feature_rows, load_jsonl


def _parse_time(value: Any) -> float:
    if value is None:
        raise ValueError("timestamp is required")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def select_session_rate_limit(
    human_recent_counts: Iterable[int],
    *,
    max_human_fpr: float = 0.01,
) -> tuple[int, float]:
    """Choose the smallest prior-attempt limit within a Human FPR budget."""
    counts = np.asarray(list(human_recent_counts), dtype=int)
    if counts.ndim != 1 or not len(counts) or (counts < 0).any():
        raise ValueError("human_recent_counts must be non-empty non-negative integers")
    if not 0.0 <= max_human_fpr <= 1.0:
        raise ValueError("max_human_fpr must be between 0 and 1")
    for limit in range(1, int(counts.max()) + 2):
        fpr = float(np.mean(counts >= limit))
        if fpr <= max_human_fpr:
            return limit, fpr
    return int(counts.max()) + 1, 0.0


def _recent_counts(rows: list[dict[str, Any]], window_s: float = 60.0) -> dict[str, int]:
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        participant = row.get("anonymous_participant_id")
        if participant:
            by_participant[participant].append(row)

    output: dict[str, int] = {}
    for participant_rows in by_participant.values():
        recent: deque[float] = deque()
        for row in sorted(participant_rows, key=lambda item: item["epoch_s"]):
            now = float(row["epoch_s"])
            while recent and now - recent[0] > window_s:
                recent.popleft()
            output[row["attempt_id"]] = len(recent)
            recent.append(now)
    return output


def _duration_ms(events: list[dict[str, Any]]) -> float:
    values = [float(event.get("t_ms", 0.0)) for event in events]
    return max(values) - min(values) if len(values) >= 2 else 0.0


def _historical(events: list[dict[str, Any]], epoch_s: float) -> HistoricalAttempt:
    path = path_from_events(events)
    endpoint = tuple(path[-1]) if len(path) else (0.0, 0.0)
    return HistoricalAttempt(
        path=path,
        duration_ms=_duration_ms(events),
        endpoint=(float(endpoint[0]), float(endpoint[1])),
        created_at_epoch_s=epoch_s,
        path_fingerprint=trace_fingerprint(path),
    )


def _empty_replay(*, attempts_per_minute: float = 0.0) -> ReplayFeatures:
    return ReplayFeatures(
        path_similarity_score=0.0,
        exact_replay_detected=False,
        repeated_duration_count=0,
        attempts_per_minute=attempts_per_minute,
        recent_attempt_count=int(attempts_per_minute),
        repeated_endpoint_count=0,
    )


def _human_sequence_replay_features(
    rows: list[dict[str, Any]],
    *,
    history_depth: int,
) -> dict[str, ReplayFeatures]:
    comparator = DynamicTimeWarpingComparator(max_points=48)
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        participant = row.get("anonymous_participant_id")
        if participant:
            by_participant[participant].append(row)

    output: dict[str, ReplayFeatures] = {}
    for participant_rows in by_participant.values():
        recent_times: deque[float] = deque()
        recent_paths: deque[HistoricalAttempt] = deque(maxlen=history_depth)
        fingerprints: set[str] = set()
        for row in sorted(participant_rows, key=lambda item: item["epoch_s"]):
            now = float(row["epoch_s"])
            while recent_times and now - recent_times[0] > 60.0:
                recent_times.popleft()
            events = row["events"]
            features = compute_replay_features(
                events,
                duration_ms=_duration_ms(events),
                now_epoch_s=now,
                history=list(recent_paths),
                comparator=comparator,
            )
            fingerprint = trace_fingerprint_from_events(events)
            output[row["attempt_id"]] = replace(
                features,
                exact_replay_detected=bool(fingerprint and fingerprint in fingerprints),
                attempts_per_minute=float(len(recent_times)),
                recent_attempt_count=len(recent_times),
            )
            if fingerprint:
                fingerprints.add(fingerprint)
            recent_times.append(now)
            recent_paths.append(_historical(events, now))
    return output


def _score_rows(bundle: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    names = bundle["feature_names"]
    frame = pd.DataFrame(
        [[float(row.get(name, 0.0)) for name in names] for row in rows],
        columns=names,
    )
    return positive_proba(bundle["model"], frame)


def _decision_metrics(
    scores: np.ndarray,
    replay_features: list[ReplayFeatures],
    policy: RiskFusionPolicy,
    *,
    human: bool,
) -> dict[str, Any]:
    baseline_blocked = scores < policy.model_human_threshold
    decisions = [
        fuse_behavior_risk(float(score), replay, policy)
        for score, replay in zip(scores, replay_features)
    ]
    fused_blocked = np.asarray([decision.high_risk for decision in decisions], dtype=bool)
    reasons = Counter(reason for decision in decisions for reason in decision.reasons)
    output = {
        "rows": len(scores),
        "baseline_blocked": int(baseline_blocked.sum()),
        "fused_blocked": int(fused_blocked.sum()),
        "reason_counts": dict(sorted(reasons.items())),
    }
    if human:
        output["baseline_human_frr"] = float(baseline_blocked.mean())
        output["fused_human_frr"] = float(fused_blocked.mean())
    else:
        output["baseline_bot_asr"] = float((~baseline_blocked).mean())
        output["fused_bot_asr"] = float((~fused_blocked).mean())
    return output


def _calibrate_dtw(
    development_human_features: list[ReplayFeatures],
    development_replays: list[dict[str, Any]],
    source_by_fingerprint: dict[str, HistoricalAttempt],
    *,
    max_human_fpr: float,
    max_pairs: int = 1000,
) -> dict[str, Any]:
    comparator = DynamicTimeWarpingComparator(max_points=48)
    positives: list[float] = []
    for row in development_replays:
        source_fingerprint = (row.get("collection") or {}).get("replay_source_fingerprint")
        source = source_by_fingerprint.get(source_fingerprint)
        if source is None:
            continue
        positives.append(comparator.similarity(path_from_events(row["events"]), source.path))
        if len(positives) >= max_pairs:
            break

    negatives = [item.path_similarity_score for item in development_human_features]
    threshold, recall, human_fpr = select_similarity_threshold(
        positives,
        negatives,
        max_human_fpr=max_human_fpr,
    )
    return {
        "threshold": threshold,
        "development_replay_pairs": len(positives),
        "development_human_sequence_rows": len(negatives),
        "development_replay_recall": recall,
        "development_human_sequence_fpr": human_fpr,
        "max_human_fpr": max_human_fpr,
        "human_similarity_percentiles": {
            str(percentile): float(np.percentile(negatives, percentile))
            for percentile in (90, 95, 99, 99.5, 99.9, 100)
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    combined_rows = load_jsonl(Path(args.combined_features))
    split_manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    split_of = split_manifest["attempt_to_split"]
    feature_by_id = {row["attempt_id"]: row for row in combined_rows}

    human_attempts = load_jsonl(Path(args.human_attempts))
    human_attempt_by_id = {row["attempt_id"]: row for row in human_attempts}
    snapshot_rows = load_jsonl(Path(args.human_snapshot))
    human_meta: dict[str, dict[str, Any]] = {}
    for row in snapshot_rows:
        attempt_id = row["record_id"]
        attempt = human_attempt_by_id.get(attempt_id)
        if attempt_id not in feature_by_id or attempt is None:
            continue
        timestamp = row["summary"].get("occurred_at") or row["summary"].get("created_at")
        participant = row.get("anonymous_participant_id")
        if not participant or not timestamp:
            continue
        human_meta[attempt_id] = {
            "attempt_id": attempt_id,
            "anonymous_participant_id": participant,
            "epoch_s": _parse_time(timestamp),
            "events": attempt["events"],
        }

    development_humans = [
        row for attempt_id, row in human_meta.items() if split_of.get(attempt_id) != "test"
    ]
    test_humans = [
        row for attempt_id, row in human_meta.items() if split_of.get(attempt_id) == "test"
    ]
    rate_counts = _recent_counts(development_humans)
    rate_limit, development_rate_fpr = select_session_rate_limit(
        rate_counts.values(), max_human_fpr=args.max_session_human_fpr
    )

    source_by_fingerprint: dict[str, HistoricalAttempt] = {}
    for row in human_meta.values():
        historical = _historical(row["events"], row["epoch_s"])
        if historical.path_fingerprint:
            source_by_fingerprint.setdefault(historical.path_fingerprint, historical)

    bot_attempts = load_jsonl(Path(args.bot_attempts))
    bot_attempt_by_id = {row["attempt_id"]: row for row in bot_attempts}
    development_replays = [
        row
        for row in bot_attempts
        if (row.get("collection") or {}).get("bot_family") == "replay_warp"
        and split_of.get(row["attempt_id"]) != "test"
    ]
    development_human_replay_map = _human_sequence_replay_features(
        development_humans,
        history_depth=args.dtw_history_depth,
    )
    dtw = _calibrate_dtw(
        [development_human_replay_map[row["attempt_id"]] for row in development_humans],
        development_replays,
        source_by_fingerprint,
        max_human_fpr=args.max_dtw_human_fpr,
    )

    human_replay = _human_sequence_replay_features(
        test_humans,
        history_depth=args.dtw_history_depth,
    )
    test_human_feature_rows = [feature_by_id[row["attempt_id"]] for row in test_humans]
    test_human_replay_rows = [human_replay[row["attempt_id"]] for row in test_humans]

    test_bot_feature_rows = [
        row
        for row in combined_rows
        if row["label"] == "bot" and split_of.get(row["attempt_id"]) == "test"
    ]
    test_bot_raw_rows = [bot_attempt_by_id[row["attempt_id"]] for row in test_bot_feature_rows]
    test_bot_replay: list[ReplayFeatures] = []
    for row in test_bot_raw_rows:
        source_fingerprint = (row.get("collection") or {}).get("replay_source_fingerprint")
        source = source_by_fingerprint.get(source_fingerprint)
        if source is None:
            test_bot_replay.append(_empty_replay())
            continue
        test_bot_replay.append(
            compute_replay_features(
                row["events"],
                duration_ms=_duration_ms(row["events"]),
                now_epoch_s=source.created_at_epoch_s + 120.0,
                history=[source],
                comparator=DynamicTimeWarpingComparator(max_points=48),
            )
        )

    external_raw = load_jsonl(Path(args.external_bots))
    external_features = build_bot_feature_rows(
        external_raw,
        groups_per_family=3,
        profile=get_feature_profile(args.feature_schema_version),
    )
    external_replay = [_empty_replay() for _ in external_features]

    exact_rows = test_human_feature_rows[: args.attack_sample_size]
    exact_replay = [
        ReplayFeatures(
            path_similarity_score=1.0,
            exact_replay_detected=True,
            repeated_duration_count=1,
            attempts_per_minute=0.0,
            recent_attempt_count=0,
            repeated_endpoint_count=1,
        )
        for _ in exact_rows
    ]

    report: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_policy": {
            "fusion": "high risk when ML OR exact fingerprint OR DTW OR session rate fires",
            "dtw_similarity_threshold": dtw["threshold"],
            "session_prior_attempt_limit_per_60s": rate_limit,
            "dtw_history_depth": args.dtw_history_depth,
            "test_set_used_for_calibration": False,
        },
        "calibration": {
            "dtw": dtw,
            "session_rate": {
                "development_human_rows": len(rate_counts),
                "max_human_fpr": args.max_session_human_fpr,
                "selected_prior_attempt_limit": rate_limit,
                "development_human_fpr": development_rate_fpr,
                "warning": (
                    "The collection workflow is bursty and is not a production-session sample; "
                    "recalibrate this limit in Shadow Mode."
                ),
            },
        },
        "data": {
            "test_human_rows": len(test_humans),
            "test_known_bot_rows": len(test_bot_feature_rows),
            "external_browser_bot_rows": len(external_features),
            "exact_replay_rows": len(exact_rows),
        },
        "models": {},
        "caveats": [
            "DTW can only detect transformed reuse when a comparable source trace is retained.",
            "Known bots are evaluated as independent attempts except replay_warp source history.",
            "The burst-rate result is a controlled 60-second stress scenario, not observed bot traffic.",
            "No candidate is promoted by this evaluator.",
        ],
    }

    for model_path in sorted(Path(args.candidate_dir).glob("*.joblib")):
        bundle = joblib.load(model_path)
        name = bundle["model_name"]
        policy = RiskFusionPolicy(
            model_human_threshold=float(bundle["threshold"]),
            dtw_similarity_threshold=float(dtw["threshold"]),
            max_attempts_per_minute=float(rate_limit),
        )
        human_scores = _score_rows(bundle, test_human_feature_rows)
        bot_scores = _score_rows(bundle, test_bot_feature_rows)
        external_scores = _score_rows(bundle, external_features)
        exact_scores = _score_rows(bundle, exact_rows)

        known_by_family: dict[str, Any] = {}
        for family in sorted({row.get("bot_family") or "unknown" for row in test_bot_feature_rows}):
            indices = [
                index
                for index, row in enumerate(test_bot_feature_rows)
                if (row.get("bot_family") or "unknown") == family
            ]
            known_by_family[family] = _decision_metrics(
                bot_scores[indices],
                [test_bot_replay[index] for index in indices],
                policy,
                human=False,
            )

        external_by_family: dict[str, Any] = {}
        for family in sorted({row.get("bot_family") or "unknown" for row in external_features}):
            indices = [
                index
                for index, row in enumerate(external_features)
                if (row.get("bot_family") or "unknown") == family
            ]
            external_by_family[family] = _decision_metrics(
                external_scores[indices],
                [external_replay[index] for index in indices],
                policy,
                human=False,
            )

        replay_indices = [
            index
            for index, row in enumerate(test_bot_feature_rows)
            if row.get("bot_family") == "replay_warp"
        ]
        worst_case_replay = _decision_metrics(
            np.ones(len(replay_indices), dtype=float),
            [test_bot_replay[index] for index in replay_indices],
            policy,
            human=False,
        )

        burst_count = min(args.burst_attempts, len(test_bot_feature_rows))
        burst_order = np.random.default_rng(args.seed).permutation(len(bot_scores))[:burst_count]
        burst_scores = bot_scores[burst_order]
        burst_replay = [
            _empty_replay(attempts_per_minute=float(index))
            for index in range(burst_count)
        ]
        report["models"][name] = {
            "threshold": float(bundle["threshold"]),
            "human_test": _decision_metrics(
                human_scores, test_human_replay_rows, policy, human=True
            ),
            "known_bot_test": _decision_metrics(
                bot_scores, test_bot_replay, policy, human=False
            ),
            "known_bot_by_family": known_by_family,
            "external_browser_bot": _decision_metrics(
                external_scores, external_replay, policy, human=False
            ),
            "external_browser_bot_by_family": external_by_family,
            "replay_warp_worst_case_ml_pass": worst_case_replay,
            "exact_replay_attack": _decision_metrics(
                exact_scores, exact_replay, policy, human=False
            ),
            "simulated_60s_bot_burst": _decision_metrics(
                burst_scores, burst_replay, policy, human=False
            ),
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-features", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--human-attempts", required=True)
    parser.add_argument("--human-snapshot", required=True)
    parser.add_argument("--bot-attempts", required=True)
    parser.add_argument("--external-bots", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtw-history-depth", type=int, default=5)
    parser.add_argument("--max-dtw-human-fpr", type=float, default=0.01)
    parser.add_argument("--max-session-human-fpr", type=float, default=0.005)
    parser.add_argument("--attack-sample-size", type=int, default=1000)
    parser.add_argument("--burst-attempts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-schema-version", choices=("1.0", "2.0"), default="2.0")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
