"""Offline grouped-CV evaluation of additional replay-pair signals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.services.feature_profiles import get_feature_profile
from app.services.replay_detector import trace_fingerprint_from_events
from training.evaluate_models import positive_proba
from training.replay_signals import (
    SIGNAL_NAMES,
    ReplayPairSignals,
    compute_replay_pair_signals,
    signal_vector,
)
from training.run_local_training import build_bot_feature_rows, load_jsonl


@dataclass
class PairRecord:
    attempt_id: str
    participant: str
    label: int  # replay=1, normal historical comparison=0
    signals: ReplayPairSignals


@dataclass
class HumanContext:
    attempt_id: str
    participant: str
    exact_replay: bool
    recent_attempt_count: int


def select_grouped_replay_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    *,
    max_human_fpr: float,
) -> float:
    """Maximize replay recall while every fold respects the Human FPR budget."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    fold_ids = np.asarray(fold_ids, dtype=int)
    if not (scores.ndim == labels.ndim == fold_ids.ndim == 1):
        raise ValueError("scores, labels, and fold_ids must be 1D")
    if not (len(scores) == len(labels) == len(fold_ids)) or not len(scores):
        raise ValueError("scores, labels, and fold_ids must have equal non-zero length")
    if not np.isfinite(scores).all() or not 0.0 <= max_human_fpr <= 1.0:
        raise ValueError("invalid scores or max_human_fpr")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("threshold selection requires Human and replay rows")

    candidates = np.unique(np.concatenate([scores, [0.5, 1.0]]))
    human_scores = np.sort(scores[labels == 0])
    replay_scores = np.sort(scores[labels == 1])
    human_fpr = (len(human_scores) - np.searchsorted(human_scores, candidates, side="left")) / len(
        human_scores
    )
    valid = human_fpr <= max_human_fpr
    for fold in np.unique(fold_ids):
        fold_human = np.sort(scores[(fold_ids == fold) & (labels == 0)])
        fold_replay = scores[(fold_ids == fold) & (labels == 1)]
        if not len(fold_human) or not len(fold_replay):
            raise ValueError(f"fold {fold} must contain Human and replay attempts")
        fold_fpr = (
            len(fold_human) - np.searchsorted(fold_human, candidates, side="left")
        ) / len(fold_human)
        valid &= fold_fpr <= max_human_fpr

    if not valid.any():
        return 1.0
    replay_recall = (
        len(replay_scores) - np.searchsorted(replay_scores, candidates, side="left")
    ) / len(replay_scores)
    valid_indices = np.flatnonzero(valid)
    best_recall = replay_recall[valid_indices].max()
    best = valid_indices[replay_recall[valid_indices] == best_recall]
    return float(candidates[best[-1]])


def _parse_time(value: Any) -> float:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _build_human_rows(
    human_attempts: list[dict[str, Any]],
    snapshot_rows: list[dict[str, Any]],
    split_of: dict[str, str],
) -> list[dict[str, Any]]:
    attempt_by_id = {row["attempt_id"]: row for row in human_attempts}
    output: list[dict[str, Any]] = []
    for snapshot in snapshot_rows:
        attempt_id = snapshot["record_id"]
        attempt = attempt_by_id.get(attempt_id)
        participant = snapshot.get("anonymous_participant_id")
        timestamp = snapshot["summary"].get("occurred_at") or snapshot["summary"].get(
            "created_at"
        )
        split_name = split_of.get(attempt_id)
        if not attempt or not participant or not timestamp or split_name is None:
            continue
        output.append(
            {
                "attempt_id": attempt_id,
                "participant": participant,
                "epoch_s": _parse_time(timestamp),
                "events": attempt["events"],
                "split": split_name,
            }
        )
    return output


def build_human_pair_records(
    rows: list[dict[str, Any]],
    *,
    history_depth: int,
) -> tuple[list[PairRecord], list[HumanContext]]:
    by_participant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_participant[row["participant"]].append(row)

    pairs: list[PairRecord] = []
    contexts: list[HumanContext] = []
    processed = 0
    for participant_rows in by_participant.values():
        history: deque[dict[str, Any]] = deque(maxlen=history_depth)
        recent_times: deque[float] = deque()
        fingerprints: set[str] = set()
        for current in sorted(participant_rows, key=lambda item: item["epoch_s"]):
            now = current["epoch_s"]
            while recent_times and now - recent_times[0] > 60.0:
                recent_times.popleft()
            fingerprint = trace_fingerprint_from_events(current["events"])
            contexts.append(
                HumanContext(
                    attempt_id=current["attempt_id"],
                    participant=current["participant"],
                    exact_replay=bool(fingerprint and fingerprint in fingerprints),
                    recent_attempt_count=len(recent_times),
                )
            )
            for source in history:
                pairs.append(
                    PairRecord(
                        attempt_id=current["attempt_id"],
                        participant=current["participant"],
                        label=0,
                        signals=compute_replay_pair_signals(
                            current["events"], source["events"]
                        ),
                    )
                )
            if fingerprint:
                fingerprints.add(fingerprint)
            history.append(current)
            recent_times.append(now)
            processed += 1
            if processed % 2000 == 0:
                print(f"human replay signals: {processed}/{len(rows)}", flush=True)
    return pairs, contexts


def build_replay_pair_records(
    replay_rows: Iterable[dict[str, Any]],
    source_by_fingerprint: dict[str, dict[str, Any]],
) -> tuple[list[PairRecord], int]:
    output: list[PairRecord] = []
    missing = 0
    for row in replay_rows:
        fingerprint = (row.get("collection") or {}).get("replay_source_fingerprint")
        source = source_by_fingerprint.get(fingerprint)
        if source is None:
            missing += 1
            continue
        output.append(
            PairRecord(
                attempt_id=row["attempt_id"],
                participant=source["participant"],
                label=1,
                signals=compute_replay_pair_signals(row["events"], source["events"]),
            )
        )
    return output, missing


def _make_meta_model(seed: int, model_type: str) -> Pipeline | ExtraTreesClassifier:
    if model_type == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if model_type == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.8,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unsupported meta model: {model_type}")


def _pair_matrix(records: list[PairRecord]) -> np.ndarray:
    return np.vstack([signal_vector(record.signals) for record in records])


def _aggregate_attempt_scores(
    pair_records: list[PairRecord],
    pair_scores: np.ndarray,
    human_contexts: list[HumanContext],
    participant_fold: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    maximum: dict[str, float] = defaultdict(float)
    for record, score in zip(pair_records, pair_scores):
        maximum[record.attempt_id] = max(maximum[record.attempt_id], float(score))

    scores: list[float] = []
    labels: list[int] = []
    folds: list[int] = []
    attempt_ids: list[str] = []
    for context in human_contexts:
        scores.append(maximum.get(context.attempt_id, 0.0))
        labels.append(0)
        folds.append(participant_fold[context.participant])
        attempt_ids.append(context.attempt_id)
    for record, score in zip(pair_records, pair_scores):
        if record.label != 1:
            continue
        scores.append(float(score))
        labels.append(1)
        folds.append(participant_fold[record.participant])
        attempt_ids.append(record.attempt_id)
    return (
        np.asarray(scores, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(folds, dtype=int),
        attempt_ids,
    )


def _rates(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    blocked = scores >= threshold
    human = labels == 0
    replay = labels == 1
    return {
        "human_rows": int(human.sum()),
        "replay_rows": int(replay.sum()),
        "human_fpr": float(blocked[human].mean()) if human.any() else 0.0,
        "replay_recall": float(blocked[replay].mean()) if replay.any() else 0.0,
        "replay_asr": float((~blocked[replay]).mean()) if replay.any() else 0.0,
    }


def _distribution(records: list[PairRecord]) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for label, label_name in ((0, "human_pair"), (1, "replay_pair")):
        selected = [record for record in records if record.label == label]
        if not selected:
            continue
        matrix = _pair_matrix(selected)
        output[label_name] = {
            f"{name}_{stat}": float(function(matrix[:, index]))
            for index, name in enumerate(SIGNAL_NAMES)
            for stat, function in (
                ("median", np.median),
                ("p05", lambda values: np.percentile(values, 5)),
                ("p95", lambda values: np.percentile(values, 95)),
            )
        }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    split_manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    split_of = split_manifest["attempt_to_split"]
    human_attempts = load_jsonl(Path(args.human_attempts))
    snapshot_rows = load_jsonl(Path(args.human_snapshot))
    human_rows = _build_human_rows(human_attempts, snapshot_rows, split_of)
    development_humans = [row for row in human_rows if row["split"] != "test"]
    test_humans = [row for row in human_rows if row["split"] == "test"]
    source_by_fingerprint = {
        fingerprint: row
        for row in human_rows
        if (fingerprint := trace_fingerprint_from_events(row["events"]))
    }

    development_human_pairs, development_contexts = build_human_pair_records(
        development_humans,
        history_depth=args.history_depth,
    )
    training_replays = [
        row
        for path in args.training_replays
        for row in load_jsonl(Path(path))
    ]
    development_replays = [
        row
        for row in training_replays
        if (
            source := source_by_fingerprint.get(
                (row.get("collection") or {}).get("replay_source_fingerprint")
            )
        )
        and source["split"] != "test"
    ]
    development_replay_pairs, missing_development_sources = build_replay_pair_records(
        development_replays,
        source_by_fingerprint,
    )
    development_pairs = [*development_human_pairs, *development_replay_pairs]
    X = _pair_matrix(development_pairs)
    y = np.asarray([record.label for record in development_pairs], dtype=int)
    groups = np.asarray([record.participant for record in development_pairs], dtype=object)

    splitter = StratifiedGroupKFold(
        n_splits=args.cv_folds,
        shuffle=True,
        random_state=args.seed,
    )
    pair_oof = np.full(len(development_pairs), np.nan, dtype=float)
    participant_fold: dict[str, int] = {}
    for fold, (train_index, validation_index) in enumerate(splitter.split(X, y, groups)):
        model = _make_meta_model(args.seed + fold, args.meta_model)
        model.fit(X[train_index], y[train_index])
        pair_oof[validation_index] = model.predict_proba(X[validation_index])[:, 1]
        for participant in set(groups[validation_index]):
            participant_fold[str(participant)] = fold

    if not np.isfinite(pair_oof).all():
        raise RuntimeError("not every development pair received an OOF score")
    attempt_oof_scores, attempt_oof_labels, attempt_oof_folds, _ = _aggregate_attempt_scores(
        development_pairs,
        pair_oof,
        development_contexts,
        participant_fold,
    )
    threshold = select_grouped_replay_threshold(
        attempt_oof_scores,
        attempt_oof_labels,
        attempt_oof_folds,
        max_human_fpr=args.max_replay_human_fpr,
    )
    oof_metrics = _rates(attempt_oof_scores, attempt_oof_labels, threshold)
    fold_metrics = {
        str(fold): _rates(
            attempt_oof_scores[attempt_oof_folds == fold],
            attempt_oof_labels[attempt_oof_folds == fold],
            threshold,
        )
        for fold in sorted(set(attempt_oof_folds))
    }

    final_model = _make_meta_model(args.seed, args.meta_model)
    final_model.fit(X, y)

    test_human_pairs, test_contexts = build_human_pair_records(
        test_humans,
        history_depth=args.history_depth,
    )
    external_replays = load_jsonl(Path(args.external_replays))
    external_replay_pairs, missing_external_sources = build_replay_pair_records(
        external_replays,
        source_by_fingerprint,
    )
    test_pairs = [*test_human_pairs, *external_replay_pairs]
    test_pair_scores = final_model.predict_proba(_pair_matrix(test_pairs))[:, 1]
    test_pair_score_by_id: dict[str, float] = defaultdict(float)
    for record, score in zip(test_pairs, test_pair_scores):
        test_pair_score_by_id[record.attempt_id] = max(
            test_pair_score_by_id[record.attempt_id], float(score)
        )

    human_meta_scores = np.asarray(
        [test_pair_score_by_id.get(context.attempt_id, 0.0) for context in test_contexts]
    )
    replay_meta_scores = np.asarray(
        [test_pair_score_by_id[record.attempt_id] for record in external_replay_pairs]
    )
    replay_dtw_scores = np.asarray(
        [record.signals.dtw_similarity for record in external_replay_pairs], dtype=float
    )

    combined_features = {
        row["attempt_id"]: row for row in load_jsonl(Path(args.combined_features))
    }
    bundle = joblib.load(Path(args.model_bundle))
    human_feature_rows = [combined_features[context.attempt_id] for context in test_contexts]
    human_frame = pd.DataFrame(
        [[float(row.get(name, 0.0)) for name in bundle["feature_names"]] for row in human_feature_rows],
        columns=bundle["feature_names"],
    )
    human_ml_scores = positive_proba(bundle["model"], human_frame)
    replay_feature_rows = build_bot_feature_rows(
        external_replays,
        groups_per_family=3,
        profile=get_feature_profile(bundle["feature_schema_version"]),
    )
    replay_frame = pd.DataFrame(
        [[float(row.get(name, 0.0)) for name in bundle["feature_names"]] for row in replay_feature_rows],
        columns=bundle["feature_names"],
    )
    replay_ml_scores = positive_proba(bundle["model"], replay_frame)
    model_threshold = float(bundle["threshold"])

    human_ml_block = human_ml_scores < model_threshold
    human_meta_block = human_meta_scores >= threshold
    human_exact_block = np.asarray([context.exact_replay for context in test_contexts])
    human_rate_block = np.asarray(
        [context.recent_attempt_count >= args.session_rate_limit for context in test_contexts]
    )
    human_combined_block = human_ml_block | human_meta_block | human_exact_block | human_rate_block

    replay_ml_block = replay_ml_scores < model_threshold
    replay_meta_block = replay_meta_scores >= threshold
    replay_combined_block = replay_ml_block | replay_meta_block
    dtw_only_block = replay_dtw_scores >= args.dtw_threshold

    if args.meta_model == "logistic":
        logistic = final_model.named_steps["model"]
        meta_model_details: dict[str, Any] = {
            "type": "StandardScaler + class-balanced LogisticRegression",
            "coefficients": {
                name: float(value) for name, value in zip(SIGNAL_NAMES, logistic.coef_[0])
            },
            "intercept": float(logistic.intercept_[0]),
        }
    else:
        meta_model_details = {
            "type": "class-balanced ExtraTreesClassifier",
            "feature_importances": {
                name: float(value)
                for name, value in zip(SIGNAL_NAMES, final_model.feature_importances_)
            },
        }
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_decision": (
            "PASS"
            if float((~replay_combined_block).mean()) <= 0.05
            and float(human_combined_block.mean()) <= 0.03
            else "FAIL"
        ),
        "offline_only": True,
        "production_or_api_changed": False,
        "policy": {
            "history_depth": args.history_depth,
            "cv_folds": args.cv_folds,
            "max_replay_layer_human_fpr": args.max_replay_human_fpr,
            "meta_threshold": threshold,
            "model_name": bundle["model_name"],
            "model_threshold": model_threshold,
            "session_prior_attempt_limit_per_60s": args.session_rate_limit,
            "legacy_dtw_threshold": args.dtw_threshold,
            "test_used_for_threshold_selection": False,
        },
        "development": {
            "training_replay_files": [str(path) for path in args.training_replays],
            "human_participants": len({row["participant"] for row in development_humans}),
            "human_attempts": len(development_contexts),
            "human_pairs": len(development_human_pairs),
            "replay_pairs": len(development_replay_pairs),
            "missing_replay_sources": missing_development_sources,
            "oof_metrics": oof_metrics,
            "fold_metrics": fold_metrics,
            "worst_fold_human_fpr": max(
                float(item["human_fpr"]) for item in fold_metrics.values()
            ),
            "signal_distribution": _distribution(development_pairs),
        },
        "test": {
            "human_participants": len({row["participant"] for row in test_humans}),
            "human_attempts": len(test_contexts),
            "human_pairs": len(test_human_pairs),
            "external_replay_attempts": len(external_replay_pairs),
            "missing_external_sources": missing_external_sources,
            "human": {
                "ml_frr": float(human_ml_block.mean()),
                "advanced_replay_fpr": float(human_meta_block.mean()),
                "exact_fingerprint_fpr": float(human_exact_block.mean()),
                "session_rate_fpr": float(human_rate_block.mean()),
                "combined_frr": float(human_combined_block.mean()),
                "combined_blocked": int(human_combined_block.sum()),
            },
            "external_replay": {
                "legacy_dtw_asr": float((~dtw_only_block).mean()),
                "advanced_replay_asr": float((~replay_meta_block).mean()),
                "ml_asr": float((~replay_ml_block).mean()),
                "combined_asr": float((~replay_combined_block).mean()),
                "combined_passed": int((~replay_combined_block).sum()),
                "worst_case_ml_pass_asr": float((~replay_meta_block).mean()),
            },
        },
        "meta_model": {"name": args.meta_model, "signal_names": list(SIGNAL_NAMES), **meta_model_details},
        "acceptance": {
            "human_frr_max": 0.03,
            "replay_warp_asr_max": 0.05,
            "human_frr_passed": float(human_combined_block.mean()) <= 0.03,
            "replay_warp_asr_passed": float((~replay_combined_block).mean()) <= 0.05,
        },
        "limitations": [
            "The external holdout uses an isolated transform profile and untouched test participants.",
            "Only seven Human participants are present in the untouched participant split.",
            "Session-rate calibration still requires production-like Shadow Mode timestamps.",
            "The meta detector is an offline candidate and is not connected to the app or API.",
        ],
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "advanced_replay_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    joblib.dump(
        {
            "model": final_model,
            "signal_names": list(SIGNAL_NAMES),
            "threshold": threshold,
            "offline_only": True,
            "production_eligible": False,
        },
        output_dir / "advanced_replay_detector_offline.joblib",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-attempts", required=True)
    parser.add_argument("--human-snapshot", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument(
        "--training-replays",
        required=True,
        nargs="+",
        help="One or more development-only replay datasets. Test-source rows are excluded.",
    )
    parser.add_argument("--external-replays", required=True)
    parser.add_argument("--combined-features", required=True)
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history-depth", type=int, default=5)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--max-replay-human-fpr", type=float, default=0.015)
    parser.add_argument("--session-rate-limit", type=int, default=55)
    parser.add_argument("--dtw-threshold", type=float, default=0.9966927763431609)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--meta-model", choices=("logistic", "extra_trees"), default="logistic")
    return parser


def main(argv: list[str] | None = None) -> int:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
