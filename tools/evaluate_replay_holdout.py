"""Score an unseen replay holdout with the frozen offline replay detector."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.services.feature_profiles import get_feature_profile
from app.services.replay_detector import trace_fingerprint_from_events
from training.evaluate_models import positive_proba
from training.replay_signals import compute_replay_pair_signals, signal_vector
from training.run_local_training import build_bot_feature_rows, load_jsonl


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + (z * z / (2.0 * total))) / denominator
    margin = z * math.sqrt(
        (proportion * (1.0 - proportion) / total) + (z * z / (4.0 * total * total))
    ) / denominator
    lower = center - margin
    upper = center + margin
    return [0.0 if abs(lower) < 1e-15 else max(0.0, lower), min(1.0, upper)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    detector = joblib.load(Path(args.replay_detector))
    meta_model = detector["model"]
    meta_threshold = float(detector["threshold"])
    detector_signal_names = tuple(detector.get("signal_names") or ())
    if not detector.get("offline_only"):
        raise ValueError("expected an offline replay detector bundle")
    if not detector_signal_names:
        raise ValueError("replay detector bundle is missing its signal schema")

    human_attempts = load_jsonl(Path(args.human_attempts))
    source_by_fingerprint = {
        fingerprint: row
        for row in human_attempts
        if (fingerprint := trace_fingerprint_from_events(row["events"]))
    }
    holdout_rows = load_jsonl(Path(args.holdout))
    vectors: list[np.ndarray] = []
    dtw_scores: list[float] = []
    missing_source = 0
    exact_fingerprint_matches = 0
    transforms: list[dict[str, Any]] = []
    matched_rows: list[dict[str, Any]] = []
    for row in holdout_rows:
        collection = row.get("collection") or {}
        source = source_by_fingerprint.get(collection.get("replay_source_fingerprint"))
        if source is None:
            missing_source += 1
            continue
        signals = compute_replay_pair_signals(row["events"], source["events"])
        vectors.append(signal_vector(signals, detector_signal_names))
        dtw_scores.append(signals.dtw_similarity)
        matched_rows.append(row)
        exact_fingerprint_matches += int(
            trace_fingerprint_from_events(row["events"])
            == collection.get("replay_source_fingerprint")
        )
        transforms.append(collection.get("transform") or {})
    if not vectors:
        raise ValueError("no holdout rows could be matched to a retained source")

    X = np.vstack(vectors)
    meta_scores = meta_model.predict_proba(X)[:, 1]
    meta_blocked = meta_scores >= meta_threshold
    dtw_blocked = np.asarray(dtw_scores) >= args.dtw_threshold

    model_bundle = joblib.load(Path(args.model_bundle))
    feature_rows = build_bot_feature_rows(
        matched_rows,
        groups_per_family=3,
        profile=get_feature_profile(model_bundle["feature_schema_version"]),
    )
    model_frame = pd.DataFrame(
        [
            [float(row.get(name, 0.0)) for name in model_bundle["feature_names"]]
            for row in feature_rows
        ],
        columns=model_bundle["feature_names"],
    )
    human_scores = positive_proba(model_bundle["model"], model_frame)
    ml_blocked = human_scores < float(model_bundle["threshold"])
    combined_blocked = meta_blocked | ml_blocked
    count = len(meta_scores)
    advanced_passed = int((~meta_blocked).sum())
    combined_passed = int((~combined_blocked).sum())
    dtw_passed = int((~dtw_blocked).sum())

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "offline_only": True,
        "production_or_api_changed": False,
        "holdout_rows": count,
        "missing_source_rows": missing_source,
        "attack_variant": (matched_rows[0].get("collection") or {}).get("attack_variant"),
        "frozen_detector": {
            "signal_names": list(detector_signal_names),
            "meta_threshold": meta_threshold,
            "legacy_dtw_threshold": args.dtw_threshold,
            "model_name": model_bundle["model_name"],
            "model_threshold": float(model_bundle["threshold"]),
        },
        "results": {
            "exact_fingerprint_asr": float(1.0 - (exact_fingerprint_matches / count)),
            "legacy_dtw_asr": dtw_passed / count,
            "advanced_replay_asr": advanced_passed / count,
            "combined_asr": combined_passed / count,
            "advanced_replay_asr_95_wilson": wilson_interval(advanced_passed, count),
            "combined_asr_95_wilson": wilson_interval(combined_passed, count),
            "ml_asr": float((~ml_blocked).mean()),
        },
        "signal_summary": {
            "meta_score_percentiles": {
                str(percentile): float(np.percentile(meta_scores, percentile))
                for percentile in (0, 5, 50, 95, 100)
            },
            "dtw_score_percentiles": {
                str(percentile): float(np.percentile(dtw_scores, percentile))
                for percentile in (0, 5, 50, 95, 100)
            },
            "rotation_abs_degrees_percentiles": {
                str(percentile): float(
                    np.percentile([abs(float(item.get("rotation_degrees", 0.0))) for item in transforms], percentile)
                )
                for percentile in (0, 50, 100)
            },
            "event_count_ratio_percentiles": {
                str(percentile): float(
                    np.percentile(
                        [
                            float(item.get("target_event_count", 0))
                            / max(float(item.get("source_event_count", 1)), 1.0)
                            for item in transforms
                        ],
                        percentile,
                    )
                )
                for percentile in (0, 50, 100)
            },
        },
        "acceptance": {
            "replay_warp_asr_max": 0.05,
            "advanced_replay_passed": (advanced_passed / count) <= 0.05,
            "combined_passed": (combined_passed / count) <= 0.05,
        },
        "limitations": [
            "The detector was frozen before this adversarial holdout was generated.",
            "This test does not remeasure Human FRR because the frozen detector and threshold are unchanged.",
            "The holdout remains a synthetic local security test, not observed production attacker traffic.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-detector", required=True)
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--human-attempts", required=True)
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtw-threshold", type=float, default=0.9966927763431609)
    return parser


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(build_parser().parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
