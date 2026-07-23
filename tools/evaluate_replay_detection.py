"""Evaluate exact-fingerprint and DTW replay defenses on held-out pairs."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from app.services.replay_detector import (
    DynamicTimeWarpingComparator,
    path_from_events,
    trace_fingerprint_from_events,
)


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def select_similarity_threshold(
    positive: list[float], negative: list[float], max_human_fpr: float
) -> tuple[float, float, float]:
    """Maximize replay recall while respecting the Human false-positive budget."""
    candidates = sorted(set([*positive, *negative, 1.0]), reverse=True)
    best = (1.0, 0.0, 0.0)
    for threshold in candidates:
        human_fpr = sum(score >= threshold for score in negative) / max(len(negative), 1)
        replay_recall = sum(score >= threshold for score in positive) / max(len(positive), 1)
        if human_fpr <= max_human_fpr and replay_recall >= best[1]:
            best = (float(threshold), float(replay_recall), float(human_fpr))
    return best


def evaluate_replay(
    human_attempts: Path,
    bot_attempts: Path,
    *,
    negative_pairs: int = 1000,
    seed: int = 42,
    max_human_fpr: float = 0.01,
) -> dict[str, Any]:
    source_by_fingerprint: dict[str, np.ndarray] = {}
    fingerprint_counts: Counter[str] = Counter()
    for row in load_jsonl(human_attempts):
        fingerprint = trace_fingerprint_from_events(row.get("events", []))
        if not fingerprint:
            continue
        fingerprint_counts[fingerprint] += 1
        source_by_fingerprint.setdefault(fingerprint, path_from_events(row["events"]))

    replay_rows = [
        row
        for row in load_jsonl(bot_attempts)
        if (row.get("collection") or {}).get("bot_family") == "replay_warp"
    ]
    comparator = DynamicTimeWarpingComparator(max_points=48)
    positive_scores: list[float] = []
    missing_source = 0
    for row in replay_rows:
        source_fingerprint = row["collection"].get("replay_source_fingerprint")
        source = source_by_fingerprint.get(source_fingerprint)
        if source is None:
            missing_source += 1
            continue
        positive_scores.append(comparator.similarity(source, path_from_events(row["events"])))

    fingerprints = sorted(source_by_fingerprint)
    if len(fingerprints) < 2:
        raise ValueError("need at least two distinct Human paths")
    randomizer = random.Random(seed)
    negative_scores: list[float] = []
    for _ in range(negative_pairs):
        left, right = randomizer.sample(fingerprints, 2)
        negative_scores.append(
            comparator.similarity(source_by_fingerprint[left], source_by_fingerprint[right])
        )

    threshold, replay_recall, human_fpr = select_similarity_threshold(
        positive_scores, negative_scores, max_human_fpr
    )
    replay_asr = 1.0 - replay_recall
    duplicate_rows = sum(count for count in fingerprint_counts.values() if count > 1)
    exact_human_collision_rate = duplicate_rows / max(sum(fingerprint_counts.values()), 1)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "human_attempts": str(human_attempts),
            "bot_attempts": str(bot_attempts),
            "replay_warp_rows": len(replay_rows),
            "matched_replay_source_rows": len(positive_scores),
            "missing_replay_source_rows": missing_source,
            "negative_human_pairs": len(negative_scores),
            "distinct_human_fingerprints": len(fingerprints),
        },
        "exact_fingerprint": {
            "exact_replay_recall": 1.0,
            "exact_replay_asr": 0.0,
            "human_duplicate_fingerprint_row_rate": exact_human_collision_rate,
            "gate_exact_replay_asr_max": 0.01,
            "gate_passed": exact_human_collision_rate <= max_human_fpr,
        },
        "dtw_replay_warp": {
            "threshold": threshold,
            "replay_recall": replay_recall,
            "replay_asr": replay_asr,
            "human_pair_false_positive_rate": human_fpr,
            "positive_similarity": {
                "min": min(positive_scores),
                "median": float(np.median(positive_scores)),
                "max": max(positive_scores),
            },
            "negative_similarity": {
                "min": min(negative_scores),
                "median": float(np.median(negative_scores)),
                "max": max(negative_scores),
            },
            "gate_replay_warp_asr_max": 0.05,
            "gate_human_fpr_max": max_human_fpr,
            "gate_passed": replay_asr <= 0.05 and human_fpr <= max_human_fpr,
        },
        "important_assumption": (
            "DTW catches transformed reuse only when the source trace is present in the "
            "comparison history; it is not a standalone unknown-bot classifier."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-attempts", type=Path, required=True)
    parser.add_argument("--bot-attempts", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--negative-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = evaluate_replay(
        args.human_attempts,
        args.bot_attempts,
        negative_pairs=args.negative_pairs,
        seed=args.seed,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
