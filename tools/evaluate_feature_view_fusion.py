"""Evaluate a conservative fusion of two trajectory-only feature views.

Both models are fitted without the requested bot family. Their Human scores
are combined with ``min`` so either view may raise risk. The fused threshold is
then calibrated on validation Humans, keeping the false-rejection budget in
the same place as a single-model evaluation.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
import joblib

from app.services.feature_profiles import get_feature_profile
from training.build_dataset import Dataset, build_dataset
from training.evaluate_models import evaluate_scores, positive_proba, select_threshold_from_scores
from training.run_local_training import _subset_split, build_local_split, load_jsonl


def _model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=120,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        n_jobs=1,
        random_state=seed,
        verbose=-1,
    )


def _split_indices(rows: list[dict[str, Any]], manifest: dict[str, Any], family: str):
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    split_of = manifest["attempt_to_split"]
    for index, row in enumerate(rows):
        split_name = split_of[row["attempt_id"]]
        if row["label"] == "human":
            {"train": train_indices, "val": validation_indices, "test": test_indices}[split_name].append(index)
        elif row.get("bot_family") == family:
            test_indices.append(index)
        elif split_name == "train":
            train_indices.append(index)
        elif split_name == "val":
            validation_indices.append(index)
    return train_indices, validation_indices, test_indices


def _load_rows(human_path: str, bot_path: str) -> list[dict[str, Any]]:
    return [*load_jsonl(Path(human_path)), *load_jsonl(Path(bot_path))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-view-a", required=True)
    parser.add_argument("--bot-view-a", required=True)
    parser.add_argument("--human-view-b", required=True)
    parser.add_argument("--bot-view-b", required=True)
    parser.add_argument("--family")
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows_a = _load_rows(args.human_view_a, args.bot_view_a)
    rows_b = _load_rows(args.human_view_b, args.bot_view_b)
    identity_a = [(row["attempt_id"], row["label"], row.get("bot_family")) for row in rows_a]
    identity_b = [(row["attempt_id"], row["label"], row.get("bot_family")) for row in rows_b]
    if identity_a != identity_b:
        raise ValueError("feature views must contain the same rows in the same order")

    profile = get_feature_profile("2.3", trajectory_only=True)
    dataset_a = build_dataset(rows_a, feature_names=profile.names, expected_schema_version="2.3")
    dataset_b = build_dataset(rows_b, feature_names=profile.names, expected_schema_version="2.3")
    primary_split = build_local_split(dataset_a, rows_a, seed=args.seed)
    if args.family:
        indices = _split_indices(rows_a, primary_split.manifest, args.family)
        split_a = _subset_split(dataset_a, *indices)
        split_b = _subset_split(dataset_b, *indices)
        metrics_on = "family_holdout_test"
    else:
        split_a = primary_split
        split_b = build_local_split(dataset_b, rows_b, seed=args.seed)
        indices = (split_a.X_train.index.tolist(), split_a.X_val.index.tolist(), split_a.X_test.index.tolist())
        metrics_on = "untouched_test"

    model_a = _model(args.seed)
    model_b = _model(args.seed + 1)
    model_a.fit(split_a.X_train, split_a.y_train)
    model_b.fit(split_b.X_train, split_b.y_train)
    validation_scores = np.minimum(
        positive_proba(model_a, split_a.X_val),
        positive_proba(model_b, split_b.X_val),
    )
    threshold = select_threshold_from_scores(validation_scores, split_a.y_val.to_numpy())
    test_scores = np.minimum(
        positive_proba(model_a, split_a.X_test),
        positive_proba(model_b, split_b.X_test),
    )
    evaluation = evaluate_scores(
        test_scores,
        split_a.y_test.to_numpy(),
        model_name="lightgbm_two_view_min_fusion",
        threshold=threshold,
        metrics_on=metrics_on,
    )
    result: dict[str, Any] = {
        "scope": "focused score-fusion experiment; not production selection",
        "feature_schema_version": "2.3",
        "score_fusion": "min(P_human_view_a, P_human_view_b)",
        "threshold_calibration": "validation split only, Human FRR <= 3%",
        "held_out_bot_family": args.family,
        "bot_asr": 1.0 - evaluation.bot_recall,
        "metrics": asdict(evaluation),
        "train_rows": len(split_a.X_train),
        "validation_rows": len(split_a.X_val),
        "test_rows": len(split_a.X_test),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.model_output:
        output_path = Path(args.model_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models": [model_a, model_b],
                "threshold": threshold,
                "feature_names": list(profile.names),
                "feature_schema_version": profile.version,
                "feature_input_scope": profile.input_scope,
                "score_fusion": "min(P_human_view_a, P_human_view_b)",
            },
            output_path,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
