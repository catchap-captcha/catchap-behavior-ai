"""Run a bounded Random Forest check for selected unseen bot families.

This is intentionally an offline candidate experiment. It uses precomputed
development features, keeps external holdouts score-only, and calibrates its
threshold on the validation split before scoring test data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
from lightgbm import LGBMClassifier

from app.services.feature_profiles import get_feature_profile
from training.build_dataset import Dataset, build_dataset
from training.evaluate_models import evaluate, select_threshold
from training.run_local_training import (
    _subset_split,
    build_bot_feature_rows,
    build_local_split,
    load_jsonl,
)


def _fast_lightgbm(seed: int) -> LGBMClassifier:
    """Bounded smoke-test model; it is not the 300-tree release candidate."""
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _family_holdout(
    dataset: Dataset,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    family: str,
    seed: int,
) -> dict[str, Any]:
    split_of = manifest["attempt_to_split"]
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    for index, row in enumerate(rows):
        split_name = split_of[row["attempt_id"]]
        if row["label"] == "human":
            target = {"train": train_indices, "val": validation_indices, "test": test_indices}[split_name]
            target.append(index)
        elif row.get("bot_family") == family:
            test_indices.append(index)
        elif split_name == "train":
            train_indices.append(index)
        elif split_name == "val":
            validation_indices.append(index)

    split = _subset_split(dataset, train_indices, validation_indices, test_indices)
    model = _fast_lightgbm(seed)
    model.fit(split.X_train, split.y_train)
    threshold = select_threshold(model, split.X_val, split.y_val)
    evaluation = evaluate(
        model,
        f"lightgbm_holdout_{family}",
        split.X_test,
        split.y_test,
        threshold,
        "family_holdout_test",
    )
    result = asdict(evaluation)
    result.update(
        {
            "held_out_bot_family": family,
            "bot_asr": 1.0 - evaluation.bot_recall,
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "test_rows": len(test_indices),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-features", required=True)
    parser.add_argument("--bot-features", required=True)
    parser.add_argument("--external-bot-holdout")
    parser.add_argument("--report", required=True)
    parser.add_argument("--model-output")
    parser.add_argument("--feature-schema-version", default="2.3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--family", action="append")
    parser.add_argument("--skip-external", action="store_true")
    parser.add_argument("--skip-family-holdout", action="store_true")
    parser.add_argument("--family-only", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    def progress(message: str) -> None:
        if args.progress:
            print(message, flush=True)

    progress("[1/7] loading precomputed development features")
    profile = get_feature_profile(args.feature_schema_version, trajectory_only=True)
    human_rows = load_jsonl(Path(args.human_features))
    bot_rows = load_jsonl(Path(args.bot_features))
    rows = [*human_rows, *bot_rows]
    progress("[2/7] preparing grouped split")
    dataset = build_dataset(
        rows,
        feature_names=profile.names,
        expected_schema_version=profile.version,
    )
    split = build_local_split(dataset, rows, seed=args.seed)
    model: LGBMClassifier | None = None
    threshold: float | None = None
    test_output: dict[str, Any] | None = None
    if not args.family_only:
        progress("[3/7] fitting focused LightGBM")
        model = _fast_lightgbm(args.seed)
        model.fit(split.X_train, split.y_train)
        threshold = select_threshold(model, split.X_val, split.y_val)
        test = evaluate(model, "lightgbm", split.X_test, split.y_test, threshold, "untouched_test")
        test_output = asdict(test)
    if args.model_output:
        if model is None or threshold is None:
            raise ValueError("--model-output cannot be used with --family-only")
        model_path = Path(args.model_output)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "threshold": threshold,
                "feature_names": list(profile.names),
                "feature_schema_version": profile.version,
                "feature_input_scope": profile.input_scope,
            },
            model_path,
        )

    external_output: dict[str, Any] | None = None
    if not args.skip_external:
        if args.family_only:
            raise ValueError("--family-only requires --skip-external")
        if not args.external_bot_holdout:
            raise ValueError("--external-bot-holdout is required unless --skip-external is set")
        progress("[4/7] extracting external holdout features")
        external_payloads = load_jsonl(Path(args.external_bot_holdout))
        external_rows = build_bot_feature_rows(
            external_payloads,
            groups_per_family=3,
            profile=profile,
            allow_external_holdout=True,
        )
        external_dataset = build_dataset(
            external_rows,
            feature_names=profile.names,
            expected_schema_version=profile.version,
        )
        external = evaluate(
            model,
            "lightgbm_external",
            external_dataset.X,
            external_dataset.y,
            threshold,
            "external_bot_holdout",
        )
        external_output = {
            **asdict(external),
            "bot_asr": 1.0 - external.bot_recall,
            "rows": len(external_rows),
        }
    progress("[5/7] evaluating selected family holdouts")
    output = {
        "scope": "focused candidate evaluation; not production selection",
        "model": "lightgbm_120_tree_smoke_test",
        "model_limitations": "This bounded model is for directional family checks only, not release selection.",
        "feature_schema_version": profile.version,
        "threshold_calibration": "validation split only",
        "external_holdout_training_usage": "score_only",
        "test": test_output,
        "external_holdout": external_output,
        "family_holdouts": [
            _family_holdout(dataset, rows, split.manifest, family, args.seed + index + 1)
            for index, family in enumerate(
                [] if args.skip_family_holdout else args.family or ["pca_gmm_surrogate", "stop_go"]
            )
        ],
    }
    progress("[6/7] writing report")
    _write_json(Path(args.report), output)
    progress("[7/7] done")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
