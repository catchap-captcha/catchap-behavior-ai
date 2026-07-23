"""Run bounded stages of the formal grouped-OOF validation pipeline.

Each command preserves the same split, model parameters, and threshold logic
as ``training.run_local_training``. It exists for environments where one
long-lived process cannot complete the entire 5-fold Random Forest run.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from app.services.feature_profiles import get_feature_profile
from training.build_dataset import build_dataset
from training.evaluate_models import (
    evaluate,
    evaluate_scores,
    positive_proba,
    select_threshold,
    select_threshold_from_scores,
)
from training.group_threshold_cv import row_group
from training.run_local_training import _subset_split, build_local_split, load_jsonl
from training.train_models import build_models


def _load_data(human_path: str, bot_path: str):
    profile = get_feature_profile("2.3", trajectory_only=True)
    rows = [*load_jsonl(Path(human_path)), *load_jsonl(Path(bot_path))]
    dataset = build_dataset(
        rows,
        feature_names=profile.names,
        expected_schema_version=profile.version,
    )
    split = build_local_split(dataset, rows, seed=42)
    development_indices = [
        index
        for index, row in enumerate(rows)
        if split.manifest["attempt_to_split"][row["attempt_id"]] != "test"
    ]
    return profile, rows, dataset, split, development_indices


def _oof_inputs(dataset, rows, development_indices):
    eligible: list[int] = []
    groups: list[str] = []
    for index in development_indices:
        group = row_group(rows[index])
        if group is not None:
            eligible.append(index)
            groups.append(group)
    X = dataset.X.iloc[eligible].reset_index(drop=True)
    y = dataset.y.iloc[eligible].reset_index(drop=True)
    return eligible, X, y, np.asarray(groups, dtype=object)


def _fold_splits(X, y, groups):
    labels = y.to_numpy(dtype=int)
    class_group_counts = [len(set(groups[labels == label])) for label in (0, 1)]
    actual_splits = min(5, *class_group_counts)
    splitter = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    return list(splitter.split(X, labels, groups=groups)), actual_splits


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_fold(args: argparse.Namespace) -> int:
    _, rows, dataset, _, development_indices = _load_data(args.human_features, args.bot_features)
    _, X, y, groups = _oof_inputs(dataset, rows, development_indices)
    folds, actual_splits = _fold_splits(X, y, groups)
    if args.fold < 0 or args.fold >= actual_splits:
        raise ValueError(f"fold must be between 0 and {actual_splits - 1}")
    train_rel, validation_rel = folds[args.fold]
    model = build_models(y.iloc[train_rel], seed=42 + args.fold)[args.model]
    model.fit(X.iloc[train_rel], y.iloc[train_rel])
    started = time.perf_counter()
    scores = positive_proba(model, X.iloc[validation_rel])
    avg_ms = (time.perf_counter() - started) * 1000.0 / max(len(validation_rel), 1)
    output = Path(args.work_dir) / f"{args.model}_fold_{args.fold}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        validation_rel=validation_rel,
        scores=scores,
        avg_inference_ms=avg_ms,
    )
    print(json.dumps({"fold": args.fold, "rows": len(validation_rel), "output": str(output)}))
    return 0


def assemble_oof(args: argparse.Namespace) -> int:
    _, rows, dataset, _, development_indices = _load_data(args.human_features, args.bot_features)
    _, X, y, groups = _oof_inputs(dataset, rows, development_indices)
    folds, actual_splits = _fold_splits(X, y, groups)
    scores = np.full(len(X), np.nan, dtype=float)
    fold_ids = np.full(len(X), -1, dtype=int)
    timings: dict[int, float] = {}
    for fold, (_, validation_rel) in enumerate(folds):
        path = Path(args.work_dir) / f"{args.model}_fold_{fold}.npz"
        if not path.exists():
            raise ValueError(f"missing OOF fold artifact: {path}")
        payload = np.load(path)
        stored_rel = payload["validation_rel"]
        if not np.array_equal(stored_rel, validation_rel):
            raise ValueError(f"OOF fold artifact does not match deterministic split: {path}")
        scores[validation_rel] = payload["scores"]
        fold_ids[validation_rel] = fold
        timings[fold] = float(payload["avg_inference_ms"])
    labels = y.to_numpy(dtype=int)
    threshold = select_threshold_from_scores(scores, labels, max_frr=0.03, fold_ids=fold_ids)
    pooled = evaluate_scores(
        scores,
        labels,
        model_name=args.model,
        threshold=threshold,
        metrics_on="group_cv_oof",
        avg_inference_ms=float(np.mean(list(timings.values()))),
    )
    fold_reports = []
    standalone_thresholds = []
    for fold in range(actual_splits):
        mask = fold_ids == fold
        standalone = select_threshold_from_scores(scores[mask], labels[mask], max_frr=0.03)
        standalone_thresholds.append(standalone)
        metrics = evaluate_scores(
            scores[mask],
            labels[mask],
            model_name=args.model,
            threshold=threshold,
            metrics_on=f"group_cv_fold_{fold}",
            avg_inference_ms=timings[fold],
        )
        fold_reports.append(
            {
                "fold": fold,
                "rows": int(mask.sum()),
                "human_rows": int((labels[mask] == 1).sum()),
                "bot_rows": int((labels[mask] == 0).sum()),
                "human_groups": len(set(groups[mask & (labels == 1)])),
                "bot_groups": len(set(groups[mask & (labels == 0)])),
                "standalone_threshold": standalone,
                "metrics_at_shared_threshold": asdict(metrics),
            }
        )
    report = {
        "model_name": args.model,
        "feature_schema_version": "2.3",
        "threshold": threshold,
        "requested_splits": 5,
        "actual_splits": actual_splits,
        "max_human_frr": 0.03,
        "eligible_rows": len(X),
        "excluded_anonymous_human_rows": len(development_indices) - len(X),
        "linked_human_groups": len(set(groups[labels == 1])),
        "bot_groups": len(set(groups[labels == 0])),
        "pooled_oof_metrics": asdict(pooled),
        "worst_fold_human_frr": max(
            item["metrics_at_shared_threshold"]["human_frr"] for item in fold_reports
        ),
        "fold_threshold_min": float(np.min(standalone_thresholds)),
        "fold_threshold_median": float(np.median(standalone_thresholds)),
        "fold_threshold_max": float(np.max(standalone_thresholds)),
        "folds": fold_reports,
        "policy": "all-fold Human FRR constraint on out-of-fold scores",
    }
    _write_json(Path(args.report), report)
    print(json.dumps({"threshold": threshold, "report": args.report}))
    return 0


def run_final_test(args: argparse.Namespace) -> int:
    profile, rows, dataset, split, development_indices = _load_data(
        args.human_features, args.bot_features
    )
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    model = build_models(dataset.y.iloc[development_indices], seed=42)[args.model]
    model.fit(dataset.X.iloc[development_indices], dataset.y.iloc[development_indices])
    evaluation = evaluate(
        model,
        args.model,
        split.X_test,
        split.y_test,
        float(calibration["threshold"]),
        "untouched_test",
    )
    bundle = {
        "model": model,
        "model_name": args.model,
        "threshold": float(calibration["threshold"]),
        "feature_names": list(profile.names),
        "feature_schema_version": profile.version,
        "feature_input_scope": profile.input_scope,
    }
    model_path = Path(args.model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    report = {"model_name": args.model, "test": asdict(evaluation), "model_path": str(model_path)}
    _write_json(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def run_family_holdout(args: argparse.Namespace) -> int:
    _, rows, dataset, split, _ = _load_data(args.human_features, args.bot_features)
    families = sorted({row.get("bot_family") for row in rows if row["label"] == "bot"})
    if any(family not in families for family in args.family):
        unknown = sorted(set(args.family) - set(families))
        raise ValueError(f"unknown Bot family: {unknown}")
    if len(args.family) > 1 and args.report_dir is None:
        raise ValueError("multiple families require --report-dir")

    summaries = []
    split_of = split.manifest["attempt_to_split"]
    for family in args.family:
        family_index = families.index(family)
        train_indices: list[int] = []
        validation_indices: list[int] = []
        test_indices: list[int] = []
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

        held_out_split = _subset_split(dataset, train_indices, validation_indices, test_indices)
        model = build_models(held_out_split.y_train, seed=42 + family_index)[args.model]
        model.fit(held_out_split.X_train, held_out_split.y_train)
        threshold = select_threshold(model, held_out_split.X_val, held_out_split.y_val)
        evaluation = evaluate(
            model,
            f"{args.model}_holdout_{family}",
            held_out_split.X_test,
            held_out_split.y_test,
            threshold,
            "family_holdout_test",
        )
        report = {
            **asdict(evaluation),
            "held_out_bot_family": family,
            "bot_asr": 1.0 - evaluation.bot_recall,
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "test_rows": len(test_indices),
        }
        report_path = (
            Path(args.report)
            if args.report is not None
            else Path(args.report_dir) / f"{family}.json"
        )
        _write_json(report_path, report)
        summaries.append({"family": family, "bot_asr": report["bot_asr"], "report": str(report_path)})
    print(json.dumps(summaries, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fold", "assemble", "final-test", "family-holdout"))
    parser.add_argument("--human-features", required=True)
    parser.add_argument("--bot-features", required=True)
    parser.add_argument("--model", required=True, choices=("random_forest", "extra_trees", "xgboost", "lightgbm"))
    parser.add_argument("--work-dir")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--report")
    parser.add_argument("--report-dir")
    parser.add_argument("--calibration")
    parser.add_argument("--model-output")
    parser.add_argument("--family", action="append")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.stage == "fold":
        if args.work_dir is None or args.fold is None:
            raise ValueError("fold stage requires --work-dir and --fold")
        return run_fold(args)
    if args.stage == "assemble":
        if args.work_dir is None or args.report is None:
            raise ValueError("assemble stage requires --work-dir and --report")
        return assemble_oof(args)
    if args.stage == "family-holdout":
        if not args.family or (args.report is None and args.report_dir is None):
            raise ValueError("family-holdout stage requires --family and --report or --report-dir")
        return run_family_holdout(args)
    if args.calibration is None or args.report is None or args.model_output is None:
        raise ValueError("final-test stage requires --calibration, --report, and --model-output")
    return run_final_test(args)


if __name__ == "__main__":
    raise SystemExit(main())
