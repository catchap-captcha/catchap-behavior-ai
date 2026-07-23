"""Run the formal schema-2.3 two-view LightGBM validation stages.

The detector trains only on the supplied development Human/Bot rows.  Every
stage uses the same participant and Bot-generator grouping as the established
formal runner.  The fusion score is ``min(P_human_general, P_human_dynamics)``:
either independent view can flag a trace as risky.

External holdouts are score-only.  They are never accepted as fitting or
threshold-calibration inputs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedGroupKFold

from app.services.feature_profiles import get_feature_profile
from app.services.trajectory_feature_views import get_feature_view
from training.build_dataset import Dataset, build_dataset
from training.evaluate_models import (
    evaluate_scores,
    positive_proba,
    select_threshold_per_human_group,
    select_threshold_from_scores,
)
from training.group_threshold_cv import row_group
from training.holdout_registry import assert_not_sealed_training_inputs, sealed_holdout_reason
from training.run_local_training import build_bot_feature_rows, build_local_split, load_jsonl


MODEL_NAME = "lightgbm_general_dynamics_min_fusion"
VIEW_A = "general_without_physics"
VIEW_B = "dynamics_physics"
FUSION_RULE = "min(P_human_general_without_physics, P_human_dynamics_physics)"


def _model(seed: int) -> LGBMClassifier:
    """Use the formal 300-tree LightGBM settings for both independent views."""
    return LGBMClassifier(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_data(
    human_path: str,
    bot_path: str,
    *,
    all_input_development: bool = False,
):
    profile = get_feature_profile("2.3", trajectory_only=True)
    human = Path(human_path)
    bot = Path(bot_path)
    assert_not_sealed_training_inputs([human, bot])
    rows = [*load_jsonl(human), *load_jsonl(bot)]
    all_dataset = build_dataset(
        rows,
        feature_names=profile.names,
        expected_schema_version=profile.version,
    )
    split = build_local_split(all_dataset, rows, seed=42)
    development_indices = (
        list(range(len(rows)))
        if all_input_development
        else [
            index
            for index, row in enumerate(rows)
            if split.manifest["attempt_to_split"][row["attempt_id"]] != "test"
        ]
    )
    view_a = Dataset(
        X=all_dataset.X.loc[:, get_feature_view(VIEW_A)].copy(),
        y=all_dataset.y,
        groups=all_dataset.groups,
        meta=all_dataset.meta,
    )
    view_b = Dataset(
        X=all_dataset.X.loc[:, get_feature_view(VIEW_B)].copy(),
        y=all_dataset.y,
        groups=all_dataset.groups,
        meta=all_dataset.meta,
    )
    return profile, rows, view_a, view_b, split, development_indices


def _oof_inputs(dataset: Dataset, rows: list[dict[str, Any]], development_indices: list[int]):
    eligible: list[int] = []
    groups: list[str] = []
    for index in development_indices:
        group = row_group(rows[index])
        if group is not None:
            eligible.append(index)
            groups.append(group)
    return (
        eligible,
        dataset.X.iloc[eligible].reset_index(drop=True),
        dataset.y.iloc[eligible].reset_index(drop=True),
        np.asarray(groups, dtype=object),
    )


def _fold_splits(X, y, groups):
    labels = y.to_numpy(dtype=int)
    class_group_counts = [len(set(groups[labels == label])) for label in (0, 1)]
    actual_splits = min(5, *class_group_counts)
    if actual_splits < 2:
        raise ValueError("need at least two Human and Bot groups for grouped OOF")
    splitter = StratifiedGroupKFold(n_splits=actual_splits, shuffle=True, random_state=42)
    return list(splitter.split(X, labels, groups=groups)), actual_splits


def _fused_scores(model_a: Any, X_a, model_b: Any, X_b) -> np.ndarray:
    return np.minimum(positive_proba(model_a, X_a), positive_proba(model_b, X_b))


def _model_name(hard_negative_weight: float | None) -> str:
    if hard_negative_weight is None:
        return MODEL_NAME
    return f"{MODEL_NAME}_hard_negative_w{hard_negative_weight:g}"


def _select_calibration_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    fold_ids: np.ndarray,
    *,
    max_human_frr: float,
    human_frr_policy: str,
) -> float:
    if human_frr_policy == "per_participant":
        return select_threshold_per_human_group(
            scores,
            labels,
            groups,
            max_frr=max_human_frr,
        )
    return select_threshold_from_scores(
        scores,
        labels,
        max_frr=max_human_frr,
        fold_ids=fold_ids,
    )


def _worst_human_group_frr(
    scores: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    threshold: float,
) -> float:
    human_groups = np.unique(groups[labels == 1])
    return max(
        float((scores[(labels == 1) & (groups == group)] < threshold).mean())
        for group in human_groups
    )


def _hard_negative_weights(
    view_a: Dataset,
    view_b: Dataset,
    rows: list[dict[str, Any]],
    indices: list[int],
    *,
    hard_negative_weight: float,
    seed_offset: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find difficult development Bots without leaking an outer fold.

    This nested OOF pass only scores rows inside ``indices``.  A Bot receives
    additional weight when its inner-OOF fused Human score would have passed
    the inner Human-safe threshold.  Rows without a safe group key (anonymous
    Humans) retain weight one and never participate in the calibration.
    """
    if hard_negative_weight <= 1.0:
        raise ValueError("hard-negative weight must be greater than 1")

    eligible_positions: list[int] = []
    groups: list[str] = []
    for position, index in enumerate(indices):
        group = row_group(rows[index])
        if group is not None:
            eligible_positions.append(position)
            groups.append(group)
    y = view_a.y.iloc[[indices[position] for position in eligible_positions]].reset_index(drop=True)
    X_a = view_a.X.iloc[[indices[position] for position in eligible_positions]].reset_index(drop=True)
    X_b = view_b.X.iloc[[indices[position] for position in eligible_positions]].reset_index(drop=True)
    group_array = np.asarray(groups, dtype=object)
    labels = y.to_numpy(dtype=int)
    class_group_counts = [len(set(group_array[labels == label])) for label in (0, 1)]
    actual_splits = min(4, *class_group_counts)
    if actual_splits < 2:
        raise ValueError("hard-negative weighting needs at least two Human and Bot groups")

    splitter = StratifiedGroupKFold(
        n_splits=actual_splits,
        shuffle=True,
        random_state=2042 + seed_offset,
    )
    scores = np.full(len(X_a), np.nan, dtype=float)
    fold_ids = np.full(len(X_a), -1, dtype=int)
    for fold, (train_rel, validation_rel) in enumerate(splitter.split(X_a, labels, groups=group_array)):
        model_a = _model(3042 + seed_offset * 10 + fold)
        model_b = _model(4042 + seed_offset * 10 + fold)
        model_a.fit(X_a.iloc[train_rel], y.iloc[train_rel])
        model_b.fit(X_b.iloc[train_rel], y.iloc[train_rel])
        scores[validation_rel] = _fused_scores(
            model_a, X_a.iloc[validation_rel], model_b, X_b.iloc[validation_rel]
        )
        fold_ids[validation_rel] = fold
    if not np.isfinite(scores).all():
        raise RuntimeError("inner grouped OOF did not produce every score")

    threshold = select_threshold_from_scores(scores, labels, max_frr=0.03, fold_ids=fold_ids)
    hard_mask = (labels == 0) & (scores >= threshold)
    weights = np.ones(len(indices), dtype=float)
    eligible_weights = np.where(hard_mask, hard_negative_weight, 1.0)
    weights[np.asarray(eligible_positions, dtype=int)] = eligible_weights
    return weights, {
        "inner_oof_splits": actual_splits,
        "inner_oof_threshold": float(threshold),
        "eligible_rows": len(eligible_positions),
        "excluded_anonymous_human_rows": len(indices) - len(eligible_positions),
        "hard_negative_bot_rows": int(hard_mask.sum()),
        "hard_negative_weight": hard_negative_weight,
    }


def _fit_pair(
    view_a: Dataset,
    view_b: Dataset,
    indices: list[int],
    *,
    seed_offset: int,
    sample_weight: np.ndarray | None = None,
):
    y = view_a.y.iloc[indices]
    model_a = _model(42 + seed_offset)
    model_b = _model(1042 + seed_offset)
    fit_kwargs = {"sample_weight": sample_weight} if sample_weight is not None else {}
    model_a.fit(view_a.X.iloc[indices], y, **fit_kwargs)
    model_b.fit(view_b.X.iloc[indices], y, **fit_kwargs)
    return model_a, model_b


def run_fold(args: argparse.Namespace) -> int:
    _, rows, view_a, view_b, _, development_indices = _load_data(
        args.human_features,
        args.bot_features,
        all_input_development=args.all_input_development,
    )
    _, X_a, y, groups = _oof_inputs(view_a, rows, development_indices)
    _, X_b, _, _ = _oof_inputs(view_b, rows, development_indices)
    folds, actual_splits = _fold_splits(X_a, y, groups)
    if args.fold < 0 or args.fold >= actual_splits:
        raise ValueError(f"fold must be between 0 and {actual_splits - 1}")
    train_rel, validation_rel = folds[args.fold]
    train_indices = [development_indices[index] for index in train_rel]
    sample_weight = None
    hard_negative = None
    if args.hard_negative_weight is not None:
        sample_weight, hard_negative = _hard_negative_weights(
            view_a,
            view_b,
            rows,
            train_indices,
            hard_negative_weight=args.hard_negative_weight,
            seed_offset=args.fold,
        )
    model_a, model_b = _fit_pair(
        view_a,
        view_b,
        train_indices,
        seed_offset=args.fold,
        sample_weight=sample_weight,
    )
    started = time.perf_counter()
    scores = _fused_scores(
        model_a, X_a.iloc[validation_rel], model_b, X_b.iloc[validation_rel]
    )
    avg_ms = (time.perf_counter() - started) * 1000.0 / max(len(validation_rel), 1)
    output = Path(args.work_dir) / f"two_view_fold_{args.fold}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        validation_rel=validation_rel,
        scores=scores,
        avg_inference_ms=avg_ms,
        hard_negative_bot_rows=(hard_negative or {}).get("hard_negative_bot_rows", 0),
    )
    print(json.dumps({"fold": args.fold, "rows": len(validation_rel), "hard_negative": hard_negative, "output": str(output)}))
    return 0


def assemble_oof(args: argparse.Namespace) -> int:
    _, rows, view_a, _, _, development_indices = _load_data(
        args.human_features,
        args.bot_features,
        all_input_development=args.all_input_development,
    )
    _, X_a, y, groups = _oof_inputs(view_a, rows, development_indices)
    folds, actual_splits = _fold_splits(X_a, y, groups)
    scores = np.full(len(X_a), np.nan, dtype=float)
    fold_ids = np.full(len(X_a), -1, dtype=int)
    timings: dict[int, float] = {}
    hard_negative_rows: dict[int, int] = {}
    for fold, (_, validation_rel) in enumerate(folds):
        path = Path(args.work_dir) / f"two_view_fold_{fold}.npz"
        if not path.exists():
            raise ValueError(f"missing OOF fold artifact: {path}")
        payload = np.load(path)
        if not np.array_equal(payload["validation_rel"], validation_rel):
            raise ValueError(f"OOF fold artifact does not match deterministic split: {path}")
        scores[validation_rel] = payload["scores"]
        fold_ids[validation_rel] = fold
        timings[fold] = float(payload["avg_inference_ms"])
        hard_negative_rows[fold] = int(payload["hard_negative_bot_rows"])
    labels = y.to_numpy(dtype=int)
    threshold = _select_calibration_threshold(
        scores,
        labels,
        groups,
        fold_ids,
        max_human_frr=args.max_human_frr,
        human_frr_policy=args.human_frr_policy,
    )
    pooled = evaluate_scores(
        scores,
        labels,
        model_name=_model_name(args.hard_negative_weight),
        threshold=threshold,
        metrics_on="group_cv_oof",
        avg_inference_ms=float(np.mean(list(timings.values()))),
    )
    fold_reports = []
    standalone_thresholds = []
    for fold in range(actual_splits):
        mask = fold_ids == fold
        standalone = _select_calibration_threshold(
            scores[mask],
            labels[mask],
            groups[mask],
            fold_ids[mask],
            max_human_frr=args.max_human_frr,
            human_frr_policy=args.human_frr_policy,
        )
        standalone_thresholds.append(standalone)
        metrics = evaluate_scores(
            scores[mask],
            labels[mask],
            model_name=_model_name(args.hard_negative_weight),
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
                "hard_negative_bot_rows_in_outer_train": hard_negative_rows[fold],
                "metrics_at_shared_threshold": asdict(metrics),
            }
        )
    report = {
        "model_name": _model_name(args.hard_negative_weight),
        "feature_schema_version": "2.3",
        "all_input_development": args.all_input_development,
        "feature_views": {VIEW_A: list(get_feature_view(VIEW_A)), VIEW_B: list(get_feature_view(VIEW_B))},
        "score_fusion": FUSION_RULE,
        "hard_negative_weight": args.hard_negative_weight,
        "hard_negative_policy": (
            "nested_inner_grouped_oof_only; Bot passing inner Human-safe threshold gets extra weight"
            if args.hard_negative_weight is not None
            else "disabled"
        ),
        "threshold": threshold,
        "requested_splits": 5,
        "actual_splits": actual_splits,
        "max_human_frr": args.max_human_frr,
        "human_frr_policy": args.human_frr_policy,
        "eligible_rows": len(X_a),
        "excluded_anonymous_human_rows": len(development_indices) - len(X_a),
        "linked_human_groups": len(set(groups[labels == 1])),
        "bot_groups": len(set(groups[labels == 0])),
        "pooled_oof_metrics": asdict(pooled),
        "worst_fold_human_frr": max(
            item["metrics_at_shared_threshold"]["human_frr"] for item in fold_reports
        ),
        "worst_human_group_frr": _worst_human_group_frr(scores, labels, groups, threshold),
        "fold_threshold_min": float(np.min(standalone_thresholds)),
        "fold_threshold_median": float(np.median(standalone_thresholds)),
        "fold_threshold_max": float(np.max(standalone_thresholds)),
        "folds": fold_reports,
        "policy": (
            "every Human participant group must satisfy the FRR constraint on two-view "
            "out-of-fold scores"
            if args.human_frr_policy == "per_participant"
            else "all-fold Human FRR constraint on two-view out-of-fold scores"
        ),
    }
    _write_json(Path(args.report), report)
    print(json.dumps({"threshold": threshold, "report": args.report}))
    return 0


def _development_indices(split, rows: list[dict[str, Any]]) -> list[int]:
    return [
        index
        for index, row in enumerate(rows)
        if split.manifest["attempt_to_split"][row["attempt_id"]] != "test"
    ]


def run_final_test(args: argparse.Namespace) -> int:
    if args.all_input_development:
        raise ValueError("final-test cannot use --all-input-development; use fit-lockbox-candidate")
    profile, rows, view_a, view_b, split, development_indices = _load_data(
        args.human_features, args.bot_features
    )
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    sample_weight = None
    hard_negative = None
    if args.hard_negative_weight is not None:
        sample_weight, hard_negative = _hard_negative_weights(
            view_a,
            view_b,
            rows,
            development_indices,
            hard_negative_weight=args.hard_negative_weight,
            seed_offset=99,
        )
    model_a, model_b = _fit_pair(
        view_a,
        view_b,
        development_indices,
        seed_offset=0,
        sample_weight=sample_weight,
    )
    test_indices = [
        index
        for index, row in enumerate(rows)
        if split.manifest["attempt_to_split"][row["attempt_id"]] == "test"
    ]
    started = time.perf_counter()
    scores = _fused_scores(
        model_a, view_a.X.iloc[test_indices], model_b, view_b.X.iloc[test_indices]
    )
    avg_ms = (time.perf_counter() - started) * 1000.0 / max(len(test_indices), 1)
    evaluation = evaluate_scores(
        scores,
        view_a.y.iloc[test_indices].to_numpy(),
        model_name=_model_name(args.hard_negative_weight),
        threshold=float(calibration["threshold"]),
        metrics_on="untouched_test",
        avg_inference_ms=avg_ms,
    )
    model_version = args.model_version or Path(args.model_output).parent.name
    bundle = {
        "models": {VIEW_A: model_a, VIEW_B: model_b},
        "model_name": _model_name(args.hard_negative_weight),
        "model_version": model_version,
        "threshold": float(calibration["threshold"]),
        "feature_schema_version": profile.version,
        "feature_names": list(profile.names),
        "feature_input_scope": profile.input_scope,
        "feature_views": {VIEW_A: list(get_feature_view(VIEW_A)), VIEW_B: list(get_feature_view(VIEW_B))},
        "score_fusion": FUSION_RULE,
        "hard_negative": hard_negative,
    }
    output = Path(args.model_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    report = {
        "model_name": _model_name(args.hard_negative_weight),
        "hard_negative": hard_negative,
        "test": asdict(evaluation),
        "model_path": str(output),
    }
    _write_json(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def run_fit_lockbox_candidate(args: argparse.Namespace) -> int:
    """Fit on every non-lockbox development row after OOF calibration."""
    profile, rows, view_a, view_b, _, development_indices = _load_data(
        args.human_features,
        args.bot_features,
        all_input_development=True,
    )
    calibration = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    if calibration.get("all_input_development") is not True:
        raise ValueError("fit-lockbox-candidate requires calibration from --all-input-development")
    threshold_calibration = {
        "source": str(args.calibration),
        "max_human_frr": calibration.get("max_human_frr"),
        "human_frr_policy": calibration.get("human_frr_policy", "per_fold"),
        "policy": calibration.get("policy"),
    }
    sample_weight = None
    hard_negative = None
    if args.hard_negative_weight is not None:
        sample_weight, hard_negative = _hard_negative_weights(
            view_a,
            view_b,
            rows,
            development_indices,
            hard_negative_weight=args.hard_negative_weight,
            seed_offset=99,
        )
    model_a, model_b = _fit_pair(
        view_a,
        view_b,
        development_indices,
        seed_offset=0,
        sample_weight=sample_weight,
    )
    model_version = args.model_version or Path(args.model_output).parent.name
    bundle = {
        "models": {VIEW_A: model_a, VIEW_B: model_b},
        "model_name": _model_name(args.hard_negative_weight),
        "model_version": model_version,
        "threshold": float(calibration["threshold"]),
        "feature_schema_version": profile.version,
        "feature_names": list(profile.names),
        "feature_input_scope": profile.input_scope,
        "feature_views": {VIEW_A: list(get_feature_view(VIEW_A)), VIEW_B: list(get_feature_view(VIEW_B))},
        "score_fusion": FUSION_RULE,
        "hard_negative": hard_negative,
        "threshold_calibration": threshold_calibration,
        "fit_scope": "all non-lockbox revalidation development rows",
    }
    output = Path(args.model_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    report = {
        "model_name": _model_name(args.hard_negative_weight),
        "fit_scope": bundle["fit_scope"],
        "development_rows": len(development_indices),
        "model_version": model_version,
        "hard_negative": hard_negative,
        "threshold_calibration": threshold_calibration,
        "model_path": str(output),
    }
    _write_json(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _family_indices(rows: list[dict[str, Any]], split, family: str):
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    split_of = split.manifest["attempt_to_split"]
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


def run_family_holdout(args: argparse.Namespace) -> int:
    _, rows, view_a, view_b, split, _ = _load_data(args.human_features, args.bot_features)
    families = sorted({row.get("bot_family") for row in rows if row["label"] == "bot"})
    if any(family not in families for family in args.family):
        unknown = sorted(set(args.family) - set(families))
        raise ValueError(f"unknown Bot family: {unknown}")
    if len(args.family) > 1 and args.report_dir is None:
        raise ValueError("multiple families require --report-dir")
    summaries = []
    for family in args.family:
        family_index = families.index(family)
        train_indices, validation_indices, test_indices = _family_indices(rows, split, family)
        sample_weight = None
        hard_negative = None
        if args.hard_negative_weight is not None:
            sample_weight, hard_negative = _hard_negative_weights(
                view_a,
                view_b,
                rows,
                train_indices,
                hard_negative_weight=args.hard_negative_weight,
                seed_offset=200 + family_index,
            )
        model_a, model_b = _fit_pair(
            view_a,
            view_b,
            train_indices,
            seed_offset=family_index,
            sample_weight=sample_weight,
        )
        validation_scores = _fused_scores(
            model_a,
            view_a.X.iloc[validation_indices],
            model_b,
            view_b.X.iloc[validation_indices],
        )
        threshold = select_threshold_from_scores(
            validation_scores, view_a.y.iloc[validation_indices].to_numpy(), max_frr=0.03
        )
        test_scores = _fused_scores(
            model_a, view_a.X.iloc[test_indices], model_b, view_b.X.iloc[test_indices]
        )
        evaluation = evaluate_scores(
            test_scores,
            view_a.y.iloc[test_indices].to_numpy(),
            model_name=f"{_model_name(args.hard_negative_weight)}_holdout_{family}",
            threshold=threshold,
            metrics_on="family_holdout_test",
        )
        report = {
            **asdict(evaluation),
            "held_out_bot_family": family,
            "bot_asr": 1.0 - evaluation.bot_recall,
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "test_rows": len(test_indices),
            "hard_negative": hard_negative,
        }
        report_path = (
            Path(args.report) if args.report is not None else Path(args.report_dir) / f"{family}.json"
        )
        _write_json(report_path, report)
        summaries.append({"family": family, "bot_asr": report["bot_asr"], "report": str(report_path)})
    print(json.dumps(summaries, ensure_ascii=False))
    return 0


def run_external_score(args: argparse.Namespace) -> int:
    bundle = joblib.load(args.model)
    if bundle.get("score_fusion") != FUSION_RULE:
        raise ValueError("expected formal two-view min-fusion model bundle")
    profile = get_feature_profile(bundle["feature_schema_version"], trajectory_only=True)
    payloads = load_jsonl(Path(args.external_bot_holdout))
    expected_usage = (
        "redteam_only" if args.score_scope == "redteam_calibration" else "external_holdout_only"
    )
    actual_usages = {payload.get("collection", {}).get("training_usage") for payload in payloads}
    if actual_usages != {expected_usage}:
        raise ValueError(
            f"{args.score_scope} requires only {expected_usage} payloads, found {sorted(actual_usages)}"
        )
    rows = build_bot_feature_rows(
        payloads,
        groups_per_family=3,
        profile=profile,
        allow_external_holdout=True,
    )
    all_dataset = build_dataset(rows, feature_names=profile.names, expected_schema_version=profile.version)
    X_a = all_dataset.X.loc[:, bundle["feature_views"][VIEW_A]]
    X_b = all_dataset.X.loc[:, bundle["feature_views"][VIEW_B]]
    scores = _fused_scores(bundle["models"][VIEW_A], X_a, bundle["models"][VIEW_B], X_b)
    model_suffix = "redteam" if args.score_scope == "redteam_calibration" else "external"
    evaluation = evaluate_scores(
        scores,
        all_dataset.y.to_numpy(),
        model_name=f"{bundle['model_name']}_{model_suffix}",
        threshold=float(bundle["threshold"]),
        metrics_on=args.score_scope,
    )
    scope = (
        "red-team calibration scoring only; no fitting or threshold tuning"
        if args.score_scope == "redteam_calibration"
        else "sealed external holdout scoring only; no fitting or threshold tuning"
    )
    report = {
        "scope": scope,
        "score_scope": args.score_scope,
        "model_name": bundle["model_name"],
        "feature_schema_version": profile.version,
        "score_fusion": FUSION_RULE,
        "rows": len(rows),
        "bot_asr": 1.0 - evaluation.bot_recall,
        "metrics": asdict(evaluation),
    }
    _write_json(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False))
    return 0


def run_feature_lockbox_score(args: argparse.Namespace) -> int:
    """Score a sealed precomputed Human or Bot lockbox without fitting."""
    lockbox = Path(args.lockbox_features)
    companion_path = lockbox.with_suffix(lockbox.suffix + ".manifest.json")
    if sealed_holdout_reason(lockbox) is None or not companion_path.exists():
        raise ValueError("feature lockbox score requires an external_holdout_only manifest")
    companion = json.loads(companion_path.read_text(encoding="utf-8"))
    if companion.get("evaluation_consumed"):
        raise ValueError("feature lockbox has already been scored and is no longer reusable")
    bundle = joblib.load(args.model)
    if bundle.get("score_fusion") != FUSION_RULE:
        raise ValueError("expected formal two-view min-fusion model bundle")
    profile = get_feature_profile(bundle["feature_schema_version"], trajectory_only=True)
    rows = load_jsonl(lockbox)
    labels = {row.get("label") for row in rows}
    if labels not in ({"human"}, {"bot"}):
        raise ValueError("feature lockbox must contain exactly one labelled class")
    dataset = build_dataset(rows, feature_names=profile.names, expected_schema_version=profile.version)
    X_a = dataset.X.loc[:, bundle["feature_views"][VIEW_A]]
    X_b = dataset.X.loc[:, bundle["feature_views"][VIEW_B]]
    scores = _fused_scores(bundle["models"][VIEW_A], X_a, bundle["models"][VIEW_B], X_b)
    evaluation = evaluate_scores(
        scores,
        dataset.y.to_numpy(),
        model_name=f"{bundle['model_name']}_feature_lockbox",
        threshold=float(bundle["threshold"]),
        metrics_on="sealed_feature_lockbox",
    )
    report = {
        "scope": "sealed feature lockbox scoring only; no fitting or threshold tuning",
        "model_name": bundle["model_name"],
        "feature_schema_version": profile.version,
        "score_fusion": FUSION_RULE,
        "lockbox_label": next(iter(labels)),
        "rows": len(rows),
        "human_frr": evaluation.human_frr if labels == {"human"} else None,
        "bot_asr": 1.0 - evaluation.bot_recall if labels == {"bot"} else None,
        "metrics": asdict(evaluation),
    }
    _write_json(Path(args.report), report)
    companion["evaluation_consumed"] = {
        "model_path": str(args.model),
        "report_path": str(args.report),
        "scored_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    companion_path.write_text(
        json.dumps(companion, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "fold",
            "assemble",
            "final-test",
            "fit-lockbox-candidate",
            "family-holdout",
            "external-score",
            "feature-lockbox-score",
        ),
    )
    parser.add_argument("--human-features")
    parser.add_argument("--bot-features")
    parser.add_argument("--work-dir")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--report")
    parser.add_argument("--report-dir")
    parser.add_argument("--calibration")
    parser.add_argument("--model-output")
    parser.add_argument("--model-version")
    parser.add_argument("--family", action="append")
    parser.add_argument("--model")
    parser.add_argument("--external-bot-holdout")
    parser.add_argument(
        "--score-scope",
        choices=("redteam_calibration", "external_holdout"),
        default="external_holdout",
    )
    parser.add_argument("--lockbox-features")
    parser.add_argument(
        "--all-input-development",
        action="store_true",
        help="Use every supplied non-lockbox row for OOF and lockbox-candidate fitting.",
    )
    parser.add_argument(
        "--hard-negative-weight",
        type=float,
        help="Pre-set additional weight for development Bots missed by nested inner OOF.",
    )
    parser.add_argument(
        "--max-human-frr",
        type=float,
        default=0.03,
        help="Human FRR budget used only for development OOF threshold calibration.",
    )
    parser.add_argument(
        "--human-frr-policy",
        choices=("per_fold", "per_participant"),
        default="per_fold",
        help="Apply the OOF Human FRR budget per CV fold or per Human participant group.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.max_human_frr <= 1.0:
        raise ValueError("--max-human-frr must be between 0 and 1")
    if args.hard_negative_weight is not None and args.hard_negative_weight <= 1.0:
        raise ValueError("--hard-negative-weight must be greater than 1")
    if args.all_input_development and args.stage not in {
        "fold",
        "assemble",
        "fit-lockbox-candidate",
    }:
        raise ValueError("--all-input-development is only valid for fold, assemble, and fit-lockbox-candidate")
    if args.stage in {"fold", "assemble", "final-test", "fit-lockbox-candidate", "family-holdout"}:
        if args.human_features is None or args.bot_features is None:
            raise ValueError(f"{args.stage} requires --human-features and --bot-features")
    if args.stage == "fold":
        if args.work_dir is None or args.fold is None:
            raise ValueError("fold requires --work-dir and --fold")
        return run_fold(args)
    if args.stage == "assemble":
        if args.work_dir is None or args.report is None:
            raise ValueError("assemble requires --work-dir and --report")
        return assemble_oof(args)
    if args.stage == "final-test":
        if args.calibration is None or args.report is None or args.model_output is None:
            raise ValueError("final-test requires --calibration, --report, and --model-output")
        return run_final_test(args)
    if args.stage == "fit-lockbox-candidate":
        if args.calibration is None or args.report is None or args.model_output is None:
            raise ValueError("fit-lockbox-candidate requires --calibration, --report, and --model-output")
        return run_fit_lockbox_candidate(args)
    if args.stage == "family-holdout":
        if not args.family or (args.report is None and args.report_dir is None):
            raise ValueError("family-holdout requires --family and --report or --report-dir")
        return run_family_holdout(args)
    if args.stage == "feature-lockbox-score":
        if args.model is None or args.lockbox_features is None or args.report is None:
            raise ValueError("feature-lockbox-score requires --model, --lockbox-features, and --report")
        return run_feature_lockbox_score(args)
    if args.model is None or args.external_bot_holdout is None or args.report is None:
        raise ValueError("external-score requires --model, --external-bot-holdout, and --report")
    return run_external_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
