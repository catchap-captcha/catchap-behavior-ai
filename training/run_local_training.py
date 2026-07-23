"""Train Human/Bot candidate models from local JSONL snapshots.

This runner keeps the production DB pipeline untouched. It converts bot
pointer payloads to a versioned feature profile, combines them with confirmed
Human feature rows, performs a leakage-aware local split, trains four tree
models, and writes versioned candidates and reports.

Anonymous Human rows are training-only because person-level grouping is not
possible for them. Linked Human rows are split by participant. Rule bots are
split by deterministic family/batch groups, then the selected model receives a
separate leave-one-family-out stress test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from app.services.feature_extractor import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, extract_features
from app.services.feature_profiles import FeatureProfile, get_feature_profile
from training.build_dataset import Dataset, build_dataset
from training.check_data_readiness import Thresholds, compute_readiness
from training.evaluate_models import Evaluation, evaluate, select_threshold
from training.group_threshold_cv import (
    GroupThresholdCalibration,
    calibrate_grouped_threshold,
    fit_development_model,
)
from training.holdout_registry import assert_not_sealed_training_inputs
from training.select_best_model import select_best
from training.split_dataset import SplitData
from training.train_models import build_models


DEFAULT_HUMAN = Path("data/processed/human_confirmed_4786_20260713/human_features.jsonl")
DEFAULT_BOTS = Path("data/interim/rule_bots_3000.jsonl")
DEFAULT_VERSION = "local_h4786_b3000_20260713"
EXPERIMENT_MAX_HUMAN_FRR = 0.03
DEPLOYMENT_MAX_HUMAN_FRR = 0.01
MAX_KNOWN_BOT_ASR = 0.05
MAX_UNSEEN_BOT_ASR = 0.10
MAX_REPLAY_WARP_ASR = 0.05
FRESH_EXTERNAL_EVALUATION_ROLE = "fresh_participant_external_holdout"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_bucket(value: str, count: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % count


def build_bot_feature_rows(
    payloads: Iterable[dict[str, Any]],
    groups_per_family: int = 12,
    *,
    profile: FeatureProfile | None = None,
    allow_external_holdout: bool = False,
) -> list[dict[str, Any]]:
    """Convert rule-bot collect payloads into model-ready rows.

    ``generator_version`` includes a deterministic batch suffix solely for
    leakage-aware splitting. ``generator_version_base`` preserves provenance.
    """
    if groups_per_family < 3:
        raise ValueError("groups_per_family must be at least 3")

    profile = profile or get_feature_profile(FEATURE_SCHEMA_VERSION)
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        collection = payload["collection"]
        training_usage = collection.get("training_usage")
        if (
            training_usage not in (None, "development_only")
            and not allow_external_holdout
        ):
            raise ValueError(
                "external-holdout or red-team bot data must not be used for detector training"
            )
        family = collection["bot_family"]
        base_version = collection.get("generator_version") or "unknown_generator"
        attempt_id = payload["attempt_id"]
        bucket = _stable_bucket(attempt_id, groups_per_family)
        interaction = payload.get("interaction") or {}
        row = {
            "attempt_id": attempt_id,
            "challenge_id": payload.get("challenge_id"),
            "session_id": payload.get("session_id"),
            "anonymous_participant_id": None,
            "label": "bot",
            "label_source": collection.get("label_source") or "rule_bot",
            "bot_family": family,
            "generator_version": f"{base_version}_batch_{bucket:02d}",
            "generator_version_base": base_version,
            "evaluation_role": collection.get("evaluation_role"),
            "schema_version": payload.get("schema_version", "1.0"),
            "feature_schema_version": profile.version,
            "quality_status": "valid",
            "position_correct": payload.get("position_correct"),
            "interaction_success": payload.get("interaction_success"),
            "final_drop_error": payload.get("final_drop_error"),
        }
        row.update(profile.extractor(payload.get("events", []), interaction))
        rows.append(row)
    return rows


def _allocate_groups(groups: list[str], seed: int) -> dict[str, str]:
    """Allocate groups approximately 70/15/15 with every split non-empty."""
    unique = sorted(set(groups))
    if len(unique) < 3:
        raise ValueError(f"need at least 3 groups, found {len(unique)}")
    random.Random(seed).shuffle(unique)
    n = len(unique)
    n_val = max(1, round(n * 0.15))
    n_test = max(1, round(n * 0.15))
    while n - n_val - n_test < 1:
        if n_test > 1:
            n_test -= 1
        else:
            n_val -= 1
    n_train = n - n_val - n_test
    return {
        **{group: "train" for group in unique[:n_train]},
        **{group: "val" for group in unique[n_train : n_train + n_val]},
        **{group: "test" for group in unique[n_train + n_val :]},
    }


def build_local_split(ds: Dataset, rows: list[dict[str, Any]], seed: int = 42) -> SplitData:
    """Split linked Humans by person, anonymous Humans train-only, bots by batch."""
    if len(ds) != len(rows):
        raise ValueError("dataset and source row counts differ")

    assignment: dict[int, str] = {}
    group_to_split: dict[str, str] = {}

    human_groups: dict[str, list[int]] = {}
    anonymous_human: list[int] = []
    bot_by_family: dict[str, dict[str, list[int]]] = {}

    for index, row in enumerate(rows):
        if row["label"] == "human":
            participant = row.get("anonymous_participant_id")
            if participant:
                human_groups.setdefault(f"human::{participant}", []).append(index)
            else:
                anonymous_human.append(index)
        else:
            family = row.get("bot_family") or "unknown_family"
            generator = row.get("generator_version") or "unknown_generator"
            group = f"bot::{family}::{generator}"
            bot_by_family.setdefault(family, {}).setdefault(group, []).append(index)

    human_map = _allocate_groups(list(human_groups), seed)
    for group, indices in human_groups.items():
        split_name = human_map[group]
        group_to_split[group] = split_name
        assignment.update({index: split_name for index in indices})

    for index in anonymous_human:
        assignment[index] = "train"
        group_to_split[f"human_anonymous_train_only::{rows[index]['attempt_id']}"] = "train"

    for family_index, family in enumerate(sorted(bot_by_family)):
        family_seed = seed + 1009 * (family_index + 1)
        family_map = _allocate_groups(list(bot_by_family[family]), family_seed)
        for group, indices in bot_by_family[family].items():
            split_name = family_map[group]
            group_to_split[group] = split_name
            assignment.update({index: split_name for index in indices})

    if set(assignment) != set(range(len(rows))):
        raise ValueError("not every row received a split assignment")

    indices_by_split = {
        name: [index for index, split_name in assignment.items() if split_name == name]
        for name in ("train", "val", "test")
    }
    for name, indices in indices_by_split.items():
        labels = set(ds.y.iloc[indices].tolist())
        if labels != {0, 1}:
            raise ValueError(f"{name} split must contain Human and Bot rows, found labels={labels}")

    attempt_ids = [row["attempt_id"] for row in rows]
    class_counts = {}
    for name, indices in indices_by_split.items():
        y = ds.y.iloc[indices]
        class_counts[name] = {
            "human": int((y == 1).sum()),
            "bot": int((y == 0).sum()),
        }

    manifest = {
        "seed": seed,
        "policy": {
            "linked_human": "participant-group split",
            "anonymous_human": "train-only",
            "rule_bot": "family-stratified deterministic batch-group split",
        },
        "counts": {name: len(indices) for name, indices in indices_by_split.items()},
        "class_counts": class_counts,
        "linked_human_groups": len(human_groups),
        "anonymous_human_train_only": len(anonymous_human),
        "bot_families": sorted(bot_by_family),
        "group_to_split": group_to_split,
        "attempt_to_split": {
            attempt_ids[index]: split_name for index, split_name in assignment.items()
        },
    }

    def part(frame: pd.DataFrame | pd.Series, name: str):
        return frame.iloc[indices_by_split[name]].reset_index(drop=True)

    return SplitData(
        X_train=part(ds.X, "train"),
        y_train=part(ds.y, "train"),
        X_val=part(ds.X, "val"),
        y_val=part(ds.y, "val"),
        X_test=part(ds.X, "test"),
        y_test=part(ds.y, "test"),
        manifest=manifest,
    )


def _subset_split(ds: Dataset, train_idx: list[int], val_idx: list[int], test_idx: list[int]) -> SplitData:
    return SplitData(
        X_train=ds.X.iloc[train_idx].reset_index(drop=True),
        y_train=ds.y.iloc[train_idx].reset_index(drop=True),
        X_val=ds.X.iloc[val_idx].reset_index(drop=True),
        y_val=ds.y.iloc[val_idx].reset_index(drop=True),
        X_test=ds.X.iloc[test_idx].reset_index(drop=True),
        y_test=ds.y.iloc[test_idx].reset_index(drop=True),
        manifest={},
    )


def family_holdout_stress_test(
    ds: Dataset,
    rows: list[dict[str, Any]],
    primary_manifest: dict[str, Any],
    model_name: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Test the selected architecture against each entirely unseen bot family."""
    split_of = primary_manifest["attempt_to_split"]
    families = sorted({row.get("bot_family") for row in rows if row["label"] == "bot"})
    results: list[dict[str, Any]] = []

    for fold, held_out in enumerate(families):
        train_idx: list[int] = []
        val_idx: list[int] = []
        test_idx: list[int] = []
        for index, row in enumerate(rows):
            split_name = split_of[row["attempt_id"]]
            if row["label"] == "human":
                if split_name == "train":
                    train_idx.append(index)
                elif split_name == "val":
                    val_idx.append(index)
                else:
                    test_idx.append(index)
            elif row.get("bot_family") == held_out:
                test_idx.append(index)
            elif split_name == "train":
                train_idx.append(index)
            elif split_name == "val":
                val_idx.append(index)

        stress_split = _subset_split(ds, train_idx, val_idx, test_idx)
        model = build_models(stress_split.y_train, seed=seed + fold)[model_name]
        model.fit(stress_split.X_train, stress_split.y_train)
        threshold = select_threshold(model, stress_split.X_val, stress_split.y_val)
        result = evaluate(
            model,
            f"{model_name}_holdout_{held_out}",
            stress_split.X_test,
            stress_split.y_test,
            threshold,
            "family_holdout_test",
        )
        item = asdict(result)
        item["held_out_bot_family"] = held_out
        item["train_rows"] = len(train_idx)
        item["validation_rows"] = len(val_idx)
        item["test_rows"] = len(test_idx)
        results.append(item)
    return results


def external_bot_holdout_test(
    models: dict[str, Any],
    thresholds: dict[str, float],
    payloads: list[dict[str, Any]],
    profile: FeatureProfile,
) -> dict[str, list[dict[str, Any]]]:
    """Score browser-generated families that were never used for training."""
    if not payloads:
        return {name: [] for name in models}
    rows = build_bot_feature_rows(
        payloads,
        groups_per_family=3,
        profile=profile,
        allow_external_holdout=True,
    )
    families = sorted({row["bot_family"] for row in rows})
    output: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    for family in families:
        family_rows = [row for row in rows if row["bot_family"] == family]
        dataset = build_dataset(
            family_rows,
            feature_names=profile.names,
            expected_schema_version=profile.version,
        )
        for name, model in models.items():
            metrics = evaluate(
                model,
                f"{name}_external_{family}",
                dataset.X,
                dataset.y,
                thresholds[name],
                "external_bot_holdout",
            )
            item = asdict(metrics)
            item["held_out_bot_family"] = family
            item["evaluation_role"] = next(
                (row.get("evaluation_role") for row in family_rows if row.get("evaluation_role")),
                None,
            )
            item["bot_asr"] = 1.0 - metrics.bot_recall
            item["test_rows"] = len(family_rows)
            output[name].append(item)
    return output


def select_robust_candidate(
    test: dict[str, Evaluation],
    holdout_by_model: dict[str, list[dict[str, Any]]],
    external_by_model: dict[str, list[dict[str, Any]]] | None = None,
    *,
    human_participants: int | None = None,
) -> dict[str, Any]:
    """Rank candidates and enforce the explicit CatChap FRR/ASR gates."""
    external_by_model = external_by_model or {name: [] for name in test}
    ranked = []
    for name, test_metrics in test.items():
        holdout = holdout_by_model[name]
        external = external_by_model.get(name, [])
        worst_frr = max([test_metrics.human_frr, *(item["human_frr"] for item in holdout)])
        recalls = [item["bot_recall"] for item in holdout]
        if worst_frr > EXPERIMENT_MAX_HUMAN_FRR:
            continue
        known_asr = 1.0 - test_metrics.bot_recall
        family_asrs = [1.0 - recall for recall in recalls]
        external_asrs = [item["bot_asr"] for item in external]
        fresh_external = [
            item
            for item in external
            if item.get("evaluation_role") == FRESH_EXTERNAL_EVALUATION_ROLE
        ]
        fresh_external_asrs = [item["bot_asr"] for item in fresh_external]
        unseen_asrs = [*family_asrs, *external_asrs]
        replay_item = next(
            (item for item in holdout if item.get("held_out_bot_family") == "replay_warp"),
            None,
        )
        replay_asr = 1.0 - replay_item["bot_recall"] if replay_item else 1.0
        worst_unseen_asr = max(unseen_asrs) if unseen_asrs else 1.0
        experiment_eligible = bool(
            known_asr <= MAX_KNOWN_BOT_ASR
            and worst_unseen_asr <= MAX_UNSEEN_BOT_ASR
            and replay_asr <= MAX_REPLAY_WARP_ASR
        )
        observation_only_eligible = bool(
            test_metrics.human_frr <= EXPERIMENT_MAX_HUMAN_FRR
        )
        shadow_mode_eligible = bool(
            experiment_eligible
            and test_metrics.human_frr <= EXPERIMENT_MAX_HUMAN_FRR
            and bool(fresh_external)
            and max(fresh_external_asrs, default=1.0) <= MAX_KNOWN_BOT_ASR
        )
        deployment_eligible = bool(
            shadow_mode_eligible
            and test_metrics.human_frr <= DEPLOYMENT_MAX_HUMAN_FRR
        )
        ranked.append(
            {
                "model_name": name,
                "minimum_family_holdout_bot_recall": min(recalls, default=0.0),
                "average_family_holdout_bot_recall": (
                    sum(recalls) / len(recalls) if recalls else 0.0
                ),
                "maximum_family_holdout_human_frr": worst_frr,
                "primary_test_bot_recall": test_metrics.bot_recall,
                "primary_test_human_frr": test_metrics.human_frr,
                "primary_test_human_f1": test_metrics.human_f1,
                "known_bot_asr": known_asr,
                "worst_unseen_bot_asr": worst_unseen_asr,
                "replay_warp_asr": replay_asr,
                "external_worst_bot_asr": max(external_asrs) if external_asrs else None,
                "fresh_external_worst_bot_asr": (
                    max(fresh_external_asrs) if fresh_external_asrs else None
                ),
                "fresh_external_holdout_present": bool(fresh_external),
                "observation_only_eligible": observation_only_eligible,
                "experiment_eligible": experiment_eligible,
                "shadow_mode_eligible": shadow_mode_eligible,
                "deployment_eligible": deployment_eligible,
                "human_participants": human_participants,
                "avg_inference_ms": test_metrics.avg_inference_ms,
            }
        )

    ranked.sort(
        key=lambda item: (
            -int(item["experiment_eligible"]),
            item["worst_unseen_bot_asr"],
            item["known_bot_asr"],
            -item["average_family_holdout_bot_recall"],
            -item["primary_test_human_f1"],
            item["avg_inference_ms"],
        )
    )
    selected = ranked[0] if ranked else None
    return {
        "selected_model": selected["model_name"] if selected else None,
        "observation_only_eligible": bool(selected and selected["observation_only_eligible"]),
        "experiment_eligible": bool(selected and selected["experiment_eligible"]),
        "shadow_mode_eligible": bool(selected and selected["shadow_mode_eligible"]),
        "deployment_eligible": bool(selected and selected["deployment_eligible"]),
        "acceptance_criteria": {
            "experiment_human_frr_max": EXPERIMENT_MAX_HUMAN_FRR,
            "deployment_human_frr_max": DEPLOYMENT_MAX_HUMAN_FRR,
            "known_bot_asr_max": MAX_KNOWN_BOT_ASR,
            "unseen_bot_worst_asr_max": MAX_UNSEEN_BOT_ASR,
            "replay_warp_asr_max": MAX_REPLAY_WARP_ASR,
            "human_participants": "reported_only_not_a_deployment_gate",
            "observation_mode_blocks_or_allows": False,
            "external_browser_holdout_required": True,
            "fresh_participant_external_holdout_required_for_deployment": True,
            "shadow_mode_human_frr_max": EXPERIMENT_MAX_HUMAN_FRR,
            "shadow_mode_fresh_external_bot_asr_max": MAX_KNOWN_BOT_ASR,
        },
        "reason": (
            "selected_by_security_gate_then_worst_unseen_asr"
            if selected
            else "no_model_meets_human_frr_budget"
        ),
        "ranked_candidates": ranked,
    }


def _library_versions() -> dict[str, str]:
    import lightgbm
    import numpy
    import sklearn
    import xgboost

    return {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "numpy": numpy.__version__,
        "pandas": pd.__version__,
    }


def _save_candidates(
    directory: Path,
    models: dict[str, Any],
    thresholds: dict[str, float],
    validation: dict[str, Evaluation],
    test: dict[str, Evaluation],
    holdout_by_model: dict[str, list[dict[str, Any]]],
    external_by_model: dict[str, list[dict[str, Any]]],
    robust_selection: dict[str, Any],
    dataset_version: str,
    feature_names: tuple[str, ...],
    feature_schema_version: str,
    feature_input_scope: str,
    threshold_calibrations: dict[str, GroupThresholdCalibration],
) -> None:
    import joblib

    directory.mkdir(parents=True, exist_ok=True)
    versions = _library_versions()
    for name, model in models.items():
        bundle = {
            "model": model,
            "model_name": name,
            "model_version": f"{name}_local_v1",
            "feature_names": list(feature_names),
            "feature_schema_version": feature_schema_version,
            "feature_input_scope": feature_input_scope,
            "dataset_version": dataset_version,
            "trained_at": _utcnow_iso(),
            "threshold": thresholds[name],
            "threshold_calibration": threshold_calibrations[name].to_dict(),
            "validation_metrics": asdict(validation[name]),
            "test_metrics": asdict(test[name]),
            "family_holdout_metrics": holdout_by_model[name],
            "external_bot_holdout_metrics": external_by_model[name],
            "library_versions": versions,
            "promotion_status": "candidate_only",
            "robust_recommended": robust_selection["selected_model"] == name,
            "deployment_eligible": bool(
                robust_selection["deployment_eligible"]
                and robust_selection["selected_model"] == name
            ),
            "shadow_mode_eligible": bool(
                robust_selection["shadow_mode_eligible"]
                and robust_selection["selected_model"] == name
            ),
            "observation_only_eligible": bool(
                robust_selection["observation_only_eligible"]
                and robust_selection["selected_model"] == name
            ),
        }
        joblib.dump(bundle, directory / f"{name}.joblib")


def _plot_confusion(report_dir: Path, evaluation: Evaluation) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = evaluation.confusion_matrix
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["bot", "human"])
    ax.set_yticklabels(["bot", "human"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{evaluation.model_name} (test)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(report_dir / f"confusion_matrix_{evaluation.model_name}.png")
    plt.close(fig)


def _write_reports(
    report_dir: Path,
    readiness,
    split: SplitData,
    validation: dict[str, Evaluation],
    test: dict[str, Evaluation],
    primary_selection,
    holdout_by_model: dict[str, list[dict[str, Any]]],
    external_by_model: dict[str, list[dict[str, Any]]],
    robust_selection: dict[str, Any],
    dataset_version: str,
    feature_schema_version: str,
    feature_input_scope: str,
    threshold_calibrations: dict[str, GroupThresholdCalibration],
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "data_readiness.json").write_text(
        json.dumps(asdict(readiness), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "split_manifest.json").write_text(
        json.dumps(split.manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    fields = [
        "model_name",
        "metrics_on",
        "threshold",
        "accuracy",
        "human_precision",
        "human_recall",
        "human_f1",
        "bot_recall",
        "human_frr",
        "roc_auc",
        "pr_auc",
        "avg_inference_ms",
    ]
    with (report_dir / "model_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for metrics in test.values():
            row = asdict(metrics)
            writer.writerow({field: row.get(field) for field in fields})

    for name, metrics in test.items():
        with (report_dir / f"feature_importance_{name}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["feature", "importance"])
            for feature, importance in sorted(
                metrics.feature_importance.items(), key=lambda item: -item[1]
            ):
                writer.writerow([feature, importance])
        _plot_confusion(report_dir, metrics)

    summary = {
        "generated_at_utc": _utcnow_iso(),
        "dataset_version": dataset_version,
        "feature_schema_version": feature_schema_version,
        "feature_input_scope": feature_input_scope,
        "readiness": asdict(readiness),
        "split": {
            "counts": split.manifest["counts"],
            "class_counts": split.manifest["class_counts"],
            "policy": split.manifest["policy"],
        },
        "validation": {name: asdict(value) for name, value in validation.items()},
        "group_threshold_calibration": {
            name: value.to_dict() for name, value in threshold_calibrations.items()
        },
        "test": {name: asdict(value) for name, value in test.items()},
        "selection": {
            "primary_test_selected_model": (
                primary_selection.selected.model_name if primary_selection.selected else None
            ),
            "primary_test_reason": primary_selection.reason,
            "primary_test_warning": primary_selection.warning,
            "robust_candidate": robust_selection,
        },
        "family_holdout_stress_test": holdout_by_model,
        "external_bot_holdout": external_by_model,
        "important_caveats": [
            "Rule bots are a baseline; their synthetic distribution does not represent every real attack.",
            "Anonymous Human rows are train-only because participant-level grouping is unavailable.",
            (
                "Primary thresholds use grouped out-of-fold scores and constrain Human FRR "
                "in every fold; the untouched test split is not used for calibration."
            ),
            (
                f"Only {split.manifest['linked_human_groups']} linked Human participant groups "
                "have usable traces."
            ),
            "Candidates are not automatically promoted to production.",
        ],
    }
    (report_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def run(args: argparse.Namespace) -> int:
    profile = get_feature_profile(
        args.feature_schema_version,
        trajectory_only=args.trajectory_only,
    )
    human_path = Path(args.human_features)
    precomputed_bot_values = args.bot_features or []
    if precomputed_bot_values and args.bot_attempts:
        raise ValueError("use either --bot-attempts or --bot-features, not both")
    if isinstance(precomputed_bot_values, str):
        precomputed_bot_values = [precomputed_bot_values]
    precomputed_bot_paths = [Path(value) for value in precomputed_bot_values]
    bot_paths: list[Path] = []
    if not precomputed_bot_paths:
        bot_values = args.bot_attempts or [str(DEFAULT_BOTS)]
        if isinstance(bot_values, str):
            bot_values = [bot_values]
        bot_paths = [Path(value) for value in bot_values]
        assert_not_sealed_training_inputs(bot_paths)
    external_values = args.external_bot_holdout or []
    if isinstance(external_values, str):
        external_values = [external_values]
    external_bot_paths = [Path(value) for value in external_values]
    dataset_dir = Path(args.dataset_dir)
    report_dir = Path(args.report_dir)
    candidate_dir = Path(args.candidate_dir)

    human_rows = load_jsonl(human_path)
    if not human_rows or any(row.get("label") != "human" for row in human_rows):
        raise ValueError("human feature input must contain only labelled Human rows")
    found_versions = {row.get("feature_schema_version") for row in human_rows}
    if found_versions != {profile.version}:
        raise ValueError(
            f"Human feature schema mismatch: found {sorted(found_versions)}, expected {profile.version}"
        )
    if precomputed_bot_paths:
        bot_rows = [
            row
            for bot_feature_path in precomputed_bot_paths
            for row in load_jsonl(bot_feature_path)
        ]
        if not bot_rows or any(row.get("label") != "bot" for row in bot_rows):
            raise ValueError("precomputed bot feature input must contain only labelled Bot rows")
        bot_versions = {row.get("feature_schema_version") for row in bot_rows}
        if bot_versions != {profile.version}:
            raise ValueError(
                f"Bot feature schema mismatch: found {sorted(bot_versions)}, expected {profile.version}"
            )
        if any(row.get("evaluation_role") for row in bot_rows):
            raise ValueError("external holdout feature rows must not enter model fitting")
    else:
        bot_rows = build_bot_feature_rows(
            [
                payload
                for bot_path in bot_paths
                for payload in load_jsonl(bot_path)
            ],
            args.bot_groups_per_family,
            profile=profile,
        )
    rows = human_rows + bot_rows

    dataset_dir.mkdir(parents=True, exist_ok=True)
    bot_features_path = dataset_dir / "bot_features.jsonl"
    combined_path = dataset_dir / "combined_features.jsonl"
    if not args.skip_dataset_copy:
        write_jsonl(bot_features_path, bot_rows)
        write_jsonl(combined_path, rows)

    readiness = compute_readiness(
        rows,
        Thresholds(500, 500, 0, 3),
        feature_names=profile.names,
        expected_schema_version=profile.version,
    )
    if not readiness.ready:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "data_readiness.json").write_text(
            json.dumps(asdict(readiness), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        raise ValueError(f"data not ready: {readiness.missing}")

    ds = build_dataset(
        rows, feature_names=profile.names, expected_schema_version=profile.version
    )
    split = build_local_split(ds, rows, seed=args.seed)
    development_indices = [
        index
        for index, row in enumerate(rows)
        if split.manifest["attempt_to_split"][row["attempt_id"]] != "test"
    ]
    available_model_names = sorted(build_models(split.y_train, seed=args.seed))
    model_names = [args.model] if args.model else available_model_names
    if args.model and args.model not in available_model_names:
        raise ValueError(f"requested model is unavailable: {args.model}")
    threshold_calibrations = {
        name: calibrate_grouped_threshold(
            ds,
            rows,
            development_indices,
            name,
            seed=args.seed,
            n_splits=args.threshold_cv_folds,
            max_human_frr=EXPERIMENT_MAX_HUMAN_FRR,
        )
        for name in model_names
    }
    thresholds = {
        name: calibration.threshold
        for name, calibration in threshold_calibrations.items()
    }
    validation: dict[str, Evaluation] = {
        name: calibration.pooled_oof_metrics
        for name, calibration in threshold_calibrations.items()
    }
    models = {
        name: fit_development_model(ds, development_indices, name, seed=args.seed)
        for name in model_names
    }
    test: dict[str, Evaluation] = {}
    for name, model in models.items():
        test[name] = evaluate(
            model,
            name,
            split.X_test,
            split.y_test,
            thresholds[name],
            "untouched_test",
        )

    primary_selection = select_best(list(test.values()))
    holdout_by_model = (
        {
            name: family_holdout_stress_test(ds, rows, split.manifest, name, args.seed)
            for name in sorted(models)
        }
        if not args.skip_family_holdout
        else {name: [] for name in models}
    )
    external_by_model = (
        external_bot_holdout_test(
            models,
            thresholds,
            [
                payload
                for external_bot_path in external_bot_paths
                for payload in load_jsonl(external_bot_path)
            ],
            profile,
        )
        if not args.skip_external_holdout
        else {name: [] for name in models}
    )
    robust_selection = select_robust_candidate(
        test,
        holdout_by_model,
        external_by_model,
        human_participants=split.manifest["linked_human_groups"],
    )

    _save_candidates(
        candidate_dir,
        models,
        thresholds,
        validation,
        test,
        holdout_by_model,
        external_by_model,
        robust_selection,
        args.dataset_version,
        profile.names,
        profile.version,
        profile.input_scope,
        threshold_calibrations,
    )
    _write_reports(
        report_dir,
        readiness,
        split,
        validation,
        test,
        primary_selection,
        holdout_by_model,
        external_by_model,
        robust_selection,
        args.dataset_version,
        profile.version,
        profile.input_scope,
        threshold_calibrations,
    )
    dataset_manifest = {
        "dataset_version": args.dataset_version,
        "created_at_utc": _utcnow_iso(),
        "human_rows": len(human_rows),
        "bot_rows": len(bot_rows),
        "combined_rows": len(rows),
        "feature_count": len(profile.names),
        "feature_schema_version": profile.version,
        "feature_input_scope": profile.input_scope,
        "inputs": {
            "human_features": {"path": str(human_path), "sha256": sha256(human_path)},
            "bot_attempts": [
                {"path": str(path), "sha256": sha256(path)} for path in bot_paths
            ],
            "precomputed_bot_features": [
                {"path": str(path), "sha256": sha256(path)}
                for path in precomputed_bot_paths
            ],
            "external_bot_holdouts": [
                {"path": str(path), "sha256": sha256(path)} for path in external_bot_paths
            ],
        },
        "outputs": (
            {
                "bot_features": {"path": str(bot_features_path), "sha256": sha256(bot_features_path)},
                "combined_features": {"path": str(combined_path), "sha256": sha256(combined_path)},
            }
            if not args.skip_dataset_copy
            else {"materialized_feature_copy": False}
        ),
        "candidate_models": sorted(models),
        "selected_candidate": robust_selection["selected_model"],
        "deployment_eligible": robust_selection["deployment_eligible"],
        "shadow_mode_eligible": robust_selection["shadow_mode_eligible"],
        "observation_only_eligible": robust_selection["observation_only_eligible"],
        "production_promoted": False,
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "dataset_version": args.dataset_version,
                "rows": {"human": len(human_rows), "bot": len(bot_rows), "total": len(rows)},
                "split": split.manifest["class_counts"],
                "selected_candidate": robust_selection["selected_model"],
                "observation_only_eligible": robust_selection["observation_only_eligible"],
                "experiment_eligible": robust_selection["experiment_eligible"],
                "shadow_mode_eligible": robust_selection["shadow_mode_eligible"],
                "deployment_eligible": robust_selection["deployment_eligible"],
                "reports": str(report_dir),
                "candidates": str(candidate_dir),
                "production_promoted": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if robust_selection["selected_model"] is not None else 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train candidates from local Human/Bot JSONL files.")
    parser.add_argument("--human-features", default=str(DEFAULT_HUMAN))
    parser.add_argument(
        "--bot-attempts",
        action="append",
        default=None,
        help="Development bot JSONL; may be supplied more than once.",
    )
    parser.add_argument(
        "--bot-features",
        action="append",
        default=[],
        help="Precomputed development Bot feature JSONL; may be supplied more than once.",
    )
    parser.add_argument(
        "--external-bot-holdout",
        action="append",
        default=[],
        help="External bot JSONL for scoring only; may be supplied more than once.",
    )
    parser.add_argument("--dataset-version", default=DEFAULT_VERSION)
    parser.add_argument("--dataset-dir", default=f"data/processed/{DEFAULT_VERSION}")
    parser.add_argument("--report-dir", default=f"reports/{DEFAULT_VERSION}")
    parser.add_argument("--candidate-dir", default=f"models/candidate/{DEFAULT_VERSION}")
    parser.add_argument("--bot-groups-per-family", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold-cv-folds", type=int, default=5)
    parser.add_argument(
        "--skip-dataset-copy",
        action="store_true",
        help="Do not copy already prepared feature rows into --dataset-dir.",
    )
    parser.add_argument(
        "--skip-family-holdout",
        action="store_true",
        help="Defer family holdout evaluation to a separate formal worker.",
    )
    parser.add_argument(
        "--skip-external-holdout",
        action="store_true",
        help="Defer external holdout scoring to a separate formal worker.",
    )
    parser.add_argument(
        "--model",
        choices=("random_forest", "extra_trees", "xgboost", "lightgbm"),
        help="Run one model only; useful for bounded formal evaluations.",
    )
    parser.add_argument("--feature-schema-version", choices=("1.0", "2.0", "2.1", "2.2", "2.3"), default="1.0")
    parser.add_argument(
        "--trajectory-only",
        action="store_true",
        help="Use only features derived from one pointer trace (x, y, t).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
