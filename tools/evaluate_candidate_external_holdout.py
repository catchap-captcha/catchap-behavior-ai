"""Score a sealed external bot holdout against existing candidate bundles.

This tool never fits models or calibrates thresholds. It evaluates each saved
candidate at its already-fixed threshold, so external data cannot influence
training or model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.feature_profiles import get_feature_profile
from training.build_dataset import build_dataset
from training.evaluate_models import evaluate
from training.holdout_registry import sealed_holdout_reason
from training.run_local_training import build_bot_feature_rows, load_jsonl

MAX_BOT_ASR = 0.05
MAX_HUMAN_FRR = 0.03


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bot_metrics(
    bundle: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, Any]:
    dataset = build_dataset(
        rows,
        feature_names=bundle["feature_names"],
        expected_schema_version=bundle["feature_schema_version"],
    )
    metrics = asdict(
        evaluate(
            bundle["model"],
            f"{bundle['model_name']}_external_{label}",
            dataset.X,
            dataset.y,
            float(bundle["threshold"]),
            "external_bot_holdout",
        )
    )
    metrics["bot_asr"] = 1.0 - metrics["bot_recall"]
    metrics["test_rows"] = len(rows)
    return metrics


def _load_bundles(directory: Path) -> list[tuple[Path, dict[str, Any]]]:
    paths = sorted(directory.glob("*.joblib"))
    if not paths:
        raise ValueError(f"no candidate bundles found in {directory}")
    return [(path, joblib.load(path)) for path in paths]


def evaluate_holdout(
    candidate_dir: Path,
    holdout_path: Path,
) -> dict[str, Any]:
    """Evaluate one manifest-protected bot holdout without model fitting."""
    reason = sealed_holdout_reason(holdout_path)
    if not reason:
        raise ValueError(
            "external validation requires a manifest-protected holdout dataset"
        )

    bundles = _load_bundles(candidate_dir)
    first_bundle = bundles[0][1]
    schema_version = str(first_bundle["feature_schema_version"])
    input_scope = str(first_bundle.get("feature_input_scope", "all_behavioral_features"))
    profile = get_feature_profile(
        schema_version,
        trajectory_only=input_scope == "pointer_trajectory_only",
    )
    expected_names = list(profile.names)

    for path, bundle in bundles:
        if str(bundle["feature_schema_version"]) != schema_version:
            raise ValueError(f"schema differs in {path}")
        if str(bundle.get("feature_input_scope", "all_behavioral_features")) != input_scope:
            raise ValueError(f"input scope differs in {path}")
        if list(bundle["feature_names"]) != expected_names:
            raise ValueError(f"feature names differ from profile in {path}")

    payloads = load_jsonl(holdout_path)
    rows = build_bot_feature_rows(
        payloads,
        groups_per_family=3,
        profile=profile,
        allow_external_holdout=True,
    )
    if not rows:
        raise ValueError("external holdout has no rows")
    by_family = {
        family: [row for row in rows if row["bot_family"] == family]
        for family in sorted({row["bot_family"] for row in rows})
    }

    candidates: dict[str, Any] = {}
    for path, bundle in bundles:
        overall = _bot_metrics(bundle, rows, label="all_playwright")
        family_metrics = {
            family: _bot_metrics(bundle, family_rows, label=family)
            for family, family_rows in by_family.items()
        }
        human_test = bundle.get("test_metrics") or {}
        human_frr = float(human_test.get("human_frr", 1.0))
        worst_family_asr = max(item["bot_asr"] for item in family_metrics.values())
        candidates[bundle["model_name"]] = {
            "bundle_path": str(path),
            "threshold": float(bundle["threshold"]),
            "existing_human_test_metrics": human_test,
            "playwright_all": overall,
            "playwright_by_family": family_metrics,
            "gates": {
                "max_bot_asr": MAX_BOT_ASR,
                "max_human_frr": MAX_HUMAN_FRR,
                "normal_human_frr": human_frr,
                "normal_human_frr_pass": human_frr <= MAX_HUMAN_FRR,
                "playwright_all_bot_asr": overall["bot_asr"],
                "playwright_worst_family_bot_asr": worst_family_asr,
                "playwright_all_bot_asr_pass": overall["bot_asr"] <= MAX_BOT_ASR,
                "playwright_worst_family_bot_asr_pass": worst_family_asr <= MAX_BOT_ASR,
                "eligible_from_this_check_alone": False,
            },
        }

    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "external_holdout_evaluation_only",
        "training_or_threshold_tuning_performed": False,
        "holdout": {
            "path": str(holdout_path),
            "sha256": _sha256(holdout_path),
            "protection_reason": reason,
            "rows": len(rows),
            "families": {family: len(family_rows) for family, family_rows in by_family.items()},
        },
        "feature_profile": {
            "schema_version": schema_version,
            "input_scope": input_scope,
            "feature_count": len(expected_names),
        },
        "candidates": candidates,
        "note": (
            "Passing this one external set does not override existing replay, "
            "fresh-participant, or shadow-mode gates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--external-bot-holdout", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    report = evaluate_holdout(args.candidate_dir, args.external_bot_holdout)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
