"""Score a sealed external bot holdout with a previously fitted candidate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib

from app.services.feature_profiles import get_feature_profile
from training.build_dataset import build_dataset
from training.evaluate_models import evaluate
from training.run_local_training import build_bot_feature_rows, load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--external-bot-holdout", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    profile = get_feature_profile(bundle["feature_schema_version"], trajectory_only=True)
    if tuple(bundle["feature_names"]) != profile.names:
        raise ValueError("candidate feature schema does not match its model bundle")
    rows = build_bot_feature_rows(
        load_jsonl(Path(args.external_bot_holdout)),
        groups_per_family=3,
        profile=profile,
        allow_external_holdout=True,
    )
    dataset = build_dataset(
        rows,
        feature_names=profile.names,
        expected_schema_version=profile.version,
    )
    model_name = str(bundle.get("model_name") or "candidate")
    evaluation = evaluate(
        bundle["model"],
        f"{model_name}_external",
        dataset.X,
        dataset.y,
        float(bundle["threshold"]),
        "external_bot_holdout",
    )
    output = {
        "scope": "sealed external holdout scoring only; no fitting or threshold tuning",
        "feature_schema_version": profile.version,
        "rows": len(rows),
        "bot_asr": 1.0 - evaluation.bot_recall,
        "metrics": asdict(evaluation),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
