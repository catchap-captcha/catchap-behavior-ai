"""One-time score-only evaluation of a sealed two-view risk-policy holdout.

The holdout is never used for fitting, feature selection, or threshold tuning.
After scoring, its manifest is marked consumed to prevent accidental reuse.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.feature_profiles import get_feature_profile
from training.build_dataset import build_dataset
from training.evaluate_models import positive_proba
from training.holdout_registry import sealed_holdout_reason
from training.run_local_training import build_bot_feature_rows, load_jsonl


FUSION_RULE = "min(P_human_general_without_physics, P_human_dynamics_physics)"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_policy_scores(
    scores: np.ndarray,
    *,
    hard_threshold: float,
    step_up_threshold: float,
) -> dict[str, int | float]:
    """Summarize fixed allow and step-up outcomes for Bot-only holdout scores."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite 1D array")
    if not 0.0 <= hard_threshold <= step_up_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= hard <= step_up <= 1")

    direct_allow = values >= step_up_threshold
    model_bot = values < hard_threshold
    step_up_band = ~model_bot & ~direct_allow
    count = len(values)
    return {
        "rows": int(count),
        "hard_model_asr": float((values >= hard_threshold).mean()),
        "direct_auto_allow_asr": float(direct_allow.mean()),
        "step_up_rate": float((~direct_allow).mean()),
        "model_bot_risk_count": int(model_bot.sum()),
        "step_up_band_count": int(step_up_band.sum()),
        "direct_auto_allow_count": int(direct_allow.sum()),
    }


def evaluate_holdout(
    *,
    model_path: Path,
    holdout_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Score exactly once and then seal the holdout manifest as consumed."""
    manifest_path = holdout_path.with_suffix(holdout_path.suffix + ".manifest.json")
    if sealed_holdout_reason(holdout_path) is None or not manifest_path.exists():
        raise ValueError("a manifest-protected external holdout is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evaluation_consumed"):
        raise ValueError("external holdout has already been scored and cannot be reused")
    if manifest.get("training_usage") != "external_holdout_only":
        raise ValueError("holdout must be marked external_holdout_only")
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite report: {report_path}")

    bundle = joblib.load(model_path)
    if bundle.get("score_fusion") != FUSION_RULE:
        raise ValueError("expected a formal two-view min-fusion model bundle")
    hard_threshold = float(bundle["threshold"])
    if bundle.get("step_up_threshold") is None:
        raise ValueError("candidate bundle must define step_up_threshold")
    step_up_threshold = float(bundle["step_up_threshold"])

    profile = get_feature_profile(bundle["feature_schema_version"], trajectory_only=True)
    payloads = load_jsonl(holdout_path)
    usages = {row.get("collection", {}).get("training_usage") for row in payloads}
    if usages != {"external_holdout_only"}:
        raise ValueError("holdout rows must all be external_holdout_only")
    rows = build_bot_feature_rows(
        payloads,
        groups_per_family=3,
        profile=profile,
        allow_external_holdout=True,
    )
    dataset = build_dataset(rows, feature_names=profile.names, expected_schema_version=profile.version)
    models = bundle["models"]
    views = bundle["feature_views"]
    scores = np.minimum(
        positive_proba(models["general_without_physics"], dataset.X.loc[:, views["general_without_physics"]]),
        positive_proba(models["dynamics_physics"], dataset.X.loc[:, views["dynamics_physics"]]),
    )
    policy_metrics = summarize_policy_scores(
        scores,
        hard_threshold=hard_threshold,
        step_up_threshold=step_up_threshold,
    )
    result = {
        "scope": "sealed external holdout score-only under fixed risk policy; no fitting or threshold tuning",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "model_path": str(model_path),
            "model_name": bundle["model_name"],
            "model_version": bundle["model_version"],
            "feature_schema_version": bundle["feature_schema_version"],
            "score_fusion": bundle["score_fusion"],
            "hard_threshold": hard_threshold,
            "step_up_threshold": step_up_threshold,
        },
        "holdout": {
            "path": str(holdout_path),
            "sha256": _sha256(holdout_path),
            "bot_family": manifest.get("bot_family"),
            "rows": len(rows),
        },
        "policy_metrics": policy_metrics,
        "gates": {
            "max_direct_auto_allow_asr": 0.05,
            "direct_auto_allow_asr_pass": policy_metrics["direct_auto_allow_asr"] <= 0.05,
            "note": "Bot-only synthetic holdout cannot measure Human FRR or prove production readiness.",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["evaluation_consumed"] = {
        "scope": "external_holdout score-only under fixed step-up policy; no fitting or threshold tuning",
        "model_path": str(model_path),
        "model_version": bundle["model_version"],
        "hard_threshold": hard_threshold,
        "step_up_threshold": step_up_threshold,
        "report_path": str(report_path),
        "scored_at_utc": result["evaluated_at"],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--external-bot-holdout", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_holdout(
                model_path=args.model,
                holdout_path=args.external_bot_holdout,
                report_path=args.report,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
