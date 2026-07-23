"""Mine difficult offline red-team traces with a frozen two-view detector.

This tool is defensive research for CatChap's own behavior detector.  It never
opens a browser, solves a CAPTCHA, fits a detector, or changes a threshold.
It generates temporary PCA-GMM hybrid candidates, scores them with an explicit
frozen model, and stores only the highest ``P(Human)`` candidates as
``redteam_only`` weakness examples.

The selected output is not detector training data.  It is a diagnostic set for
finding recurring weaknesses in the current detector family.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.services.feature_profiles import get_feature_profile
from training.build_dataset import build_dataset
from training.generate_hybrid_redteam_bots import (
    BOT_FAMILY,
    GENERATOR_VERSION,
    generate_dataset,
)
from training.generate_ml_bots import GeneratorConfig, load_jsonl, sha256
from training.run_local_training import build_bot_feature_rows
from tools.run_formal_two_view_fusion import FUSION_RULE, VIEW_A, VIEW_B, _fused_scores


WEAKSET_GENERATOR_VERSION = f"{GENERATOR_VERSION}_score_guided_v1"
WEAKSET_BOT_FAMILY = "hybrid_motion_score_guided_redteam"


def score_payloads(
    payloads: list[dict[str, Any]],
    *,
    model_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return frozen two-view Human scores without fitting or calibration."""
    bundle = joblib.load(model_path)
    if bundle.get("score_fusion") != FUSION_RULE:
        raise ValueError("expected formal two-view min-fusion model bundle")
    profile = get_feature_profile(bundle["feature_schema_version"], trajectory_only=True)
    rows = build_bot_feature_rows(
        payloads,
        groups_per_family=3,
        profile=profile,
        # This explicit override is score-only. The same payloads remain
        # forbidden from detector fitting by their training_usage metadata.
        allow_external_holdout=True,
    )
    dataset = build_dataset(
        rows,
        feature_names=profile.names,
        expected_schema_version=profile.version,
    )
    scores = _fused_scores(
        bundle["models"][VIEW_A],
        dataset.X.loc[:, bundle["feature_views"][VIEW_A]],
        bundle["models"][VIEW_B],
        dataset.X.loc[:, bundle["feature_views"][VIEW_B]],
    )
    metadata = {
        "model_name": bundle["model_name"],
        "feature_schema_version": profile.version,
        "score_fusion": bundle["score_fusion"],
        "threshold": float(bundle["threshold"]),
    }
    return scores, metadata


def select_weak_payloads(
    payloads: list[dict[str, Any]],
    scores: np.ndarray,
    *,
    threshold: float,
    top_k: int,
    model_name: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the highest Human-score traces and preserve red-team provenance."""
    scores = np.asarray(scores, dtype=float)
    if len(payloads) != len(scores) or not len(payloads):
        raise ValueError("payloads and scores must be non-empty and same length")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    selected_indices = sorted(
        range(len(payloads)), key=lambda index: float(scores[index]), reverse=True
    )[: min(top_k, len(payloads))]
    selected: list[dict[str, Any]] = []
    for rank, index in enumerate(selected_indices, start=1):
        payload = copy.deepcopy(payloads[index])
        source_attempt_id = str(payload["attempt_id"])
        human_score = float(scores[index])
        payload["attempt_id"] = f"redteam_weak_{WEAKSET_GENERATOR_VERSION}_{rank:06d}"
        payload["challenge_id"] = "hybrid_redteam_weakness_challenge"
        payload["session_id"] = "hybrid_redteam_weakness"
        collection = payload["collection"]
        collection.update(
            {
                "label": "bot",
                "label_source": "hybrid_redteam_score_guided",
                "bot_family": WEAKSET_BOT_FAMILY,
                "generator_version": WEAKSET_GENERATOR_VERSION,
                "training_usage": "redteam_only",
                "evaluation_role": "redteam_weakness",
                "redteam_selection": {
                    "selection_rule": "top_human_score_from_frozen_detector",
                    "rank": rank,
                    "human_score": round(human_score, 9),
                    "detector_threshold": float(threshold),
                    "passed_detector_threshold": human_score >= threshold,
                    "frozen_model_name": model_name,
                    "source_candidate_attempt_id": source_attempt_id,
                },
            }
        )
        selected.append(payload)

    selected_scores = np.asarray([scores[index] for index in selected_indices], dtype=float)
    return selected, {
        "candidate_count": len(payloads),
        "selected_count": len(selected),
        "detector_pass_count": int(np.count_nonzero(scores >= threshold)),
        "detector_pass_rate": float(np.mean(scores >= threshold)),
        "selected_score_min": float(selected_scores.min()),
        "selected_score_median": float(np.median(selected_scores)),
        "selected_score_max": float(selected_scores.max()),
    }


def mine_weaknesses(
    *,
    human_attempts_path: Path,
    source_human_features_path: Path,
    model_path: Path,
    output_path: Path,
    candidate_count: int,
    top_k: int,
    config: GeneratorConfig,
) -> dict[str, Any]:
    """Generate, score, and retain difficult PCA-GMM hybrid traces offline."""
    if candidate_count < top_k:
        raise ValueError("candidate_count must be at least top_k")
    if candidate_count < 1 or top_k < 1:
        raise ValueError("candidate_count and top_k must be positive")

    with tempfile.TemporaryDirectory(prefix="catchap_redteam_candidates_") as directory:
        candidate_path = Path(directory) / "hybrid_candidates.jsonl"
        generation = generate_dataset(
            human_attempts_path=human_attempts_path,
            source_human_features_path=source_human_features_path,
            output_path=candidate_path,
            role="calibration",
            count=candidate_count,
            config=config,
        )
        payloads = load_jsonl(candidate_path)
        scores, score_metadata = score_payloads(payloads, model_path=model_path)

    selected, selection = select_weak_payloads(
        payloads,
        scores,
        threshold=score_metadata["threshold"],
        top_k=top_k,
        model_name=score_metadata["model_name"],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for payload in selected:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "dataset_name": output_path.stem,
        "generator_type": "PCA-GMM hybrid candidate generation + frozen detector score selection",
        "generator_version": WEAKSET_GENERATOR_VERSION,
        "bot_family": WEAKSET_BOT_FAMILY,
        "training_usage": "redteam_only",
        "detector_training_forbidden": True,
        "threshold_tuning_forbidden": True,
        "selection": {
            "rule": "top_human_score_from_frozen_detector",
            "top_k": top_k,
            **selection,
        },
        "frozen_detector": {
            "path": str(model_path),
            "sha256": sha256(model_path),
            **score_metadata,
        },
        "candidate_generation": {
            "base_generator_version": generation["generator_version"],
            "source_attempt_count": generation["source_attempt_count"],
            "source_attempt_ids_exported": False,
            "policy": generation["policy"],
            "config": asdict(config),
            "quality_rejected": generation["quality_rejected"],
            "novelty_rejected": generation["novelty_rejected"],
        },
        "inputs": {
            "human_attempts": {
                "path": str(human_attempts_path),
                "sha256": sha256(human_attempts_path),
            },
            "human_development_features": {
                "path": str(source_human_features_path),
                "sha256": sha256(source_human_features_path),
            },
        },
        "output": {"path": str(output_path), "sha256": sha256(output_path)},
        "notes": [
            "Offline defensive red-team mining only; no browser or network interaction.",
            "The detector was loaded once for scoring only; no detector fitting or threshold tuning occurred.",
            "Selected rows are the highest frozen P(Human) candidates, not Human-labelled data.",
            "This redteam_only output must not be used as detector training or threshold-tuning input.",
        ],
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-attempts", type=Path, required=True)
    parser.add_argument("--source-human-features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=3000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--point-count", type=int, default=48)
    parser.add_argument("--pca-components", type=int, default=24)
    parser.add_argument("--gmm-components", type=int, default=8)
    parser.add_argument("--min-novelty-distance", type=float, default=0.015)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = mine_weaknesses(
        human_attempts_path=args.human_attempts,
        source_human_features_path=args.source_human_features,
        model_path=args.model,
        output_path=args.out,
        candidate_count=args.candidate_count,
        top_k=args.top_k,
        config=GeneratorConfig(
            point_count=args.point_count,
            pca_components=args.pca_components,
            gmm_components=args.gmm_components,
            min_novelty_distance=args.min_novelty_distance,
            seed=args.seed,
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
