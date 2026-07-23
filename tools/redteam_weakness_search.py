"""Offline red-team weakness search against a fixed two-view detector.

Pipeline (Stage A of the red-team <-> defense loop):

    broad motion-policy sweep -> generate many red-team candidates
      -> score every candidate with a FIXED detector bundle (no fitting)
      -> auto-collect the evaders + near-miss band as a weak-set
      -> cluster the weak-set by motion policy / binding view / features
      -> write a summary report of the common evasion patterns

Every generated row is ``redteam_only`` and detector-training-forbidden.  This
tool never fits a model, never tunes a threshold, and never opens a browser or
network connection.  It only reads local JSONL + a fixed model and writes
synthetic JSONL + JSON reports.  The weak-set it saves is a diagnostic artifact
for finding *why* traces evade; it is not a training input.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from app.services.feature_profiles import get_feature_profile
from training.build_dataset import build_dataset
from training.evaluate_models import positive_proba
from training.generate_hybrid_redteam_bots import (
    BOT_FAMILY,
    GENERATOR_VERSION,
    MotionPolicy,
    _motion_events,
    _payload,
)
from training.generate_ml_bots import (
    GeneratorConfig,
    _novelty_distances,
    fit_generator,
    load_jsonl,
    sample_payloads,
    sha256,
    vectorize_attempt,
)
from app.services.quality_validator import QUALITY_REJECTED, validate_attempt
from training.run_local_training import build_bot_feature_rows

VIEW_A = "general_without_physics"
VIEW_B = "dynamics_physics"
FUSION_RULE = "min(P_human_general_without_physics, P_human_dynamics_physics)"

# A deliberately broad sweep: a superset of the calibration/external-holdout
# policies plus a little beyond, so each sampled candidate explores a wide
# region of the motion space.  This is search, not a sealed evaluation family.
SWEEP_POLICY = MotionPolicy(
    curvature=(0.006, 0.075),
    jitter=(0.0006, 0.0075),
    time_power=(0.60, 1.50),
    duration_scale=(0.70, 1.42),
    turn_slowdown=(0.06, 0.75),
    coalesce_fraction=(0.04, 0.30),
    frame_ms=(8, 10, 12, 16),
    late_correction_probability=0.6,
)


def _allowed_source_rows(human_attempts_path: Path, source_human_features_path: Path) -> list[dict[str, Any]]:
    allowed_ids = {
        str(row["attempt_id"])
        for row in load_jsonl(source_human_features_path)
        if row.get("label") == "human" and row.get("attempt_id")
    }
    rows = [
        row
        for row in load_jsonl(human_attempts_path)
        if str(row.get("attempt_id")) in allowed_ids and len(row.get("events") or []) >= 4
    ]
    if len(rows) < 20:
        raise ValueError(f"need at least 20 eligible Human source attempts, found {len(rows)}")
    return rows


def generate_candidates(
    *,
    source_rows: list[dict[str, Any]],
    count: int,
    config: GeneratorConfig,
    run_tag: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Fit the PCA+GMM generator once, then sweep motion policies per candidate."""
    generator = fit_generator(source_rows, config)
    base_candidates = sample_payloads(
        generator, count=max(256, count * 4), role="external_holdout", config=config
    )
    randomizer = random.Random(config.seed + 733)
    accepted: list[dict[str, Any]] = []
    rejected_quality = 0
    rejected_novelty = 0
    for candidate in base_candidates:
        captcha = candidate["captcha"]
        width, height = int(captcha["width"]), int(captcha["height"])
        events, mutation = _motion_events(
            candidate["events"], width=width, height=height, policy=SWEEP_POLICY, randomizer=randomizer
        )
        quality = validate_attempt(events, captcha_width=width, captcha_height=height)
        if quality.status == QUALITY_REJECTED:
            rejected_quality += 1
            continue
        vector = vectorize_attempt({"captcha": captcha, "events": events}, config.point_count)
        novelty = float(_novelty_distances(generator["nearest"], vector.reshape(1, -1))[0])
        if novelty < config.min_novelty_distance:
            rejected_novelty += 1
            continue
        attempt_id = f"redteam_search_{run_tag}_{len(accepted):06d}"
        payload = _payload(
            attempt_id,
            events,
            role="calibration",  # -> training_usage=redteam_only, evaluation_role=redteam_calibration
            width=width,
            height=height,
            novelty_distance=novelty,
            mutation=mutation,
        )
        accepted.append(payload)
        if len(accepted) == count:
            break
    stats = {
        "requested": count,
        "accepted": len(accepted),
        "quality_rejected": rejected_quality,
        "novelty_rejected": rejected_novelty,
        "source_attempt_count": generator["source_count"],
    }
    return accepted, stats


def score_candidates(
    *, model_path: Path, payloads: list[dict[str, Any]]
) -> dict[str, np.ndarray]:
    """Return per-sample fused/view scores from the FIXED detector (no fitting)."""
    bundle = joblib.load(model_path)
    if bundle.get("score_fusion") != FUSION_RULE:
        raise ValueError("expected formal two-view min-fusion model bundle")
    profile = get_feature_profile(bundle["feature_schema_version"], trajectory_only=True)
    usages = {p.get("collection", {}).get("training_usage") for p in payloads}
    if usages != {"redteam_only"}:
        raise ValueError(f"weakness search scores redteam_only payloads only, found {sorted(usages)}")
    rows = build_bot_feature_rows(payloads, groups_per_family=3, profile=profile, allow_external_holdout=True)
    if len(rows) != len(payloads):
        raise RuntimeError("feature row count does not match candidate count")
    dataset = build_dataset(rows, feature_names=profile.names, expected_schema_version=profile.version)
    X_a = dataset.X.loc[:, bundle["feature_views"][VIEW_A]]
    X_b = dataset.X.loc[:, bundle["feature_views"][VIEW_B]]
    view_a = positive_proba(bundle["models"][VIEW_A], X_a)
    view_b = positive_proba(bundle["models"][VIEW_B], X_b)
    return {
        "threshold": float(bundle["threshold"]),
        "model_name": bundle["model_name"],
        "fused": np.minimum(view_a, view_b),
        "view_a": view_a,
        "view_b": view_b,
        "features": dataset.X,
    }


def _bin(value: float, edges: list[float]) -> str:
    for index in range(len(edges) - 1):
        if value < edges[index + 1]:
            return f"[{edges[index]:g},{edges[index + 1]:g})"
    return f">={edges[-1]:g}"


def _mutation_histogram(payloads: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    mutations = [p["collection"]["mutation"] for p in payloads]
    hist: dict[str, Counter] = {
        "frame_interval_ms": Counter(),
        "correction_used": Counter(),
        "curvature_amplitude_abs": Counter(),
        "time_power": Counter(),
        "turn_slowdown": Counter(),
        "event_coalescing": Counter(),
    }
    for mutation in mutations:
        hist["frame_interval_ms"][str(mutation["frame_interval_ms"])] += 1
        hist["correction_used"][str(bool(mutation["correction_used"]))] += 1
        hist["curvature_amplitude_abs"][_bin(abs(mutation["curvature_amplitude"]), [0, 0.02, 0.04, 0.06, 0.08])] += 1
        hist["time_power"][_bin(mutation["time_power"], [0.6, 0.85, 1.0, 1.15, 1.5])] += 1
        hist["turn_slowdown"][_bin(mutation["turn_slowdown"], [0.0, 0.2, 0.4, 0.6, 0.8])] += 1
        hist["event_coalescing"][_bin(mutation["event_coalescing"], [0.0, 0.08, 0.16, 0.24, 0.32])] += 1
    return {key: dict(counter.most_common()) for key, counter in hist.items()}


def _feature_summary(features, indices: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    if len(indices) == 0:
        return {}
    subset = features.iloc[indices]
    summary: dict[str, dict[str, float]] = {}
    for name in names:
        if name not in subset.columns:
            continue
        column = subset[name].to_numpy(dtype=float)
        summary[name] = {
            "mean": float(np.mean(column)),
            "p10": float(np.quantile(column, 0.10)),
            "p50": float(np.quantile(column, 0.50)),
            "p90": float(np.quantile(column, 0.90)),
        }
    return summary


PHYSICS_FEATURES = [
    "speed_turn_abs_correlation",
    "turn_change_smoothness",
    "pause_position_entropy",
]


def run(args: argparse.Namespace) -> int:
    human_attempts = Path(args.human_attempts)
    source_features = Path(args.source_human_features)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = GeneratorConfig(
        point_count=args.point_count,
        pca_components=args.pca_components,
        gmm_components=args.gmm_components,
        min_novelty_distance=args.min_novelty_distance,
        seed=args.seed,
    )
    source_rows = _allowed_source_rows(human_attempts, source_features)
    payloads, gen_stats = generate_candidates(
        source_rows=source_rows, count=args.count, config=config, run_tag=args.run_tag
    )
    if len(payloads) != args.count:
        raise RuntimeError(
            f"generated {len(payloads)}/{args.count} candidates "
            f"(quality={gen_stats['quality_rejected']}, novelty={gen_stats['novelty_rejected']})"
        )

    scored = score_candidates(model_path=model_path, payloads=payloads)
    threshold = scored["threshold"]
    fused = scored["fused"]
    view_a = scored["view_a"]
    view_b = scored["view_b"]

    evader_mask = fused >= threshold
    margin = args.near_miss_margin
    near_mask = (fused < threshold) & (fused >= threshold - margin)
    weak_mask = evader_mask | near_mask

    binding_view = np.where(view_a <= view_b, VIEW_A, VIEW_B)

    weak_payloads: list[dict[str, Any]] = []
    for index in np.nonzero(weak_mask)[0]:
        payload = json.loads(json.dumps(payloads[index]))  # copy
        payload["collection"]["fixed_detector_score"] = {
            "model_name": scored["model_name"],
            "threshold": threshold,
            "fused_human_score": float(fused[index]),
            "view_a_human_score": float(view_a[index]),
            "view_b_human_score": float(view_b[index]),
            "binding_view": str(binding_view[index]),
            "band": "evader" if evader_mask[index] else "near_miss",
        }
        weak_payloads.append(payload)

    weak_path = out_dir / f"redteam_weakset_{args.run_tag}.jsonl"
    with weak_path.open("w", encoding="utf-8") as handle:
        for payload in weak_payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    evader_idx = np.nonzero(evader_mask)[0]
    near_idx = np.nonzero(near_mask)[0]
    weak_idx = np.nonzero(weak_mask)[0]

    report = {
        "scope": "offline red-team weakness search; scoring only, no fitting or threshold tuning",
        "run_tag": args.run_tag,
        "fixed_detector": {
            "model_path": str(model_path),
            "model_name": scored["model_name"],
            "threshold": threshold,
            "score_fusion": FUSION_RULE,
        },
        "generation": {**gen_stats, "sweep_policy": SWEEP_POLICY.__dict__},
        "counts": {
            "candidates": int(len(payloads)),
            "evaders": int(evader_mask.sum()),
            "near_miss": int(near_mask.sum()),
            "weak_set": int(weak_mask.sum()),
            "blocked": int((~weak_mask).sum()),
        },
        "rates": {
            "bot_asr_at_fixed_threshold": float(evader_mask.mean()),
            "near_miss_rate": float(near_mask.mean()),
        },
        "fused_score": {
            "max": float(fused.max()),
            "p99": float(np.quantile(fused, 0.99)),
            "p90": float(np.quantile(fused, 0.90)),
            "median": float(np.quantile(fused, 0.50)),
        },
        "binding_view_of_weak_set": dict(Counter(binding_view[weak_idx].tolist())),
        "weak_set_mutation_histogram": _mutation_histogram([payloads[i] for i in weak_idx]),
        "evader_mutation_histogram": _mutation_histogram([payloads[i] for i in evader_idx]) if len(evader_idx) else {},
        "feature_summary": {
            "weak_set": _feature_summary(scored["features"], weak_idx, PHYSICS_FEATURES),
            "blocked": _feature_summary(scored["features"], np.nonzero(~weak_mask)[0], PHYSICS_FEATURES),
        },
        "outputs": {
            "weak_set_path": str(weak_path),
            "weak_set_sha256": sha256(weak_path) if weak_payloads else None,
        },
        "guards": [
            "All rows are redteam_only and detector-training-forbidden.",
            "No model was fitted; no threshold was tuned; the detector bundle is fixed.",
            "Weak-set is a diagnostic artifact, not a training or threshold-tuning input.",
        ],
    }

    manifest = {
        "dataset_name": weak_path.stem,
        "bot_family": BOT_FAMILY,
        "generator_version": GENERATOR_VERSION,
        "role": "weakness_search",
        "training_usage": "redteam_only",
        "detector_training_forbidden": True,
        "threshold_tuning_forbidden": True,
        "source_attempt_ids_exported": False,
        "fixed_detector_model": str(model_path),
        "count": len(weak_payloads),
        "inputs": {
            "human_attempts": {"path": str(human_attempts), "sha256": sha256(human_attempts)},
            "human_development_features": {"path": str(source_features), "sha256": sha256(source_features)},
        },
        "output": {"path": str(weak_path), "sha256": sha256(weak_path) if weak_payloads else None},
        "notes": [
            "Offline weakness-search weak-set; no browser or network interaction.",
            "Contains no Human source attempt IDs; novelty-filtered after mutation.",
            "Forbidden from detector fitting and threshold tuning.",
        ],
    }
    weak_path.with_suffix(weak_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("counts", "rates", "binding_view_of_weak_set")}, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Fixed two-view detector bundle to score against.")
    parser.add_argument("--human-attempts", required=True)
    parser.add_argument("--source-human-features", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--near-miss-margin", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--point-count", type=int, default=48)
    parser.add_argument("--pca-components", type=int, default=24)
    parser.add_argument("--gmm-components", type=int, default=8)
    parser.add_argument("--min-novelty-distance", type=float, default=0.015)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
