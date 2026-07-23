"""Run repeated VAE-based offline red-team scans against a frozen detector.

A conditional VAE is fitted once from explicitly supplied non-lockbox Human
development attempts.  Each scan samples fresh VAE trajectories, applies the
same broad motion-policy sweep used by the PCA-GMM red-team scan, and scores
the resulting ``redteam_only`` traces with a fixed two-view detector.

This tool is diagnostic only: it never fits the detector, tunes its threshold,
opens an external holdout, drives a browser, or makes a network request.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

from app.services.quality_validator import QUALITY_REJECTED, validate_attempt
from training.generate_hybrid_redteam_bots import (
    MotionPolicy,
    _allowed_source_rows,
    _motion_events,
    _payload as hybrid_payload,
)
from training.generate_ml_bots import load_jsonl, sha256, vectorize_attempt
from training.generate_vae_bots import (
    GENERATOR_VERSION as VAE_GENERATOR_VERSION,
    VaeConfig,
    _condition_from_vector,
    _fit_model,
    _novelty_distances,
    _sample_vectors,
    _source_metadata,
)
VAE_REDTEAM_FAMILY = "conditional_vae_hybrid_redteam"
VAE_REDTEAM_GENERATOR_VERSION = f"{VAE_GENERATOR_VERSION}_hybrid_motion_redteam_v1"
SCOPE = "offline red-team weakness search; scoring only, no fitting or threshold tuning"

# Kept local so the generation phase does not import the detector/LightGBM runtime.
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


def fit_vae_redteam_generator(
    source_rows: list[dict[str, Any]],
    config: VaeConfig,
) -> dict[str, Any]:
    """Fit a VAE once on allowed development rows, with no source IDs retained."""
    vectors = np.vstack(
        [vectorize_attempt(row, config.point_count) for row in source_rows]
    ).astype(np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("source trajectory vectors contain non-finite values")
    conditions = np.vstack(
        [_condition_from_vector(vector, config.point_count) for vector in vectors]
    ).astype(np.float32)
    model, state = _fit_model(vectors, conditions, config)
    dimensions, event_count_choices = _source_metadata(source_rows)
    return {
        "model": model,
        "state": state,
        "source_vectors": vectors,
        "source_conditions": conditions,
        "nearest": NearestNeighbors(n_neighbors=1, metric="euclidean").fit(vectors),
        "dimensions": dimensions,
        "event_count_choices": event_count_choices,
        "source_count": len(source_rows),
        "config": config,
    }


def save_vae_redteam_generator(generator: dict[str, Any], output_path: Path) -> None:
    """Persist model state and metadata without source attempt IDs or vectors."""
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    config: VaeConfig = generator["config"]
    state = generator["state"]
    torch.save(
        {
            "generator_type": "conditional VAE + broad hybrid motion red-team",
            "generator_version": VAE_REDTEAM_GENERATOR_VERSION,
            "training_usage": "redteam_only",
            "source_scope": "explicit non-lockbox Human development attempts only",
            "source_attempt_count": generator["source_count"],
            "source_attempt_ids_exported": False,
            "config": asdict(config),
            "model_state": generator["model"].state_dict(),
            "vector_mean": state["vector_mean"],
            "vector_std": state["vector_std"],
            "condition_mean": state["condition_mean"],
            "condition_std": state["condition_std"],
            "loss_history": state["loss_history"],
            "dimensions": generator["dimensions"],
            "event_count_choices": generator["event_count_choices"],
        },
        output_path,
    )


def build_redteam_payload(
    *,
    attempt_id: str,
    events: list[dict[str, Any]],
    width: int,
    height: int,
    novelty_distance: float,
    mutation: dict[str, Any],
    vae_novelty_distance: float,
) -> dict[str, Any]:
    """Mark every VAE scan trace as detector-forbidden red-team data."""
    payload = hybrid_payload(
        attempt_id,
        events,
        role="calibration",
        width=width,
        height=height,
        novelty_distance=novelty_distance,
        mutation=mutation,
    )
    collection = payload["collection"]
    collection.update(
        {
            "label": "bot",
            "label_source": "conditional_vae_hybrid_redteam_generated",
            "bot_family": VAE_REDTEAM_FAMILY,
            "generator_version": VAE_REDTEAM_GENERATOR_VERSION,
            "training_usage": "redteam_only",
            "evaluation_role": "redteam_calibration",
            "base_generator": "conditional_vae",
            "base_vae_novelty_distance": round(vae_novelty_distance, 6),
        }
    )
    return payload


def generate_candidates(
    *,
    generator: dict[str, Any],
    count: int,
    sample_seed: int,
    run_tag: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Sample VAE candidates, apply the broad motion policy, then recheck novelty."""
    config: VaeConfig = generator["config"]
    randomizer = random.Random(sample_seed + 733)
    accepted: list[dict[str, Any]] = []
    quality_rejected = 0
    base_novelty_rejected = 0
    post_mutation_novelty_rejected = 0
    attempts = 0
    batch_index = 0
    max_attempts = count * 40

    while len(accepted) < count and attempts < max_attempts:
        remaining = count - len(accepted)
        batch_size = min(512, max(128, remaining * 3))
        sample_config = replace(config, seed=sample_seed + batch_index * 1009)
        vectors = _sample_vectors(
            generator["model"],
            generator["source_conditions"],
            generator["state"],
            count=batch_size,
            config=sample_config,
        )
        base_novelty = _novelty_distances(generator["nearest"], vectors)
        for vector, distance in zip(vectors, base_novelty):
            attempts += 1
            if float(distance) < config.min_novelty_distance:
                base_novelty_rejected += 1
                continue
            width, height = randomizer.choice(generator["dimensions"])
            event_count = randomizer.choice(generator["event_count_choices"])
            from training.generate_vae_bots import _decode_events

            base_events = _decode_events(
                vector,
                width=width,
                height=height,
                point_count=config.point_count,
                output_event_count=event_count,
            )
            events, mutation = _motion_events(
                base_events,
                width=width,
                height=height,
                policy=SWEEP_POLICY,
                randomizer=randomizer,
            )
            quality = validate_attempt(events, captcha_width=width, captcha_height=height)
            if quality.status == QUALITY_REJECTED:
                quality_rejected += 1
                continue
            mutated_vector = vectorize_attempt(
                {"captcha": {"width": width, "height": height}, "events": events},
                config.point_count,
            )
            novelty_distance = float(
                _novelty_distances(generator["nearest"], mutated_vector.reshape(1, -1))[0]
            )
            if novelty_distance < config.min_novelty_distance:
                post_mutation_novelty_rejected += 1
                continue
            accepted.append(
                build_redteam_payload(
                    attempt_id=f"vae_redteam_search_{run_tag}_{len(accepted):06d}",
                    events=events,
                    width=width,
                    height=height,
                    novelty_distance=novelty_distance,
                    mutation=mutation,
                    vae_novelty_distance=float(distance),
                )
            )
            if len(accepted) == count:
                break
        batch_index += 1

    if len(accepted) != count:
        raise RuntimeError(
            f"generated {len(accepted)}/{count} VAE red-team candidates after {attempts} draws; "
            f"quality={quality_rejected}, base_novelty={base_novelty_rejected}, "
            f"post_mutation_novelty={post_mutation_novelty_rejected}"
        )
    return accepted, {
        "requested": count,
        "accepted": len(accepted),
        "quality_rejected": quality_rejected,
        "base_novelty_rejected": base_novelty_rejected,
        "post_mutation_novelty_rejected": post_mutation_novelty_rejected,
        "source_attempt_count": int(generator["source_count"]),
    }


def write_candidate_dataset(
    *,
    payloads: list[dict[str, Any]],
    generation: dict[str, int],
    generator: dict[str, Any],
    sample_seed: int,
    run_tag: str,
    candidate_path: Path,
) -> dict[str, Any]:
    """Persist red-team candidates before a separate detector-only score phase."""
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    with candidate_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    config: VaeConfig = generator["config"]
    metadata = {
        **generation,
        "base_generator": "conditional_vae",
        "generator_version": VAE_REDTEAM_GENERATOR_VERSION,
        "sample_seed": sample_seed,
        "run_tag": run_tag,
        "vae_config": asdict(config),
        "vae_training": {
            "final_reconstruction_loss": generator["state"]["loss_history"][-1]["reconstruction_loss"],
            "final_kl_loss": generator["state"]["loss_history"][-1]["kl_loss"],
        },
        "sweep_policy": asdict(SWEEP_POLICY),
    }
    manifest = {
        "dataset_name": candidate_path.stem,
        "bot_family": VAE_REDTEAM_FAMILY,
        "generator_version": VAE_REDTEAM_GENERATOR_VERSION,
        "role": "weakness_search_candidate_pool",
        "training_usage": "redteam_only",
        "detector_training_forbidden": True,
        "threshold_tuning_forbidden": True,
        "source_attempt_ids_exported": False,
        "generation": metadata,
        "output": {"path": str(candidate_path), "sha256": sha256(candidate_path)},
        "notes": [
            "Offline VAE red-team candidate pool; no browser, network, or database interaction.",
            "Every row is redteam_only and must not be used for detector fitting or threshold tuning.",
            "This file is generated in a process separate from fixed-detector scoring to avoid ML runtime conflicts.",
        ],
    }
    candidate_path.with_suffix(candidate_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def score_candidate_payloads(
    *,
    payloads: list[dict[str, Any]],
    candidate_path: Path,
    candidate_manifest: dict[str, Any],
    model_path: Path,
    output_dir: Path,
    report_path: Path,
    near_miss_margin: float,
) -> dict[str, Any]:
    """Score a pre-generated VAE candidate pool in a detector-only process."""
    from tools.redteam_weakness_search import (
        FUSION_RULE,
        PHYSICS_FEATURES,
        VIEW_A,
        VIEW_B,
        _feature_summary,
        _mutation_histogram,
        score_candidates,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    usages = {payload.get("collection", {}).get("training_usage") for payload in payloads}
    families = {payload.get("collection", {}).get("bot_family") for payload in payloads}
    if not payloads or usages != {"redteam_only"} or families != {VAE_REDTEAM_FAMILY}:
        raise ValueError("VAE score phase requires only conditional VAE redteam_only candidates")
    generation = candidate_manifest.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("VAE candidate manifest is missing generation metadata")
    run_tag = str(generation["run_tag"])
    scored = score_candidates(model_path=model_path, payloads=payloads)
    threshold = float(scored["threshold"])
    fused = scored["fused"]
    view_a = scored["view_a"]
    view_b = scored["view_b"]
    evader_mask = fused >= threshold
    near_mask = (fused < threshold) & (fused >= threshold - near_miss_margin)
    weak_mask = evader_mask | near_mask
    binding_view = np.where(view_a <= view_b, VIEW_A, VIEW_B)

    weak_payloads: list[dict[str, Any]] = []
    for index in np.nonzero(weak_mask)[0]:
        payload = copy.deepcopy(payloads[index])
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

    weak_path = output_dir / f"vae_redteam_weakset_{run_tag}.jsonl"
    with weak_path.open("w", encoding="utf-8") as handle:
        for payload in weak_payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    weak_indices = np.nonzero(weak_mask)[0]
    evader_indices = np.nonzero(evader_mask)[0]
    report = {
        "scope": SCOPE,
        "run_tag": run_tag,
        "fixed_detector": {
            "model_path": str(model_path),
            "model_name": scored["model_name"],
            "threshold": threshold,
            "score_fusion": FUSION_RULE,
        },
        "generation": generation,
        "counts": {
            "candidates": int(len(payloads)),
            "evaders": int(np.count_nonzero(evader_mask)),
            "near_miss": int(np.count_nonzero(near_mask)),
            "weak_set": int(np.count_nonzero(weak_mask)),
            "blocked": int(np.count_nonzero(~weak_mask)),
        },
        "rates": {
            "bot_asr_at_fixed_threshold": float(np.mean(evader_mask)),
            "near_miss_rate": float(np.mean(near_mask)),
        },
        "fused_score": {
            "max": float(np.max(fused)),
            "p99": float(np.quantile(fused, 0.99)),
            "p90": float(np.quantile(fused, 0.90)),
            "median": float(np.quantile(fused, 0.50)),
        },
        "binding_view_of_weak_set": {
            view: int(np.count_nonzero(binding_view[weak_indices] == view))
            for view in (VIEW_A, VIEW_B)
            if np.count_nonzero(binding_view[weak_indices] == view)
        },
        "weak_set_mutation_histogram": _mutation_histogram([payloads[index] for index in weak_indices]),
        "evader_mutation_histogram": _mutation_histogram([payloads[index] for index in evader_indices])
        if len(evader_indices)
        else {},
        "feature_summary": {
            "weak_set": _feature_summary(scored["features"], weak_indices, PHYSICS_FEATURES),
            "blocked": _feature_summary(scored["features"], np.nonzero(~weak_mask)[0], PHYSICS_FEATURES),
        },
        "outputs": {
            "candidate_pool_path": str(candidate_path),
            "candidate_pool_sha256": sha256(candidate_path),
            "weak_set_path": str(weak_path),
            "weak_set_sha256": sha256(weak_path) if weak_payloads else None,
        },
        "guards": [
            "Every generated row is redteam_only and detector-training-forbidden.",
            "The detector bundle is loaded for scoring only; no detector fitting or threshold tuning occurs.",
            "The VAE is trained only from the explicitly supplied non-lockbox Human development rows.",
            "No sealed external holdout is read or modified.",
        ],
    }
    manifest = {
        "dataset_name": weak_path.stem,
        "bot_family": VAE_REDTEAM_FAMILY,
        "generator_version": VAE_REDTEAM_GENERATOR_VERSION,
        "role": "weakness_search",
        "training_usage": "redteam_only",
        "detector_training_forbidden": True,
        "threshold_tuning_forbidden": True,
        "source_attempt_ids_exported": False,
        "fixed_detector_model": str(model_path),
        "count": len(weak_payloads),
        "candidate_pool": {"path": str(candidate_path), "sha256": sha256(candidate_path)},
        "output": {"path": str(weak_path), "sha256": sha256(weak_path) if weak_payloads else None},
        "notes": report["guards"],
    }
    weak_path.with_suffix(weak_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_generation_phase(args: argparse.Namespace) -> int:
    source_rows = _allowed_source_rows(
        Path(args.human_attempts),
        Path(args.source_human_features),
    )
    config = VaeConfig(
        point_count=args.point_count,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        beta=args.beta,
        learning_rate=args.learning_rate,
        min_novelty_distance=args.min_novelty_distance,
        seed=args.train_seed,
    )
    generator = fit_vae_redteam_generator(source_rows, config)
    generator_path = Path(args.generator_model_out)
    save_vae_redteam_generator(generator, generator_path)
    outputs = []
    for index, seed in enumerate(args.sample_seed, start=1):
        run_tag = f"vae_seed_{seed}"
        payloads, generation = generate_candidates(
            generator=generator,
            sample_seed=seed,
            run_tag=run_tag,
            count=args.count,
        )
        candidate_path = Path(args.out_root) / f"run_{index:02d}" / f"vae_redteam_candidates_{run_tag}.jsonl"
        manifest = write_candidate_dataset(
            payloads=payloads,
            generation=generation,
            generator=generator,
            sample_seed=seed,
            run_tag=run_tag,
            candidate_path=candidate_path,
        )
        outputs.append(
            {
                "run_tag": run_tag,
                "candidates": manifest["generation"]["accepted"],
                "candidate_path": str(candidate_path),
            }
        )
    print(json.dumps({"generator_model": str(generator_path), "runs": outputs}, ensure_ascii=False))
    return 0


def run_score_phase(args: argparse.Namespace) -> int:
    reports = []
    for index, seed in enumerate(args.sample_seed, start=1):
        run_tag = f"vae_seed_{seed}"
        output_dir = Path(args.out_root) / f"run_{index:02d}"
        candidate_path = output_dir / f"vae_redteam_candidates_{run_tag}.jsonl"
        manifest_path = candidate_path.with_suffix(candidate_path.suffix + ".manifest.json")
        if not candidate_path.exists() or not manifest_path.exists():
            raise ValueError(f"missing VAE candidate pool or manifest: {candidate_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("training_usage") != "redteam_only":
            raise ValueError(f"VAE candidate pool is not redteam_only: {candidate_path}")
        if manifest.get("output", {}).get("sha256") != sha256(candidate_path):
            raise ValueError(f"VAE candidate pool hash does not match manifest: {candidate_path}")
        report = score_candidate_payloads(
            payloads=load_jsonl(candidate_path),
            candidate_path=candidate_path,
            candidate_manifest=manifest,
            model_path=Path(args.model),
            output_dir=output_dir,
            report_path=Path(args.report_root) / f"run_{index:02d}.json",
            near_miss_margin=args.near_miss_margin,
        )
        reports.append(
            {
                "run_tag": run_tag,
                "candidates": report["counts"]["candidates"],
                "evaders": report["counts"]["evaders"],
                "bot_asr": report["rates"]["bot_asr_at_fixed_threshold"],
            }
        )
    print(json.dumps({"runs": reports}, ensure_ascii=False))
    return 0


def run_repeated_scan(args: argparse.Namespace) -> int:
    if args.phase == "generate":
        return run_generation_phase(args)
    return run_score_phase(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("generate", "score"),
        required=True,
        help="Generate VAE candidate pools or score existing pools in a separate process.",
    )
    parser.add_argument("--model", required=True, help="Fixed two-view detector bundle to score.")
    parser.add_argument("--human-attempts", required=True)
    parser.add_argument("--source-human-features", required=True)
    parser.add_argument("--generator-model-out", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--sample-seed", action="append", type=int, required=True)
    parser.add_argument("--count", type=int, default=3000)
    parser.add_argument("--near-miss-margin", type=float, default=1e-6)
    parser.add_argument("--train-seed", type=int, default=20260728)
    parser.add_argument("--point-count", type=int, default=48)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--beta", type=float, default=0.003)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-novelty-distance", type=float, default=0.015)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 1 or args.point_count < 4 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("count, point-count, epochs, and batch-size must be positive")
    if args.latent_dim < 2 or args.hidden_dim < 8 or args.beta < 0 or args.min_novelty_distance < 0:
        raise ValueError("invalid VAE dimensions, beta, or novelty distance")
    return run_repeated_scan(args)


if __name__ == "__main__":
    raise SystemExit(main())
