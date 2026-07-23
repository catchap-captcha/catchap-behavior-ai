"""Train a conditional VAE on Human trajectories and create development bots.

This is an offline defensive generator. It trains only on the Human ``train``
split, never drives a browser, and marks every generated payload as
``development_only`` unless an explicit external-holdout role is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.neighbors import NearestNeighbors

from app.services.quality_validator import QUALITY_REJECTED, validate_attempt
from training.generate_ml_bots import load_jsonl, sha256, vectorize_attempt


GENERATOR_VERSION = "conditional_vae_trace_v1"
DEFAULT_POINT_COUNT = 48
DEFAULT_LATENT_DIM = 16
DEFAULT_HIDDEN_DIM = 192
DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 20
DEFAULT_BETA = 0.003
DEFAULT_MIN_NOVELTY_DISTANCE = 0.015
ROLE_SOURCE_SPLITS = {
    "development": {"train"},
    "external_holdout": {"test"},
}


@dataclass(frozen=True)
class VaeConfig:
    point_count: int = DEFAULT_POINT_COUNT
    latent_dim: int = DEFAULT_LATENT_DIM
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    beta: float = DEFAULT_BETA
    learning_rate: float = 1e-3
    min_novelty_distance: float = DEFAULT_MIN_NOVELTY_DISTANCE
    seed: int = 20260721


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _resample(values: np.ndarray, point_count: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, num=len(values))
    target = np.linspace(0.0, 1.0, num=point_count)
    return np.interp(target, source, values)


def _resample_matrix(values: np.ndarray, point_count: int) -> np.ndarray:
    return np.column_stack(
        [_resample(values[:, column], point_count) for column in range(values.shape[1])]
    )


def _condition_from_vector(vector: np.ndarray, point_count: int) -> np.ndarray:
    """Use start/end location and total duration as generation conditions."""
    trace = np.asarray(vector, dtype=float).reshape(point_count, 3)
    duration = float(np.expm1(np.clip(trace[:, 2], 0.0, math.log1p(300.0))).sum())
    return np.asarray(
        [trace[0, 0], trace[0, 1], trace[-1, 0], trace[-1, 1], math.log1p(duration)],
        dtype=np.float32,
    )


def _source_rows(
    rows: Iterable[dict[str, Any]],
    attempt_to_split: dict[str, str],
    role: str,
) -> list[dict[str, Any]]:
    if role not in ROLE_SOURCE_SPLITS:
        raise ValueError(f"unsupported role: {role}")
    selected = [
        row
        for row in rows
        if attempt_to_split.get(str(row.get("attempt_id"))) in ROLE_SOURCE_SPLITS[role]
        and len(row.get("events") or []) >= 4
    ]
    if len(selected) < 20:
        raise ValueError(f"need at least 20 {role} Human attempts, found {len(selected)}")
    return selected


def _source_metadata(source_rows: list[dict[str, Any]]) -> tuple[list[tuple[int, int]], list[int]]:
    dimensions = sorted(
        {
            (
                int((row.get("captcha") or {}).get("width") or 1),
                int((row.get("captcha") or {}).get("height") or 1),
            )
            for row in source_rows
        }
    )
    counts = np.asarray([len(row["events"]) for row in source_rows], dtype=float)
    lower, upper = np.percentile(counts, [5, 95])
    event_count_choices = sorted(
        {
            int(count)
            for count in counts
            if max(4, int(math.floor(lower))) <= count <= int(math.ceil(upper))
        }
    )
    if not event_count_choices:
        raise ValueError("no usable source event counts")
    return dimensions, event_count_choices


def _build_model(
    *,
    input_dim: int,
    condition_dim: int,
    latent_dim: int,
    hidden_dim: int,
):
    import torch
    from torch import nn

    class ConditionalVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim + condition_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.mean = nn.Linear(hidden_dim, latent_dim)
            self.log_variance = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim + condition_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def encode(self, features, conditions):
            encoded = self.encoder(torch.cat((features, conditions), dim=1))
            return self.mean(encoded), self.log_variance(encoded)

        def decode(self, latent, conditions):
            return self.decoder(torch.cat((latent, conditions), dim=1))

        def forward(self, features, conditions):
            mean, log_variance = self.encode(features, conditions)
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)
            return self.decode(latent, conditions), mean, log_variance

    return ConditionalVAE()


def _fit_model(
    vectors: np.ndarray,
    conditions: np.ndarray,
    config: VaeConfig,
) -> tuple[Any, dict[str, Any]]:
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    vector_mean = vectors.mean(axis=0).astype(np.float32)
    vector_std = np.maximum(vectors.std(axis=0), 1e-5).astype(np.float32)
    condition_mean = conditions.mean(axis=0).astype(np.float32)
    condition_std = np.maximum(conditions.std(axis=0), 1e-5).astype(np.float32)
    normalized_vectors = ((vectors - vector_mean) / vector_std).astype(np.float32)
    normalized_conditions = ((conditions - condition_mean) / condition_std).astype(np.float32)

    features = torch.tensor(normalized_vectors)
    condition_tensor = torch.tensor(normalized_conditions)
    model = _build_model(
        input_dim=features.shape[1],
        condition_dim=condition_tensor.shape[1],
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    history: list[dict[str, float]] = []

    model.train()
    for epoch in range(config.epochs):
        order = torch.randperm(len(features), generator=generator)
        reconstruction_total = 0.0
        kl_total = 0.0
        batches = 0
        for start in range(0, len(order), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            batch_features = features[batch_indices]
            batch_conditions = condition_tensor[batch_indices]
            reconstructed, mean, log_variance = model(batch_features, batch_conditions)
            reconstruction = functional.mse_loss(reconstructed, batch_features)
            kl = -0.5 * torch.mean(
                torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(), dim=1)
            )
            loss = reconstruction + config.beta * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            reconstruction_total += float(reconstruction.detach())
            kl_total += float(kl.detach())
            batches += 1
        history.append(
            {
                "epoch": epoch + 1,
                "reconstruction_loss": reconstruction_total / max(batches, 1),
                "kl_loss": kl_total / max(batches, 1),
            }
        )

    model.eval()
    state = {
        "vector_mean": vector_mean,
        "vector_std": vector_std,
        "condition_mean": condition_mean,
        "condition_std": condition_std,
        "loss_history": history,
    }
    return model, state


def _sample_vectors(
    model: Any,
    conditions: np.ndarray,
    state: dict[str, Any],
    *,
    count: int,
    config: VaeConfig,
) -> np.ndarray:
    import torch

    randomizer = np.random.default_rng(config.seed + 71)
    selected = conditions[randomizer.integers(0, len(conditions), size=count)]
    normalized_conditions = (
        (selected - state["condition_mean"]) / state["condition_std"]
    ).astype(np.float32)
    torch.manual_seed(config.seed + 72)
    with torch.no_grad():
        latent = torch.randn(count, config.latent_dim)
        decoded = model.decode(latent, torch.tensor(normalized_conditions)).numpy()
    return decoded * state["vector_std"] + state["vector_mean"]


def _novelty_distances(nearest: NearestNeighbors, vectors: np.ndarray) -> np.ndarray:
    distances, _ = nearest.kneighbors(vectors, n_neighbors=1, return_distance=True)
    return distances[:, 0] / math.sqrt(vectors.shape[1])


def _decode_events(
    vector: np.ndarray,
    *,
    width: int,
    height: int,
    point_count: int,
    output_event_count: int,
) -> list[dict[str, Any]]:
    trace = np.asarray(vector, dtype=float).reshape(point_count, 3)
    coordinates = np.clip(trace[:, :2], 0.0, 1.0)
    log_intervals = np.clip(trace[:, 2], 0.0, math.log1p(300.0))
    intervals = np.expm1(log_intervals)
    intervals[0] = 0.0
    intervals[1:] = np.clip(intervals[1:], 1.0, 300.0)
    times = np.cumsum(intervals)
    if output_event_count != point_count:
        coordinates = _resample_matrix(coordinates, output_event_count)
        times = _resample(times, output_event_count)

    events: list[dict[str, Any]] = []
    previous_time = -1
    for index, ((x_normalized, y_normalized), timestamp) in enumerate(zip(coordinates, times)):
        time_ms = 0 if index == 0 else max(int(round(float(timestamp))), previous_time + 1)
        previous_time = time_ms
        events.append(
            {
                "seq": index,
                "event_type": "pointerdown" if index == 0 else "pointerup" if index == output_event_count - 1 else "pointermove",
                "t_ms": time_ms,
                "x": round(float(x_normalized * width), 3),
                "y": round(float(y_normalized * height), 3),
                "x_normalized": round(float(x_normalized), 6),
                "y_normalized": round(float(y_normalized), 6),
                "target_role": "captcha_area",
            }
        )
    return events


def _payload(
    attempt_id: str,
    events: list[dict[str, Any]],
    *,
    role: str,
    width: int,
    height: int,
    novelty_distance: float,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "challenge_id": "vae_surrogate_challenge",
        "session_id": f"vae_surrogate_{role}",
        "anonymous_participant_id": None,
        "captcha": {"width": width, "height": height},
        "timing": {"presented_at": None, "submitted_at": None},
        "events": events,
        "interaction": {
            "regrab_count": 0,
            "retry_count": 0,
            "pointercancel_count": 0,
            "empty_click_count": 0,
            "failed_drop_count": 0,
        },
        "collection": {
            "label": "bot",
            "label_source": "vae_generated_surrogate",
            "bot_family": "conditional_vae_surrogate",
            "generator_version": GENERATOR_VERSION,
            "training_usage": "development_only" if role == "development" else "external_holdout_only",
            "novelty_distance": round(novelty_distance, 6),
            "age_group": "unknown",
        },
        "position_correct": True,
        "interaction_success": True,
        "final_drop_error": 0.0,
    }


def _sample_payloads(
    model: Any,
    state: dict[str, Any],
    source_vectors: np.ndarray,
    source_conditions: np.ndarray,
    dimensions: list[tuple[int, int]],
    event_count_choices: list[int],
    *,
    count: int,
    role: str,
    config: VaeConfig,
) -> list[dict[str, Any]]:
    nearest = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(source_vectors)
    randomizer = random.Random(config.seed + (1 if role == "development" else 2))
    output: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = count * 40

    while len(output) < count and attempts < max_attempts:
        batch_size = max(32, (count - len(output)) * 3)
        candidates = _sample_vectors(
            model,
            source_conditions,
            state,
            count=batch_size,
            config=VaeConfig(**{**asdict(config), "seed": config.seed + attempts}),
        )
        novelty = _novelty_distances(nearest, candidates)
        for vector, distance in zip(candidates, novelty):
            attempts += 1
            if distance < config.min_novelty_distance:
                continue
            width, height = randomizer.choice(dimensions)
            event_count = randomizer.choice(event_count_choices)
            events = _decode_events(
                vector,
                width=width,
                height=height,
                point_count=config.point_count,
                output_event_count=event_count,
            )
            quality = validate_attempt(events, captcha_width=width, captcha_height=height)
            if quality.status == QUALITY_REJECTED:
                continue
            output.append(
                _payload(
                    f"vaebot_{GENERATOR_VERSION}_{role}_{len(output):06d}",
                    events,
                    role=role,
                    width=width,
                    height=height,
                    novelty_distance=float(distance),
                )
            )
            if len(output) == count:
                break

    if len(output) != count:
        raise RuntimeError(
            f"generated {len(output)}/{count} novel valid samples after {attempts} draws; "
            "review the VAE loss and novelty threshold"
        )
    return output


def generate_dataset(
    *,
    human_attempts_path: Path,
    split_manifest_path: Path,
    output_path: Path,
    model_path: Path,
    role: str,
    count: int,
    config: VaeConfig,
) -> dict[str, Any]:
    """Fit one role-isolated VAE and write valid, novel bot payloads."""
    rows = load_jsonl(human_attempts_path)
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    source_rows = _source_rows(rows, split_manifest["attempt_to_split"], role)
    vectors = np.vstack([vectorize_attempt(row, config.point_count) for row in source_rows]).astype(np.float32)
    if not np.isfinite(vectors).all():
        raise ValueError("source trajectory vectors contain non-finite values")
    conditions = np.vstack(
        [_condition_from_vector(vector, config.point_count) for vector in vectors]
    ).astype(np.float32)
    dimensions, event_count_choices = _source_metadata(source_rows)
    model, state = _fit_model(vectors, conditions, config)
    payloads = _sample_payloads(
        model,
        state,
        vectors,
        conditions,
        dimensions,
        event_count_choices,
        count=count,
        role=role,
        config=config,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    import torch

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "generator_type": "conditional variational autoencoder",
            "generator_version": GENERATOR_VERSION,
            "source_role": role,
            "source_splits": sorted(ROLE_SOURCE_SPLITS[role]),
            "config": asdict(config),
            "model_state": model.state_dict(),
            "vector_mean": state["vector_mean"],
            "vector_std": state["vector_std"],
            "condition_mean": state["condition_mean"],
            "condition_std": state["condition_std"],
            "source_vectors": vectors,
            "source_conditions": conditions,
            "dimensions": dimensions,
            "event_count_choices": event_count_choices,
            "source_count": len(source_rows),
        },
        model_path,
    )

    manifest = {
        "dataset_name": output_path.stem,
        "generator_type": "conditional variational autoencoder",
        "generator_version": GENERATOR_VERSION,
        "role": role,
        "training_usage": "development_only" if role == "development" else "external_holdout_only",
        "source_split": sorted(ROLE_SOURCE_SPLITS[role]),
        "source_attempt_count": len(source_rows),
        "source_attempt_ids_exported": False,
        "config": asdict(config),
        "training": {
            "final_reconstruction_loss": state["loss_history"][-1]["reconstruction_loss"],
            "final_kl_loss": state["loss_history"][-1]["kl_loss"],
        },
        "count": len(payloads),
        "output_event_count_range": [min(event_count_choices), max(event_count_choices)],
        "inputs": {
            "human_attempts": {"path": str(human_attempts_path), "sha256": sha256(human_attempts_path)},
            "split_manifest": {"path": str(split_manifest_path), "sha256": sha256(split_manifest_path)},
        },
        "model_path": str(model_path),
        "output": {"path": str(output_path), "sha256": sha256(output_path)},
        "notes": [
            "Offline defensive surrogate data only; no browser, network, or database interaction.",
            "Nearest-neighbor novelty filtering rejects traces too close to source data.",
            "External-holdout output must never be used to fit or tune the detector.",
        ],
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-attempts", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-out", type=Path, required=True)
    parser.add_argument("--role", choices=tuple(ROLE_SOURCE_SPLITS), default="development")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--point-count", type=int, default=DEFAULT_POINT_COUNT)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-novelty-distance", type=float, default=DEFAULT_MIN_NOVELTY_DISTANCE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 1 or args.point_count < 4 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("count, epochs, batch-size must be positive and point-count >= 4")
    if args.latent_dim < 2 or args.hidden_dim < 8 or args.beta < 0.0 or args.min_novelty_distance < 0.0:
        raise ValueError("invalid VAE dimensions, beta, or novelty distance")
    manifest = generate_dataset(
        human_attempts_path=args.human_attempts,
        split_manifest_path=args.split_manifest,
        output_path=args.out,
        model_path=args.model_out,
        role=args.role,
        count=args.count,
        config=VaeConfig(
            point_count=args.point_count,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            beta=args.beta,
            learning_rate=args.learning_rate,
            min_novelty_distance=args.min_novelty_distance,
            seed=args.seed,
        ),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
