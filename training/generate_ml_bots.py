"""Generate defensive Human-like surrogate bots with a trained PCA + GMM model.

This is an offline red-team data generator for CatChap only. It never opens a
CAPTCHA, controls a browser, or calls a network service. The model learns a
compact distribution of normalized pointer traces, then samples new traces as
hard negative examples for detector research.

The generator deliberately separates development and external-holdout sources:
development samples are derived from training-partition Human traces, while
external samples use held-out Human traces and are marked as forbidden from
detector fitting. A nearest-neighbor novelty check rejects samples that are too
close to a source trace.
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

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from app.services.quality_validator import QUALITY_REJECTED, validate_attempt


GENERATOR_VERSION = "pca_gmm_trace_v2"
DEFAULT_POINT_COUNT = 48
DEFAULT_PCA_COMPONENTS = 24
DEFAULT_GMM_COMPONENTS = 8
DEFAULT_MIN_NOVELTY_DISTANCE = 0.015
ROLE_SOURCE_SPLITS = {
    "development": {"train"},
    "external_holdout": {"test"},
}


@dataclass(frozen=True)
class GeneratorConfig:
    point_count: int = DEFAULT_POINT_COUNT
    pca_components: int = DEFAULT_PCA_COMPONENTS
    gmm_components: int = DEFAULT_GMM_COMPONENTS
    min_novelty_distance: float = DEFAULT_MIN_NOVELTY_DISTANCE
    seed: int = 20260721


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _resample(values: np.ndarray, point_count: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, num=len(values))
    target = np.linspace(0.0, 1.0, num=point_count)
    return np.interp(target, source, values)


def _resample_matrix(values: np.ndarray, point_count: int) -> np.ndarray:
    return np.column_stack([_resample(values[:, column], point_count) for column in range(values.shape[1])])


def vectorize_attempt(payload: dict[str, Any], point_count: int) -> np.ndarray:
    """Encode an attempt as normalized x/y positions and log time intervals."""
    events = sorted(payload.get("events") or [], key=lambda event: event.get("seq", 0))
    if len(events) < 4:
        raise ValueError("attempt needs at least four events")
    captcha = payload.get("captcha") or {}
    width = max(float(captcha.get("width") or 1.0), 1.0)
    height = max(float(captcha.get("height") or 1.0), 1.0)
    x = np.asarray(
        [
            _clamp(float(event.get("x_normalized", float(event["x"]) / width)), 0.0, 1.0)
            for event in events
        ],
        dtype=float,
    )
    y = np.asarray(
        [
            _clamp(float(event.get("y_normalized", float(event["y"]) / height)), 0.0, 1.0)
            for event in events
        ],
        dtype=float,
    )
    times = np.asarray([float(event["t_ms"]) for event in events], dtype=float)
    times = np.maximum.accumulate(times - times[0])
    sampled_times = _resample(times, point_count)
    intervals = np.diff(sampled_times, prepend=sampled_times[0])
    intervals[0] = 0.0
    intervals = np.maximum(intervals, 0.0)
    matrix = np.column_stack(
        (
            _resample(x, point_count),
            _resample(y, point_count),
            np.log1p(intervals),
        )
    )
    return matrix.reshape(-1)


def _decode_events(
    vector: np.ndarray,
    *,
    width: int,
    height: int,
    point_count: int,
    output_event_count: int,
) -> list[dict[str, Any]]:
    matrix = np.asarray(vector, dtype=float).reshape(point_count, 3)
    coordinates = np.clip(matrix[:, :2], 0.0, 1.0)
    log_intervals = np.clip(matrix[:, 2], 0.0, math.log1p(300.0))
    intervals = np.expm1(log_intervals)
    intervals[0] = 0.0
    intervals[1:] = np.clip(intervals[1:], 1.0, 300.0)
    times = np.cumsum(intervals)
    if output_event_count != point_count:
        coordinates = _resample_matrix(coordinates, output_event_count)
        times = _resample(times, output_event_count)

    events: list[dict[str, Any]] = []
    previous_time = -1
    for index, ((x_normalized, y_normalized), time_ms) in enumerate(zip(coordinates, times)):
        timestamp = 0 if index == 0 else max(int(round(float(time_ms))), previous_time + 1)
        previous_time = timestamp
        events.append(
            {
                "seq": index,
                "event_type": (
                    "pointerdown"
                    if index == 0
                    else "pointerup" if index == output_event_count - 1 else "pointermove"
                ),
                "t_ms": timestamp,
                "x": round(float(x_normalized * width), 3),
                "y": round(float(y_normalized * height), 3),
                "x_normalized": round(float(x_normalized), 6),
                "y_normalized": round(float(y_normalized), 6),
                "target_role": "captcha_area",
            }
        )
    return events


def _source_rows(
    rows: Iterable[dict[str, Any]],
    attempt_to_split: dict[str, str],
    role: str,
) -> list[dict[str, Any]]:
    if role not in ROLE_SOURCE_SPLITS:
        raise ValueError(f"unsupported role: {role}")
    allowed_splits = ROLE_SOURCE_SPLITS[role]
    selected = [
        row
        for row in rows
        if attempt_to_split.get(str(row.get("attempt_id"))) in allowed_splits
        and len(row.get("events") or []) >= 4
    ]
    if len(selected) < 20:
        raise ValueError(f"need at least 20 {role} Human attempts, found {len(selected)}")
    return selected


def fit_generator(source_rows: list[dict[str, Any]], config: GeneratorConfig) -> dict[str, Any]:
    vectors = np.vstack([vectorize_attempt(row, config.point_count) for row in source_rows])
    max_pca = min(vectors.shape[0] - 1, vectors.shape[1])
    pca_components = min(config.pca_components, max_pca)
    if pca_components < 2:
        raise ValueError("not enough source variation for PCA")
    pca = PCA(n_components=pca_components, random_state=config.seed)
    latent = pca.fit_transform(vectors)
    gmm_components = min(config.gmm_components, max(2, len(source_rows) // 10))
    gmm = GaussianMixture(
        n_components=gmm_components,
        covariance_type="diag",
        reg_covar=1e-5,
        max_iter=300,
        n_init=2,
        random_state=config.seed,
    ).fit(latent)
    nearest = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(vectors)
    dimensions = sorted(
        {
            (int(row["captcha"]["width"]), int(row["captcha"]["height"]))
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
    return {
        "pca": pca,
        "gmm": gmm,
        "nearest": nearest,
        "vector_dimension": int(vectors.shape[1]),
        "dimensions": dimensions,
        "event_count_choices": event_count_choices,
        "source_count": len(source_rows),
        "config": asdict(config),
    }


def _novelty_distances(nearest: NearestNeighbors, vectors: np.ndarray) -> np.ndarray:
    distances, _ = nearest.kneighbors(vectors, n_neighbors=1, return_distance=True)
    return distances[:, 0] / math.sqrt(vectors.shape[1])


def _payload(
    attempt_id: str,
    events: list[dict[str, Any]],
    *,
    role: str,
    width: int,
    height: int,
    novelty_distance: float,
) -> dict[str, Any]:
    training_usage = "development_only" if role == "development" else "external_holdout_only"
    return {
        "schema_version": "1.0",
        "attempt_id": attempt_id,
        "challenge_id": "ml_surrogate_challenge",
        "session_id": f"ml_surrogate_{role}",
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
            "label_source": "ml_generated_surrogate",
            "bot_family": "pca_gmm_surrogate",
            "generator_version": GENERATOR_VERSION,
            "training_usage": training_usage,
            "novelty_distance": round(novelty_distance, 6),
            "age_group": "unknown",
        },
        "position_correct": True,
        "interaction_success": True,
        "final_drop_error": 0.0,
    }


def sample_payloads(
    generator: dict[str, Any],
    *,
    count: int,
    role: str,
    config: GeneratorConfig,
) -> list[dict[str, Any]]:
    randomizer = random.Random(config.seed + (1 if role == "development" else 2))
    pca: PCA = generator["pca"]
    gmm: GaussianMixture = generator["gmm"]
    nearest: NearestNeighbors = generator["nearest"]
    output: list[dict[str, Any]] = []
    attempts = 0
    max_attempts = count * 40

    while len(output) < count and attempts < max_attempts:
        batch_size = max(32, (count - len(output)) * 3)
        latent, _ = gmm.sample(batch_size)
        candidates = pca.inverse_transform(latent)
        novelty = _novelty_distances(nearest, candidates)
        for vector, distance in zip(candidates, novelty):
            attempts += 1
            if distance < config.min_novelty_distance:
                continue
            width, height = randomizer.choice(generator["dimensions"])
            output_event_count = randomizer.choice(generator["event_count_choices"])
            events = _decode_events(
                vector,
                width=width,
                height=height,
                point_count=config.point_count,
                output_event_count=output_event_count,
            )
            quality = validate_attempt(events, captcha_width=width, captcha_height=height)
            if quality.status == QUALITY_REJECTED:
                continue
            attempt_id = f"mlbot_{GENERATOR_VERSION}_{role}_{len(output):06d}"
            output.append(
                _payload(
                    attempt_id,
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
            "lower the novelty distance only after reviewing source similarity"
        )
    return output


def generate_dataset(
    *,
    human_attempts_path: Path,
    split_manifest_path: Path,
    output_path: Path,
    role: str,
    count: int,
    config: GeneratorConfig,
    model_path: Path | None = None,
) -> dict[str, Any]:
    rows = load_jsonl(human_attempts_path)
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    source_rows = _source_rows(rows, split_manifest["attempt_to_split"], role)
    generator = fit_generator(source_rows, config)
    payloads = sample_payloads(generator, count=count, role=role, config=config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for payload in payloads:
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")

    if model_path is not None:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "generator_type": "PCA + diagonal GaussianMixture",
                "generator_version": GENERATOR_VERSION,
                "source_role": role,
                "source_splits": sorted(ROLE_SOURCE_SPLITS[role]),
                "pca": generator["pca"],
                "gmm": generator["gmm"],
                "point_count": config.point_count,
                "dimensions": generator["dimensions"],
                "event_count_choices": generator["event_count_choices"],
                "source_count": generator["source_count"],
                "config": asdict(config),
            },
            model_path,
        )

    manifest = {
        "dataset_name": output_path.stem,
        "generator_type": "PCA + diagonal GaussianMixture",
        "generator_version": GENERATOR_VERSION,
        "role": role,
        "training_usage": "development_only" if role == "development" else "external_holdout_only",
        "source_split": sorted(ROLE_SOURCE_SPLITS[role]),
        "source_attempt_count": generator["source_count"],
        "source_attempt_ids_exported": False,
        "config": asdict(config),
        "count": len(payloads),
        "output_event_count_range": [
            min(generator["event_count_choices"]),
            max(generator["event_count_choices"]),
        ],
        "inputs": {
            "human_attempts": {"path": str(human_attempts_path), "sha256": sha256(human_attempts_path)},
            "split_manifest": {"path": str(split_manifest_path), "sha256": sha256(split_manifest_path)},
        },
        "model_path": str(model_path) if model_path is not None else None,
        "output": {"path": str(output_path), "sha256": sha256(output_path)},
        "notes": [
            "Offline defensive surrogate data only; no browser or network interaction.",
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
    parser.add_argument("--role", choices=tuple(ROLE_SOURCE_SPLITS), required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--point-count", type=int, default=DEFAULT_POINT_COUNT)
    parser.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    parser.add_argument("--gmm-components", type=int, default=DEFAULT_GMM_COMPONENTS)
    parser.add_argument("--min-novelty-distance", type=float, default=DEFAULT_MIN_NOVELTY_DISTANCE)
    parser.add_argument("--model-out", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 1 or args.point_count < 4 or args.min_novelty_distance < 0.0:
        raise ValueError("count must be positive, point-count >= 4, novelty distance >= 0")
    manifest = generate_dataset(
        human_attempts_path=args.human_attempts,
        split_manifest_path=args.split_manifest,
        output_path=args.out,
        role=args.role,
        count=args.count,
        config=GeneratorConfig(
            point_count=args.point_count,
            pca_components=args.pca_components,
            gmm_components=args.gmm_components,
            min_novelty_distance=args.min_novelty_distance,
            seed=args.seed,
        ),
        model_path=args.model_out,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
