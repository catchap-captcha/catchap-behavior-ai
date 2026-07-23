"""Describe score-guided red-team weaknesses without changing the detector.

The tool compares a selected ``redteam_only`` weak set with blocked traces from
another ``redteam_only`` calibration set.  It reports feature separation,
motion-policy differences, and descriptive clusters.  It never fits or tunes
the behavior detector and must not be used to select a production threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.services.feature_profiles import get_feature_profile
from training.build_dataset import build_dataset
from training.generate_ml_bots import load_jsonl, sha256
from training.run_local_training import build_bot_feature_rows
from tools.run_formal_two_view_fusion import FUSION_RULE, VIEW_A, VIEW_B, _fused_scores


MUTATION_NUMERIC_FIELDS = (
    "curvature_abs",
    "time_power",
    "turn_slowdown",
    "frame_interval_ms",
    "event_coalescing",
)


def _score_and_features(
    payloads: list[dict[str, Any]],
    bundle: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    profile = get_feature_profile(bundle["feature_schema_version"], trajectory_only=True)
    rows = build_bot_feature_rows(
        payloads,
        groups_per_family=3,
        profile=profile,
        # This is score-only analysis. Payload metadata still blocks fitting.
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
    return scores, dataset.X.to_numpy(dtype=float), tuple(dataset.X.columns)


def _score_summary(scores: np.ndarray, threshold: float) -> dict[str, Any]:
    return {
        "count": int(len(scores)),
        "detector_pass_count": int(np.count_nonzero(scores >= threshold)),
        "detector_pass_rate": float(np.mean(scores >= threshold)),
        "min": float(np.min(scores)),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.median(scores)),
        "p75": float(np.percentile(scores, 75)),
        "max": float(np.max(scores)),
    }


def rank_feature_separation(
    weak_features: np.ndarray,
    control_features: np.ndarray,
    feature_names: tuple[str, ...],
    *,
    limit: int = 15,
) -> list[dict[str, float | str]]:
    """Rank descriptive standardized mean differences, not model importance."""
    weak_mean = weak_features.mean(axis=0)
    control_mean = control_features.mean(axis=0)
    pooled_std = np.sqrt((weak_features.var(axis=0) + control_features.var(axis=0)) / 2.0)
    standardized = np.divide(
        weak_mean - control_mean,
        pooled_std,
        out=np.zeros_like(weak_mean),
        where=pooled_std > 1e-12,
    )
    ordered = np.argsort(np.abs(standardized))[::-1][:limit]
    return [
        {
            "feature": feature_names[index],
            "weak_mean": float(weak_mean[index]),
            "blocked_control_mean": float(control_mean[index]),
            "standardized_mean_difference": float(standardized[index]),
        }
        for index in ordered
    ]


def _mutation_matrix(payloads: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for payload in payloads:
        mutation = payload.get("collection", {}).get("mutation") or {}
        rows.append(
            [
                abs(float(mutation.get("curvature_amplitude", 0.0))),
                float(mutation.get("time_power", 0.0)),
                float(mutation.get("turn_slowdown", 0.0)),
                float(mutation.get("frame_interval_ms", 0.0)),
                float(mutation.get("event_coalescing", 0.0)),
            ]
        )
    return np.asarray(rows, dtype=float)


def _mutation_summary(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    matrix = _mutation_matrix(payloads)
    correction_count = sum(
        bool((payload.get("collection", {}).get("mutation") or {}).get("correction_used"))
        for payload in payloads
    )
    return {
        "count": len(payloads),
        "numeric_mean": {
            field: float(matrix[:, index].mean())
            for index, field in enumerate(MUTATION_NUMERIC_FIELDS)
        },
        "numeric_median": {
            field: float(np.median(matrix[:, index]))
            for index, field in enumerate(MUTATION_NUMERIC_FIELDS)
        },
        "correction_used_count": correction_count,
        "correction_used_rate": correction_count / len(payloads),
    }


def _cluster_summary(
    weak_features: np.ndarray,
    weak_payloads: list[dict[str, Any]],
    weak_scores: np.ndarray,
    threshold: float,
    clusters: int,
) -> list[dict[str, Any]]:
    if clusters < 2 or clusters > len(weak_payloads):
        raise ValueError("clusters must be between 2 and weak-set size")
    normalized = StandardScaler().fit_transform(weak_features)
    labels = KMeans(n_clusters=clusters, n_init=20, random_state=20260722).fit_predict(normalized)
    mutations = _mutation_matrix(weak_payloads)
    summary: list[dict[str, Any]] = []
    for label in range(clusters):
        mask = labels == label
        values = mutations[mask]
        scores = weak_scores[mask]
        summary.append(
            {
                "cluster": int(label + 1),
                "count": int(np.count_nonzero(mask)),
                "human_score_median": float(np.median(scores)),
                "detector_pass_count": int(np.count_nonzero(scores >= threshold)),
                "mutation_mean": {
                    field: float(values[:, index].mean())
                    for index, field in enumerate(MUTATION_NUMERIC_FIELDS)
                },
            }
        )
    return sorted(summary, key=lambda item: item["human_score_median"], reverse=True)


def _assert_redteam_only(payloads: list[dict[str, Any]], label: str) -> None:
    usages = {payload.get("collection", {}).get("training_usage") for payload in payloads}
    if usages != {"redteam_only"}:
        raise ValueError(f"{label} must contain only redteam_only payloads, found {sorted(usages)}")


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 점수 유도형 레드팀 약점 분석",
        "",
        "이 보고서는 고정 detector에 대한 설명용 분석이다. detector fitting, threshold tuning, "
        "production 판정은 수행하지 않았다.",
        "",
        "## 점수 분포",
        "",
        "| 집합 | 수량 | 통과 수 | 통과율 | 중앙 Human score |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, values in (("약점 세트", report["weak_scores"]), ("차단 control", report["blocked_control_scores"])):
        lines.append(
            f"| {label} | {values['count']} | {values['detector_pass_count']} | "
            f"{values['detector_pass_rate']:.2%} | {values['median']:.9f} |"
        )
    lines.extend(["", "## Feature 차이", "", "| Feature | 약점 평균 | 차단 control 평균 | 표준화 평균 차이 |", "|---|---:|---:|---:|"])
    for row in report["top_feature_separation"]:
        lines.append(
            f"| {row['feature']} | {row['weak_mean']:.6f} | {row['blocked_control_mean']:.6f} | "
            f"{row['standardized_mean_difference']:.3f} |"
        )
    lines.extend(["", "## 약점 군집", "", "| 군집 | 수량 | 중앙 Human score | 통과 수 |", "|---|---:|---:|---:|"])
    for cluster in report["weak_clusters"]:
        lines.append(
            f"| {cluster['cluster']} | {cluster['count']} | {cluster['human_score_median']:.9f} | "
            f"{cluster['detector_pass_count']} |"
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "- 이 결과는 `redteam_only` 합성 Bot끼리의 기술적 차이다. 정상 Human FRR을 다시 측정한 결과가 아니다.",
            "- 차이는 설명용 평균 차이이며, 특정 Feature 하나가 원인이라는 인과 결론이 아니다.",
            "- 이 보고서를 근거로 threshold를 낮추거나 약점 세트를 detector 학습에 넣으면 안 된다.",
        ]
    )
    return "\n".join(lines) + "\n"


def analyze(
    *,
    weak_set_path: Path,
    control_set_path: Path,
    model_path: Path,
    report_path: Path,
    clusters: int,
) -> dict[str, Any]:
    weak_payloads = load_jsonl(weak_set_path)
    control_payloads = load_jsonl(control_set_path)
    _assert_redteam_only(weak_payloads, "weak set")
    _assert_redteam_only(control_payloads, "control set")

    bundle = joblib.load(model_path)
    if bundle.get("score_fusion") != FUSION_RULE:
        raise ValueError("expected formal two-view min-fusion model bundle")
    threshold = float(bundle["threshold"])
    weak_scores, weak_features, feature_names = _score_and_features(weak_payloads, bundle)
    control_scores, control_features, control_feature_names = _score_and_features(control_payloads, bundle)
    if feature_names != control_feature_names:
        raise RuntimeError("weak and control feature matrices do not match")
    blocked_mask = control_scores < threshold
    if not np.any(blocked_mask):
        raise ValueError("control set has no blocked samples for comparison")
    blocked_payloads = [payload for payload, keep in zip(control_payloads, blocked_mask) if keep]

    report = {
        "scope": "descriptive red-team analysis only; no detector fitting or threshold tuning",
        "weak_set": {"path": str(weak_set_path), "sha256": sha256(weak_set_path)},
        "control_set": {"path": str(control_set_path), "sha256": sha256(control_set_path)},
        "frozen_detector": {
            "path": str(model_path),
            "sha256": sha256(model_path),
            "model_name": bundle["model_name"],
            "feature_schema_version": bundle["feature_schema_version"],
            "score_fusion": bundle["score_fusion"],
            "threshold": threshold,
        },
        "weak_scores": _score_summary(weak_scores, threshold),
        "blocked_control_scores": _score_summary(control_scores[blocked_mask], threshold),
        "top_feature_separation": rank_feature_separation(
            weak_features,
            control_features[blocked_mask],
            feature_names,
        ),
        "weak_mutations": _mutation_summary(weak_payloads),
        "blocked_control_mutations": _mutation_summary(blocked_payloads),
        "weak_clusters": _cluster_summary(
            weak_features,
            weak_payloads,
            weak_scores,
            threshold,
            clusters,
        ),
        "notes": [
            "Weak set and control set are redteam_only and remain detector-forbidden.",
            "The control comparison uses only calibration rows blocked by the same frozen detector.",
            "Feature separation is descriptive and must not be treated as causal model importance.",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weak-set", type=Path, required=True)
    parser.add_argument("--control-set", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = analyze(
        weak_set_path=args.weak_set,
        control_set_path=args.control_set,
        model_path=args.model,
        report_path=args.report,
        clusters=args.clusters,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
