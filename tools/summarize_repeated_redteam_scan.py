"""Summarize repeated fixed-detector red-team scans without model tuning.

This tool combines reports emitted by ``tools.redteam_weakness_search``.  It
only describes whether the same weak-vs-blocked feature direction recurs across
independent random seeds.  It never fits a detector, changes a threshold, or
turns ``redteam_only`` data into detector training input.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.services.feature_extractor_v23 import V23_ADDITIONAL_FEATURES


REQUIRED_SCOPE = "offline red-team weakness search; scoring only, no fitting or threshold tuning"


def summarize_feature_deltas(
    reports: list[dict[str, Any]],
    *,
    min_runs: int,
    min_absolute_delta: float,
) -> list[dict[str, Any]]:
    """Return repeated weak-set feature directions from precomputed reports."""
    if not reports:
        raise ValueError("at least one scan report is required")

    features = sorted(reports[0]["feature_summary"]["weak_set"])
    summary: list[dict[str, Any]] = []
    for feature in features:
        deltas = np.asarray(
            [
                float(report["feature_summary"]["weak_set"][feature]["mean"])
                - float(report["feature_summary"]["blocked"][feature]["mean"])
                for report in reports
            ],
            dtype=float,
        )
        positive_runs = int(np.count_nonzero(deltas >= min_absolute_delta))
        negative_runs = int(np.count_nonzero(deltas <= -min_absolute_delta))
        direction = "higher_in_weak_set" if positive_runs >= negative_runs else "lower_in_weak_set"
        support = max(positive_runs, negative_runs)
        summary.append(
            {
                "feature": feature,
                "mean_delta": float(np.mean(deltas)),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
                "positive_support_runs": positive_runs,
                "negative_support_runs": negative_runs,
                "repeated_direction": direction,
                "recurrent": support >= min_runs,
                "already_in_v23": feature in V23_ADDITIONAL_FEATURES,
            }
        )
    return sorted(
        summary,
        key=lambda item: (
            not item["recurrent"],
            -max(item["positive_support_runs"], item["negative_support_runs"]),
            -abs(item["mean_delta"]),
        ),
    )


def _read_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("scope") != REQUIRED_SCOPE:
        raise ValueError(f"not a fixed-detector red-team scan report: {path}")
    if report.get("generation", {}).get("accepted") != report.get("counts", {}).get("candidates"):
        raise ValueError(f"incomplete candidate generation in report: {path}")
    return report


def _markdown(report: dict[str, Any]) -> str:
    totals = report["totals"]
    lines = [
        "# 반복 Red-team 약점 탐색 요약",
        "",
        "이 보고서는 고정된 detector에 대한 합성 `redteam_only` 공격 진단이다. "
        "모델 fitting, feature 선택, threshold tuning, 외부 holdout 평가는 수행하지 않았다.",
        "",
        "## 반복 결과",
        "",
        f"- 독립 seed 실행: {report['run_count']}회",
        f"- 후보: {totals['candidates']:,}건, 현재 threshold 통과: {totals['evaders']:,}건 "
        f"({totals['bot_asr']:.3%})",
        f"- 실행별 ASR 범위: {report['per_run_asr']['min']:.3%} ~ {report['per_run_asr']['max']:.3%}",
        f"- 약점/경계 세트 {totals['weak_set']:,}건 중 dynamics view가 binding인 수: "
        f"{totals['dynamics_binding_count']:,}건 ({totals['dynamics_binding_rate']:.1%})",
        "",
        "## 반복 feature 방향",
        "",
        "| Feature | 약점-차단 평균 차이 | 재현 방향 | 지지 실행 | v2.3 포함 |",
        "|---|---:|---|---:|---|",
    ]
    for item in report["feature_recurrence"]:
        support = max(item["positive_support_runs"], item["negative_support_runs"])
        lines.append(
            f"| {item['feature']} | {item['mean_delta']:+.6f} | {item['repeated_direction']} | "
            f"{support}/{report['run_count']} | {'예' if item['already_in_v23'] else '아니오'} |"
        )
    lines.extend(
        [
            "",
            "## 결정",
            "",
            f"- feature 변경: **{report['feature_change_decision']['status']}**",
            f"- 이유: {report['feature_change_decision']['reason']}",
            f"- OOF 재검증: **{report['oof_decision']['status']}**",
            f"- 이유: {report['oof_decision']['reason']}",
            "",
            "이 결과는 합성 Bot 생성기 내부의 반복 패턴일 수 있다. 실제 detector 변경은 새롭고 "
            "일반화 가능한 feature 가설이 생긴 뒤, 개발 데이터 내부 OOF와 새 holdout으로만 평가한다.",
        ]
    )
    return "\n".join(lines) + "\n"


def summarize(
    *,
    report_paths: list[Path],
    output_path: Path,
    min_runs: int = 4,
    min_absolute_delta: float = 0.02,
) -> dict[str, Any]:
    if min_runs < 1 or min_runs > len(report_paths):
        raise ValueError("min_runs must be between 1 and the number of reports")
    if min_absolute_delta < 0:
        raise ValueError("min_absolute_delta must be non-negative")

    reports = [_read_report(path) for path in report_paths]
    detector_paths = {report["fixed_detector"]["model_path"] for report in reports}
    thresholds = {float(report["fixed_detector"]["threshold"]) for report in reports}
    if len(detector_paths) != 1 or len(thresholds) != 1:
        raise ValueError("all runs must use the same frozen detector and threshold")

    recurrence = summarize_feature_deltas(
        reports,
        min_runs=min_runs,
        min_absolute_delta=min_absolute_delta,
    )
    recurrent_new = [item["feature"] for item in recurrence if item["recurrent"] and not item["already_in_v23"]]
    candidates = sum(int(report["counts"]["candidates"]) for report in reports)
    evaders = sum(int(report["counts"]["evaders"]) for report in reports)
    weak_set = sum(int(report["counts"]["weak_set"]) for report in reports)
    dynamics_binding = sum(
        int(report["binding_view_of_weak_set"].get("dynamics_physics", 0)) for report in reports
    )
    asrs = np.asarray([float(report["rates"]["bot_asr_at_fixed_threshold"]) for report in reports])

    feature_decision = {
        "status": "candidate_feature_hypothesis_required" if recurrent_new else "no_new_feature_candidate",
        "reason": (
            "반복된 red-team 차이에 v2.3에 없는 feature가 있으므로, 개발 데이터 OOF 전에 "
            "일반적인 궤적 가설을 먼저 정의해야 한다."
            if recurrent_new
            else "반복된 방향은 모두 기존 v2.3 물리 feature로 이미 표현된다. 중복 feature를 추가하거나 "
            "redteam_only 궤적에 맞춰 조정할 근거가 없다."
        ),
        "recurrent_new_features": recurrent_new,
    }
    oof_decision = {
        "status": "required_after_feature_change" if recurrent_new else "not_run_no_detector_change",
        "reason": (
            "제안 feature는 holdout 점수화 전에 개발 데이터 내부 그룹 OOF로 평가해야 한다."
            if recurrent_new
            else "일반 feature나 모델 설정이 바뀌지 않았으므로, OOF를 반복해도 새 후보를 검증하는 실험이 아니다."
        ),
    }
    result = {
        "scope": "offline repeated red-team summary; no fitting, threshold tuning, or external holdout evaluation",
        "run_count": len(reports),
        "inputs": [str(path) for path in report_paths],
        "fixed_detector": {
            "model_path": next(iter(detector_paths)),
            "threshold": next(iter(thresholds)),
        },
        "criteria": {
            "minimum_same_direction_runs": min_runs,
            "minimum_absolute_per_run_mean_delta": min_absolute_delta,
        },
        "totals": {
            "candidates": candidates,
            "evaders": evaders,
            "bot_asr": evaders / candidates,
            "weak_set": weak_set,
            "dynamics_binding_count": dynamics_binding,
            "dynamics_binding_rate": dynamics_binding / weak_set if weak_set else 0.0,
        },
        "per_run_asr": {
            "min": float(np.min(asrs)),
            "median": float(np.median(asrs)),
            "max": float(np.max(asrs)),
        },
        "feature_recurrence": recurrence,
        "feature_change_decision": feature_decision,
        "oof_decision": oof_decision,
        "guards": [
            "Every source scan is redteam_only and remains forbidden from detector fitting and threshold tuning.",
            "The frozen detector and threshold are only used to score synthetic diagnostic candidates.",
            "The sealed external holdout is not read by this tool.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-runs", type=int, default=4)
    parser.add_argument("--min-absolute-delta", type=float, default=0.02)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = summarize(
        report_paths=args.report,
        output_path=args.out,
        min_runs=args.min_runs,
        min_absolute_delta=args.min_absolute_delta,
    )
    print(json.dumps({"totals": result["totals"], "decision": result["feature_change_decision"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
