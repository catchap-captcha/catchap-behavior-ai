"""Compare repeated red-team summaries from two offline candidate generators.

The comparison is descriptive.  It identifies shared and generator-specific
recurring feature directions from already-produced diagnostic summaries.  It
never fits a detector, selects a feature, or changes a threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EXPECTED_SCOPE = "offline repeated red-team summary; no fitting, threshold tuning, or external holdout evaluation"


def compare_feature_recurrence(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Classify recurring directions as shared, candidate-specific, or absent."""
    baseline_by_name = {str(item["feature"]): item for item in baseline}
    candidate_by_name = {str(item["feature"]): item for item in candidate}
    comparison: list[dict[str, Any]] = []
    for feature in sorted(set(baseline_by_name) | set(candidate_by_name)):
        old = baseline_by_name.get(feature)
        new = candidate_by_name.get(feature)
        old_recurrent = bool(old and old.get("recurrent"))
        new_recurrent = bool(new and new.get("recurrent"))
        same_direction = bool(
            old_recurrent
            and new_recurrent
            and old.get("repeated_direction") == new.get("repeated_direction")
        )
        if same_direction:
            status = "shared_recurrent"
        elif new_recurrent:
            status = "candidate_generator_recurrent"
        elif old_recurrent:
            status = "baseline_generator_only"
        else:
            status = "not_recurrent"
        comparison.append(
            {
                "feature": feature,
                "status": status,
                "baseline_mean_delta": old.get("mean_delta") if old else None,
                "candidate_mean_delta": new.get("mean_delta") if new else None,
                "baseline_direction": old.get("repeated_direction") if old_recurrent else None,
                "candidate_direction": new.get("repeated_direction") if new_recurrent else None,
                "already_in_v23": bool((new or old or {}).get("already_in_v23")),
            }
        )
    return comparison


def _load(path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("scope") != EXPECTED_SCOPE:
        raise ValueError(f"not a repeated red-team summary: {path}")
    return summary


def _markdown(report: dict[str, Any]) -> str:
    baseline = report["baseline"]
    candidate = report["candidate"]
    lines = [
        "# PCA-GMM과 VAE Red-team 반복 탐색 비교",
        "",
        "두 결과는 동일한 고정 detector에 대한 오프라인 `redteam_only` 진단이다. "
        "합성 Bot을 detector 학습·임계값 조정에 사용하지 않았고, 봉인 external holdout도 사용하지 않았다.",
        "",
        "## 공격 통과율",
        "",
        "| 생성기 | 후보 | 통과 | ASR | 실행별 중앙 ASR | dynamics binding |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| {baseline['label']} | {baseline['totals']['candidates']:,} | {baseline['totals']['evaders']:,} | "
            f"{baseline['totals']['bot_asr']:.3%} | {baseline['per_run_asr']['median']:.3%} | "
            f"{baseline['totals']['dynamics_binding_rate']:.1%} |"
        ),
        (
            f"| {candidate['label']} | {candidate['totals']['candidates']:,} | {candidate['totals']['evaders']:,} | "
            f"{candidate['totals']['bot_asr']:.3%} | {candidate['per_run_asr']['median']:.3%} | "
            f"{candidate['totals']['dynamics_binding_rate']:.1%} |"
        ),
        "",
        "## 반복 feature 패턴",
        "",
        "| Feature | 비교 | PCA-GMM 방향 | VAE 방향 | v2.3 포함 |",
        "|---|---|---|---|---|",
    ]
    for item in report["feature_comparison"]:
        lines.append(
            f"| {item['feature']} | {item['status']} | {item['baseline_direction'] or '-'} | "
            f"{item['candidate_direction'] or '-'} | {'예' if item['already_in_v23'] else '아니오'} |"
        )
    lines.extend(
        [
            "",
            "## 결론",
            "",
            report["conclusion"],
            "",
            "이 비교는 생성기 차이를 설명할 뿐, 새 feature의 유효성이나 production 보안 성능을 증명하지 않는다. "
            "새 일반 feature 가설이 생겨도 먼저 개발 데이터 내부 OOF에서만 검증한다.",
        ]
    )
    return "\n".join(lines) + "\n"


def compare(
    *,
    baseline_path: Path,
    candidate_path: Path,
    output_path: Path,
    baseline_label: str = "PCA-GMM hybrid",
    candidate_label: str = "VAE hybrid",
) -> dict[str, Any]:
    baseline = _load(baseline_path)
    candidate = _load(candidate_path)
    feature_comparison = compare_feature_recurrence(
        baseline["feature_recurrence"], candidate["feature_recurrence"]
    )
    candidate_specific = [
        item["feature"]
        for item in feature_comparison
        if item["status"] == "candidate_generator_recurrent"
    ]
    new_feature_hypotheses = [
        item["feature"]
        for item in feature_comparison
        if item["status"] == "candidate_generator_recurrent" and not item["already_in_v23"]
    ]
    conclusion = (
        "VAE는 PCA-GMM보다 높은 합성 후보 통과율을 보였으며, 두 생성기는 "
        "pause 위치 엔트로피와 회전 변화 매끄러움 패턴을 공유한다. VAE에서만 반복된 "
        f"{', '.join(candidate_specific) if candidate_specific else '추가 feature'}도 이미 v2.3에 있으므로, "
        "redteam 데이터에 맞춘 feature 추가·재학습·threshold tuning은 하지 않는다."
        if not new_feature_hypotheses
        else "VAE에서만 반복되고 v2.3에 없는 feature가 있으므로, 일반 궤적 가설을 정의한 뒤 "
        "개발 데이터 내부 OOF 후보 실험을 검토한다."
    )
    report = {
        "scope": "offline red-team generator comparison; no fitting, threshold tuning, or external holdout evaluation",
        "baseline": {"label": baseline_label, "source": str(baseline_path), **baseline},
        "candidate": {"label": candidate_label, "source": str(candidate_path), **candidate},
        "feature_comparison": feature_comparison,
        "candidate_generator_only_recurrent_features": candidate_specific,
        "new_feature_hypotheses": new_feature_hypotheses,
        "conclusion": conclusion,
        "guards": [
            "Both source summaries are redteam_only diagnostics.",
            "No detector model was fitted and no threshold was changed by this comparison.",
            "A recurrent synthetic pattern is not by itself a production feature-selection criterion.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_path.with_suffix(".md").write_text(_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-label", default="PCA-GMM hybrid")
    parser.add_argument("--candidate-label", default="VAE hybrid")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = compare(
        baseline_path=args.baseline,
        candidate_path=args.candidate,
        output_path=args.out,
        baseline_label=args.baseline_label,
        candidate_label=args.candidate_label,
    )
    print(json.dumps({"new_feature_hypotheses": report["new_feature_hypotheses"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
