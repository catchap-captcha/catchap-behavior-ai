"""Create compact CSV and Markdown comparisons for two training checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


MODEL_LABELS = {
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
}


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_rows(label: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    holdouts = summary["family_holdout_stress_test"]
    rows = []
    for model_name, metrics in summary["test"].items():
        family = holdouts[model_name]
        by_family = {item["held_out_bot_family"]: item["bot_recall"] for item in family}
        robust = next(
            (
                item
                for item in summary["selection"]["robust_candidate"]["ranked_candidates"]
                if item["model_name"] == model_name
            ),
            None,
        )
        rows.append(
            {
                "checkpoint": label,
                "human_samples": summary["readiness"]["human_samples"],
                "human_participants": summary["readiness"]["human_participants"],
                "bot_samples": summary["readiness"]["bot_samples"],
                "bot_families": summary["readiness"]["bot_families"],
                "model": MODEL_LABELS.get(model_name, model_name),
                "test_accuracy": metrics["accuracy"],
                "test_human_frr": metrics["human_frr"],
                "test_bot_recall": metrics["bot_recall"],
                "test_human_f1": metrics["human_f1"],
                "holdout_accel_bot_recall": by_family.get("accel"),
                "holdout_jitter_bot_recall": by_family.get("jitter"),
                "holdout_straight_bot_recall": by_family.get("straight"),
                "worst_holdout_bot_recall": min(by_family.values()),
                "average_holdout_bot_recall": sum(by_family.values()) / len(by_family),
                "maximum_holdout_human_frr": max(item["human_frr"] for item in family),
                "robust_candidate": bool(
                    robust
                    and summary["selection"]["robust_candidate"]["selected_model"] == model_name
                ),
                "deployment_eligible": bool(
                    robust
                    and summary["selection"]["robust_candidate"]["selected_model"] == model_name
                    and summary["selection"]["robust_candidate"]["deployment_eligible"]
                ),
            }
        )
    return rows


def pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def write_comparison(
    baseline_label: str,
    baseline: dict[str, Any],
    current_label: str,
    current: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = checkpoint_rows(baseline_label, baseline) + checkpoint_rows(current_label, current)
    csv_path = output_dir / "checkpoint_model_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def selected(summary: dict[str, Any]) -> dict[str, Any]:
        robust = summary["selection"]["robust_candidate"]
        name = robust["selected_model"]
        ranked = next(item for item in robust["ranked_candidates"] if item["model_name"] == name)
        return {"name": name, "eligible": robust["deployment_eligible"], **ranked}

    old_selected = selected(baseline)
    new_selected = selected(current)
    human_delta = current["readiness"]["human_samples"] - baseline["readiness"]["human_samples"]
    participant_delta = (
        current["readiness"]["human_participants"]
        - baseline["readiness"]["human_participants"]
    )
    worst_delta = (
        new_selected["minimum_family_holdout_bot_recall"]
        - old_selected["minimum_family_holdout_bot_recall"]
    )
    gate_gap = 0.80 - new_selected["minimum_family_holdout_bot_recall"]
    current_selected_metrics = current["test"][new_selected["name"]]
    top_features = sorted(
        current_selected_metrics["feature_importance"].items(), key=lambda item: -item[1]
    )[:3]
    importance_total = sum(current_selected_metrics["feature_importance"].values()) or 1.0
    top_feature_share = sum(value for _, value in top_features) / importance_total
    top_feature_names = ", ".join(name for name, _ in top_features)
    lines = [
        "# Catchap Human 데이터 체크포인트 비교",
        "",
        "## 핵심 변화",
        "",
        f"- Human 학습 데이터: {baseline['readiness']['human_samples']:,} -> {current['readiness']['human_samples']:,} ({human_delta:+,})",
        f"- 연결 참여자: {baseline['readiness']['human_participants']}명 -> {current['readiness']['human_participants']}명 ({participant_delta:+}명)",
        f"- 강건 후보의 미지 Bot 최저 Recall: {pct(old_selected['minimum_family_holdout_bot_recall'])} -> {pct(new_selected['minimum_family_holdout_bot_recall'])} ({100 * worst_delta:+.1f}%p)",
        f"- 현재 배포 기준 80%까지 {100 * max(0.0, gate_gap):.1f}%p 부족",
        "",
        "## 체크포인트 요약",
        "",
        "| 체크포인트 | Human | 연결 참여자 | Bot | Bot 종류 | 강건 후보 | 미지 Bot 최저 Recall | 배포 |",
        "|---|---:|---:|---:|---:|---|---:|---|",
        (
            f"| {baseline_label} | {baseline['readiness']['human_samples']:,} | "
            f"{baseline['readiness']['human_participants']} | {baseline['readiness']['bot_samples']:,} | "
            f"{baseline['readiness']['bot_families']} | {MODEL_LABELS[old_selected['name']]} | "
            f"{pct(old_selected['minimum_family_holdout_bot_recall'])} | "
            f"{'통과' if old_selected['eligible'] else '보류'} |"
        ),
        (
            f"| {current_label} | {current['readiness']['human_samples']:,} | "
            f"{current['readiness']['human_participants']} | {current['readiness']['bot_samples']:,} | "
            f"{current['readiness']['bot_families']} | {MODEL_LABELS[new_selected['name']]} | "
            f"{pct(new_selected['minimum_family_holdout_bot_recall'])} | "
            f"{'통과' if new_selected['eligible'] else '보류'} |"
        ),
        "",
        "## 모델별 비교",
        "",
        "| 체크포인트 | 모델 | 일반 정확도 | Human FRR | 일반 Bot Recall | Accel 제외 | Jitter 제외 | Straight 제외 | 최저 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {row['model']} | {pct(row['test_accuracy'])} | "
            f"{pct(row['test_human_frr'])} | {pct(row['test_bot_recall'])} | "
            f"{pct(row['holdout_accel_bot_recall'])} | {pct(row['holdout_jitter_bot_recall'])} | "
            f"{pct(row['holdout_straight_bot_recall'])} | {pct(row['worst_holdout_bot_recall'])} |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- 일반 Test는 이미 학습에 등장한 Bot 종류를 섞어 나눈 결과라 배포 판단에 충분하지 않다.",
            "- Bot 한 종류 전체를 학습에서 제외한 뒤 검사한 최저 Recall을 핵심 강건성 지표로 본다.",
            "- Human FRR 3% 이하를 유지하면서 미지 Bot 최저 Recall 80% 이상이어야 배포를 검토한다.",
            "- 행 수는 두 배가 됐지만 연결 참여자는 4명만 늘어 사람 다양성은 여전히 부족하다.",
            f"- 현재 후보의 상위 3개 특징({top_feature_names})이 전체 중요도의 {100 * top_feature_share:.1f}%를 차지해 타이밍 의존 위험이 크다.",
            "- 후보 선정은 production 배포를 의미하지 않는다.",
            "",
        ]
    )
    (output_dir / "CHECKPOINT_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    baseline_rows = rows[: len(rows) // 2]
    current_rows = rows[len(rows) // 2 :]
    models = [row["model"] for row in baseline_rows]
    x = np.arange(len(models))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    old_bars = axes[0].bar(
        x - width / 2,
        [100 * row["worst_holdout_bot_recall"] for row in baseline_rows],
        width,
        label=baseline_label,
        color="#607D8B",
    )
    new_bars = axes[0].bar(
        x + width / 2,
        [100 * row["worst_holdout_bot_recall"] for row in current_rows],
        width,
        label=current_label,
        color="#1976D2",
    )
    axes[0].axhline(80, color="#C62828", linestyle="--", linewidth=1.5, label="Deploy gate 80%")
    axes[0].set_title("Worst unseen-family Bot Recall")
    axes[0].set_xticks(x, models)
    axes[0].set_ylim(0, 105)
    axes[0].set_ylabel("Recall (%)")
    axes[0].legend(fontsize=8)
    axes[0].bar_label(old_bars, fmt="%.1f", padding=2, fontsize=8)
    axes[0].bar_label(new_bars, fmt="%.1f", padding=2, fontsize=8)

    family_names = ["Accel", "Jitter", "Straight"]
    old_model_row = next(row for row in baseline_rows if row["robust_candidate"])
    new_model_row = next(row for row in current_rows if row["robust_candidate"])
    old_family = [
        100 * old_model_row[f"holdout_{family.lower()}_bot_recall"] for family in family_names
    ]
    new_family = [
        100 * new_model_row[f"holdout_{family.lower()}_bot_recall"] for family in family_names
    ]
    old_candidate_bars = axes[1].bar(
        x - width / 2,
        old_family,
        width,
        label=f"{baseline_label} {old_model_row['model']}",
        color="#607D8B",
    )
    new_candidate_bars = axes[1].bar(
        x + width / 2,
        new_family,
        width,
        label=f"{current_label} {new_model_row['model']}",
        color="#1976D2",
    )
    axes[1].axhline(80, color="#C62828", linestyle="--", linewidth=1.5)
    axes[1].set_title("Robust candidate by held-out family")
    axes[1].set_xticks(x, family_names)
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Bot Recall (%)")
    axes[1].legend(fontsize=8)
    axes[1].bar_label(old_candidate_bars, fmt="%.1f", padding=2, fontsize=8)
    axes[1].bar_label(new_candidate_bars, fmt="%.1f", padding=2, fontsize=8)

    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Catchap model checkpoint comparison", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "checkpoint_comparison.png", dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--current-label", required=True)
    parser.add_argument("--current-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    write_comparison(
        args.baseline_label,
        load_summary(args.baseline_summary),
        args.current_label,
        load_summary(args.current_summary),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
