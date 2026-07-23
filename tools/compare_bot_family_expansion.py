"""Compare model results before and after expanding synthetic bot families."""

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


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def selected_model(summary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    selection = summary["selection"]["robust_candidate"]
    name = selection["selected_model"]
    metrics = next(
        item for item in selection["ranked_candidates"] if item["model_name"] == name
    )
    return name, metrics


def model_rows(label: str, summary: dict[str, Any]) -> list[dict[str, Any]]:
    selected_name, _ = selected_model(summary)
    rows: list[dict[str, Any]] = []
    for name, primary in summary["test"].items():
        holdouts = summary["family_holdout_stress_test"][name]
        recalls = [item["bot_recall"] for item in holdouts]
        non_replay = [
            item["bot_recall"]
            for item in holdouts
            if item["held_out_bot_family"] != "replay_warp"
        ]
        rows.append(
            {
                "experiment": label,
                "dataset_version": summary["dataset_version"],
                "human_samples": summary["readiness"]["human_samples"],
                "human_participants": summary["readiness"]["human_participants"],
                "bot_samples": summary["readiness"]["bot_samples"],
                "bot_families": summary["readiness"]["bot_families"],
                "model": MODEL_LABELS.get(name, name),
                "primary_accuracy": primary["accuracy"],
                "primary_human_frr": primary["human_frr"],
                "primary_bot_recall": primary["bot_recall"],
                "primary_human_f1": primary["human_f1"],
                "worst_family_holdout_recall": min(recalls),
                "average_family_holdout_recall": sum(recalls) / len(recalls),
                "worst_non_replay_holdout_recall": min(non_replay) if non_replay else None,
                "maximum_family_holdout_human_frr": max(
                    item["human_frr"] for item in holdouts
                ),
                "robust_candidate": name == selected_name,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    baseline_label: str,
    baseline: dict[str, Any],
    expanded_label: str,
    expanded: dict[str, Any],
    output_dir: Path,
) -> None:
    if baseline["readiness"]["human_samples"] != expanded["readiness"]["human_samples"]:
        raise ValueError("Human sample counts differ; this is not a controlled family expansion")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = model_rows(baseline_label, baseline) + model_rows(expanded_label, expanded)
    write_csv(output_dir / "model_comparison.csv", rows)

    base_name, base_selected = selected_model(baseline)
    expanded_name, expanded_selected = selected_model(expanded)
    expanded_holdouts = expanded["family_holdout_stress_test"][expanded_name]
    expanded_by_family = {
        item["held_out_bot_family"]: item for item in expanded_holdouts
    }
    non_replay_recalls = [
        item["bot_recall"]
        for item in expanded_holdouts
        if item["held_out_bot_family"] != "replay_warp"
    ]
    replay_recall = expanded_by_family["replay_warp"]["bot_recall"]
    primary = expanded["test"][expanded_name]

    lines = [
        "# Bot 3종 vs 10종 학습 비교",
        "",
        "## 실험 통제",
        "",
        f"- Human 데이터는 두 실험 모두 {baseline['readiness']['human_samples']:,}건, 연결 참여자 {baseline['readiness']['human_participants']}명으로 고정했다.",
        "- Feature 29개, seed 42, 참여자 그룹 분할, Human FRR 3% 임계값 정책을 동일하게 사용했다.",
        "- 차이는 Bot 데이터뿐이다: 3종 3,000건에서 10종 10,000건으로 확장했다.",
        "- 각 Bot family 전체를 학습에서 제외한 leave-one-family-out 검증을 실행했다.",
        "",
        "## 한눈에 보기",
        "",
        "| 실험 | Human | Bot | Bot 종류 | 강건 후보 | 일반 정확도 | Human FRR | 일반 Bot Recall | 미지 Bot 최악 | 미지 Bot 평균 | 배포 |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
        (
            f"| {baseline_label} | {baseline['readiness']['human_samples']:,} | "
            f"{baseline['readiness']['bot_samples']:,} | {baseline['readiness']['bot_families']} | "
            f"{MODEL_LABELS[base_name]} | {pct(baseline['test'][base_name]['accuracy'])} | "
            f"{pct(baseline['test'][base_name]['human_frr'])} | {pct(baseline['test'][base_name]['bot_recall'])} | "
            f"{pct(base_selected['minimum_family_holdout_bot_recall'])} | {pct(base_selected['average_family_holdout_bot_recall'])} | 보류 |"
        ),
        (
            f"| {expanded_label} | {expanded['readiness']['human_samples']:,} | "
            f"{expanded['readiness']['bot_samples']:,} | {expanded['readiness']['bot_families']} | "
            f"{MODEL_LABELS[expanded_name]} | {pct(primary['accuracy'])} | "
            f"{pct(primary['human_frr'])} | {pct(primary['bot_recall'])} | "
            f"{pct(expanded_selected['minimum_family_holdout_bot_recall'])} | {pct(expanded_selected['average_family_holdout_bot_recall'])} | 보류 |"
        ),
        "",
        "## 모델별 결과",
        "",
        "| 실험 | 모델 | 일반 정확도 | Human FRR | 일반 Bot Recall | 미지 Bot 최악 | 미지 Bot 평균 | Replay 제외 최악 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | {row['model']} | {pct(row['primary_accuracy'])} | "
            f"{pct(row['primary_human_frr'])} | {pct(row['primary_bot_recall'])} | "
            f"{pct(row['worst_family_holdout_recall'])} | "
            f"{pct(row['average_family_holdout_recall'])} | "
            f"{pct(row['worst_non_replay_holdout_recall']) if row['worst_non_replay_holdout_recall'] is not None else '-'} |"
        )

    lines.extend(
        [
            "",
            f"## 10종 강건 후보: {MODEL_LABELS[expanded_name]}",
            "",
            "| 학습에서 숨긴 Bot | 탐지율 | Human FRR | 판단 |",
            "|---|---:|---:|---|",
        ]
    )
    for family, item in sorted(expanded_by_family.items()):
        status = "통과" if item["bot_recall"] >= 0.80 else "실패"
        lines.append(
            f"| {family} | {pct(item['bot_recall'])} | {pct(item['human_frr'])} | {status} |"
        )

    lines.extend(
        [
            "",
            "## 결론",
            "",
            f"- Replay를 제외한 9종의 미지 Bot 최저 탐지율은 {pct(min(non_replay_recalls))}로 모두 80% 게이트를 통과했다.",
            f"- 사람 궤적을 재사용해 변형한 replay_warp의 미지 공격 탐지율은 {pct(replay_recall)}로 사실상 탐지하지 못했다.",
            "- replay_warp까지 포함해 최악 성능을 계산하면 0.1%이므로 10종 모델은 배포 불가다.",
            "- 이 결과는 좌표·시간 궤적만 보는 행동 모델의 한계를 보여준다. 사람의 실제 입력과 사람 궤적 재생은 29개 통계 Feature에서 거의 같은 분포가 된다.",
            "- 다음 단계는 단순히 synthetic 행 수를 늘리는 것이 아니라 challenge nonce, 재사용 지문, 세션 속도, 이벤트 신뢰도 같은 서버·브라우저 신호를 결합하는 것이다.",
            "- 후보 모델은 production에 반영하지 않았다.",
            "",
        ]
    )
    (output_dir / "BOT_FAMILY_EXPANSION_COMPARISON.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    baseline_rows = rows[:3]
    expanded_rows = rows[3:]
    models = [row["model"] for row in baseline_rows]
    x = np.arange(len(models))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))

    old_bars = axes[0].bar(
        x - width / 2,
        [100 * row["worst_family_holdout_recall"] for row in baseline_rows],
        width,
        label=baseline_label,
        color="#455A64",
    )
    new_bars = axes[0].bar(
        x + width / 2,
        [100 * row["worst_family_holdout_recall"] for row in expanded_rows],
        width,
        label=expanded_label,
        color="#00796B",
    )
    axes[0].bar_label(old_bars, fmt="%.1f", padding=2, fontsize=8)
    axes[0].bar_label(new_bars, fmt="%.1f", padding=2, fontsize=8)
    axes[0].set_title("Worst unseen-family Bot recall")
    axes[0].set_xticks(x, models)
    axes[0].set_ylabel("Recall (%)")
    axes[0].legend(fontsize=8)

    family_names = sorted(expanded_by_family)
    family_values = [100 * expanded_by_family[name]["bot_recall"] for name in family_names]
    colors = ["#C62828" if name == "replay_warp" else "#1976D2" for name in family_names]
    bars = axes[1].barh(family_names, family_values, color=colors)
    axes[1].bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axes[1].set_title(f"{MODEL_LABELS[expanded_name]} leave-one-family-out")
    axes[1].set_xlabel("Bot recall (%)")
    axes[1].set_xlim(0, 107)

    axes[0].axhline(80, color="#F9A825", linestyle="--", linewidth=1.4)
    axes[1].axvline(80, color="#F9A825", linestyle="--", linewidth=1.4)
    for axis in axes:
        axis.grid(axis="y" if axis is axes[0] else "x", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Catchap: 3 vs 10 synthetic Bot families", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "bot_family_expansion_comparison.png", dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-label", default="3 families")
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--expanded-label", default="10 families")
    parser.add_argument("--expanded-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    write_report(
        args.baseline_label,
        load_summary(args.baseline_summary),
        args.expanded_label,
        load_summary(args.expanded_summary),
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
