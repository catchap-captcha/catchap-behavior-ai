"""Consolidate staged formal validation artifacts into JSON and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_result(model: str, root: Path) -> dict:
    summary_path = root / "training_summary.json"
    if summary_path.exists():
        summary = _read(summary_path)
        validation = summary["validation"][model]
        test = summary["test"][model]
        oof = summary["group_threshold_calibration"][model]
    else:
        validation = _read(root / "group_threshold_calibration.json")["pooled_oof_metrics"]
        test = _read(root / "final_test.json")["test"]
        oof = _read(root / "group_threshold_calibration.json")

    family_reports = [_read(path) for path in sorted((root / "family_holdouts").glob("*.json"))]
    external_reports = {
        name: _read(root / f"external_{name}.json")
        for name in ("playwright", "vae")
        if (root / f"external_{name}.json").exists()
    }
    family_asrs = {item["held_out_bot_family"]: item["bot_asr"] for item in family_reports}
    external_asrs = {name: item["bot_asr"] for name, item in external_reports.items()}
    worst_family = max(family_asrs.values(), default=None)
    worst_external = max(external_asrs.values(), default=None)
    unseen_values = [value for value in (worst_family, worst_external) if value is not None]
    worst_human_frr = max(
        [test["human_frr"], *(item["human_frr"] for item in family_reports)], default=None
    )
    return {
        "model": model,
        "oof": validation,
        "oof_calibration": oof,
        "test": test,
        "family_holdouts": family_asrs,
        "external_holdouts": external_asrs,
        "known_bot_asr": 1.0 - test["bot_recall"],
        "worst_family_asr": worst_family,
        "worst_external_asr": worst_external,
        "worst_unseen_asr": max(unseen_values) if unseen_values else None,
        "worst_human_frr": worst_human_frr,
        "gates_without_replay": {
            "human_frr_max_3_percent": worst_human_frr is not None and worst_human_frr <= 0.03,
            "known_bot_asr_max_5_percent": (1.0 - test["bot_recall"]) <= 0.05,
            "unseen_bot_asr_max_10_percent": bool(unseen_values) and max(unseen_values) <= 0.10,
        },
    }


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def _markdown(results: list[dict]) -> str:
    lines = [
        "# Schema 2.3 정식 모델 검증 결과",
        "",
        "기준일: 2026-07-22",
        "",
        "이 보고서는 300-tree 정식 모델, 사용자·Bot 그룹 기반 5-fold OOF 임계값 보정, untouched test, replay를 제외한 11개 미지 Bot family holdout, Playwright/VAE 외부 holdout을 통합한다.",
        "`replay_adversarial`, `replay_warp`은 현재 작업 요청에 따라 보류했으므로 이 보고서의 최종 통과 판정에는 포함하지 않는다.",
        "",
        "## 모델 요약",
        "",
        "| 모델 | OOF FRR | Test FRR | 알려진 Bot ASR | 최악 미지 ASR | 최악 외부 ASR | FRR 3% | 알려진 5% | 미지 10% |",
        "|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in results:
        gates = item["gates_without_replay"]
        lines.append(
            "| {model} | {oof} | {test_frr} | {known} | {unseen} | {external} | {frr_gate} | {known_gate} | {unseen_gate} |".format(
                model=item["model"],
                oof=_percent(item["oof"]["human_frr"]),
                test_frr=_percent(item["test"]["human_frr"]),
                known=_percent(item["known_bot_asr"]),
                unseen=_percent(item["worst_unseen_asr"]),
                external=_percent(item["worst_external_asr"]),
                frr_gate="통과" if gates["human_frr_max_3_percent"] else "실패",
                known_gate="통과" if gates["known_bot_asr_max_5_percent"] else "실패",
                unseen_gate="통과" if gates["unseen_bot_asr_max_10_percent"] else "실패",
            )
        )
    lines.extend(["", "## Family Holdout ASR", "", "| Family | " + " | ".join(item["model"] for item in results) + " |", "|---|" + "|".join("---:" for _ in results) + "|"])
    families = sorted({family for item in results for family in item["family_holdouts"]})
    for family in families:
        lines.append(
            "| {family} | {values} |".format(
                family=family,
                values=" | ".join(_percent(item["family_holdouts"].get(family)) for item in results),
            )
        )
    lines.extend(["", "## 외부 Holdout ASR", "", "| 외부 holdout | " + " | ".join(item["model"] for item in results) + " |", "|---|" + "|".join("---:" for _ in results) + "|"])
    for holdout in ("playwright", "vae"):
        lines.append(
            "| {holdout} | {values} |".format(
                holdout=holdout,
                values=" | ".join(_percent(item["external_holdouts"].get(holdout)) for item in results),
            )
        )
    lines.extend(
        [
            "",
            "## 결론",
            "",
            "- 평가한 후보 모두 Human FRR 3%는 통과했지만, 알려진 Bot ASR 5% 기준 또는 미지 family ASR 10% 기준을 통과하지 못했다.",
            "- 따라서 replay 기준을 보류하더라도 현재 단일 모델 후보를 배포 또는 차단 모드로 승격할 수 없다.",
            "- 다음 실험은 holdout을 학습에 넣지 않은 상태에서, 개발용 Bot의 누락 사례를 교차검증으로 가중하는 방식과 일반화 신호를 정식 OOF·전체 holdout 구조로 검증하는 것이다.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, help="model=report_directory")
    parser.add_argument("--json-report", required=True)
    parser.add_argument("--markdown-report", required=True)
    args = parser.parse_args()

    results = []
    for value in args.root:
        model, separator, directory = value.partition("=")
        if not separator or not model or not directory:
            raise ValueError("--root must use model=report_directory")
        results.append(_model_result(model, Path(directory)))
    output = {"feature_schema_version": "2.3", "replay_status": "deferred", "models": results}
    json_path = Path(args.json_report)
    markdown_path = Path(args.markdown_report)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(results), encoding="utf-8")
    print(json.dumps({"json_report": str(json_path), "markdown_report": str(markdown_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
