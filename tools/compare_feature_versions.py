"""Compare Feature v1/v2 security metrics on the same split and bot sets."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    external = summary.get("external_browser_bot_holdout", {})
    for model_name, metrics in summary["test"].items():
        holdout = summary["family_holdout_stress_test"][model_name]
        replay = next(
            (item for item in holdout if item.get("held_out_bot_family") == "replay_warp"),
            None,
        )
        family_asrs = [1.0 - item["bot_recall"] for item in holdout]
        external_asrs = [item["bot_asr"] for item in external.get(model_name, [])]
        rows.append(
            {
                "feature_schema_version": summary["feature_schema_version"],
                "model_name": model_name,
                "human_frr": metrics["human_frr"],
                "known_bot_asr": 1.0 - metrics["bot_recall"],
                "worst_family_holdout_asr": max(family_asrs),
                "replay_warp_asr": 1.0 - replay["bot_recall"] if replay else 1.0,
                "external_worst_asr": max(external_asrs) if external_asrs else None,
                "maximum_holdout_human_frr": max(item["human_frr"] for item in holdout),
            }
        )
    return rows


def compare(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    rows = [*_model_rows(v1), *_model_rows(v2)]
    criteria = v2["selection"]["robust_candidate"]["acceptance_criteria"]
    for row in rows:
        row["experiment_gate_passed"] = bool(
            row["human_frr"] <= criteria["experiment_human_frr_max"]
            and row["maximum_holdout_human_frr"] <= criteria["experiment_human_frr_max"]
            and row["known_bot_asr"] <= criteria["known_bot_asr_max"]
            and row["worst_family_holdout_asr"] <= criteria["unseen_bot_worst_asr_max"]
            and row["replay_warp_asr"] <= criteria["replay_warp_asr_max"]
            and row["external_worst_asr"] is not None
            and row["external_worst_asr"] <= criteria["unseen_bot_worst_asr_max"]
        )

    best_frr = min(rows, key=lambda row: row["human_frr"])
    best_known_asr = min(rows, key=lambda row: row["known_bot_asr"])
    passed = [row for row in rows if row["experiment_gate_passed"]]
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "same_split_required": v1["split"] == v2["split"],
        "acceptance_criteria": criteria,
        "rows": rows,
        "best_observed_human_frr": best_frr,
        "best_observed_known_bot_asr": best_known_asr,
        "passing_candidates": passed,
        "recommendation": (
            "promote_best_passing_candidate"
            if passed
            else "keep_feature_v2_experimental_and_do_not_promote_any_model"
        ),
    }


def _percent(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Feature v1 vs v2 Security Comparison",
        "",
        "Both versions use the same Human participants, Bot families, split seed, and external browser holdout.",
        "",
        "| Feature | Model | Human FRR | Known Bot ASR | Worst Family ASR | Replay ASR | Browser Worst ASR | Gate |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {feature} | {model} | {frr} | {known} | {worst} | {replay} | {browser} | {gate} |".format(
                feature=row["feature_schema_version"],
                model=row["model_name"],
                frr=_percent(row["human_frr"]),
                known=_percent(row["known_bot_asr"]),
                worst=_percent(row["worst_family_holdout_asr"]),
                replay=_percent(row["replay_warp_asr"]),
                browser=_percent(row["external_worst_asr"]),
                gate="PASS" if row["experiment_gate_passed"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            f"Recommendation: `{report['recommendation']}`.",
            "",
            "Feature v2 is not a replacement unless it passes every FRR/ASR gate, including unseen replay and browser automation.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    report = compare(_load(args.v1), _load(args.v2))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "feature_version_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "FEATURE_VERSION_COMPARISON.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
