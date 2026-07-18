"""One-command training pipeline.

    python -m training.run_training_pipeline [options]

Order:
  1. verify MySQL connection
  2. check data readiness (gate)
  3. if not ready -> write report and STOP (no training, no model overwrite)
  4. load valid data, build dataset
  5. grouped split (train/val/test)
  6. train RandomForest / ExtraTrees / XGBoost / LightGBM
  7. choose threshold on validation
  8. evaluate once on test
  9. select best model
  10. save candidate bundles + reports
  11. promote to production ONLY if a model satisfies the selection criteria

Any error during training leaves the existing production model untouched, because
promotion is the very last step and only touches models/production/.

Options: --check-only, --dataset-version, --seed, --min-human-samples,
--min-bot-samples, --min-human-participants, --min-bot-families, --no-promote.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

from app.config import get_settings
from app.services.feature_extractor import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from training.build_dataset import build_dataset
from training.check_data_readiness import (
    REPORT_PATH,
    Thresholds,
    compute_readiness,
    write_report,
)
from training.evaluate_models import evaluate, select_threshold
from training.select_best_model import select_best
from training.split_dataset import split_dataset
from training.train_models import train_all

REPORTS_DIR = "reports"
CANDIDATE_DIR = "models/candidate"
PRODUCTION_DIR = "models/production"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _library_versions() -> dict[str, str]:
    import lightgbm
    import numpy
    import pandas
    import sklearn
    import xgboost

    return {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }


def _fetch_rows() -> list[dict[str, Any]] | None:
    from app.database.connection import check_connection, get_sessionmaker
    from app.database.repositories import TrainingDatasetRepository

    if not check_connection():
        print("MySQL 연결 실패: 학습을 중단합니다.", file=sys.stderr)
        return None
    session = get_sessionmaker()()
    try:
        repo = TrainingDatasetRepository(session)
        if not repo.view_exists():
            print("ai_training_dataset 뷰가 없습니다. DB 담당자에게 요청하세요.", file=sys.stderr)
            return None
        return repo.fetch_all()
    finally:
        session.close()


def run_pipeline(args: argparse.Namespace) -> int:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    thr = _resolve_thresholds(args)

    # 1-2) connect + readiness
    rows = _fetch_rows()
    if rows is None:
        return 3
    report = compute_readiness(rows, thr)
    write_report(report)

    if args.check_only:
        _print_readiness(report)
        return 0 if report.ready else 2

    # 3) gate
    if not report.ready:
        print("데이터가 준비되지 않아 학습을 건너뜁니다 (data_not_ready). 기존 모델은 그대로 유지됩니다.")
        _print_readiness(report)
        return 2

    # 4-5) dataset + grouped split
    ds = build_dataset(rows)
    split = split_dataset(ds, seed=args.seed)
    _save_split_manifest(split.manifest, args.dataset_version)

    # 6) train candidate models
    models = train_all(split, seed=args.seed)

    # 7-8) threshold on validation, evaluate once on test
    evaluations = []
    thresholds: dict[str, float] = {}
    val_metrics: dict[str, Any] = {}
    for name, model in models.items():
        t = select_threshold(model, split.X_val, split.y_val)
        thresholds[name] = t
        val_metrics[name] = evaluate(model, name, split.X_val, split.y_val, t, "validation")
        evaluations.append(evaluate(model, name, split.X_test, split.y_test, t, "test"))

    # 9) select
    selection = select_best(evaluations)

    # 10) persist candidates + reports
    dataset_version = args.dataset_version or f"auto_{len(ds)}rows"
    _write_reports(evaluations)
    _save_candidates(models, evaluations, thresholds, val_metrics, dataset_version)
    _write_summary(report, evaluations, selection, dataset_version)

    # 11) promote only if a model was selected
    if selection.selected is None:
        print("경고: 선택 기준을 만족하는 모델이 없습니다. production 모델을 교체하지 않습니다.")
        print(f"  사유: {selection.warning}")
        return 4
    if args.no_promote:
        print(f"선택된 모델: {selection.selected.model_name} (--no-promote: production 미교체)")
        return 0

    _promote(models, selection, thresholds, val_metrics, dataset_version)
    print(f"production 모델 교체 완료: {selection.selected.model_name}")
    return 0


# --------------------------------------------------------------------------- #
def _resolve_thresholds(args: argparse.Namespace) -> Thresholds:
    thr = Thresholds.from_settings()
    for attr in (
        "min_human_samples",
        "min_bot_samples",
        "min_human_participants",
        "min_bot_families",
    ):
        val = getattr(args, attr, None)
        if val is not None:
            setattr(thr, attr, val)
    return thr


def _print_readiness(report) -> None:
    print(f"ready={report.ready} human={report.human_samples}/{report.required_human_samples} "
          f"bot={report.bot_samples}/{report.required_bot_samples}")
    for item in report.missing:
        print(f"  - {item}")


def _bundle(model, name, threshold, val_metric, dataset_version) -> dict[str, Any]:
    from dataclasses import asdict

    return {
        "model": model,
        "model_name": name,
        "model_version": f"{name}_v1",
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "trained_at": _utcnow_iso(),
        "threshold": float(threshold),
        "validation_metrics": asdict(val_metric),
        "library_versions": _library_versions(),
    }


def _save_candidates(models, evaluations, thresholds, val_metrics, dataset_version) -> None:
    import joblib

    os.makedirs(CANDIDATE_DIR, exist_ok=True)
    for name, model in models.items():
        bundle = _bundle(model, name, thresholds[name], val_metrics[name], dataset_version)
        joblib.dump(bundle, os.path.join(CANDIDATE_DIR, f"{name}.joblib"))


def _promote(models, selection, thresholds, val_metrics, dataset_version) -> None:
    import joblib

    os.makedirs(PRODUCTION_DIR, exist_ok=True)
    name = selection.selected.model_name
    bundle = _bundle(models[name], name, thresholds[name], val_metrics[name], dataset_version)
    # single production bundle (model_service loads the newest *.joblib)
    for old in os.listdir(PRODUCTION_DIR):
        if old.endswith(".joblib"):
            os.remove(os.path.join(PRODUCTION_DIR, old))
    joblib.dump(bundle, os.path.join(PRODUCTION_DIR, f"production_{name}.joblib"))


def _save_split_manifest(manifest: dict, dataset_version: str | None) -> None:
    os.makedirs("data/metadata", exist_ok=True)
    name = f"split_manifest_{dataset_version or 'auto'}.json"
    with open(os.path.join("data/metadata", name), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def _write_reports(evaluations) -> None:
    import csv

    from dataclasses import asdict

    # comparison CSV
    fields = [
        "model_name", "metrics_on", "threshold", "accuracy", "human_precision",
        "human_recall", "human_f1", "bot_recall", "human_frr", "roc_auc",
        "pr_auc", "avg_inference_ms",
    ]
    with open(os.path.join(REPORTS_DIR, "model_comparison.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for e in evaluations:
            d = asdict(e)
            w.writerow({k: d.get(k) for k in fields})

    for e in evaluations:
        # feature importance CSV
        with open(os.path.join(REPORTS_DIR, f"feature_importance_{e.model_name}.csv"),
                  "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["feature", "importance"])
            for feat, imp in sorted(e.feature_importance.items(), key=lambda kv: -kv[1]):
                w.writerow([feat, imp])
        _plot_confusion(e)


def _plot_confusion(e) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cm = e.confusion_matrix
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["bot", "human"]); ax.set_yticklabels(["bot", "human"])
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{e.model_name} (test)")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(REPORTS_DIR, f"confusion_matrix_{e.model_name}.png"))
        plt.close(fig)
    except Exception:
        pass  # plotting is non-essential


def _write_summary(readiness, evaluations, selection, dataset_version) -> None:
    from dataclasses import asdict

    summary = {
        "generated_at": _utcnow_iso(),
        "dataset_version": dataset_version,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "readiness": asdict(readiness),
        "evaluations": [asdict(e) for e in evaluations],
        "selection": {
            "selected_model": selection.selected.model_name if selection.selected else None,
            "reason": selection.reason,
            "warning": selection.warning,
        },
        "note": "테스트 fixture로 생성된 결과는 최종 성능이 아닙니다.",
    }
    with open(os.path.join(REPORTS_DIR, "training_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the Human/Bot training pipeline.")
    p.add_argument("--check-only", action="store_true", help="readiness report only, no training")
    p.add_argument("--dataset-version", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-human-samples", type=int, default=None)
    p.add_argument("--min-bot-samples", type=int, default=None)
    p.add_argument("--min-human-participants", type=int, default=None)
    p.add_argument("--min-bot-families", type=int, default=None)
    p.add_argument("--no-promote", action="store_true", help="train + evaluate but do not replace production")
    return p


def main(argv: list[str] | None = None) -> int:
    return run_pipeline(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
