"""Final model selection from the three test-set evaluations.

Selection rules (in order):
  1. Only models with Human False Rejection Rate <= max_frr (default 3%) are
     candidates.
  2. Among candidates, pick the highest Bot recall.
  3. Tie -> highest Human F1.
  4. Still tied -> fastest average inference time.
  5. If NO model meets the FRR budget, select nothing (the caller must not
     replace the production model) and emit a warning report.
"""

from __future__ import annotations

from dataclasses import dataclass

from training.evaluate_models import DEFAULT_MAX_FRR, Evaluation


@dataclass
class Selection:
    selected: Evaluation | None
    reason: str
    warning: str | None = None


def select_best(
    evaluations: list[Evaluation], max_frr: float = DEFAULT_MAX_FRR
) -> Selection:
    """Apply the selection rules to test-set evaluations."""
    if not evaluations:
        return Selection(None, "no_models_evaluated", "평가된 모델이 없습니다.")

    candidates = [e for e in evaluations if e.human_frr <= max_frr]
    if not candidates:
        best_frr = min(evaluations, key=lambda e: e.human_frr)
        return Selection(
            None,
            "no_model_meets_frr_budget",
            (
                f"Human 오탐률 {max_frr:.0%} 이하를 만족하는 모델이 없습니다 "
                f"(최소 FRR={best_frr.human_frr:.4f}, 모델={best_frr.model_name}). "
                "production 모델을 교체하지 않습니다."
            ),
        )

    # rank: bot_recall desc, human_f1 desc, avg_inference_ms asc
    best = sorted(
        candidates,
        key=lambda e: (-e.bot_recall, -e.human_f1, e.avg_inference_ms),
    )[0]
    return Selection(best, "selected_by_bot_recall_then_f1_then_speed")
