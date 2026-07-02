"""Rule-based weak-problem recommendation (phase 1).

Public entry point: :func:`learning.service.diagnose`. See
``docs/LEARNING_RECOMMENDATION.md`` for the design and roadmap.
"""

from learning.models import (
    CountedOutcome,
    DiagnoseResult,
    Judgment,
    MasteryResult,
    Outcome,
    Question,
    RawAttempt,
    Recommendation,
    WeakConcept,
)
from learning.service import diagnose

__all__ = [
    "diagnose",
    "RawAttempt",
    "Question",
    "Judgment",
    "Outcome",
    "CountedOutcome",
    "MasteryResult",
    "WeakConcept",
    "Recommendation",
    "DiagnoseResult",
]
