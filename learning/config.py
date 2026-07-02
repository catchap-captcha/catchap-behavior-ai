"""Tunable constants for the rule-based weak-problem recommendation.

Every number here is a STARTING default agreed during design, not a tuned
value. Once real data accumulates these should be revisited using measured
"recommendation effectiveness" (did the recommended practice actually raise
mastery?) and observed retry patterns.
"""

from __future__ import annotations

# --- mastery smoothing (additive / Laplace) ---
# observed_accuracy = (correct + SMOOTHING_CORRECT) / (valid + SMOOTHING_TOTAL)
# prevents "1 correct out of 1" from reading as 100% mastery.
SMOOTHING_CORRECT = 2
SMOOTHING_TOTAL = 4

# a concept with fewer valid attempts than this is "diagnosis needed",
# never ranked as a confirmed weak concept.
MIN_VALID_ATTEMPTS_FOR_CONFIDENCE = 3

# --- operation-error / retry handling ---
RETRY_MAX = 2               # attempts beyond this for one presentation are ignored
RETRY_WINDOW_S = 5.0        # a same-answer success within this window confirms
                            # the prior attempt was an operation error

# --- weakness score weights (must sum to 1.0) ---
WEIGHT_MASTERY = 0.45       # (1 - mastery): how much they can't do it (primary)
WEIGHT_RECENT = 0.30        # recent wrong ratio: current wobble
WEIGHT_REVIEW = 0.15        # time since last study: forgetting / review need
WEIGHT_HARD_FAIL = 0.10     # failure on above-level questions
assert abs(WEIGHT_MASTERY + WEIGHT_RECENT + WEIGHT_REVIEW + WEIGHT_HARD_FAIL - 1.0) < 1e-9

RECENT_WINDOW = 5           # "recent" = last N counted outcomes
REVIEW_SATURATION_DAYS = 30.0  # review_urgency hits 1.0 at this many days
HARD_DIFFICULTY = 0.7       # difficulty >= this counts as "hard" for hard_fail

# --- difficulty bands (difficulty is a float in [0, 1]) ---
EASY_MAX = 0.4              # [0.0, 0.4)   -> easy
MEDIUM_MAX = 0.7            # [0.4, 0.7)   -> medium; [0.7, 1.0] -> hard

# --- recommendation ---
DEFAULT_TOP_K_WEAK = 3      # weakest concepts to focus on
DEFAULT_N_RECOMMEND = 4     # questions per training set (design said 3~5)
MASTERED_THRESHOLD = 0.85   # at/above this a concept is treated as mastered
