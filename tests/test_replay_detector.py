"""Tests for exact and warped pointer-path replay detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.services.replay_detector import (
    DynamicTimeWarpingComparator,
    HistoricalAttempt,
    NormalizedPathComparator,
    ProcrustesPathComparator,
    compute_replay_features,
    trace_fingerprint,
    trace_fingerprint_from_events,
)


def _curve(n: int = 31) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return np.column_stack((t, 0.3 * np.sin(np.pi * t)))


def _events(path: np.ndarray) -> list[dict[str, float | int]]:
    return [
        {"seq": i, "x_normalized": float(x), "y_normalized": float(y)}
        for i, (x, y) in enumerate(path)
    ]


def test_trace_fingerprint_ignores_translation_but_not_scale():
    path = _curve()
    translated = path + np.array([5.0, -2.0])
    scaled = path * 1.1

    assert trace_fingerprint(path) == trace_fingerprint(translated)
    assert trace_fingerprint(path) != trace_fingerprint(scaled)


def test_event_fingerprint_uses_normalized_coordinates():
    events = [
        {"seq": 0, "x": 10, "y": 20, "x_normalized": 0.1, "y_normalized": 0.2},
        {"seq": 1, "x": 20, "y": 30, "x_normalized": 0.2, "y_normalized": 0.3},
    ]
    expected = trace_fingerprint(np.asarray([[0.1, 0.2], [0.2, 0.3]]))

    assert trace_fingerprint_from_events(events) == expected


def test_dtw_handles_resampling_translation_and_uniform_scale():
    comparator = DynamicTimeWarpingComparator()
    path = _curve(31)
    warped = (_curve(67) * 1.2) + np.array([0.3, -0.4])
    different = np.column_stack((np.linspace(0.0, 1.0, 31), np.zeros(31)))

    same_score = comparator.similarity(path, warped)
    different_score = comparator.similarity(path, different)

    assert comparator.similarity(path, path) == pytest.approx(1.0)
    assert same_score > 0.97
    assert same_score > different_score


def test_compute_replay_features_separates_exact_and_warped_replays():
    path = _curve()
    translated = path + np.array([0.2, -0.1])
    scaled = path * 1.15
    history = [
        HistoricalAttempt(
            path=translated,
            duration_ms=900.0,
            endpoint=tuple(translated[-1]),
            created_at_epoch_s=970.0,
        ),
        HistoricalAttempt(
            path=scaled,
            duration_ms=750.0,
            endpoint=tuple(scaled[-1]),
            created_at_epoch_s=980.0,
        ),
    ]

    features = compute_replay_features(
        _events(path),
        duration_ms=900.0,
        now_epoch_s=1000.0,
        history=history,
    )

    assert features.exact_replay_detected is True
    assert features.path_similarity_score > 0.99
    assert features.repeated_duration_count == 1
    assert features.recent_attempt_count == 2
    assert features.attempts_per_minute == pytest.approx(2.0)


def test_short_or_invalid_paths_do_not_look_identical():
    dtw = DynamicTimeWarpingComparator()
    normalized = NormalizedPathComparator()
    point = np.array([[0.0, 0.0]])
    invalid = np.array([[0.0, np.nan], [1.0, 1.0]])

    assert dtw.similarity(point, point) == 0.0
    assert dtw.similarity(invalid, _curve()) == 0.0
    assert normalized.similarity(point, point) == 0.0
    assert trace_fingerprint(point) is None


def _rotate(path: np.ndarray, radians: float, scale: float = 1.0) -> np.ndarray:
    """What a replay attacker does to reuse a captured path on a new target."""
    rotation = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)],
    ])
    return (path - path[0]) @ rotation.T * scale + path[0]


def test_procrustes_sees_through_the_rotation_a_replay_must_apply():
    """The transform that defines the attack must not be the one that hides it.

    An attacker reusing a captured trajectory has to rotate it, because the
    object is somewhere else this time. DTW normalizes away translation and
    scale but not rotation, so it reads a rotated replay as an unrelated path —
    measured at ~0.61 on real data, far below any threshold worth setting.
    """
    procrustes = ProcrustesPathComparator()
    dtw = DynamicTimeWarpingComparator()
    original = _curve()

    for radians in (0.4, 1.2, 2.5, 4.0):
        replayed = _rotate(original, radians, scale=1.3)
        assert procrustes.similarity(original, replayed) > 0.99
        assert dtw.similarity(original, replayed) < 0.9

    # Invariance must not become blindness: a genuinely different shape stays
    # far away no matter how it is turned.
    other = np.column_stack((np.linspace(0.0, 1.0, 31), np.zeros(31)))
    assert procrustes.similarity(original, _rotate(other, 1.0)) < 0.9


def test_procrustes_is_symmetric_and_refuses_degenerate_paths():
    procrustes = ProcrustesPathComparator()
    a, b = _curve(), _rotate(_curve(), 0.8, scale=0.7)
    assert procrustes.similarity(a, b) == pytest.approx(procrustes.similarity(b, a))

    point = np.array([[0.0, 0.0]])
    stationary = np.tile([0.5, 0.5], (20, 1))
    invalid = np.array([[0.0, np.nan], [1.0, 1.0]])
    assert procrustes.similarity(point, point) == 0.0
    assert procrustes.similarity(stationary, _curve()) == 0.0
    assert procrustes.similarity(invalid, _curve()) == 0.0


def test_rotated_replay_scores_above_the_deployed_threshold():
    """The comparator and the threshold have to move together.

    `risk_dtw_similarity_threshold` was recalibrated for this comparator; if
    either is changed alone the signal goes quiet without failing anything.
    """
    from app.config import Settings

    threshold = Settings().risk_dtw_similarity_threshold
    original = _curve()
    history = [HistoricalAttempt(
        path=original, duration_ms=700.0,
        endpoint=(float(original[-1][0]), float(original[-1][1])),
        created_at_epoch_s=1000.0,
    )]
    replayed = _rotate(original, 1.1, scale=1.2)
    events = [{"seq": i, "event_type": "pointer_move", "t_ms": i * 50.0,
               "x": float(x), "y": float(y)} for i, (x, y) in enumerate(replayed)]

    features = compute_replay_features(
        events, duration_ms=700.0, now_epoch_s=1001.0, history=history)
    assert features.path_similarity_score >= threshold
    assert not features.exact_replay_detected  # rotation changes the hash


def test_joined_path_gates_on_length_and_drops_the_seam_duplicate():
    """The aim/drag join must not manufacture a zero-length step at the seam.

    `pointerdown` fires where the pointer already is, so the last aim sample and
    the first drag sample can coincide. Left in, that step has undefined
    direction and step length — the two quantities the fingerprint rests on.
    """
    from tools.aim_drag_path import MIN_POINTS_FOR_FINGERPRINT, join

    aim = [{"x": i / 40.0, "y": 0.1 * i} for i in range(20)]
    drag = [{"x": aim[-1]["x"], "y": aim[-1]["y"]}] + [
        {"x": 0.5 + i / 40.0, "y": 2.0 + 0.1 * i} for i in range(15)]

    joined = join(aim, drag)
    assert joined.aim_points == 20
    steps = np.linalg.norm(np.diff(joined.points, axis=0), axis=1)
    assert steps.min() > 0.0             # seam duplicate removed
    # drag carried 16 samples, one of them coinciding with the last aim sample.
    assert joined.drag_points == 15      # counted after the seam is dropped
    assert joined.total_points == 35

    assert joined.judgeable()
    # A drag on its own is exactly the case that cannot be judged: ~12 points,
    # where warped replays are caught 1.4% of the time.
    assert not join([], drag).judgeable()
    assert MIN_POINTS_FOR_FINGERPRINT == 31


def test_joined_path_survives_missing_halves():
    from tools.aim_drag_path import join

    drag = [{"x": i / 20.0, "y": 0.0} for i in range(12)]
    assert join([], drag).total_points == 12
    assert join(drag, []).total_points == 12
    assert join([], []).total_points == 0
    assert not join([], []).judgeable()
    # Rows where the widget recorded no coordinates must not crash the join.
    assert join([{"x": None, "y": None}], drag).total_points == 12


def test_density_veto_is_applied_when_the_bundle_carries_one():
    """A veto in the bundle must actually run, and its absence must change nothing.

    The threshold in a veto-equipped bundle is calibrated assuming the veto
    rejects its share of humans. Loading such a bundle into a scorer that ignores
    the veto does not merely lose the veto — it leaves an operating point that no
    longer means anything. Measured on 200 real production traces, the same
    bundle flagged 8.5% with the veto and 0.5% without, and the second number is
    the model waving bots through, not a false-reject improvement.
    """
    from app.services.model_service import ModelService

    class _AlwaysOutside:
        def score(self, X):
            return np.zeros(len(X))

    features = {"speed_std": 1.0}
    plain = {"threshold": 0.5}
    assert ModelService._apply_density_veto(plain, features, 0.9) == 0.9

    vetoed = {"threshold": 0.5, "density_veto": _AlwaysOutside(),
              "density_feature_names": ("speed_std",), "veto_below": 0.5}
    assert ModelService._apply_density_veto(vetoed, features, 0.9) == 0.0

    class _Inside:
        def score(self, X):
            return np.ones(len(X))

    inside = dict(vetoed, density_veto=_Inside())
    assert ModelService._apply_density_veto(inside, features, 0.9) == 0.9

    # A broken veto must not reject every human — failing open is the safe
    # direction when the alternative is rejecting everyone.
    class _Broken:
        def score(self, X):
            raise ValueError("boom")

    broken = dict(vetoed, density_veto=_Broken())
    assert ModelService._apply_density_veto(broken, features, 0.9) == 0.9


def test_shipped_bundle_loads_without_a_patched_main():
    """A bundle must unpickle in a process that never ran the training script.

    Every local check used to do

        sys.modules["__main__"].DensityVeto = DensityVeto

    before loading, which is exactly the condition production does not have. So
    a bundle that pickled the class as `__main__.DensityVeto` passed every check
    here and failed in the cluster with

        AttributeError: Can't get attribute 'DensityVeto' on <module 'main'>

    while `/health` still returned 200 and every prediction recorded as
    `unavailable`. The harness was patching away the failure it existed to catch.

    This test asserts the property directly: the class the bundle references must
    live at an importable module path, not in whatever `__main__` happens to be.
    """
    import subprocess
    from pathlib import Path

    # Only what the repo actually ships. `models/candidate/*` is gitignored with
    # per-directory exceptions, and the Dockerfile copies exactly those — local
    # experiment bundles are not deployed and must not fail the build.
    tracked = subprocess.run(
        ["git", "ls-files", "models/candidate/"],
        capture_output=True, text=True, check=False).stdout.split()
    shipped = [Path(p) for p in tracked if p.endswith("two_view_fusion.joblib")]
    assert shipped, "배포 대상 번들이 없다 — .gitignore 예외를 확인할 것"

    for path in shipped:
        blob = path.read_bytes()
        # joblib writes a pickle; module paths appear verbatim in the opcodes.
        assert b"__main__" not in blob, (
            f"{path} 가 __main__ 을 참조한다 — 학습 스크립트 밖에서는 풀리지 않는다. "
            "커스텀 클래스는 app/ 아래 import 가능한 모듈에 두고 다시 저장할 것."
        )


def test_density_veto_class_is_importable_from_the_app():
    """Where this class lives is part of the bundle contract, not a detail."""
    from app.services.density_veto import DensityVeto

    assert DensityVeto.__module__ == "app.services.density_veto"


def test_model_service_exposes_model_version_as_a_property():
    """`model_version` is a property; calling it raises TypeError.

    The startup logger added to catch a *silent* model failure called it with
    parentheses, so on a healthy boot the log line itself threw and the outer
    handler reported `[MODEL] ★모델 적재 중 예외 — TypeError: 'str' object is not
    callable`. The model was fine; the log lied about it, in the one direction
    that wastes an operator's time.

    `app/api/health.py` had it right all along, which is what makes this worth a
    test: two call sites, one convention, and nothing enforcing agreement.
    """
    import inspect
    from app.services.model_service import ModelService

    assert isinstance(inspect.getattr_static(ModelService, "model_version"), property)

    source = Path("app/main.py").read_text(encoding="utf-8")
    assert "model_service.model_version()" not in source, (
        "app/main.py 가 property 를 호출하고 있다 — 괄호를 빼야 한다"
    )
