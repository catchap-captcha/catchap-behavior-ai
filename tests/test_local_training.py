"""Local JSONL training preparation and split tests."""

from __future__ import annotations

from training.build_dataset import build_dataset
from training.evaluate_models import Evaluation
from training.run_local_training import (
    build_bot_feature_rows,
    build_local_split,
    select_robust_candidate,
)
from app.services.feature_profiles import get_feature_profile
from tests.conftest import human_like_events, make_row


def _bot_payload(index: int, family: str) -> dict:
    return {
        "schema_version": "1.0",
        "attempt_id": f"bot_{family}_{index}",
        "challenge_id": "challenge",
        "session_id": "session",
        "events": human_like_events(),
        "interaction": {},
        "collection": {
            "label": "bot",
            "label_source": "rule_bot",
            "bot_family": family,
            "generator_version": "rule_v1",
        },
    }


def test_bot_payloads_become_complete_feature_rows():
    rows = build_bot_feature_rows([_bot_payload(1, "straight")], groups_per_family=4)
    assert len(rows) == 1
    assert rows[0]["label"] == "bot"
    assert rows[0]["generator_version"].startswith("rule_v1_batch_")
    assert rows[0]["feature_schema_version"] == "1.0"
    assert rows[0]["event_count"] > 0


def test_external_holdout_bot_is_scoreable_but_not_trainable():
    payload = _bot_payload(1, "external")
    payload["collection"]["training_usage"] = "external_holdout_only"

    try:
        build_bot_feature_rows([payload])
    except ValueError as error:
        assert "external-holdout" in str(error)
    else:
        raise AssertionError("external holdout must not enter model fitting")

    profile = get_feature_profile("2.0", trajectory_only=True)
    rows = build_bot_feature_rows(
        [payload], profile=profile, allow_external_holdout=True
    )
    assert rows[0]["feature_schema_version"] == "2.0"
    assert len(profile.names) == 39


def test_local_split_is_class_complete_and_anonymous_human_is_train_only():
    rows = []
    for participant in range(9):
        rows.append(make_row("human", participant=f"p_{participant}", attempt_id=f"h_{participant}"))
    rows.append(make_row("human", participant=None, attempt_id="h_anonymous"))

    payloads = []
    for family in ("straight", "accel", "jitter"):
        payloads.extend(_bot_payload(index, family) for index in range(30))
    rows.extend(build_bot_feature_rows(payloads, groups_per_family=6))

    ds = build_dataset(rows)
    split = build_local_split(ds, rows, seed=7)

    assert split.manifest["attempt_to_split"]["h_anonymous"] == "train"
    for name in ("train", "val", "test"):
        assert split.manifest["class_counts"][name]["human"] > 0
        assert split.manifest["class_counts"][name]["bot"] > 0
    assert sum(split.manifest["counts"].values()) == len(rows)


def _evaluation(name: str, frr: float = 0.0) -> Evaluation:
    return Evaluation(
        model_name=name,
        threshold=0.5,
        accuracy=1.0,
        human_precision=1.0,
        human_recall=1.0 - frr,
        human_f1=1.0,
        bot_recall=1.0,
        human_frr=frr,
        roc_auc=1.0,
        pr_auc=1.0,
        confusion_matrix=[[1, 0], [0, 1]],
        avg_inference_ms=0.1,
        feature_importance={},
        metrics_on="test",
    )


def test_robust_selection_uses_worst_family_recall_and_deployment_gate():
    tests = {"fast_brittle": _evaluation("fast_brittle"), "robust": _evaluation("robust")}
    holdout = {}
    for name, values in {
        "fast_brittle": (1.0, 1.0, 0.3),
        "robust": (0.95, 0.92, 0.97),
    }.items():
        holdout[name] = [
            {
                "human_frr": 0.0,
                "bot_recall": value,
                "held_out_bot_family": "replay_warp" if index == 2 else f"family_{index}",
            }
            for index, value in enumerate(values)
        ]
    external = {
        name: [
            {
                "bot_asr": 0.02,
                "bot_recall": 0.98,
                "evaluation_role": "fresh_participant_external_holdout",
            }
        ]
        for name in ("fast_brittle", "robust")
    }
    selected = select_robust_candidate(
        tests, holdout, external, human_participants=120
    )
    assert selected["selected_model"] == "robust"
    assert selected["observation_only_eligible"] is True
    assert selected["shadow_mode_eligible"] is True
    assert selected["deployment_eligible"] is True


def test_shadow_mode_requires_fresh_participant_external_holdout():
    tests = {"candidate": _evaluation("candidate")}
    holdout = {
        "candidate": [
            {
                "human_frr": 0.0,
                "bot_recall": 0.98,
                "held_out_bot_family": "replay_warp",
            }
        ]
    }
    external = {"candidate": [{"bot_asr": 0.02, "bot_recall": 0.98}]}

    selected = select_robust_candidate(
        tests, holdout, external, human_participants=100
    )

    assert selected["experiment_eligible"] is True
    assert selected["observation_only_eligible"] is True
    assert selected["shadow_mode_eligible"] is False
    assert selected["deployment_eligible"] is False


def test_participant_count_is_reported_but_not_a_deployment_gate():
    tests = {"candidate": _evaluation("candidate")}
    holdout = {
        "candidate": [
            {
                "human_frr": 0.0,
                "bot_recall": 0.98,
                "held_out_bot_family": "replay_warp",
            }
        ]
    }
    external = {
        "candidate": [
            {
                "bot_asr": 0.02,
                "bot_recall": 0.98,
                "evaluation_role": "fresh_participant_external_holdout",
            }
        ]
    }

    selected = select_robust_candidate(
        tests, holdout, external, human_participants=1
    )

    assert selected["shadow_mode_eligible"] is True
    assert selected["deployment_eligible"] is True
