from __future__ import annotations

import pytest

from training.holdout_registry import assert_not_sealed_training_inputs


def test_external_holdout_manifest_is_rejected_as_training_input(tmp_path):
    payload = tmp_path / "external.jsonl"
    payload.write_text('{"attempt_id":"x"}\n', encoding="utf-8")
    manifest = payload.with_suffix(payload.suffix + ".manifest.json")
    manifest.write_text(
        '{"training_usage":"external_holdout_only"}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="sealed holdout"):
        assert_not_sealed_training_inputs([payload])
