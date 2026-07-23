"""Formal two-view fitting must reject feature files marked as lockboxes."""

import pytest

from tools.run_formal_two_view_fusion import _load_data


def test_formal_two_view_rejects_sealed_human_lockbox(tmp_path):
    human_lockbox = tmp_path / "human_lockbox.jsonl"
    human_lockbox.write_text('{"label":"human"}\n', encoding="utf-8")
    human_lockbox.with_suffix(".jsonl.manifest.json").write_text(
        '{"training_usage":"external_holdout_only"}\n', encoding="utf-8"
    )
    bot_features = tmp_path / "bot_features.jsonl"
    bot_features.write_text('{"label":"bot"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="sealed holdout"):
        _load_data(str(human_lockbox), str(bot_features))
