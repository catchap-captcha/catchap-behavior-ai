"""Protect sealed external evaluation files from accidental model fitting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


SEALED_HOLDOUTS = {
    "2d1854d0c21f1927025bfec91470d6b3a93d572e2ee0ecc4f348c6a57662ae92": (
        "replay_warp_external_1000_20260721"
    ),
    "572b129e07b7d2d2a04dfb74fac282af66367b7a90c764397f247fdeafaa1857": (
        "adversarial_replay_external_1000_20260721"
    ),
    "f08c6b7e3f66e04716c80821ce5173650d4f695c5d22e6e36645052cef771f41": (
        "ml_pca_gmm_external_holdout_1000_20260721"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed_holdout_reason(path: Path) -> str | None:
    digest = sha256(path)
    if digest in SEALED_HOLDOUTS:
        return f"sealed external holdout: {SEALED_HOLDOUTS[digest]}"

    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("training_usage") == "external_holdout_only":
        return "manifest marks this data as external_holdout_only"
    if manifest.get("source_role") == "external_holdout":
        return "manifest marks this data as an external_holdout source"
    return None


def assert_not_sealed_training_inputs(paths: Iterable[Path]) -> None:
    for path in paths:
        reason = sealed_holdout_reason(path)
        if reason:
            raise ValueError(f"refusing sealed holdout as training input ({path}): {reason}")


__all__ = [
    "SEALED_HOLDOUTS",
    "assert_not_sealed_training_inputs",
    "sealed_holdout_reason",
]
