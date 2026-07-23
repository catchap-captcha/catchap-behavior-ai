"""Turn training-view rows into a model-ready dataset.

Input rows come from the ``ai_training_dataset`` MySQL view (already filtered to
quality_status='valid', label in human/bot, label_source present). This module:

  * enforces feature-schema-version consistency,
  * maps Human -> 1 and Bot -> 0,
  * separates model-input features (the 29) from identifiers/metadata,
  * exposes the group key used for leakage-free splitting.

It is DB-agnostic: any iterable of dicts works, so tests feed fixtures directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from app.services.feature_extractor import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    MODEL_INPUT_EXCLUDE_COLUMNS,
)

LABEL_MAP = {"human": 1, "bot": 0}


@dataclass
class Dataset:
    """A prepared dataset ready for grouped splitting and training."""

    X: pd.DataFrame          # exactly FEATURE_NAMES columns, in order
    y: pd.Series             # 1 = human, 0 = bot
    groups: pd.Series        # leakage-free split key per row
    meta: pd.DataFrame       # identifiers + provenance (never fed to the model)

    def __len__(self) -> int:
        return len(self.y)


def to_dataframe(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    return df


def group_key(row: pd.Series) -> str:
    """Leakage-free group id for one row.

    Rules:
      * Human: keyed by participant, so all attempts from one person stay in a
        single split.
      * GAN bots derived from a human keep that human's participant id (when the
        generator tagged it), so a synthetic bot never leaks its origin human
        across splits.
      * Other bots: keyed by generator_version + bot_family (template group), so
        one generator/template group stays in a single split. Seeded/replay
        copies from the same generator therefore travel together.
    """
    label = row.get("label")
    participant = row.get("anonymous_participant_id")
    if label == "human" and participant:
        return f"human::{participant}"
    if label == "bot":
        # GAN bot carrying an origin participant follows that human
        if row.get("bot_family") == "gan" and participant:
            return f"human::{participant}"
        gen = row.get("generator_version") or "unknown_gen"
        fam = row.get("bot_family") or "unknown_family"
        return f"bot::{gen}::{fam}"
    # fallback: attempt-level group (its own island)
    return f"attempt::{row.get('attempt_id')}"


def build_dataset(
    rows: Iterable[dict[str, Any]],
    *,
    feature_names: Iterable[str] = FEATURE_NAMES,
    expected_schema_version: str = FEATURE_SCHEMA_VERSION,
) -> Dataset:
    """Build a :class:`Dataset` from training-view rows.

    Raises:
        ValueError: on empty input, unknown labels, missing features, or a
            feature-schema-version mismatch (rows produced under different
            feature definitions must never be mixed).
    """
    df = to_dataframe(rows)
    if df.empty:
        raise ValueError("no rows to build a dataset from")

    if "label" not in df or df["label"].isna().any():
        raise ValueError("every row must carry a label")
    bad = set(df["label"].unique()) - set(LABEL_MAP)
    if bad:
        raise ValueError(f"unexpected labels present: {sorted(bad)}")

    feature_names = list(feature_names)

    # feature schema consistency
    if "feature_schema_version" in df:
        versions = set(df["feature_schema_version"].dropna().unique())
        if versions and versions != {expected_schema_version}:
            raise ValueError(
                f"feature_schema_version mismatch: found {sorted(versions)}, "
                f"expected {expected_schema_version}"
            )

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"dataset is missing feature columns: {missing}")

    X = df[feature_names].astype(float).copy()
    if X.isnull().to_numpy().any():
        raise ValueError("feature matrix contains NULL/NaN values")

    y = df["label"].map(LABEL_MAP).astype(int)
    groups = df.apply(group_key, axis=1)

    meta_cols = [c for c in MODEL_INPUT_EXCLUDE_COLUMNS if c in df.columns]
    meta = df[meta_cols].copy()
    meta["_group"] = groups
    return Dataset(X=X, y=y, groups=groups, meta=meta)
