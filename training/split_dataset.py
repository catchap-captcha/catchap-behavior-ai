"""Leakage-free grouped train/validation/test split (70/15/15).

Splitting is by GROUP, never by row: a whole participant (or bot generator /
GAN-origin group) lands entirely in one split. A simple row-level random split
is explicitly forbidden because attempts from the same person/generator are
correlated and would leak information across splits.

After splitting, an automatic leakage check asserts no group crosses splits.
A manifest (group -> split, attempt -> split) is returned for persistence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from training.build_dataset import Dataset

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


@dataclass
class SplitData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    manifest: dict


class LeakageError(RuntimeError):
    """Raised when the same group appears in more than one split."""


def split_dataset(ds: Dataset, seed: int = 42) -> SplitData:
    """Split into train/val/test by group with a fixed seed.

    Raises:
        ValueError: if there are too few groups to form three non-empty splits.
        LeakageError: if the automatic post-split check finds a shared group.
    """
    groups = ds.groups.to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 3:
        raise ValueError(
            f"need at least 3 distinct groups for a grouped split, found {n_groups}. "
            "Collect more distinct participants / bot generators."
        )

    idx = np.arange(len(ds.y))

    # 1) train vs (val+test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(VAL_FRAC + TEST_FRAC), random_state=seed)
    train_idx, temp_idx = next(gss1.split(idx, ds.y, groups))

    # 2) split temp into val vs test (test share of the temp block)
    temp_groups = groups[temp_idx]
    rel_test = TEST_FRAC / (VAL_FRAC + TEST_FRAC)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
    val_rel, test_rel = next(gss2.split(temp_idx, ds.y.iloc[temp_idx], temp_groups))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    split_of = {}
    for name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for i in indices:
            split_of[int(i)] = name

    _assert_no_leakage(groups, train_idx, val_idx, test_idx)

    attempt_ids = (
        ds.meta["attempt_id"].tolist()
        if "attempt_id" in ds.meta.columns
        else [str(i) for i in idx]
    )
    manifest = {
        "seed": seed,
        "fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "counts": {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "group_to_split": _group_to_split(groups, split_of),
        "attempt_to_split": {attempt_ids[i]: split_of[i] for i in idx},
    }

    return SplitData(
        X_train=ds.X.iloc[train_idx].reset_index(drop=True),
        y_train=ds.y.iloc[train_idx].reset_index(drop=True),
        X_val=ds.X.iloc[val_idx].reset_index(drop=True),
        y_val=ds.y.iloc[val_idx].reset_index(drop=True),
        X_test=ds.X.iloc[test_idx].reset_index(drop=True),
        y_test=ds.y.iloc[test_idx].reset_index(drop=True),
        manifest=manifest,
    )


def _group_to_split(groups: np.ndarray, split_of: dict[int, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for i, g in enumerate(groups):
        out[str(g)] = split_of[i]
    return out


def _assert_no_leakage(groups, train_idx, val_idx, test_idx) -> None:
    g_train = set(groups[train_idx])
    g_val = set(groups[val_idx])
    g_test = set(groups[test_idx])
    if g_train & g_val or g_train & g_test or g_val & g_test:
        raise LeakageError(
            "group leakage detected across splits: "
            f"train∩val={g_train & g_val}, train∩test={g_train & g_test}, "
            f"val∩test={g_val & g_test}"
        )
