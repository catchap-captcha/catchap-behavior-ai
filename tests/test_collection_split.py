"""The split must be reproducible and beyond anyone's reach, including mine."""

from __future__ import annotations

from tools.collection_split import SALT, person_of, split

CODES = [
    "jy-mouse", "jy-trackpad", "ms-mouse", "ms-trackpad",
    "my-mouse", "my-trackpad", "sw-mouse", "sw-trackpad",
    "th-mouse", "th-trackpad",
]


def test_same_input_same_assignment():
    assert split(CODES, 3) == split(CODES, 3)


def test_order_of_the_input_does_not_matter():
    assert split(CODES, 3)["holdout_people"] == split(list(reversed(CODES)), 3)["holdout_people"]


def test_a_person_never_lands_on_both_sides():
    """`jy-mouse` and `jy-trackpad` are one human.

    Splitting by code instead of by person is exactly the leakage that made
    2026-08-03's cross-participant FRR look better than it was.
    """
    result = split(CODES, 3)
    by_person = {}
    for code, side in result["assignment"].items():
        by_person.setdefault(person_of(code), set()).add(side)
    assert all(len(sides) == 1 for sides in by_person.values())


def test_adding_a_person_does_not_reshuffle_the_others():
    """The rank of a person depends only on their name, so a late joiner cannot
    move anyone else across the line — which would otherwise be a way to steer
    the split by choosing who to invite last."""
    before = split(CODES, 3)
    after = split(CODES + ["zz-mouse"], 3)
    for person, digest in before["ranks"].items():
        assert after["ranks"][person] == digest


def test_holdout_size_is_respected():
    for n in (2, 3, 4):
        assert len(split(CODES, n)["holdout_people"]) == n


def test_salt_is_the_published_one():
    # Changing it changes every assignment; it must show up as a diff, not as a
    # quietly different result.
    assert SALT == "catchap-collection-split-20260804"
