"""Reserve a future known-Bot lockbox without using prior test batches.

One deterministic previous-training generator batch is reserved per requested
Bot family.  Previous test batches are excluded from both the next development
set and the new lockbox, because their outcome has already been observed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reserve_lockbox(
    rows: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    *,
    seed: str,
    excluded_families: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if any(row.get("label") != "bot" for row in rows):
        raise ValueError("Bot lockbox source must contain Bot rows only")
    split_of = split_manifest["attempt_to_split"]
    scoped = [row for row in rows if row.get("bot_family") not in excluded_families]
    if not scoped:
        raise ValueError("no Bot rows remain after excluded families")

    by_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in scoped:
        by_family[row["bot_family"]][row["generator_version"]].append(row)

    selected: dict[str, str] = {}
    for family, groups in sorted(by_family.items()):
        eligible = [
            generator
            for generator, group_rows in groups.items()
            if all(split_of[row["attempt_id"]] == "train" for row in group_rows)
        ]
        if not eligible:
            raise ValueError(f"no previous-train generator group available for {family}")
        selected[family] = min(
            eligible,
            key=lambda generator: hashlib.sha256(f"{seed}:{family}:{generator}".encode()).digest(),
        )

    lockbox_rows = [
        row for row in scoped if row["generator_version"] == selected[row["bot_family"]]
    ]
    development_rows = [
        row
        for row in scoped
        if split_of[row["attempt_id"]] != "test"
        and row["generator_version"] != selected[row["bot_family"]]
    ]
    if not lockbox_rows or not development_rows:
        raise ValueError("lockbox reservation produced an empty partition")
    prior_test = [row for row in scoped if split_of[row["attempt_id"]] == "test"]
    family_counts = Counter(row["bot_family"] for row in lockbox_rows)
    manifest = {
        "kind": "prospective_internal_bot_lockbox",
        "training_usage": "external_holdout_only",
        "source_role": "revalidation_bot_lockbox",
        "limitations": [
            "re-reserved from an existing local development Bot snapshot; not an independently generated future attack set",
            "generator batches may have appeared in historic experiments before this reservation",
        ],
        "selection": {
            "excluded_families": sorted(excluded_families),
            "previous_test_policy": "excluded from next development and lockbox",
            "eligible_population": "previous train generator batches only",
            "hash_seed": seed,
            "selection_rule": "lowest SHA-256-ranked previous-train generator batch per family",
            "lockbox_generator_by_family": selected,
        },
        "counts": {
            "source_bot_rows": len(rows),
            "excluded_family_rows": len(rows) - len(scoped),
            "previous_test_bot_rows_excluded": len(prior_test),
            "lockbox_bot_rows": len(lockbox_rows),
            "lockbox_families": len(family_counts),
            "lockbox_rows_by_family": dict(sorted(family_counts.items())),
            "development_bot_rows": len(development_rows),
        },
    }
    return development_rows, lockbox_rows, manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.bot_features)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    rows = _load_jsonl(source)
    split_manifest = json.loads(Path(args.previous_split_manifest).read_text(encoding="utf-8"))
    development, lockbox, manifest = reserve_lockbox(
        rows,
        split_manifest,
        seed=args.seed,
        excluded_families=set(args.exclude_family),
    )
    output_dir.mkdir(parents=True)
    development_path = output_dir / "bot_development.jsonl"
    lockbox_path = output_dir / "bot_lockbox.jsonl"
    development_count = _write_jsonl(development_path, development)
    lockbox_count = _write_jsonl(lockbox_path, lockbox)
    manifest["source"] = {
        "bot_features": str(source),
        "bot_features_sha256": _sha256(source),
        "previous_split_manifest": str(Path(args.previous_split_manifest)),
        "previous_split_manifest_sha256": _sha256(Path(args.previous_split_manifest)),
    }
    manifest["files"] = {
        development_path.name: {"rows": development_count, "sha256": _sha256(development_path)},
        lockbox_path.name: {"rows": lockbox_count, "sha256": _sha256(lockbox_path)},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lockbox_path.with_suffix(lockbox_path.suffix + ".manifest.json").write_text(
        json.dumps(
            {
                "training_usage": "external_holdout_only",
                "source_role": "revalidation_bot_lockbox",
                "lockbox_manifest": str(output_dir / "manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bot-features", required=True)
    parser.add_argument("--previous-split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", default="catChap-bot-lockbox-v1")
    parser.add_argument("--exclude-family", action="append", default=[])
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
