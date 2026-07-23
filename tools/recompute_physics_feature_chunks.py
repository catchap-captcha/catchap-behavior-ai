"""Recompute schema-2.3 physics features in bounded JSONL chunks.

This migration updates only the three schema-2.3 trajectory physics values.
It keeps all labels and provenance untouched, so it can refresh large local
snapshots without connecting to a database or fitting a model.
"""

from __future__ import annotations

import argparse
import json
from itertools import chain, islice
from pathlib import Path

from app.services.feature_extractor_v23 import _trajectory_physics_features


def _lines(paths: list[str]):
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            yield from handle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-input", required=True)
    parser.add_argument("--attempt-input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or args.limit <= 0:
        raise ValueError("--start must be non-negative and --limit must be positive")

    feature_lines = _lines([args.feature_input])
    attempt_lines = _lines(args.attempt_input)
    pairs = islice(zip(feature_lines, attempt_lines), args.start, args.start + args.limit)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.append else "w"
    count = 0
    with output_path.open(mode, encoding="utf-8") as handle:
        for feature_line, attempt_line in pairs:
            feature_row = json.loads(feature_line)
            attempt_row = json.loads(attempt_line)
            if feature_row.get("attempt_id") != attempt_row.get("attempt_id"):
                raise ValueError("feature and raw attempt rows are not aligned")
            if feature_row.get("feature_schema_version") != "2.3":
                raise ValueError("only schema 2.3 feature rows may be refreshed")
            feature_row.update(_trajectory_physics_features(attempt_row.get("events") or []))
            handle.write(json.dumps(feature_row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    print(json.dumps({"start": args.start, "rows_written": count, "output": str(output_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
