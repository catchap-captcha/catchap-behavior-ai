"""Record the sha256 of every usable holdout, so later edits are detectable.

What this can and cannot do
---------------------------
It cannot certify the past. Hashing today only says "this is the file as of
2026-08-08"; if a holdout was regenerated last week, that is already invisible
and stays invisible. What it buys is the future: from now on, a holdout that
changes stops matching, and `lockbox_audit --strict` fails instead of quietly
scoring a candidate against a file nobody meant to touch.

That failure mode is not hypothetical here. Three datasets in this directory
were labelled holdouts while sitting 100% inside the training set, one manifest
had a stray brace and did not parse at all, and three genuine holdouts were
missing their `training_usage` and had been silently reclassified as development
data. Every one of those was found by a checker, not by a person reading files.

Refuses to touch anything already marked consumed. A spent holdout's hash is a
record of what it was when it was spent, and rewriting it would erase that.

    .venv/bin/python tools/seal_hashes.py            # 기록
    .venv/bin/python tools/seal_hashes.py --verify   # 대조만
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_ROOT = Path("data")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_of(path: Path) -> int:
    with path.open() as f:
        return sum(1 for _ in f)


def main() -> int:
    verify_only = "--verify" in sys.argv
    recorded = changed = skipped = 0
    problems: list[str] = []

    for manifest_path in sorted(DATA_ROOT.rglob("*.manifest.json")):
        try:
            doc = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{manifest_path} — JSON 파손: {e}")
            continue

        if doc.get("training_usage") != "external_holdout_only":
            continue

        data_file = manifest_path.parent / manifest_path.name.replace(".manifest.json", "")
        if not data_file.exists():
            problems.append(f"{manifest_path} — 데이터 파일 없음: {data_file.name}")
            continue

        actual = sha256(data_file)
        seal = doc.get("content_seal")

        if seal:
            if seal.get("sha256") == actual:
                print(f"  일치   {data_file.name[:52]}")
            else:
                changed += 1
                problems.append(
                    f"{data_file.name} — 봉인 이후 파일이 바뀌었다\n"
                    f"      기록 {seal.get('sha256')}\n      실제 {actual}")
            continue

        if verify_only:
            print(f"  미기록 {data_file.name[:52]}")
            skipped += 1
            continue

        if doc.get("evaluation_consumed"):
            # A spent holdout's hash belongs to the moment it was spent. Writing
            # today's value over it would claim a verification that never happened.
            print(f"  소진   {data_file.name[:52]} — 기록하지 않음")
            skipped += 1
            continue

        doc["content_seal"] = {
            "sha256": actual,
            "rows": rows_of(data_file),
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": ("2026-08-08 시점의 파일을 봉인한다. 그 이전의 변경은 "
                     "확인할 수 없고, 이후의 변경만 잡힌다."),
        }
        manifest_path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n")
        recorded += 1
        print(f"  기록   {data_file.name[:52]}  {actual[:16]}")

    print(f"\n기록 {recorded}건 · 건너뜀 {skipped}건 · 불일치 {changed}건")
    if problems:
        print("\n⚠️  문제")
        for p in problems:
            print(f"  {p}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
