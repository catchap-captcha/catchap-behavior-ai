"""Audit the sealed-holdout ledger: what is still sealed, what was spent, on what.

Why this exists
---------------
A sealed holdout is only worth something if we can say, without hedging, that
the candidate never saw it. Today that claim rests on a scatter of loose JSON
files that nothing validates. One of them (`revalidation_bot_lockbox_b13300`)
has a stray closing brace and does not parse — a scan that skips unreadable
files would have reported a *spent* lockbox as available, and we would have
scored a candidate on data it had already been tuned against without noticing.

So this fails loudly instead. Anything it cannot read, cannot hash, or cannot
explain is an error, not a gap in a table.

Three questions, answered from the files rather than from memory:

  1. Which holdouts are still sealed?
  2. Which were spent, and on which model?
  3. Does the data on disk still match what was sealed?  (sha256)

    .venv/bin/python tools/lockbox_audit.py            # report
    .venv/bin/python tools/lockbox_audit.py --strict   # nonzero exit on problems
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

DATA_ROOT = Path("data")


def read_json(path: Path) -> tuple[dict | None, str | None]:
    try:
        return json.loads(path.read_text()), None
    except json.JSONDecodeError as e:
        return None, f"JSON 파손: {e}"
    except OSError as e:
        return None, f"읽기 실패: {e}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


TRAINING_SETS = (
    "data/interim/bot_features_v23corr_20260722.jsonl",
    "data/interim/human_features_v23corr_20260722.jsonl",
)


def row_ids(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = r.get("attempt_id") or r.get("session_id")
            if key:
                out.add(key)
    return out


def main() -> int:
    strict = "--strict" in sys.argv
    verify_hashes = "--no-hash" not in sys.argv

    # `training_usage` is a label somebody wrote, and two datasets carry the wrong
    # one: extended_bots_10000 and adversarial_replay_development_2000 both say
    # `external_holdout_only` while sitting 100% inside the training set. Scoring a
    # candidate on either would produce a flattering number that means nothing. So
    # the label is a hint; the overlap check is the answer.
    trained_ids: set[str] = set()
    for p in TRAINING_SETS:
        path = Path(p)
        if path.exists():
            trained_ids |= row_ids(path)

    problems: list[str] = []
    sealed: list[tuple[str, int | None, bool]] = []
    development: list[str] = []
    unlabelled: list[str] = []
    spent: list[tuple[str, str]] = []

    for manifest_path in sorted(DATA_ROOT.rglob("*.manifest.json")):
        name = manifest_path.name.replace(".jsonl.manifest.json", "")
        rel = manifest_path.relative_to(DATA_ROOT)
        doc, err = read_json(manifest_path)
        if err:
            problems.append(f"{rel} — {err}")
            continue

        consumed = doc.get("evaluation_consumed")
        data_file = manifest_path.parent / manifest_path.name.replace(".manifest.json", "")
        rows = None
        hash_checked = False

        # Only `external_holdout_only` sets are holdouts. The rest are development
        # data the model is *supposed* to have seen; listing them as "still sealed"
        # would inflate the count of holdouts we can still spend.
        #
        # An ABSENT field is not a holdout either. Until 2026-08-08 a missing
        # `training_usage` fell through to the sealed branch, so three datasets
        # that sit 100% inside the training set were listed as spendable
        # holdouts purely because nobody had labelled them. The safe default for
        # "we do not know what this is" is "not a holdout" — claiming a holdout
        # we do not have is the expensive direction of that error.
        usage = doc.get("training_usage")
        if usage is None:
            unlabelled.append(name)
            development.append(name)
            continue
        if usage != "external_holdout_only":
            development.append(name)
            continue

        # The directory-level manifest is the seal: it records each file's sha256.
        seal_path = doc.get("lockbox_manifest")
        if seal_path:
            seal, seal_err = read_json(Path(seal_path))
            if seal_err:
                problems.append(f"{seal_path} (봉인 매니페스트) — {seal_err}")
            elif verify_hashes:
                entry = (seal.get("files") or {}).get(data_file.name)
                if entry is None:
                    problems.append(f"{rel} — 봉인 매니페스트에 {data_file.name} 항목이 없음")
                else:
                    rows = entry.get("rows")
                    if not data_file.exists():
                        problems.append(f"{rel} — 데이터 파일 없음: {data_file}")
                    else:
                        actual = sha256(data_file)
                        hash_checked = actual == entry.get("sha256")
                        if not hash_checked:
                            problems.append(
                                f"{rel} — 봉인 해시 불일치. 데이터가 바뀌었거나 다시 만들어짐\n"
                                f"      기록 {entry.get('sha256')}\n      실제 {actual}")

        if isinstance(consumed, dict) and consumed.get("model_path"):
            model = Path(consumed["model_path"]).parent.name
            spent.append((name, model))
        elif consumed:
            problems.append(f"{rel} — evaluation_consumed 형식이 이상함: {consumed!r}")
        else:
            contaminated = None
            if trained_ids and data_file.exists():
                own = row_ids(data_file)
                if own:
                    overlap = len(own & trained_ids)
                    contaminated = overlap / len(own)
                    if contaminated > 0:
                        problems.append(
                            f"{rel} — 홀드아웃으로 표시돼 있으나 학습 데이터와 "
                            f"{overlap}/{len(own)}행 ({contaminated*100:.1f}%) 겹침. "
                            f"이 데이터로 채점하면 학습 데이터로 채점하는 것이다")
            sealed.append((name, rows if rows is not None else doc.get("rows"),
                           hash_checked, contaminated))

    usable = [s_ for s_ in sealed if not s_[3]]
    verified = sum(1 for *_, ok, _ in sealed if ok)
    print(f"봉인 유지 표시 {len(sealed)}건 · 실제로 쓸 수 있는 것 {len(usable)}건 "
          f"· 해시 검증된 것 {verified}건")
    for name, rows, ok, cont in sealed:
        state = ("오염 %.0f%%" % (cont * 100)) if cont else "깨끗"
        mark = "해시확인" if ok else "해시미확인"
        print(f"  {name[:52]:54s}{'' if rows is None else f'{rows}행':>8s}  {state:9s} {mark}")
    print(f"\n개발용 (홀드아웃 아님) {len(development)}건")
    if unlabelled:
        print(f"  그중 training_usage 미기재 {len(unlabelled)}건 — 라벨이 없어서 "
              "개발용으로 처리했다. 홀드아웃이면 매니페스트에 명시하라")
        for name in sorted(unlabelled):
            print(f"    {name}")

    print(f"\n소진 {len(spent)}건")
    by_model: dict[str, list[str]] = {}
    for name, model in spent:
        by_model.setdefault(model, []).append(name)
    for model, names in sorted(by_model.items()):
        print(f"  {model}")
        for n in names:
            print(f"    {n}")

    if problems:
        print(f"\n⚠️  문제 {len(problems)}건 — 해결 전에는 어떤 락박스도 신뢰할 수 없습니다")
        for p in problems:
            print(f"  {p}")
    else:
        print("\n문제 없음")

    return 1 if (problems and strict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
