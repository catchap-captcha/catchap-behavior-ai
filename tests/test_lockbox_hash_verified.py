"""봉인 홀드아웃의 해시가 **실제로 대조되는지** 지킨다 (2026-08-18).

왜 이 시험이 있나
-----------------
`tools/lockbox_audit.py` 는 스스로 세 가지 질문에 답한다고 적어 두었고, 그 셋째가
"디스크의 데이터가 봉인 당시 그대로인가(sha256)" 다. 그런데 해시 대조가
`lockbox_manifest` 키가 있는 경우에만 돌았다. 그 형식을 쓰는 것은 **이미 소진된
세 벌뿐**이고, 아직 봉인된 9벌은 자기 매니페스트의 `output.sha256` 에 해시를 적는
형식이다. 그래서 감사는 9벌 전부를 "해시미확인" 으로 내보내면서도 `--strict` 에서
정상 종료했다.

★그 상태의 위험은 "해시가 틀렸다" 가 아니라 **"틀렸는지 아무도 모른다"** 였다.
승격 심사에서 "후보가 이 데이터를 본 적 없다" 를 뒷받침하는 물증이 이 해시다.
물증 없이 통과하는 심사는 심사가 아니다.

봉인이 하나라도 대조되지 않으면 실패시킨다 — 개수가 아니라 **미확인이 0인지**를 본다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA_ROOT = Path("data")


def _sealed_manifests() -> list[Path]:
    found = []
    for manifest in sorted(DATA_ROOT.rglob("*.manifest.json")):
        try:
            doc = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError):
            continue  # 파손은 감사 도구가 따로 잡는다
        if doc.get("training_usage") == "external_holdout_only":
            found.append(manifest)
    return found


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="데이터 디렉터리가 없는 환경")
def test_모든_봉인_홀드아웃이_대조할_해시를_가진다():
    """해시가 없는 봉인은 봉인이 아니다 — 바뀌었는지 확인할 방법이 없다."""
    missing = []
    for manifest in _sealed_manifests():
        doc = json.loads(manifest.read_text())
        has_seal = bool(doc.get("lockbox_manifest"))
        has_output_hash = bool((doc.get("output") or {}).get("sha256"))
        if not (has_seal or has_output_hash):
            missing.append(manifest.name)
    assert not missing, (
        "봉인으로 표시됐는데 대조할 해시가 없다. 이 데이터로 채점하면 "
        f"'후보가 본 적 없다'를 증명할 수 없다: {missing}"
    )


@pytest.mark.skipif(not DATA_ROOT.exists(), reason="데이터 디렉터리가 없는 환경")
def test_감사_도구가_봉인을_빠짐없이_대조한다():
    """★핵심. 감사가 "해시미확인" 을 하나라도 남기면 실패한다.

    도구를 **바깥에서 그대로 돌린다.** 내부 함수를 부르면 도구가 실제로 내보내는
    보고서와 시험이 갈라질 수 있는데, 여기서 지키려는 것은 사람이 읽는 그 보고서다.
    (봉인 파일 전체를 해시하므로 다른 시험보다 느리다 — 데이터 무결성 시험이라
    감수한다.)
    """
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "tools/lockbox_audit.py", "--strict"],
        capture_output=True, text=True, timeout=600,
    )
    assert done.returncode == 0, f"감사가 문제를 보고했다:\n{done.stdout}\n{done.stderr}"
    assert "해시미확인" not in done.stdout, (
        "봉인인데 해시가 대조되지 않은 것이 있다. 승격 심사에서 "
        f"'후보가 본 적 없다'를 증명할 수 없다:\n{done.stdout}"
    )
