"""quality_status 매핑이 검증기가 실제로 내는 값과 맞는지 고정한다.

이 매핑은 조용히 틀릴 수 있다. 배포된 CHECK 가 'review' 를 허용하기 때문에,
매핑에 없는 값이 들어와도 INSERT 는 성공하고 예외도 로그도 남지 않는다.
실제로 그랬다 — 매핑이 "accepted" 를 키로 갖고 있었는데 검증기는 "valid" 를
내므로, 정상 시도가 전부 'review' 로 저장됐다. quality_status='valid' 로
학습 데이터를 뽑으면 0건이 나오는 상태였고, 아무 데서도 안 터졌다.
"""

from __future__ import annotations

from app.database.repositories import _QUALITY_STATUS, _quality_status
from app.services.quality_validator import (
    QUALITY_PENDING,
    QUALITY_REJECTED,
    QUALITY_VALID,
)

# 배포된 chk_ai_behavior_attempts_quality_status 가 허용하는 값.
DEPLOYED_ALLOWED = {"pending", "valid", "invalid", "review"}


def test_every_validator_status_has_an_explicit_mapping():
    """검증기가 내는 세 값 모두 매핑에 있어야 한다 — fallback 으로 새면 안 된다."""
    for status in (QUALITY_VALID, QUALITY_PENDING, QUALITY_REJECTED):
        assert status in _QUALITY_STATUS, (
            f"검증기가 내는 {status!r} 가 매핑에 없다. "
            "fallback 'review' 로 조용히 새고, DB CHECK 가 그걸 받아준다."
        )


def test_valid_attempts_are_stored_as_valid_not_review():
    """정상 시도는 'valid' 로 저장돼야 한다. 이게 학습 데이터 선별 기준이다."""
    assert _quality_status(QUALITY_VALID) == "valid"
    assert _quality_status(QUALITY_REJECTED) == "invalid"
    assert _quality_status(QUALITY_PENDING) == "pending"


def test_mapped_values_satisfy_the_deployed_check():
    for source, stored in _QUALITY_STATUS.items():
        assert stored in DEPLOYED_ALLOWED, f"{source} -> {stored} 는 CHECK 위반"


def test_unknown_status_falls_back_to_review():
    """모르는 값은 review 로 보내되, 그건 예외 경로여야 한다."""
    assert _quality_status("wat") == "review"
    assert _quality_status(None) == "pending"
