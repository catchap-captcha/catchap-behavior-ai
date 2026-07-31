-- PointerEvent 원천 신호용 컬럼 (2026-07-31 적용 완료)
--
-- pointer_type / pressure / buttons_mask 는 배포 스키마에 이미 있었는데 ORM 매핑이
-- 없어 계속 NULL 이었다. 나머지 셋만 새로 추가했다.
--
-- 전부 nullable 인 것이 중요하다. 미지원 브라우저는 값을 못 보내고, 그 결측을
-- 실제 0 과 섞으면 판별이 왜곡된다(7/17 계획서 4.2 주의사항과 같은 이유).
ALTER TABLE ai_pointer_events ADD COLUMN is_trusted      TINYINT(1) NULL;
ALTER TABLE ai_pointer_events ADD COLUMN is_primary      TINYINT(1) NULL;
ALTER TABLE ai_pointer_events ADD COLUMN coalesced_count INT        NULL;
