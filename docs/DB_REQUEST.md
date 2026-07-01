# DB 담당자 전달 문서 (MySQL 8.0)

ai-service는 **테이블을 직접 생성하지 않습니다.** 아래 DDL을 검토·적용해 주세요.

## 적용 방법

```bash
mysql -h <host> -u <admin> -p <database> < db/schema_mysql.sql
```

- 엔진 `InnoDB`, 문자셋 `utf8mb4` 기준입니다.
- 모든 테이블/뷰는 `ai_` 접두사를 사용합니다 (기존 테이블과 충돌 방지).

## 생성 대상

| 객체 | 종류 | 용도 |
|------|------|------|
| `ai_behavior_attempts` | table | 시도 1건 메타 + 라벨 + 품질상태 |
| `ai_pointer_events` | table | 원본 포인터 이벤트 (attempt별 다건) |
| `ai_interaction_summaries` | table | 조작 카운터 요약 |
| `ai_attempt_features` | table | 행동 Feature 29개 |
| `ai_security_features` | table | replay/세션 위험 신호 |
| `ai_model_predictions` | table | 추론 결과 로그 |
| `ai_training_dataset` | **view** | 학습용 조회 (valid + human/bot + label_source 존재) |

## ⚠️ FK 연결 요청 (확인 필요)

`ai_behavior_attempts.challenge_id` 와 `session_id` 는 현재 **VARCHAR(64)** 로 저장합니다.
다른 팀의 CAPTCHA/세션 테이블 **이름과 자료형이 확정되면**, 아래를 확인 후 FK를
연결해 주세요.

- CAPTCHA 챌린지 테이블: 테이블명 `?`, PK 컬럼/타입 `?`
- 세션 테이블: 테이블명 `?`, PK 컬럼/타입 `?`

자료형이 정수형이라면 애플리케이션 매핑도 함께 바꿔야 하므로 **연결 전 ai-service
팀에 알려주세요.** (지금은 문자열 저장이라 FK 없이도 정상 동작합니다.)

## 계정 권한

ai-service 애플리케이션 계정은 다음 권한만 필요합니다 (DDL 권한 불필요):

```
SELECT, INSERT, UPDATE  ON <database>.ai_*
```

- `DELETE`는 사용하지 않습니다 (품질이 낮은 원본도 삭제하지 않고 상태만 기록).
- 접속 정보는 애플리케이션의 환경변수(`MYSQL_*`)로 주입되며 코드에 하드코딩하지
  않습니다.

## 인덱스 요약 (DDL에 포함됨)

- `ai_behavior_attempts`: challenge_id, session_id, participant, label, quality_status, label_source
- `ai_pointer_events`: `UNIQUE(attempt_id, seq)` + attempt_id
- 각 자식 테이블: `attempt_id` FK (ON DELETE CASCADE)

문의: 스키마 컬럼명은 애플리케이션이 이름으로 매핑하므로 **컬럼명 변경 시 반드시
사전 협의**가 필요합니다.
