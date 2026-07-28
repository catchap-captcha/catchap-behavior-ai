# DB팀 요청 — 행동 AI 서비스 연동 (2026-07-28)

작성: 최성우 (행동 기반 봇 판별 AI 담당)
대상 DB: `catchap_dev_db` @ `210.109.52.114:3306` (MySQL 8.0.46)

---

## 0. 요약

행동 AI 서비스를 DB에 붙이려고 점검하다가 **배포된 스키마와 AI 코드가 서로 다른 설계**라는 걸 확인했습니다.
**AI 쪽 코드를 배포된 스키마에 맞춰 이미 수정했습니다.** DB 변경은 최소한만 요청드립니다.

요청은 2건이었고 **둘 다 반영 완료**입니다. 처음 있던 계정 발급 요청(A)은 철회했습니다.

| # | 요청 | 급함 | 상태 |
|---|---|---|---|
| B | `ai_shadow_outcomes` 테이블 생성 | 🟡 권장 | ✅ **반영 완료 (07-28)** |
| C | `ai_interaction_summaries` 컬럼 5개 추가 | 🟢 선택 | ✅ **반영 완료 (07-28)** |
| ~~A~~ | ~~`ai_service` DB 계정 발급~~ | — | ❌ **철회** — 아래 2절 참고 |

**B·C 처리해주셔서 감사합니다.** 요청드린 형태 그대로 적용된 것 확인했고
(FK·CHECK 3개·인덱스 2개 포함), 저희 쪽 코드도 맞춰서 수정했습니다:

- 요청 C 컬럼 5개 — `metadata` JSON 저장에서 **실제 컬럼 저장으로 전환 완료**.
  약속드린 대로 코드를 바꿨으니 이제 인덱싱·집계 바로 쓰실 수 있습니다.
- 요청 B — shadow 결과가 정상 저장되는 것까지 확인했습니다.

**DB팀에 더 요청드릴 것은 없습니다.**

추가로 **확인만** 부탁드릴 사항이 3건 있습니다 (6절).

> **이 문서의 SQL 은 전부 실행 검증했습니다.**
> 프로덕션 스키마를 로컬 MySQL 8.0 에 그대로 복제한 뒤 실행해
> 문법·제약이 의도대로 적용되는 것을 확인했고, 검증용 임시 DB 는 삭제했습니다.
> 프로덕션에는 `SELECT` / `SHOW` 만 실행했습니다. 쓰기는 하지 않았습니다.
>
> 각 SQL 은 **줄마다 왜 필요한지 주석**을 달아뒀습니다. 불필요하다고 판단되는 항목은
> 빼고 적용하셔도 됩니다 — 어떤 기능이 빠지는지도 주석에 적어뒀습니다.

---

## 1. 배경 — 무엇을 발견했는지

`ai-service` 레포는 `db/schema_mysql.sql`이라는 DDL을 넘겨드린 것으로 알고 있었고,
코드는 그 DDL 기준으로 작성돼 있었습니다. 그런데 실제 배포본은 다른 설계였습니다.

`ai_*` 테이블 **7개 전부** 컬럼이 일치하지 않았습니다. 가장 큰 차이:

- `ai_behavior_attempts`의 PK가 **`id CHAR(36)`** — 코드는 `attempt_id VARCHAR(64)`를 PK로 가정
- FK 5개가 전부 `ai_behavior_attempts.id`를 참조
- 이름 차이: `participant_id`↔`anonymous_participant_id`, `bot_type`↔`bot_family`,
  `quality_reason`↔`rejection_reason`, `human_probability`↔`human_score`,
  `decision_threshold`↔`threshold`, `x_pixel`/`elapsed_ms`↔`x`/`t_ms` 등

읽기조차 실패하는 상태였습니다:

```
recent_session_history() → (1054, "Unknown column 'ai_behavior_attempts.attempt_id' in 'field list'")
```

**중요: 이건 DB팀 실수가 아닙니다.** 배포된 스키마가 오히려 더 촘촘합니다
(`webdriver_detected`, `headless_browser_detected`, `nonce_reuse_detected`,
`event_order_anomaly_detected`, `device_fingerprint_hash` 등). 저희 쪽 DDL이 뒤처진 것이라
판단해서 **AI 코드를 배포본에 맞추는 방향으로 수정했습니다.**

`db/schema_mysql.sql`은 더 이상 기준이 아닙니다. 폐기 예정입니다.

### 저희가 이미 처리한 것 (DB 작업 불필요)

- ORM 전체를 배포 스키마에 맞춰 재매핑
- CAPTCHA가 보내는 42자 `ms-{challenge_id}-a{n}`를 `id CHAR(36)`에 맞추기 위해
  **UUIDv5로 결정적 변환**. 원본 문자열은 `metadata.source_attempt_id`에 보존
- 배포된 CHECK 제약 준수:
  - `label` — AI 추론 트래픽은 `'unknown'`이 아니라 **NULL**로 저장 (원본은 `metadata.raw_label`)
  - `quality_status` — `accepted`→`valid`, `rejected`→`invalid`로 매핑
  - `predicted_label` — `human`/`bot`/`uncertain`으로 정규화
  - `x_normalized`/`y_normalized`/`straightness_ratio`/확률값 — `[0,1]` 클램프
- `UNIQUE KEY (challenge_id, attempt_number)` 준수 — 재시도 시 `attempt_number` 자동 증가
- 배포 스키마에 자리가 없는 값은 각 테이블의 JSON 컬럼에 보존
  (`metadata` / `extra_features` / `security_flags` / `model_metadata` / `event_metadata`)

검증: 프로덕션 스키마를 로컬에 그대로 복제해 테스트했고, 저장·재생탐지 모두 정상 동작 확인했습니다.
프로덕션에는 **읽기만** 했습니다 (SELECT / SHOW).

---

## 2. ~~요청 A — `ai_service` DB 계정~~ (철회)

**철회합니다. 계정 발급하지 말아주세요.**

행동 AI 서비스를 **백엔드 서버에 합치기로** 했습니다. 합치고 나면 백엔드가 이미 쓰는
DB 연결을 그대로 쓰게 되므로 별도 계정은 관리 대상만 늘립니다.

합병이 가능한 이유 (확인 완료):

- 추론 경로에 `torch`/CUDA 사용처가 없습니다. **GPU 가 필요 없습니다.**
- 모델은 `two_view_fusion.joblib` 1.1MB (LightGBM) 하나뿐입니다.
- 지금 GPU 서버(`61.109.239.231`)에 있는 건 그 서버에 자리가 있어서였지 필요해서가 아닙니다.
- `requirements.txt` 의 `torch` 는 학습·GAN 전용이라 서빙 배포에서는 제외 가능합니다.

**다만 한 가지만 챙겨주세요** — 백엔드가 쓰는 계정으로 아래 7개 테이블에
`SELECT, INSERT, UPDATE` 가 되는지 확인 부탁드립니다. 합병 후 AI 코드가 이 테이블들에 씁니다.

```sql
-- 확인용 (실행해도 아무것도 바뀌지 않습니다)
SHOW GRANTS FOR CURRENT_USER();
```

| 테이블 | 필요 권한 | 왜 |
|---|---|---|
| `ai_behavior_attempts` | SELECT, INSERT, UPDATE | 시도 1건당 부모 행. SELECT 는 재시도 판별용 |
| `ai_pointer_events` | SELECT, INSERT | 궤적 원본. **SELECT 필수** — 재생 공격 탐지가 과거 궤적을 다시 읽어 DTW 비교 |
| `ai_interaction_summaries` | INSERT | 조작 카운터 요약 |
| `ai_attempt_features` | SELECT, INSERT, UPDATE | 모델 입력 feature. 재학습 때 다시 읽음 |
| `ai_security_features` | SELECT, INSERT, UPDATE | 재생 유사도·시도 빈도 |
| `ai_model_predictions` | SELECT, INSERT | 판정 결과 |
| `ai_shadow_outcomes` | SELECT, INSERT | 오탐률 집계 (요청 B 로 생성된 테이블) |

현재 확인된 `catchap_backend` 권한은 `catchap_dev_db.*` 에 대한
`SELECT, INSERT, UPDATE, DELETE` 라 위 요건을 이미 충족합니다.
**따라서 추가 작업이 필요 없을 가능성이 높습니다.** 백엔드가 다른 계정을 쓴다면 알려주세요.

## 3. 요청 B — `ai_shadow_outcomes` 테이블 🟡 ✅ 반영 완료

> 07-28 적용 확인했습니다. 아래는 요청 당시 기록으로 남겨둡니다.

**요청 당시 상태**: 테이블이 없었습니다. `POST /api/v1/behavior/shadow/outcomes`가 500을 반환합니다.
(`db/migrations/20260723_shadow_mode.sql`을 드린 적이 있는데 적용되지 않은 것 같습니다.)

**왜 필요한지**: 지금 AI는 **shadow 모드**입니다. 즉 판정을 내리기만 하고 실제 차단은 하지 않습니다.
active로 전환하려면 "AI가 차단했을 사용자 중 실제로는 정상이었던 비율"(오탐률)을 먼저 측정해야 하는데,
그러려면 **AI의 판정**과 **캡차의 실제 최종 결과**를 짝지어 저장해야 합니다. 그 짝을 담는 테이블입니다.

이게 없으면 오탐률을 모른 채로 active를 켜야 하고, 그건 정상 사용자를 차단할 위험이 있어
저희 쪽에서 금지하고 있는 상태입니다.

**요청 DDL** (배포된 다른 `ai_*` 테이블 스타일에 맞췄습니다):

```sql
CREATE TABLE ai_shadow_outcomes (
    -- 왜: 어느 시도에 대한 결과인지. 다른 ai_* 테이블과 동일하게 CHAR(36) 이고
    --     ai_behavior_attempts.id 를 참조합니다. PK 로 둔 이유는 한 시도당
    --     결과가 정확히 하나여야 하고, 캡차가 같은 요청을 재전송해도
    --     중복 적재되지 않아야 하기 때문입니다(멱등성).
    attempt_id           CHAR(36)     NOT NULL,

    -- 왜: 캡차가 실제로 내린 판정(정답 맞췄는지). 오탐률 계산의 '정답지'입니다.
    --     "AI 는 봇이라 했는데 캡차는 통과시킨" 건수를 세려면 이 값이 있어야 합니다.
    main_captcha_verdict VARCHAR(16)  NOT NULL,

    -- 왜: 사용자에게 최종적으로 나간 결과. shadow 모드에서는 위와 같아야 정상이며,
    --     다르면 AI 가 실수로 사용자 결과에 개입했다는 뜻이라 즉시 잡아내야 합니다.
    --     (애플리케이션에서도 검증하지만 DB 에도 기록해 사후 감사 가능하게)
    final_verdict        VARCHAR(16)  NOT NULL,

    -- 왜: "active 였다면 AI 가 취했을 조치". 이 컬럼과 main_captcha_verdict 를
    --     교차하면 오탐률이 바로 나옵니다:
    --       오탐 = would_have_action != 'allow' AND main_captcha_verdict = 'passed'
    --     이 문서에서 요청드리는 핵심이 사실상 이 한 컬럼입니다.
    would_have_action    VARCHAR(32)  NOT NULL,

    -- 왜: 위험도 구간별로 오탐률이 어떻게 다른지 봐야 임계값을 조정할 수 있습니다.
    risk_level           VARCHAR(16)  NOT NULL,

    -- 왜: 모델을 교체하면 오탐률도 달라집니다. 버전을 같이 남겨야
    --     "어느 모델의 성적인지" 구분됩니다. 이게 없으면 모델 교체 시점을 기준으로
    --     데이터가 섞여 이전 측정치가 전부 무의미해집니다.
    model_version        VARCHAR(64)  NOT NULL,

    -- 왜: 측정 기간을 잘라서 보기 위함(예: "최근 7일 오탐률").
    recorded_at          DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),
    created_at           DATETIME(6)  NOT NULL DEFAULT CURRENT_TIMESTAMP(6),

    PRIMARY KEY (attempt_id),

    -- 왜: 오탐률 집계 쿼리가 would_have_action 으로 GROUP BY 합니다.
    KEY idx_ai_shadow_outcomes_action (would_have_action),
    -- 왜: 기간별 조회(최근 N일)에서 풀스캔을 막습니다.
    KEY idx_ai_shadow_outcomes_recorded_at (recorded_at),

    -- 왜: 존재하지 않는 시도에 대한 결과가 들어오는 걸 DB 레벨에서 차단합니다.
    --     CASCADE 인 이유는 시도가 지워지면 그 결과만 남아도 의미가 없기 때문입니다.
    CONSTRAINT fk_ai_shadow_outcomes_attempt
        FOREIGN KEY (attempt_id) REFERENCES ai_behavior_attempts (id) ON DELETE CASCADE,

    -- 왜: 다른 ai_* 테이블이 CHECK 로 값 범위를 강제하고 있어 같은 방식을 따랐습니다.
    --     오타나 코드 버그로 이상한 값이 들어가면 오탐률 계산이 조용히 틀어집니다.
    CONSTRAINT chk_ai_shadow_outcomes_main_verdict
        CHECK (main_captcha_verdict IN ('passed','failed')),
    CONSTRAINT chk_ai_shadow_outcomes_final_verdict
        CHECK (final_verdict IN ('passed','failed')),
    -- 왜: ai_model_predictions.recommended_action 의 CHECK 와 동일한 값 집합입니다.
    CONSTRAINT chk_ai_shadow_outcomes_action
        CHECK (would_have_action IN ('allow','step_up','step_up_and_rate_limit'))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
```

**이 DDL 은 검증했습니다.** 프로덕션 스키마를 로컬에 그대로 복제한 DB 에 적용한 뒤
실제 요청을 흘려보내 정상 저장되는 것을 확인했습니다. 미검증 SQL 을 보내지 않으려고 했습니다.

**참고 — 요청하지 않는 것**: `ai_model_predictions.policy_mode` 컬럼.
`20260723_shadow_mode.sql` 에 같이 들어있었지만, 저희가 `model_metadata` JSON 에 넣는 것으로
처리했으므로 **ALTER TABLE 하지 말아주세요.** 운영 테이블 변경을 하나라도 줄이는 게 낫다고 봤습니다.

---

## 4. 요청 C — `ai_interaction_summaries` 컬럼 5개 🟢 ✅ 반영 완료

> 07-28 적용 확인했고, 저희 코드도 JSON 저장 → 컬럼 저장으로 전환했습니다.
> 아래는 요청 당시 기록입니다.

**요청 당시 상태**: AI가 산출하는 조작 카운터 5개에 대응하는 컬럼이 없습니다.
배포된 테이블에는 `wrong_click_count`, `wrong_drag_count`, `object_revisit_count` 등이 있는데
**세는 대상이 미묘하게 다릅니다.** 의미가 다른 컬럼에 억지로 넣으면 나중에 분석이 틀어지므로,
현재는 `metadata` JSON에 그대로 보존하고 있습니다.

**동작에는 지장이 없습니다.** 인덱싱·집계가 필요해지면 그때 요청드리는 것도 괜찮습니다.

| AI 카운터 | 의미 | 배포된 유사 컬럼과의 차이 |
|---|---|---|
| `regrab_count` | 드래그 중 놓았다가 다시 잡은 횟수 | `object_revisit_count`는 "다시 방문"이라 범위가 넓음 |
| `retry_count` | 같은 챌린지 재시도 횟수 | 대응 컬럼 없음 |
| `pointercancel_count` | `pointercancel` 이벤트 수 | 대응 컬럼 없음 |
| `empty_click_count` | 아무 객체도 없는 곳 클릭 | `wrong_click_count`는 "틀린 객체 클릭" |
| `failed_drop_count` | 드롭했으나 선택되지 않음 | `wrong_drag_count`는 "틀린 객체 드래그" |

필요하다고 판단하시면 아래를 적용해주세요. **급하지 않습니다.**

```sql
-- 왜(공통): 지금은 metadata JSON 에 들어있어 조회·집계는 되지만 인덱싱이 안 되고
--          JSON_EXTRACT 를 거쳐야 합니다. 이 다섯 값으로 대시보드를 만들거나
--          WHERE 조건으로 거를 일이 생기면 그때 컬럼이 있는 편이 낫습니다.
--          전부 NOT NULL DEFAULT 0 이라 기존 행에 영향 없이 추가됩니다.
ALTER TABLE ai_interaction_summaries
  -- 왜: 봇은 한 번에 정확히 집어 옮기지만, 사람은 놓쳤다가 다시 잡는 일이 잦습니다.
  --     사람다움을 나타내는 신호라 단독 집계 수요가 가장 큽니다.
  ADD COLUMN regrab_count        INT UNSIGNED NOT NULL DEFAULT 0 AFTER object_revisit_count,
  -- 왜: 같은 문제를 몇 번 다시 풀었는지. 문제 난이도 평가에도 씁니다.
  ADD COLUMN retry_count         INT UNSIGNED NOT NULL DEFAULT 0 AFTER regrab_count,
  -- 왜: pointercancel 은 실제 브라우저에서만 발생하는 이벤트라
  --     합성 궤적(봇)에서는 거의 0 입니다. 봇 판별에 유용한 신호입니다.
  ADD COLUMN pointercancel_count INT UNSIGNED NOT NULL DEFAULT 0 AFTER retry_count,
  -- 왜: 빈 공간 클릭. 기존 wrong_click_count(틀린 '객체' 클릭)와 대상이 다릅니다.
  ADD COLUMN empty_click_count   INT UNSIGNED NOT NULL DEFAULT 0 AFTER pointercancel_count,
  -- 왜: 드롭했으나 선택 처리가 안 된 횟수. 기존 wrong_drag_count(틀린 '객체' 드래그)와
  --     구분되는 값이라 같은 컬럼에 합칠 수 없습니다.
  ADD COLUMN failed_drop_count   INT UNSIGNED NOT NULL DEFAULT 0 AFTER empty_click_count;
```

**→ 코드 전환 완료 (07-28).** 이제 이 컬럼들에 실제 값이 들어갑니다. 집계·인덱싱 바로 쓰실 수 있습니다.

---

## 5. 참고 — 캡차 행동 방어 테이블 (김민서 담당)

`captcha_behavior_sessions` / `captcha_behavior_batches` 두 테이블이 `catchap_dev_db`에 없습니다.
이건 제 담당이 아니라 **캡차 서비스(김민서)** 쪽 배포 항목입니다.
캡차의 행동 이벤트 신뢰 프로토콜(재생 공격 방어)이 이 테이블을 씁니다.
캡차 배포 일정 잡으실 때 같이 챙기시면 좋겠습니다. DDL은 캡차 레포에 있습니다.

---

## 6. 확인만 부탁드릴 사항

**(1) `ai_attempt_features`의 feature 컬럼 33개를 저희가 못 채웁니다.**

배포된 테이블의 feature 집합과 저희 모델이 쓰는 집합이 다릅니다
(예: `movement_entropy`, `average_curvature`, `hesitation_count` — 저희 추출기에 없는 값).
반대로 저희 v2.3 모델은 feature 55개를 쓰는데 그중 배포 컬럼과 정확히 같은 양은 10개뿐입니다.

현재 처리: **55개 전부를 `extra_features` JSON에 저장**하고, 이름만 다르고 계산이 동일한 10개
(`average_speed`, `speed_stddev`, `average_acceleration`, `average_jerk`,
`direction_change_count`, `total_path_length`, `straight_line_distance`,
`straightness_ratio`, `pointer_event_count`, `total_duration_ms`)는 명명 컬럼에도 같이 채웁니다.

의미가 애매한 컬럼은 **일부러 비워뒀습니다.** 잘못 매핑하는 것보다 비는 게 낫다고 판단했습니다.
혹시 이 컬럼들을 채우는 다른 주체(백엔드? 별도 배치?)가 있다면 알려주세요. 충돌 방지가 필요합니다.

**(2) `ai_behavior_attempts` / `ai_predictions` 를 쓰는 다른 서비스가 있나요?**

현재 `ai_*` 테이블은 전부 **0행**이라 충돌 걱정은 없어 보입니다만,
AI가 쓰기 시작하면 `(challenge_id, attempt_number)` UNIQUE KEY를 공유하게 됩니다.
다른 팀도 이 테이블에 넣을 계획이 있다면 미리 조율이 필요합니다.
(`ai_predictions`는 2026-07-10 생성 후 0행인 구버전 테이블로 보이는데, 폐기 대상인지 확인 부탁드립니다.)

**(3) 사소함** — `catchap_backend` 계정에 `SHOW VIEW` 권한이 없어
`ai_training_dataset` 뷰의 정의를 확인하지 못했습니다. AI가 그 뷰를 쓰진 않으니 급하진 않습니다.

---

## 7. 다음 단계

DB팀에 요청드릴 것은 없습니다. 이후는 저희와 백엔드 쪽 작업입니다.

1. 행동 AI 를 백엔드 서버로 이전 (GPU 불필요 — 2절 참고)
2. 백엔드 DB 연결로 `/health` 가 `mysql_connected:true` 인지 확인
3. shadow 모드로 실트래픽 수집 시작 → 오탐률 측정
4. 오탐률이 목표(3% 이하)를 만족하면 그때 active 전환 논의

문의: 최성우 (wwdhogo@gmail.com)
