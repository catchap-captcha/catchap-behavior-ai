-- =============================================================================
-- catchap 취약문제 추천 (learning) — MySQL 8.0 schema  [DRAFT]
-- =============================================================================
-- DB 팀 전달용 초안. 애플리케이션은 DDL을 직접 실행하지 않습니다.
--
-- DB 팀 참고:
--   * 학습 도메인 테이블은 `learning_` 접두사를 씁니다 (봇탐지 `ai_` 와 분리).
--   * CAPTCHA 테이블과는 분리하고, learning_attempts.captcha_attempt_id 로
--     ai_behavior_attempts.attempt_id 와 연결합니다(같은 드래그가 봇탐지+학습
--     양쪽에 쓰이므로). 타입/FK는 확정 후 연결 (아래 주석 참조).
--   * 컬럼명은 애플리케이션이 이름으로 매핑하므로 변경 시 사전 협의 필요.
--   * 엔진 InnoDB + utf8mb4.
--
-- 데이터 출처 주의:
--   learning_attempts 의 "답 의미" 컬럼(grabbed_answer_id, correct_answer_id,
--   released_target_id, concept_id, difficulty, answer_options_count)은 프론트/
--   CAPTCHA 수집 payload가 함께 실어 보내야 채워집니다 (수집 계약 확장 필요).
-- =============================================================================

SET NAMES utf8mb4;

-- 1) 학생 (익명) -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS learning_students (
    student_id       VARCHAR(64)  NOT NULL,   -- 익명 ID (개인식별정보 저장 금지)
    age_group        VARCHAR(16)  NOT NULL DEFAULT 'unknown',  -- adult|child|unknown
    grade_level      VARCHAR(16)  NULL,        -- 예: 초1, 초2 (선택)
    consent_version  VARCHAR(32)  NULL,        -- 보호자 동의서 버전
    created_at       DATETIME     NOT NULL,
    updated_at       DATETIME     NOT NULL,
    PRIMARY KEY (student_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2) 개념 --------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS learning_concepts (
    concept_id         VARCHAR(64)  NOT NULL,
    name               VARCHAR(128) NOT NULL,
    subject            VARCHAR(64)  NULL,       -- 과목 (예: 수학)
    prerequisite_ids   VARCHAR(512) NULL,       -- 선행 개념 id 목록(콤마구분, 임시)
    description        VARCHAR(512) NULL,
    created_at         DATETIME     NOT NULL,
    PRIMARY KEY (concept_id),
    KEY idx_concepts_subject (subject)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3) 문제 (난이도·선택지·정답) ------------------------------------------------
CREATE TABLE IF NOT EXISTS learning_questions (
    question_id           VARCHAR(64)  NOT NULL,
    concept_id            VARCHAR(64)  NOT NULL,   -- 주(主) 개념
    subject               VARCHAR(64)  NULL,
    difficulty            DOUBLE       NOT NULL,   -- 0.0(쉬움)~1.0(어려움)
    answer_options_count  INT          NOT NULL,   -- 선택지 개수 (찍기보정용); 0/1=주관식
    correct_answer_id     VARCHAR(64)  NOT NULL,   -- 정답 타일 id
    answer_slot_id        VARCHAR(64)  NOT NULL DEFAULT 'slot',  -- 정답 드롭 영역 id
    created_at            DATETIME     NOT NULL,
    PRIMARY KEY (question_id),
    KEY idx_questions_concept (concept_id),
    KEY idx_questions_difficulty (difficulty),
    CONSTRAINT fk_questions_concept FOREIGN KEY (concept_id)
        REFERENCES learning_concepts (concept_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4) 문제-개념 연결 (다개념 대비 M:N) ----------------------------------------
CREATE TABLE IF NOT EXISTS learning_question_concepts (
    question_id  VARCHAR(64) NOT NULL,
    concept_id   VARCHAR(64) NOT NULL,
    weight       DOUBLE      NOT NULL DEFAULT 1.0,  -- 이 문제에서 개념의 비중
    PRIMARY KEY (question_id, concept_id),
    CONSTRAINT fk_qc_question FOREIGN KEY (question_id)
        REFERENCES learning_questions (question_id) ON DELETE CASCADE,
    CONSTRAINT fk_qc_concept FOREIGN KEY (concept_id)
        REFERENCES learning_concepts (concept_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5) 풀이 원본 (RawAttempt) ---------------------------------------------------
CREATE TABLE IF NOT EXISTS learning_attempts (
    attempt_id            VARCHAR(64)  NOT NULL,
    student_id            VARCHAR(64)  NOT NULL,
    question_id           VARCHAR(64)  NOT NULL,
    concept_id            VARCHAR(64)  NOT NULL,   -- 조회 편의를 위한 비정규화
    difficulty            DOUBLE       NOT NULL,
    answer_options_count  INT          NOT NULL,
    correct_answer_id     VARCHAR(64)  NOT NULL,

    -- 학생이 한 것 (WHAT — 수집 payload에서 옴)
    grabbed_answer_id     VARCHAR(64)  NULL,       -- 집은 타일 = 의도한 답
    released_target_id    VARCHAR(64)  NULL,       -- 놓은 영역; NULL=드롭 실패
    answer_slot_id        VARCHAR(64)  NOT NULL DEFAULT 'slot',

    -- 드래그 조작 신호 (CAPTCHA 행동 신호 재사용)
    pointercancel_count   INT          NOT NULL DEFAULT 0,
    regrab_count          INT          NOT NULL DEFAULT 0,
    failed_drop_count     INT          NOT NULL DEFAULT 0,
    retry_count           INT          NOT NULL DEFAULT 0,
    final_drop_error_px   DOUBLE       NULL,
    response_time_ms      INT          NULL,
    system_error          TINYINT(1)   NOT NULL DEFAULT 0,

    presentation_id       VARCHAR(64)  NULL,       -- 같은 문제 제시의 재시도 묶음

    -- 판정 결과 (learning.operation_error 로 계산해 저장)
    outcome               VARCHAR(20)  NULL,       -- correct|concept_error|operation_error|system_error|ambiguous
    valid_for_learning    TINYINT(1)   NULL,       -- 숙련도 반영 여부
    is_correct            TINYINT(1)   NULL,       -- 개념 레벨일 때만

    -- 봇탐지 CAPTCHA 시도와의 연결 (같은 드래그) — 타입/FK 확정 후 연결
    captcha_attempt_id    VARCHAR(64)  NULL,       -- -> ai_behavior_attempts.attempt_id
    session_id            VARCHAR(64)  NULL,

    answered_at           DATETIME     NOT NULL,
    created_at            DATETIME     NOT NULL,

    PRIMARY KEY (attempt_id),
    KEY idx_attempts_student (student_id),
    KEY idx_attempts_question (question_id),
    KEY idx_attempts_concept (concept_id),
    KEY idx_attempts_student_concept (student_id, concept_id),
    KEY idx_attempts_presentation (presentation_id),
    KEY idx_attempts_captcha (captcha_attempt_id),
    CONSTRAINT fk_attempts_student FOREIGN KEY (student_id)
        REFERENCES learning_students (student_id),
    CONSTRAINT fk_attempts_question FOREIGN KEY (question_id)
        REFERENCES learning_questions (question_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 6) 개념 숙련도 (계산 결과 캐시) --------------------------------------------
CREATE TABLE IF NOT EXISTS learning_concept_mastery (
    student_id         VARCHAR(64) NOT NULL,
    concept_id         VARCHAR(64) NOT NULL,
    mastery            DOUBLE      NOT NULL,     -- 0..1 (찍기보정)
    valid_attempts     INT         NOT NULL,
    correct            INT         NOT NULL,
    guess_baseline     DOUBLE      NOT NULL,
    diagnostic_needed  TINYINT(1)  NOT NULL DEFAULT 1,
    updated_at         DATETIME    NOT NULL,
    PRIMARY KEY (student_id, concept_id),
    CONSTRAINT fk_mastery_student FOREIGN KEY (student_id)
        REFERENCES learning_students (student_id),
    CONSTRAINT fk_mastery_concept FOREIGN KEY (concept_id)
        REFERENCES learning_concepts (concept_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 7) 추천 (추천한 문제 + 이유) ------------------------------------------------
CREATE TABLE IF NOT EXISTS learning_recommendations (
    recommendation_id  BIGINT       NOT NULL AUTO_INCREMENT,
    student_id         VARCHAR(64)  NOT NULL,
    question_id        VARCHAR(64)  NOT NULL,
    concept_id         VARCHAR(64)  NOT NULL,
    target_band        VARCHAR(16)  NOT NULL,    -- easy|medium|hard
    reason             VARCHAR(255) NULL,
    mastery_before     DOUBLE       NOT NULL,
    created_at         DATETIME     NOT NULL,
    PRIMARY KEY (recommendation_id),
    KEY idx_reco_student (student_id),
    CONSTRAINT fk_reco_student FOREIGN KEY (student_id)
        REFERENCES learning_students (student_id),
    CONSTRAINT fk_reco_question FOREIGN KEY (question_id)
        REFERENCES learning_questions (question_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 8) 추천 결과 (추천 문제를 푼 결과) ------------------------------------------
CREATE TABLE IF NOT EXISTS learning_recommendation_results (
    result_id           BIGINT       NOT NULL AUTO_INCREMENT,
    recommendation_id   BIGINT       NOT NULL,
    attempt_id          VARCHAR(64)  NOT NULL,   -- 그 추천을 푼 learning_attempts
    answer_correct      TINYINT(1)   NULL,
    interaction_success TINYINT(1)   NULL,
    mastery_after       DOUBLE       NULL,
    created_at          DATETIME     NOT NULL,
    PRIMARY KEY (result_id),
    KEY idx_recoresult_reco (recommendation_id),
    CONSTRAINT fk_recoresult_reco FOREIGN KEY (recommendation_id)
        REFERENCES learning_recommendations (recommendation_id) ON DELETE CASCADE,
    CONSTRAINT fk_recoresult_attempt FOREIGN KEY (attempt_id)
        REFERENCES learning_attempts (attempt_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
