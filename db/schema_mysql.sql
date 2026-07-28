-- =============================================================================
-- ⚠️ 폐기됨 (2026-07-28) — DO NOT APPLY. 참고용으로만 남겨둡니다.
-- =============================================================================
-- 이 DDL은 배포된 `catchap_dev_db` 와 다릅니다. DB팀이 더 촘촘한 설계를 이미
-- 적용했고, AI 코드는 그 배포본에 맞춰 재매핑됐습니다 (app/database/mysql_models.py).
-- 이 파일을 적용하면 배포 스키마와 충돌합니다.
--
--   실제 스키마 스냅샷 : local-test-data/prod_ai_schema.sql
--   차이와 요청 사항   : docs/DB_REQUEST_20260728.md
-- =============================================================================

-- =============================================================================
-- catchap ai-service — MySQL 8.0 schema (구버전)
-- =============================================================================
-- This DDL is HANDED TO THE DB TEAM. The application never runs DDL itself.
--
-- Notes for the DB team:
--   * challenge_id and session_id are stored as VARCHAR here. Once the other
--     teams' table names and column types are confirmed, add real FOREIGN KEYs
--     to those tables (see docs/DB_REQUEST.md). Do NOT change the AI tables'
--     column names — the application maps to them by name.
--   * All AI tables are prefixed `ai_` to avoid clashing with existing tables.
--   * Engine InnoDB + utf8mb4 assumed (MySQL 8.0 default).
-- =============================================================================

SET NAMES utf8mb4;

-- 0) One-time CAPTCHA challenge state -----------------------------------------
-- Nonce and problem binding are SHA-256 digests. The browser never calls the
-- issue/consume API directly; only the CAPTCHA backend holds its API key.
CREATE TABLE IF NOT EXISTS ai_captcha_challenges (
    challenge_id          VARCHAR(64)  NOT NULL,
    nonce_hash            CHAR(64)     NOT NULL,
    session_id            VARCHAR(64)  NOT NULL,
    site_key              VARCHAR(128) NOT NULL,
    purpose               VARCHAR(64)  NOT NULL,
    problem_binding_hash  CHAR(64)     NOT NULL,
    status                VARCHAR(16)  NOT NULL DEFAULT 'issued', -- issued|consumed
    expires_at            DATETIME     NOT NULL,
    consumed_at           DATETIME     NULL,
    verdict               VARCHAR(16)  NULL, -- passed|failed
    created_at            DATETIME     NOT NULL,

    PRIMARY KEY (challenge_id),
    KEY idx_challenges_session (session_id),
    KEY idx_challenges_status_expiry (status, expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 1) Attempt-level record ------------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_behavior_attempts (
    attempt_id                VARCHAR(64)  NOT NULL,
    challenge_id              VARCHAR(64)  NOT NULL,   -- FK target TBD (CAPTCHA team)
    session_id                VARCHAR(64)  NOT NULL,   -- FK target TBD (session table)
    anonymous_participant_id  VARCHAR(64)  NULL,
    schema_version            VARCHAR(16)  NOT NULL,

    captcha_width             INT          NOT NULL,
    captcha_height            INT          NOT NULL,
    presented_at              DATETIME     NULL,
    submitted_at              DATETIME     NULL,

    position_correct          TINYINT(1)   NULL,       -- CAPTCHA pass signal, NOT a model feature
    interaction_success       TINYINT(1)   NULL,
    final_drop_error          DOUBLE       NULL,

    label                     VARCHAR(16)  NOT NULL DEFAULT 'unknown',  -- human | bot | unknown
    label_source              VARCHAR(32)  NULL,       -- controlled_collection|playwright|selenium|rule_bot|gan_bot|replay_bot
    bot_family                VARCHAR(64)  NULL,
    generator_version         VARCHAR(64)  NULL,
    age_group                 VARCHAR(16)  NOT NULL DEFAULT 'unknown',  -- adult | child | unknown
    consent_version           VARCHAR(32)  NULL,

    quality_status            VARCHAR(16)  NOT NULL DEFAULT 'pending',  -- valid | pending | rejected
    rejection_reason          VARCHAR(255) NULL,

    created_at                DATETIME     NOT NULL,
    updated_at                DATETIME     NOT NULL,

    PRIMARY KEY (attempt_id),
    KEY idx_attempts_challenge (challenge_id),
    KEY idx_attempts_session (session_id),
    KEY idx_attempts_participant (anonymous_participant_id),
    KEY idx_attempts_label (label),
    KEY idx_attempts_quality (quality_status),
    KEY idx_attempts_labelsrc (label_source)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2) Raw pointer events --------------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_pointer_events (
    event_id       BIGINT       NOT NULL AUTO_INCREMENT,
    attempt_id     VARCHAR(64)  NOT NULL,
    seq            INT          NOT NULL,
    event_type     VARCHAR(16)  NOT NULL,   -- pointerdown|pointermove|pointerup|pointercancel
    t_ms           INT          NOT NULL,   -- ms since drag start
    x              DOUBLE       NOT NULL,   -- CAPTCHA-area px
    y              DOUBLE       NOT NULL,
    x_normalized   DOUBLE       NULL,       -- 0..1
    y_normalized   DOUBLE       NULL,       -- 0..1
    target_role    VARCHAR(32)  NULL,       -- slider_handle|puzzle_piece|empty_area|...
    created_at     DATETIME     NOT NULL,

    PRIMARY KEY (event_id),
    UNIQUE KEY uq_attempt_seq (attempt_id, seq),
    KEY idx_events_attempt (attempt_id),
    CONSTRAINT fk_events_attempt FOREIGN KEY (attempt_id)
        REFERENCES ai_behavior_attempts (attempt_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 3) Interaction summary -------------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_interaction_summaries (
    attempt_id           VARCHAR(64) NOT NULL,
    regrab_count         INT NOT NULL DEFAULT 0,
    retry_count          INT NOT NULL DEFAULT 0,
    pointercancel_count  INT NOT NULL DEFAULT 0,
    empty_click_count    INT NOT NULL DEFAULT 0,
    failed_drop_count    INT NOT NULL DEFAULT 0,

    PRIMARY KEY (attempt_id),
    CONSTRAINT fk_summary_attempt FOREIGN KEY (attempt_id)
        REFERENCES ai_behavior_attempts (attempt_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 4) Behavioral features (29) --------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_attempt_features (
    attempt_id              VARCHAR(64) NOT NULL,
    feature_schema_version  VARCHAR(16) NOT NULL,

    -- A. basic (15)
    event_count             DOUBLE NULL,
    duration_ms             DOUBLE NULL,
    total_distance          DOUBLE NULL,
    displacement            DOUBLE NULL,
    avg_speed               DOUBLE NULL,
    max_speed               DOUBLE NULL,
    speed_std               DOUBLE NULL,
    avg_acceleration        DOUBLE NULL,
    max_acceleration        DOUBLE NULL,
    jerk_mean               DOUBLE NULL,
    direction_changes       DOUBLE NULL,
    pause_count             DOUBLE NULL,
    pause_ratio             DOUBLE NULL,
    linearity               DOUBLE NULL,
    y_deviation             DOUBLE NULL,

    -- B. interval (4)
    interval_mean_ms        DOUBLE NULL,
    interval_std_ms         DOUBLE NULL,
    interval_cv             DOUBLE NULL,
    duplicate_interval_ratio DOUBLE NULL,

    -- C. correction (5)
    overshoot_count         DOUBLE NULL,
    overshoot_distance      DOUBLE NULL,
    correction_count        DOUBLE NULL,
    endpoint_adjustment_time DOUBLE NULL,
    final_segment_speed     DOUBLE NULL,

    -- D. interaction (5)
    regrab_count            DOUBLE NULL,
    retry_count             DOUBLE NULL,
    pointercancel_count     DOUBLE NULL,
    empty_click_count       DOUBLE NULL,
    failed_drop_count       DOUBLE NULL,

    calculated_at           DATETIME NOT NULL,

    PRIMARY KEY (attempt_id),
    CONSTRAINT fk_features_attempt FOREIGN KEY (attempt_id)
        REFERENCES ai_behavior_attempts (attempt_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 5) Security / replay features ------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_security_features (
    attempt_id               VARCHAR(64) NOT NULL,
    path_similarity_score    DOUBLE  NULL,
    exact_replay_detected    TINYINT(1) NULL,
    repeated_duration_count  INT     NULL,
    attempts_per_minute      DOUBLE  NULL,
    recent_attempt_count     INT     NULL,
    repeated_endpoint_count  INT     NULL,
    calculated_at            DATETIME NOT NULL,

    PRIMARY KEY (attempt_id),
    CONSTRAINT fk_security_attempt FOREIGN KEY (attempt_id)
        REFERENCES ai_behavior_attempts (attempt_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 6) Model predictions ---------------------------------------------------------
CREATE TABLE IF NOT EXISTS ai_model_predictions (
    prediction_id           BIGINT      NOT NULL AUTO_INCREMENT,
    attempt_id              VARCHAR(64) NOT NULL,
    human_score             DOUBLE      NOT NULL,
    bot_risk_score          DOUBLE      NOT NULL,
    bot_decision            VARCHAR(16) NOT NULL,
    risk_score              DOUBLE      NOT NULL, -- advisory policy score, not P(bot)
    risk_level              VARCHAR(16) NOT NULL, -- low | medium | high
    recommended_action      VARCHAR(32) NOT NULL, -- allow | step_up | step_up_and_rate_limit
    policy_mode             VARCHAR(16) NOT NULL DEFAULT 'shadow', -- shadow | active
    risk_reasons            JSON        NOT NULL,
    threshold               DOUBLE      NOT NULL,
    model_name              VARCHAR(64) NOT NULL,
    model_version           VARCHAR(64) NOT NULL,
    feature_schema_version  VARCHAR(16) NOT NULL,
    predicted_at            DATETIME    NOT NULL,

    PRIMARY KEY (prediction_id),
    KEY idx_pred_attempt (attempt_id),
    KEY idx_pred_policy_mode (policy_mode),
    CONSTRAINT fk_pred_attempt FOREIGN KEY (attempt_id)
        REFERENCES ai_behavior_attempts (attempt_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 7) Shadow-mode observed outcomes ---------------------------------------------
CREATE TABLE IF NOT EXISTS ai_shadow_outcomes (
    attempt_id               VARCHAR(64) NOT NULL,
    main_captcha_verdict     VARCHAR(16) NOT NULL, -- passed | failed
    final_verdict            VARCHAR(16) NOT NULL, -- equals main verdict in shadow mode
    would_have_action        VARCHAR(32) NOT NULL, -- copied from ai_model_predictions
    risk_level               VARCHAR(16) NOT NULL,
    model_version            VARCHAR(64) NOT NULL,
    recorded_at              DATETIME    NOT NULL,

    PRIMARY KEY (attempt_id),
    KEY idx_shadow_action (would_have_action),
    CONSTRAINT fk_shadow_attempt FOREIGN KEY (attempt_id)
        REFERENCES ai_behavior_attempts (attempt_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 8) Training dataset view -----------------------------------------------------
-- Only quality_status='valid', label in ('human','bot'), label_source present.
-- Joins the 29 features and split-relevant metadata for the training pipeline.
CREATE OR REPLACE VIEW ai_training_dataset AS
SELECT
    a.attempt_id,
    a.challenge_id,
    a.session_id,
    a.anonymous_participant_id,
    a.label,
    a.label_source,
    a.bot_family,
    a.generator_version,
    a.age_group,
    a.schema_version,
    a.position_correct,
    a.interaction_success,
    a.final_drop_error,
    f.feature_schema_version,
    f.event_count, f.duration_ms, f.total_distance, f.displacement,
    f.avg_speed, f.max_speed, f.speed_std, f.avg_acceleration,
    f.max_acceleration, f.jerk_mean, f.direction_changes, f.pause_count,
    f.pause_ratio, f.linearity, f.y_deviation,
    f.interval_mean_ms, f.interval_std_ms, f.interval_cv, f.duplicate_interval_ratio,
    f.overshoot_count, f.overshoot_distance, f.correction_count,
    f.endpoint_adjustment_time, f.final_segment_speed,
    f.regrab_count, f.retry_count, f.pointercancel_count,
    f.empty_click_count, f.failed_drop_count
FROM ai_behavior_attempts a
JOIN ai_attempt_features f ON f.attempt_id = a.attempt_id
WHERE a.quality_status = 'valid'
  AND a.label IN ('human', 'bot')
  AND a.label_source IS NOT NULL;
