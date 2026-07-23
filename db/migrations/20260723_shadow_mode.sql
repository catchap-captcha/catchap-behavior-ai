-- Apply once to an existing MySQL 8.0 ai-service schema before shadow testing.
-- The application account does not need DDL privileges; DB admin applies this.

ALTER TABLE ai_model_predictions
  ADD COLUMN policy_mode VARCHAR(16) NOT NULL DEFAULT 'shadow' AFTER recommended_action,
  ADD KEY idx_pred_policy_mode (policy_mode);

CREATE TABLE ai_shadow_outcomes (
    attempt_id               VARCHAR(64) NOT NULL,
    main_captcha_verdict     VARCHAR(16) NOT NULL,
    final_verdict            VARCHAR(16) NOT NULL,
    would_have_action        VARCHAR(32) NOT NULL,
    risk_level               VARCHAR(16) NOT NULL,
    model_version            VARCHAR(64) NOT NULL,
    recorded_at              DATETIME    NOT NULL,

    PRIMARY KEY (attempt_id),
    KEY idx_shadow_action (would_have_action),
    CONSTRAINT fk_shadow_attempt FOREIGN KEY (attempt_id)
        REFERENCES ai_behavior_attempts (attempt_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
