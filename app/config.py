"""Application configuration.

All connection info and secrets are read from environment variables only.
Nothing here is hardcoded; missing secrets simply disable the guarded path
(e.g. an unset COLLECT_API_KEY makes /collect reject every request).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.secrets_loader import load_secrets_into_env


class Settings(BaseSettings):
    """Environment-backed settings. See `.env.example` for the full list."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- MySQL ---
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_user: str = "ai_service"
    mysql_password: str = ""
    # ⚠️★기본값을 "catchap" 에서 "catchap_ai" 로 바꿨다 (2026-08-10).
    #
    #   운영은 ConfigMap 이 MYSQL_DATABASE=catchap_ai 를 주므로 지금도 옳은 DB 에 붙는다.
    #   ★문제는 그 값이 빠졌을 때다 — 옛 기본값 "catchap" 에는 ★2026-08-09 컷오버 때
    #   복원된 ai_* 표 10개가 ★그대로 있다(7/29~7/30 데이터 9행).
    #   그러면 앱이 ★오류 없이 뜨고, ★엉뚱한 표에 쓰기 시작한다. 아무도 모른다.
    #
    #   ★"틀리면 시끄럽게 죽는다"가 안 되는 자리라, ★맞는 곳을 기본값으로 둔다.
    mysql_database: str = "catchap_ai"
    mysql_pool_size: int = 5

    # --- Auth (interfaces; empty means "no key configured") ---
    collect_api_key: str = ""
    admin_api_key: str = ""
    captcha_backend_api_key: str = ""

    # --- CAPTCHA protocol security ---
    captcha_challenge_ttl_seconds: int = 120
    captcha_challenge_max_ttl_seconds: int = 300

    # --- Model serving ---
    production_model_dir: str = "models/production"
    default_threshold: float = 0.55

    # --- Scoring unit ---
    # "session" scores the whole trajectory at once, which is what the model was
    # trained and calibrated on. "per_drag" scores each press-to-release and takes
    # the median, which removes the interaction-scale lever: session aggregates
    # grow with the number of drags, so a ruler-straight path passed by being
    # repeated (measured 2026-07-31).
    #
    # On main-captcha data (411 attempts we can reproduce production scores for):
    #   session,  threshold 0.99995   human FRR 16.4%   bot ASR 14.2%
    #   session,  threshold 0.05      human FRR  1.5%   bot ASR 45.8%
    #   per_drag, threshold 0.01      human FRR  1.5%   bot ASR  0.0%
    #
    # Default stays "session": that number was chosen on the same data it is
    # measured on, so it is a candidate until a sealed holdout scores it. Both
    # scores are recorded either way, so flipping this is a config change with
    # evidence behind it rather than a leap.
    scoring_unit: str = "session"
    per_drag_threshold: float = 0.01

    # --- Advisory risk policy ---
    # These are policy weights, not calibrated probabilities. The backend uses
    # the returned action to step up verification; it remains the decision maker.
    # Read against `ProcrustesPathComparator` since 2026-08-10 — the name still
    # says dtw because the deployed ConfigMap sets this key, and renaming it
    # would silently fall back to the default in production.
    #
    # Calibrated on the surface it actually runs on: 593 collection drags from
    # five people. Innocent cross-person pairs top out at 0.9805; replays
    # rotated onto a new target bottom out at 0.9846.
    #
    #     0.98  ->  98.4% of rotated replays caught, 0.017% of human pairs hit
    #     0.99  ->  83.5%                          , 0.000%
    #
    # 0.99 is the conservative pick. Per-pair false hits accumulate over a
    # session (10 attempts is 45 pairs), and this signal steps up verification
    # for real people, so the pair-level rate is what has to stay near zero.
    # The old 0.996693 belonged to DTW, whose scores for the same rotated
    # replays sat around 0.61 — that threshold caught 4.2% of them.
    risk_dtw_similarity_threshold: float = 0.99
    risk_max_attempts_per_minute: float = 20.0
    risk_history_window_seconds: int = 60
    risk_history_max_attempts: int = 50
    # Shadow is the safe default: the backend records what the AI would have
    # requested but does not alter the current CAPTCHA outcome.
    risk_policy_mode: Literal["shadow", "active"] = "shadow"

    # --- Schema versions ---
    api_schema_version: str = "1.0"
    feature_schema_version: str = "1.0"

    # --- Training readiness gates (project defaults, not research thresholds) ---
    min_human_samples: int = 500
    min_bot_samples: int = 500
    min_human_participants: int = 0
    min_bot_families: int = 3

    # --- GAN readiness gates ---
    min_gan_human_samples: int = 2000
    min_gan_human_participants: int = 50

    @property
    def sqlalchemy_url(self) -> str:
        """Build the SQLAlchemy URL from parts (password never logged)."""
        from urllib.parse import quote_plus

        pwd = quote_plus(self.mysql_password)
        return (
            f"mysql+pymysql://{self.mysql_user}:{pwd}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
            "?charset=utf8mb4"
        )


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (one instance per process)."""
    # ★Settings 를 만들기 전에 Secrets Manager 를 읽어 환경변수로 넣는다.
    # 비밀값이 Settings 의 재료라 순서가 반대면 의미가 없다.
    # SECRETS_BACKEND 기본값이 none 이라 로컬 개발·시험에서는 아무 일도 일어나지 않는다.
    load_secrets_into_env()
    return Settings()
