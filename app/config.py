"""Application configuration.

All connection info and secrets are read from environment variables only.
Nothing here is hardcoded; missing secrets simply disable the guarded path
(e.g. an unset COLLECT_API_KEY makes /collect reject every request).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    mysql_database: str = "catchap"
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

    # --- Advisory risk policy ---
    # These are policy weights, not calibrated probabilities. The backend uses
    # the returned action to step up verification; it remains the decision maker.
    risk_dtw_similarity_threshold: float = 0.996693
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
    return Settings()
