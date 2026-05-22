"""
config/settings.py
Centralised environment-variable management using Pydantic v2 BaseSettings.
All secrets are read from the environment (or a .env file) – never hard-coded.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ────────────────────────────────────────────────────────────
    app_name: str = Field(default="EscalationSync", description="Service display name")
    app_version: str = Field(default="1.0.0")
    environment: Literal["development", "staging", "production"] = Field(
        default="development"
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )

    # ── API Server ─────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)
    api_workers: int = Field(default=1, ge=1)

    # ── LLM Provider Keys ──────────────────────────────────────────────────────
    google_api_key: SecretStr = Field(
        default=SecretStr("REPLACE_ME"),
        description="Google AI Studio / Vertex API key for Gemini models",
    )
    anthropic_api_key: SecretStr = Field(
        default=SecretStr("REPLACE_ME"),
        description="Anthropic API key for Claude models",
    )

    # ── Observability ──────────────────────────────────────────────────────────
    langfuse_secret_key: SecretStr = Field(
        default=SecretStr(""),
        description="Langfuse secret key (leave blank to disable tracing)",
    )
    langfuse_public_key: str = Field(
        default="",
        description="Langfuse public key",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse ingestion endpoint",
    )

    # ── LangGraph / Agent Tuning ───────────────────────────────────────────────
    triage_model: str = Field(default="gemini-1.5-flash")
    escalation_model: str = Field(default="claude-3-5-sonnet-20241022")
    resolution_model: str = Field(default="gemini-1.5-flash")

    # Retry configuration (applied to every LLM node)
    llm_max_retries: int = Field(default=3, ge=1, le=10)
    llm_retry_wait_seconds: float = Field(default=2.0, ge=0.5)

    # Routing thresholds
    escalation_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    # ── n8n Webhook security (optional shared secret) ──────────────────────────
    n8n_webhook_secret: SecretStr = Field(
        default=SecretStr(""),
        description="If set, every inbound request must include X-N8N-Signature header",
    )

    @field_validator("api_port", mode="before")
    @classmethod
    def _coerce_port(cls, v: object) -> int:
        return int(v)  # type: ignore[arg-type]

    @property
    def observability_enabled(self) -> bool:
        """True when Langfuse credentials are configured."""
        return bool(
            self.langfuse_secret_key.get_secret_value()
            and self.langfuse_public_key
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
