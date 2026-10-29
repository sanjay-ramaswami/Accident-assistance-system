from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"


def load_project_environment(env_file: Path = ENV_FILE) -> Path:
    """Load the repository dotenv file without replacing nonempty shell values."""
    for name, value in dotenv_values(env_file).items():
        if value is not None and not os.environ.get(name):
            os.environ[name] = value
    return env_file


load_project_environment()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    gradium_api_key: str | None = Field(None, alias="GRADIUM_API_KEY")
    ai_coustics_api_key: str | None = Field(None, alias="AI_COUSTICS_API_KEY")
    google_cloud_project: str | None = Field(None, alias="GOOGLE_CLOUD_PROJECT")
    google_cloud_location: str = Field("us-central1", alias="GOOGLE_CLOUD_LOCATION")
    google_model: str = Field("gemini-2.5-flash", alias="GOOGLE_MODEL")
    # Gemini Live API (google-genai SDK). The SDK also accepts GOOGLE_API_KEY, but
    # this project standardises on GEMINI_API_KEY to avoid colliding with GCP
    # Application Default Credentials (GOOGLE_CLOUD_PROJECT/... are already in use).
    gemini_api_key: str | None = Field(None, alias="GEMINI_API_KEY")
    gemini_live_model: str = Field("gemini-2.5-flash-native-audio-preview-12-2025", alias="GEMINI_LIVE_MODEL")
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(None, alias="OPENAI_BASE_URL")
    openai_model: str | None = Field(None, alias="OPENAI_MODEL")
    openai_transcription_model: str = "gpt-4o-mini-transcribe"
    triage_api_key: str | None = Field(None, alias="TRIAGE_API_KEY")
    triage_base_url: str = Field("https://api.pioneer.ai/v1", alias="TRIAGE_BASE_URL")
    triage_decoder_model: str | None = Field(None, alias="TRIAGE_DECODER_MODEL")
    live_fact_ledger_model: str | None = Field(None, alias="LIVE_FACT_LEDGER_MODEL")
    triage_gliner_model: str | None = Field(None, alias="TRIAGE_GLINER_MODEL")
    triage_gliner_live_enabled: bool = Field(False, alias="TRIAGE_GLINER_LIVE_ENABLED")
    triage_gliner_timeout_s: float = Field(2.0, alias="TRIAGE_GLINER_TIMEOUT_S")
    triage_request_timeout_s: float = Field(15.0, alias="TRIAGE_REQUEST_TIMEOUT_S")
    deepgram_api_key: str | None = Field(None, alias="DEEPGRAM_API_KEY")
    soniox_api_key: str | None = Field(None, alias="SONIOX_API_KEY")
    stt_rescue_continuous_compare: bool = Field(False, alias="STT_RESCUE_CONTINUOUS_COMPARE")
    tavily_api_key: str | None = Field(None, alias="TAVILY_API_KEY")
    telli_api_key: str | None = Field(None, alias="TELLI_API_KEY")
    ai_coustics_model_id: str = "quail-vf-2.1-l-16khz"
    ai_coustics_model_dir: Path = REPO_ROOT / ".cache" / "ai-coustics-models"
    triage_engine: str = Field("heuristic", alias="TRIAGE_ENGINE")
    live_dispatch_mode: str = Field("slm", alias="LIVE_DISPATCH_MODE")


def get_settings() -> Settings:
    return Settings()
