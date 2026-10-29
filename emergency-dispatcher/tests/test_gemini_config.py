"""Gemini Live configuration tests.

Hermetic: every key used here is a synthetic placeholder. No real secret is read,
asserted on, or printed, and no network call is made (genai.Client does not dial
during construction).
"""

from __future__ import annotations

import logging

import pytest
from fastapi.testclient import TestClient

from emergency_dispatcher import gemini_live
from emergency_dispatcher import server
from emergency_dispatcher.gemini_live import GEMINI_API_KEY_ENV
from emergency_dispatcher.gemini_live import GeminiConfigError
from emergency_dispatcher.gemini_live import _require_api_key
from emergency_dispatcher.gemini_live import build_gemini_client
from emergency_dispatcher.gemini_live import gemini_live_missing
from emergency_dispatcher.gemini_live import gemini_live_ready
from emergency_dispatcher.gemini_live import gemini_live_status
from emergency_dispatcher.server import app
from emergency_dispatcher.settings import Settings
from emergency_dispatcher.settings import get_settings


FAKE_KEY = "fake-gemini-key-for-tests-0001"
client = TestClient(app)


# --- settings loading -----------------------------------------------------

def test_settings_loads_gemini_api_key_from_exact_env_var(monkeypatch):
    monkeypatch.setenv(GEMINI_API_KEY_ENV, FAKE_KEY)
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key == FAKE_KEY
    assert GEMINI_API_KEY_ENV == "GEMINI_API_KEY"


def test_settings_missing_gemini_key_defaults_to_none(monkeypatch):
    monkeypatch.delenv(GEMINI_API_KEY_ENV, raising=False)
    settings = Settings(_env_file=None)
    assert settings.gemini_api_key is None


def test_existing_environment_variables_are_preserved():
    """Every pre-existing setting must still exist after the Gemini addition."""
    fields = set(Settings.model_fields)
    for name in (
        "gradium_api_key",
        "ai_coustics_api_key",
        "google_cloud_project",
        "google_cloud_location",
        "google_model",
        "openai_api_key",
        "deepgram_api_key",
        "soniox_api_key",
        "triage_engine",
        "live_dispatch_mode",
    ):
        assert name in fields, f"existing setting removed: {name}"
    # New Gemini settings are present with the documented defaults.
    assert Settings(_env_file=None).gemini_live_model == "gemini-2.5-flash-native-audio-preview-12-2025"


# --- readiness helpers (missing key path) ---------------------------------

def test_missing_key_reports_name_only():
    settings = Settings(_env_file=None)
    monkeyed = settings.model_copy(update={"gemini_api_key": None})
    assert gemini_live_missing(monkeyed) == ["GEMINI_API_KEY"]
    assert gemini_live_ready(monkeyed) is False


def test_missing_key_raises_clear_error_naming_variable():
    settings = Settings(_env_file=None).model_copy(update={"gemini_api_key": "  "})
    with pytest.raises(GeminiConfigError) as excinfo:
        _require_api_key(settings)
    message = str(excinfo.value)
    assert "GEMINI_API_KEY" in message
    assert "GEMINI_LIVE_MODEL" not in message  # message names only what's required


def test_ready_when_key_present():
    settings = Settings(_env_file=None).model_copy(update={"gemini_api_key": FAKE_KEY})
    assert gemini_live_missing(settings) == []
    assert gemini_live_ready(settings) is True


def test_build_gemini_client_uses_configured_key():
    settings = Settings(_env_file=None).model_copy(update={"gemini_api_key": FAKE_KEY})
    client_obj = build_gemini_client(settings)
    assert client_obj is not None


def test_build_gemini_client_without_key_fails_before_any_request():
    settings = Settings(_env_file=None).model_copy(update={"gemini_api_key": None})
    with pytest.raises(GeminiConfigError) as excinfo:
        build_gemini_client(settings)
    assert "GEMINI_API_KEY" in str(excinfo.value)


# --- status payload never leaks the key -----------------------------------

def test_status_payload_is_names_only():
    settings = Settings(_env_file=None).model_copy(update={"gemini_api_key": FAKE_KEY})
    status = gemini_live_status(settings)
    assert status["api_key_env"] == "GEMINI_API_KEY"
    assert status["ready"] is True
    assert status["api_key_configured"] is True
    assert status["api_key_preview"] == "[redacted]"
    # The fake key must appear nowhere in the serialised status.
    assert FAKE_KEY not in repr(status)


def test_status_payload_when_missing_has_no_secrets():
    settings = Settings(_env_file=None).model_copy(update={"gemini_api_key": None})
    status = gemini_live_status(settings)
    assert status["missing_config"] == ["GEMINI_API_KEY"]
    assert status["ready"] is False
    assert FAKE_KEY not in repr(status)


# --- HTTP surface: no key ever reaches the frontend -----------------------

def test_audio_config_never_contains_the_key(monkeypatch):
    settings = get_settings().model_copy(update={"gemini_api_key": FAKE_KEY})
    monkeypatch.setattr(server, "get_settings", lambda: settings)
    monkeypatch.setattr(gemini_live, "get_settings", lambda: settings)
    body = client.get("/api/audio-config").json()
    assert body["gemini_live_ready"] is True
    assert body["gemini_live_missing_config"] == []
    assert FAKE_KEY not in client.get("/api/audio-config").text


def test_gemini_status_endpoint_never_contains_the_key(monkeypatch):
    settings = get_settings().model_copy(update={"gemini_api_key": FAKE_KEY})
    monkeypatch.setattr(server, "get_settings", lambda: settings)
    monkeypatch.setattr(gemini_live, "get_settings", lambda: settings)
    response = client.get("/api/gemini-live/status")
    assert response.status_code == 200
    body = response.json()
    assert body["api_key_env"] == "GEMINI_API_KEY"
    assert body["api_key_preview"] == "[redacted]"
    assert FAKE_KEY not in response.text


def test_gemini_status_endpoint_reports_missing_name_when_unset(monkeypatch):
    settings = get_settings().model_copy(update={"gemini_api_key": None})
    monkeypatch.setattr(server, "get_settings", lambda: settings)
    monkeypatch.setattr(gemini_live, "get_settings", lambda: settings)
    body = client.get("/api/gemini-live/status").json()
    assert body["ready"] is False
    assert body["missing_config"] == ["GEMINI_API_KEY"]


# --- startup logging: clear error, never the key --------------------------

def test_startup_logs_clear_error_when_key_missing(monkeypatch, caplog):
    settings = get_settings().model_copy(update={"gemini_api_key": None})
    monkeypatch.setattr(server, "get_settings", lambda: settings)
    monkeypatch.setattr(gemini_live, "get_settings", lambda: settings)
    with caplog.at_level(logging.INFO, logger="emergency_dispatcher.server"):
        with client:
            pass
    assert "GEMINI_API_KEY" in caplog.text
    assert "not configured" in caplog.text
    assert FAKE_KEY not in caplog.text


def test_startup_logs_presence_but_never_value_when_key_set(monkeypatch, caplog):
    settings = get_settings().model_copy(update={"gemini_api_key": FAKE_KEY})
    monkeypatch.setattr(server, "get_settings", lambda: settings)
    monkeypatch.setattr(gemini_live, "get_settings", lambda: settings)
    with caplog.at_level(logging.INFO, logger="emergency_dispatcher.server"):
        with client:
            pass
    assert "Gemini Live API configured" in caplog.text
    assert FAKE_KEY not in caplog.text
