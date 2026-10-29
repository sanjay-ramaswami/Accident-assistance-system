import asyncio
from pathlib import Path
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from emergency_dispatcher import server
from emergency_dispatcher import session_store
from emergency_dispatcher.server import app
from emergency_dispatcher.settings import get_settings


client = TestClient(app)


def test_dispatch_simulation_reuses_state_until_inputs_or_milestone_changes():
    now = datetime(2026, 10, 29, tzinfo=UTC)
    stable = {
        "session_id": "incident-1",
        "plan": {"services": {"ambulance": {"needed": True}}},
        "location_text": "Dubai, UAE",
        "location_confirmed": False,
        "response_bases": {},
    }
    latest = {"args": stable}
    state = {"services": {"ambulance": {"activated_at": now.isoformat(), "status": "locating"}}}

    assert server._dispatch_simulation_due(latest, state, stable, now=now) is False
    assert server._dispatch_simulation_due(latest, state, stable, now=now + timedelta(seconds=1.5)) is True
    changed = stable | {"location_text": "Abu Dhabi, UAE"}
    assert server._dispatch_simulation_due(latest, state, changed, now=now) is True


def test_unchanged_auto_dispatch_update_does_not_rerun_simulator(monkeypatch):
    now = datetime.now(UTC)
    dispatch_inputs = {
        "transcript": "There was an accident and someone is injured.",
        "issue_type": "TRAFFIC",
        "priority": "HIGH",
        "issue_cues": ["injury"],
        "lat": None,
        "lon": None,
        "location_confirmed": False,
        "location_text": None,
        "dispatch_location_text": None,
    }
    plan = {"priority": "HIGH", "needed_services": [], "services": {}}
    stable_args = {
        "session_id": "stable-incident",
        "plan": plan,
        "location_text": None,
        "location_confirmed": False,
        "response_bases": {},
    }
    prior = {
        "services": {"ambulance": {"activated_at": now.isoformat(), "status": "locating"}},
        "response_bases": {},
        "dispatchable": False,
        "version": 3,
    }
    session = {
        "session_id": "stable-incident",
        "server": {"live_dispatch_mode": "llm_only"},
        "dispatch_services": prior,
        "tool_calls": [{"name": "simulate_dispatch_services", "args": stable_args, "result": prior}],
    }
    calls = []
    monkeypatch.setattr(server, "_derive_dispatch_inputs", lambda _session: dispatch_inputs)

    def fake_dispatch(name, _args):
        calls.append(name)
        return plan

    monkeypatch.setattr(server, "dispatch_tool_call", fake_dispatch)

    result = server._auto_dispatch_services(session, None)

    assert calls == ["plan_response_services"]
    assert result["version"] == 3


def _settings_without_live_credentials():
    return get_settings().model_copy(
        update={
            "gradium_api_key": None,
            "ai_coustics_api_key": None,
            "openai_api_key": None,
            "openai_model": None,
            "triage_api_key": None,
            "triage_decoder_model": None,
            "live_fact_ledger_model": None,
            "live_dispatch_mode": "llm_only",
        }
    )


async def _preflight_ok(_settings):
    """Keep WebSocket tests hermetic: never probe the real LLM endpoint."""
    return {"status": "ok"}


def test_audio_config_reports_exact_missing_live_dependencies(monkeypatch):
    monkeypatch.setattr(server, "get_settings", _settings_without_live_credentials)

    response = client.get("/api/audio-config?triage_mode=slm")

    assert response.status_code == 200
    payload = response.json()
    assert payload["default_dispatch_mode"] == "llm_only"
    assert payload["live_voice_ready"] is False
    assert payload["live_voice_missing_config"] == [
        "GRADIUM_API_KEY",
        "AI_COUSTICS_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_API_KEY",
    ]
    assert payload["triage_ready"] is False
    assert payload["triage_missing_config"] == ["TRIAGE_API_KEY", "LIVE_FACT_LEDGER_MODEL"]


def test_websocket_rejects_before_accept_with_actionable_credential_reason(monkeypatch):
    monkeypatch.setattr(server, "get_settings", _settings_without_live_credentials)

    with pytest.raises(WebSocketDisconnect) as disconnected:
        with client.websocket_connect("/ws"):
            pytest.fail("WebSocket must not accept a live session without its providers")

    assert disconnected.value.code == 1008
    assert "GRADIUM_API_KEY" in disconnected.value.reason
    assert "AI_COUSTICS_API_KEY" in disconnected.value.reason
    assert "OPENAI_API_KEY" in disconnected.value.reason


def test_websocket_rejects_non_object_start_message(monkeypatch):
    ready_settings = get_settings().model_copy(
        update={
            "gradium_api_key": "test-only",
            "ai_coustics_api_key": "test-only",
            "openai_api_key": "test-only",
            "openai_model": "test-only",
        }
    )
    monkeypatch.setattr(server, "_ensure_llm_ready", lambda: ready_settings)
    monkeypatch.setattr(server, "make_enhancer", lambda settings: object())

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(["not", "a", "start-object"])
        closed = websocket.receive()

    assert closed["type"] == "websocket.close"
    assert closed["code"] == 1008
    assert closed["reason"] == "Expected a JSON start message"


def test_websocket_reports_provider_initialization_failure_without_exception_text(monkeypatch, tmp_path):
    ready_settings = get_settings().model_copy(
        update={
            "gradium_api_key": "test-only",
            "ai_coustics_api_key": "test-only",
            "openai_api_key": "test-only",
            "openai_model": "test-only",
        }
    )
    monkeypatch.setattr(server, "_ensure_llm_ready", lambda: ready_settings)
    monkeypatch.setattr(server, "make_enhancer", lambda settings: object())
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(server, "_llm_preflight_check", _preflight_ok)

    async def fail_provider_start(**kwargs):
        raise RuntimeError("private-token-must-not-be-returned")

    monkeypatch.setattr(server.gradbot, "run", fail_provider_start)
    session_id = "provider-init-failure"

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "session_id": session_id,
                "triage_mode": "llm_only",
                "client": {},
            }
        )
        error = websocket.receive_json()
        closed = websocket.receive()

    assert error == {
        "type": "error",
        "code": "provider_error",
        "message": "Live provider initialization failed (RuntimeError).",
    }
    assert "private-token" not in error["message"]
    assert closed["type"] == "websocket.close"
    assert closed["code"] == 1011
    saved = session_store.get_session(session_id)
    assert saved["server"]["status"] == "error"
    assert saved["server"]["provider_initialization_error"] == "RuntimeError"


def test_websocket_marks_connected_only_after_provider_initializes(monkeypatch, tmp_path):
    ready_settings = get_settings().model_copy(
        update={
            "gradium_api_key": "test-only",
            "ai_coustics_api_key": "test-only",
            "openai_api_key": "test-only",
            "openai_model": "test-only",
        }
    )
    monkeypatch.setattr(server, "_ensure_llm_ready", lambda: ready_settings)
    monkeypatch.setattr(server, "make_enhancer", lambda settings: type("Enhancer", (), {"flush": lambda self: b""})())
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(server, "_llm_preflight_check", _preflight_ok)
    initialized = False

    class InputHandle:
        def __init__(self):
            self.closed = asyncio.Event()

        async def close(self):
            self.closed.set()
            return None

    class OutputHandle:
        def __init__(self, input_handle):
            self.input_handle = input_handle

        async def receive(self):
            await self.input_handle.closed.wait()
            return None

    async def initialize_provider(**kwargs):
        nonlocal initialized
        initialized = True
        input_handle = InputHandle()
        return input_handle, OutputHandle(input_handle)

    async def no_artifacts(**kwargs):
        return {}

    monkeypatch.setattr(server.gradbot, "run", initialize_provider)
    monkeypatch.setattr(server, "finalize_audio_artifacts", no_artifacts)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "session_id": "provider-init-success",
                "triage_mode": "llm_only",
                "client": {},
            }
        )
        ready = websocket.receive_json()
        websocket.send_json({"type": "stop"})
        closed = websocket.receive()

    assert initialized is True
    assert ready == {"type": "connection_status", "status": "connected"}
    assert closed["type"] == "websocket.close"


def test_websocket_records_provider_output_eof_without_private_error_text(monkeypatch, tmp_path, caplog):
    ready_settings = get_settings().model_copy(
        update={
            "gradium_api_key": "test-only",
            "ai_coustics_api_key": "test-only",
            "openai_api_key": "test-only",
            "openai_model": "test-only",
        }
    )
    monkeypatch.setattr(server, "_ensure_llm_ready", lambda: ready_settings)
    monkeypatch.setattr(server, "make_enhancer", lambda settings: type("Enhancer", (), {"flush": lambda self: b""})())
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(server, "_llm_preflight_check", _preflight_ok)

    class InputHandle:
        async def close(self):
            return None

    class OutputHandle:
        async def receive(self):
            if not hasattr(self, "sent"):
                self.sent = 1
                return SimpleNamespace(
                    msg_type="stt_text",
                    text="PRIVATE_CALLER_TRANSCRIPT_SENTINEL",
                    start_s=0.0,
                    stop_s=1.0,
                    turn_idx=1,
                    interrupted=False,
                    event=None,
                    data=None,
                )
            if self.sent == 1:
                self.sent = 2
                return SimpleNamespace(msg_type="event", event=SimpleNamespace(event_type="llm_started"), text=None, data=None)
            if self.sent == 2:
                self.sent = 3
                return SimpleNamespace(
                    msg_type="tts_text",
                    text="PRIVATE_ASSISTANT_TEXT_SENTINEL",
                    start_s=1.0,
                    stop_s=2.0,
                    turn_idx=1,
                    interrupted=False,
                    event=None,
                    data=None,
                )
            if self.sent == 3:
                self.sent = 4
                return SimpleNamespace(
                    msg_type="audio",
                    text=None,
                    start_s=1.0,
                    stop_s=1.1,
                    turn_idx=1,
                    interrupted=False,
                    event=None,
                    data=b"\x00\x00\x01\x00",
                )
            if self.sent == 4:
                self.sent = 5
                return SimpleNamespace(msg_type="event", event=SimpleNamespace(event_type="end_of_turn"), text=None, data=None)
            return None

    async def initialize_provider(**kwargs):
        return InputHandle(), OutputHandle()

    async def no_artifacts(**kwargs):
        return {}

    monkeypatch.setattr(server.gradbot, "run", initialize_provider)
    monkeypatch.setattr(server, "finalize_audio_artifacts", no_artifacts)
    caplog.set_level("INFO", logger="emergency_dispatcher.server")
    session_id = "provider-output-eof"

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "session_id": session_id,
                "triage_mode": "llm_only",
                "client": {},
            }
        )
        ready = websocket.receive_json()
        caller_text = websocket.receive_json()
        llm_started = websocket.receive_json()
        assistant_text = websocket.receive_json()
        audio_timing = websocket.receive_json()
        audio_bytes = websocket.receive()
        end_of_turn = websocket.receive_json()
        error = websocket.receive_json()
        closed = websocket.receive()

    assert ready == {"type": "connection_status", "status": "connected"}
    assert caller_text["type"] == "user_text"
    assert llm_started == {"type": "event", "event": "llm_started"}
    assert assistant_text["type"] == "agent_text"
    assert audio_timing["type"] == "audio_timing"
    assert audio_bytes["type"] == "websocket.send"
    assert audio_bytes["bytes"] == b"\x00\x00\x01\x00"
    assert end_of_turn == {"type": "event", "event": "end_of_turn"}
    assert error == {
        "type": "error",
        "code": "provider_output_eof",
        "message": "Provider output ended before the live response completed.",
    }
    assert closed["type"] == "websocket.close"
    assert closed["code"] == 1011
    diagnostics = session_store.get_session(session_id)["server"]["provider_diagnostics"]
    assert diagnostics["stream_close"] == {
        "reason": "provider_output_eof",
        "websocket_code": 1011,
        "response_completed": True,
    }
    assert diagnostics["input_audio_format"] == "s16le_mono_24000hz"
    assert diagnostics["caller_transcript_events"] == 1
    assert diagnostics["assistant_text_events"] == 1
    assert diagnostics["assistant_audio_frames_forwarded"] == 1
    assert diagnostics["assistant_audio_bytes_forwarded"] == 4
    assert diagnostics["llm_started_events"] == 1
    assert diagnostics["end_of_turn_events"] == 1
    assert diagnostics["completed_llm_responses"] == 1
    assert diagnostics["provider_event_counts"] == {"llm_started": 1, "end_of_turn": 1}
    assert "PRIVATE" not in repr(diagnostics)
    assert "PRIVATE_CALLER_TRANSCRIPT_SENTINEL" not in repr(session_store.get_session(session_id)["server"])
    assert "PRIVATE_CALLER_TRANSCRIPT_SENTINEL" not in caplog.text


def test_websocket_classifies_openai_quota_failure_and_does_not_retry(monkeypatch, tmp_path, caplog):
    ready_settings = get_settings().model_copy(
        update={
            "gradium_api_key": "test-only",
            "ai_coustics_api_key": "test-only",
            "openai_api_key": "test-only",
            "openai_model": "test-only",
        }
    )
    monkeypatch.setattr(server, "_ensure_llm_ready", lambda: ready_settings)
    monkeypatch.setattr(server, "make_enhancer", lambda settings: type("Enhancer", (), {"flush": lambda self: b""})())
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(server, "_llm_preflight_check", _preflight_ok)

    class InputHandle:
        async def close(self):
            return None

    class QuotaError(Exception):
        status_code = 429
        body = {
            "error": {
                "type": "insufficient_quota",
                "code": "credit_balance_exhausted",
                "message": "PRIVATE_QUOTA_ERROR_SENTINEL",
            }
        }

    class OutputHandle:
        def __init__(self):
            self.calls = 0

        async def receive(self):
            self.calls += 1
            raise QuotaError("PRIVATE_EXCEPTION_SENTINEL")

    output_handle = OutputHandle()

    async def initialize_provider(**kwargs):
        return InputHandle(), output_handle

    async def no_artifacts(**kwargs):
        return {}

    monkeypatch.setattr(server.gradbot, "run", initialize_provider)
    monkeypatch.setattr(server, "finalize_audio_artifacts", no_artifacts)
    caplog.set_level("INFO", logger="emergency_dispatcher.server")
    session_id = "provider-quota-failure"

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "session_id": session_id,
                "triage_mode": "llm_only",
                "client": {},
            }
        )
        ready = websocket.receive_json()
        error = websocket.receive_json()
        closed = websocket.receive()

    assert ready == {"type": "connection_status", "status": "connected"}
    assert error == {
        "type": "error",
        "code": "llm_quota_exhausted",
        "message": "The language model quota is exhausted; no live response was generated.",
    }
    assert closed["type"] == "websocket.close"
    assert closed["code"] == 1011
    assert output_handle.calls == 1
    diagnostics = session_store.get_session(session_id)["server"]["provider_diagnostics"]
    assert diagnostics["last_failure"] == {
        "provider": "openai",
        "category": "llm_quota_exhausted",
        "exception_type": "QuotaError",
        "http_status": 429,
        "error_type": "insufficient_quota",
        "error_code": "credit_balance_exhausted",
        "stage": "provider_output",
    }
    assert "PRIVATE" not in repr(diagnostics)
    assert "PRIVATE" not in caplog.text


def test_session_report_is_saved_without_claiming_unavailable_slm(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "get_settings", _settings_without_live_credentials)
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")
    report = {
        "session_id": "test-session-report-no-slm",
        "status": "setup_error",
        "triage_mode": "slm",
        "transcripts": {"user": [], "agent": []},
    }

    response = client.post("/api/session-report", json=report)

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "saved"
    assert payload["triage_enabled"] is False
    assert payload["dispatch_update"] is None
    assert payload["triage_meta"]["runtime_provider"] == "unavailable"
    assert payload["triage_meta"]["missing_config"] == ["TRIAGE_API_KEY", "LIVE_FACT_LEDGER_MODEL"]
    saved = session_store.get_session(report["session_id"])
    assert saved["client"]["status"] == "setup_error"
    assert saved["server"]["triage_status"] == "unavailable"
    assert saved["server"]["triage_unavailable_reason"] == (
        "Live triage unavailable; missing configuration: TRIAGE_API_KEY, LIVE_FACT_LEDGER_MODEL."
    )
    repeated_response = client.post("/api/session-report", json=report)
    assert repeated_response.status_code == 200
    assert len(list((tmp_path / "sessions").glob("*.json"))) == 1


def test_session_report_persists_sanitized_browser_audio_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "get_settings", lambda: get_settings().model_copy(update={"live_dispatch_mode": "llm_only"}))
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")
    response = client.post(
        "/api/session-report",
        json={
            "session_id": "browser-audio-diagnostics",
            "triage_mode": "llm_only",
            "status": "closed",
            "transcripts": {"user": [], "agent": []},
            "events": [
                {"text": "PRIVATE_TRANSCRIPT_SENTINEL"},
                {"text": "audio stream or playback failed (Error)"},
                {"text": "websocket closed (1011)"},
            ],
        },
    )

    assert response.status_code == 200
    saved = session_store.get_session("browser-audio-diagnostics")
    diagnostics = saved["server"]["browser_audio_diagnostics"]
    assert diagnostics["playback_error_types"] == {"Error": 1}
    assert diagnostics["websocket_close_codes"] == [1011]
    assert "PRIVATE_TRANSCRIPT_SENTINEL" not in repr(saved["server"])


def test_session_report_rejects_unsafe_session_id_before_persistence(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "get_settings", _settings_without_live_credentials)
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")

    response = client.post(
        "/api/session-report",
        json={"session_id": "../outside", "triage_mode": "llm_only"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid session_id"
    assert not (tmp_path / "outside.json").exists()


def test_session_report_returns_sanitized_error_when_storage_write_fails(monkeypatch, tmp_path):
    session_dir = tmp_path / "sessions"
    monkeypatch.setattr(session_store, "SESSION_DIR", session_dir)
    original_replace = session_store.os.replace

    def fail_session_write(source, target):
        if Path(target).parent == session_dir:
            raise OSError("private filesystem detail")
        return original_replace(source, target)

    monkeypatch.setattr(session_store.os, "replace", fail_session_write)

    response = client.post(
        "/api/session-report",
        json={"session_id": "storage-failure", "triage_mode": "llm_only"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Session storage is unavailable; the report could not be saved."}
    assert "private filesystem detail" not in response.text
