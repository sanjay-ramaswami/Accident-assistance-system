"""Live-voice LLM failure surfacing: pre-flight probe and runtime watchdog.

The gradbot native SDK swallows upstream LLM failures (they only appear in its
native stderr logs), so this server probes the LLM once before starting a
session and watches for push cycles that never start an LLM response. These
tests pin both behaviors with hermetic fakes; no real provider is contacted.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from emergency_dispatcher import server
from emergency_dispatcher import session_store
from emergency_dispatcher.server import app
from emergency_dispatcher.settings import get_settings


client = TestClient(app)


def _ready_settings():
    return get_settings().model_copy(
        update={
            "gradium_api_key": "test-only",
            "ai_coustics_api_key": "test-only",
            "openai_api_key": "test-only",
            "openai_model": "test-only",
        }
    )


async def _preflight_ok(_settings):
    return {"status": "ok"}


async def _preflight_quota(_settings):
    return {
        "status": "failed",
        "http_status": 429,
        "error_type": "insufficient_quota",
        "error_code": "credit_balance_exhausted",
        "quota_exhausted": True,
    }


def _event(name):
    return SimpleNamespace(
        msg_type="event",
        event=SimpleNamespace(event_type=name),
        text=None,
        data=None,
    )


class _QuietInput:
    async def close(self):
        return None


# ---------------------------------------------------------------------------
# _llm_preflight_check unit behavior
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, status_code: int, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {}

    def json(self):
        return self._body


class _FakeAsyncClient:
    def __init__(self, *, response=None, exc=None, capture=None):
        self._response = response
        self._exc = exc
        self._capture = capture if capture is not None else {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def post(self, url, headers=None, json=None):
        self._capture["url"] = url
        self._capture["headers"] = headers or {}
        self._capture["json"] = json
        if self._exc is not None:
            raise self._exc
        return self._response


def _preflight_settings():
    return get_settings().model_copy(
        update={
            "openai_api_key": "test-only",
            "openai_model": "test-model",
            "openai_base_url": "https://llm.example/v1",
        }
    )


def test_preflight_success_probes_chat_completions_with_bearer(monkeypatch):
    capture = {}
    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(response=_FakeResponse(200), capture=capture),
    )

    result = asyncio.run(server._llm_preflight_check(_preflight_settings()))

    assert result == {"status": "ok"}
    assert capture["url"] == "https://llm.example/v1/chat/completions"
    assert capture["headers"]["Authorization"] == "Bearer test-only"
    assert capture["json"]["max_tokens"] == 1


def test_preflight_classifies_quota_exhaustion_from_error_body(monkeypatch):
    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(
            response=_FakeResponse(
                429,
                {
                    "error": {
                        "type": "insufficient_quota",
                        "code": "credit_balance_exhausted",
                        "message": "secret detail must not be retained",
                    }
                },
            )
        ),
    )

    result = asyncio.run(server._llm_preflight_check(_preflight_settings()))

    assert result["status"] == "failed"
    assert result["http_status"] == 429
    assert result["error_type"] == "insufficient_quota"
    assert result["error_code"] == "credit_balance_exhausted"
    assert result["quota_exhausted"] is True
    assert "secret" not in repr(result)


def test_preflight_distinguishes_non_quota_429_and_server_errors(monkeypatch):
    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(
            response=_FakeResponse(429, {"error": {"type": "rate_limit_error", "code": None}})
        ),
    )
    rate_limited = asyncio.run(server._llm_preflight_check(_preflight_settings()))
    assert rate_limited["status"] == "failed"
    assert rate_limited["quota_exhausted"] is False

    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(response=_FakeResponse(500, {})),
    )
    server_error = asyncio.run(server._llm_preflight_check(_preflight_settings()))
    assert server_error["status"] == "failed"
    assert server_error["http_status"] == 500
    assert server_error["quota_exhausted"] is False


def test_preflight_transport_failure_reports_only_exception_type(monkeypatch):
    monkeypatch.setattr(
        server.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(exc=httpx.ConnectError("private network detail")),
    )

    result = asyncio.run(server._llm_preflight_check(_preflight_settings()))

    assert result == {"status": "unreachable", "error_type": "ConnectError"}
    assert "private" not in repr(result)


# ---------------------------------------------------------------------------
# WebSocket behavior when pre-flight or watchdog detect a dead LLM
# ---------------------------------------------------------------------------


def test_quota_exhausted_preflight_refuses_session_before_provider_start(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_ensure_llm_ready", _ready_settings)
    monkeypatch.setattr(server, "make_enhancer", lambda settings: type("Enhancer", (), {"flush": lambda self: b""})())
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(server, "_llm_preflight_check", _preflight_quota)
    provider_started = False

    async def fake_run(**kwargs):
        nonlocal provider_started
        provider_started = True
        raise AssertionError("provider must not start when the LLM quota is exhausted")

    monkeypatch.setattr(server.gradbot, "run", fake_run)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start", "session_id": "preflight-quota", "triage_mode": "llm_only", "client": {}})
        error = websocket.receive_json()
        closed = websocket.receive()

    assert provider_started is False
    assert error["type"] == "error"
    assert error["code"] == "llm_quota_exhausted"
    assert "quota is exhausted" in error["message"]
    assert "first-aid" in error["message"]
    assert closed["type"] == "websocket.close"
    assert closed["code"] == 1011
    saved = session_store.get_session("preflight-quota")
    assert saved["server"]["status"] == "error"
    preflight = saved["server"]["provider_diagnostics"]["llm_preflight"]
    assert preflight["quota_exhausted"] is True
    assert preflight["error_code"] == "credit_balance_exhausted"
    assert saved["server"]["provider_diagnostics"]["stream_close"]["reason"] == "llm_quota_exhausted_preflight"


def test_llm_watchdog_terminates_session_with_repeated_silent_push_cycles(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_ensure_llm_ready", _ready_settings)
    monkeypatch.setattr(server, "make_enhancer", lambda settings: type("Enhancer", (), {"flush": lambda self: b""})())
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(server, "_llm_preflight_check", _preflight_ok)
    monkeypatch.setattr(server, "LLM_START_TIMEOUT_S", 0.05)
    monkeypatch.setattr(server, "LLM_NO_RESPONSE_LIMIT", 2)

    class SilentOutput:
        """Emits push_to_llm like the real provider but never starts an LLM."""

        async def receive(self):
            await asyncio.sleep(1.2)
            return _event("push_to_llm")

    async def fake_run(**kwargs):
        return _QuietInput(), SilentOutput()

    async def no_artifacts(**kwargs):
        return {}

    monkeypatch.setattr(server.gradbot, "run", fake_run)
    monkeypatch.setattr(server, "finalize_audio_artifacts", no_artifacts)
    session_id = "llm-watchdog-silent"

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start", "session_id": session_id, "triage_mode": "llm_only", "client": {}})
        messages = []
        closed = None
        for _ in range(20):
            frame = websocket.receive()
            if frame.get("type") == "websocket.close":
                closed = frame
                break
            if frame.get("type") == "websocket.send" and isinstance(frame.get("text"), str):
                try:
                    messages.append(json.loads(frame["text"]))
                except ValueError:
                    continue

    assert any(m.get("type") == "error" and m.get("code") == "llm_no_response" for m in messages), messages
    for message in messages:
        if message.get("type") == "error":
            assert message["code"] == "llm_no_response"
            assert "first-aid" in message["message"]
    assert closed is not None and closed.get("code") == 1011
    saved = session_store.get_session(session_id)["server"]["provider_diagnostics"]
    assert saved["llm_watchdog"]["silent_push_cycles"] == 2
    assert saved["stream_close"]["reason"] == "llm_watchdog_no_response"
    assert saved["llm_started_events"] == 0


def test_llm_watchdog_stays_silent_when_llm_responds(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "_ensure_llm_ready", _ready_settings)
    monkeypatch.setattr(server, "make_enhancer", lambda settings: type("Enhancer", (), {"flush": lambda self: b""})())
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(server, "_llm_preflight_check", _preflight_ok)
    monkeypatch.setattr(server, "LLM_START_TIMEOUT_S", 0.05)

    class HealthyOutput:
        def __init__(self):
            self._steps = [
                _event("push_to_llm"),
                _event("llm_started"),
                _event("end_of_turn"),
                _event("push_to_llm"),
                _event("llm_started"),
                _event("end_of_turn"),
            ]

        async def receive(self):
            if self._steps:
                await asyncio.sleep(0.05)
                return self._steps.pop(0)
            await asyncio.sleep(30)

    async def fake_run(**kwargs):
        return _QuietInput(), HealthyOutput()

    async def no_artifacts(**kwargs):
        return {}

    monkeypatch.setattr(server.gradbot, "run", fake_run)
    monkeypatch.setattr(server, "finalize_audio_artifacts", no_artifacts)
    session_id = "llm-watchdog-healthy"

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"type": "start", "session_id": session_id, "triage_mode": "llm_only", "client": {}})
        assert websocket.receive_json() == {"type": "connection_status", "status": "connected"}
        for _ in range(6):
            websocket.receive_json()
        websocket.send_json({"type": "stop"})
        closed = websocket.receive()

    assert closed["type"] == "websocket.close"
    saved = session_store.get_session(session_id)["server"]["provider_diagnostics"]
    assert saved["llm_watchdog"]["silent_push_cycles"] == 0
    assert saved["llm_started_events"] == 2
    assert saved["completed_llm_responses"] == 2
    assert saved["stream_close"]["reason"] == "client_stop"
