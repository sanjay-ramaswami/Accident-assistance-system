"""Live-voice session ownership guard.

Only one live WebSocket may own a session_id at a time. A duplicate or stale
"start" that reuses an already-owned session_id must be rejected instead of
interleaving writes into the same persisted session file. These tests pin the
acquire/release registry and the WebSocket rejection path; no real provider is
contacted.
"""

from __future__ import annotations

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


def _drain_registry():
    # Keep tests independent even if a prior test leaked an entry.
    server._ACTIVE_LIVE_SESSIONS.clear()


def test_acquire_rejects_duplicate_and_release_allows_reacquire():
    _drain_registry()
    assert server._acquire_live_session("guard-1") is True
    assert server._acquire_live_session("guard-1") is False
    assert server._acquire_live_session("guard-2") is True
    server._release_live_session("guard-1")
    assert "guard-1" not in server._ACTIVE_LIVE_SESSIONS
    assert server._acquire_live_session("guard-1") is True
    _drain_registry()


def test_duplicate_start_is_rejected_without_touching_the_active_session(monkeypatch, tmp_path):
    _drain_registry()
    monkeypatch.setattr(server, "_ensure_llm_ready", _ready_settings)
    monkeypatch.setattr(
        server, "make_enhancer", lambda settings: type("Enhancer", (), {"flush": lambda self: b""})()
    )
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(server, "_llm_preflight_check", _preflight_ok)

    # Simulate an already-live session owning this session_id.
    assert server._acquire_live_session("guarded-session") is True

    started = False

    async def fake_run(**kwargs):
        nonlocal started
        started = True
        raise AssertionError("provider must not start for a duplicate session_id")

    monkeypatch.setattr(server.gradbot, "run", fake_run)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "session_id": "guarded-session",
                "triage_mode": "llm_only",
                "client": {},
            }
        )
        closed = websocket.receive()

    assert started is False
    assert closed["type"] == "websocket.close"
    assert closed["code"] == 1008
    assert closed["reason"] == "Session already active"
    # The original owner still holds the session; the duplicate did not release it.
    assert "guarded-session" in server._ACTIVE_LIVE_SESSIONS
    _drain_registry()


def test_session_id_is_released_after_a_normal_stop(monkeypatch, tmp_path):
    _drain_registry()
    monkeypatch.setattr(server, "_ensure_llm_ready", _ready_settings)
    monkeypatch.setattr(
        server, "make_enhancer", lambda settings: type("Enhancer", (), {"flush": lambda self: b""})()
    )
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")
    monkeypatch.setattr(server, "_llm_preflight_check", _preflight_ok)

    class QuietInput:
        async def close(self):
            return None

    class IdleOutput:
        async def receive(self):
            import asyncio

            await asyncio.sleep(30)

    async def fake_run(**kwargs):
        return QuietInput(), IdleOutput()

    async def no_artifacts(**kwargs):
        return {}

    monkeypatch.setattr(server.gradbot, "run", fake_run)
    monkeypatch.setattr(server, "finalize_audio_artifacts", no_artifacts)
    session_id = "guard-release-after-stop"

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json(
            {"type": "start", "session_id": session_id, "triage_mode": "llm_only", "client": {}}
        )
        assert websocket.receive_json() == {"type": "connection_status", "status": "connected"}
        # While the session is live, the id is owned.
        assert session_id in server._ACTIVE_LIVE_SESSIONS
        websocket.send_json({"type": "stop"})
        closed = websocket.receive()

    assert closed["type"] == "websocket.close"
    assert session_id not in server._ACTIVE_LIVE_SESSIONS
    _drain_registry()
