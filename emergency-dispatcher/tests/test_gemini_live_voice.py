"""Gemini voice path contract and WebSocket integration coverage."""

import asyncio
from collections import deque
from pathlib import Path
import re
from types import SimpleNamespace

from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect
from google.genai.errors import APIError
import pytest

from emergency_dispatcher import gemini_live
from emergency_dispatcher import server


def test_demo_defaults_to_gemini_and_keeps_gradbot_selectable():
    page = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")
    match = re.search(r'<select id="voice-provider-select"[^>]*>(.*?)</select>', page, re.DOTALL)
    assert match
    options = re.findall(r'<option value="([^"]+)"([^>]*)>', match.group(1))
    assert options[0] == ("gemini", " selected")
    assert ("gradbot", "") in options


def test_gemini_output_pcm_is_resampled_to_browser_rate():
    # Four 24 kHz samples become eight 48 kHz samples, with PCM16 LE retained.
    pcm = b"\x00\x00\xe8\x03\xd0\x07\xb8\x0b"
    result = gemini_live._resample_pcm16(pcm, 24000, 48000)
    assert len(result) == 16
    assert int.from_bytes(result[0:2], "little", signed=True) == 0
    assert int.from_bytes(result[-2:], "little", signed=True) == 3000


def test_gemini_adapter_sends_explicit_raw_pcm_rate():
    class Session:
        def __init__(self):
            self.audio = []

        async def send_realtime_input(self, **kwargs):
            self.audio.append(kwargs["audio"])

    adapter = gemini_live.GeminiLiveAdapter.__new__(gemini_live.GeminiLiveAdapter)
    adapter.session = Session()
    adapter._closed = False
    asyncio.run(adapter.send_audio(b"\x00\x00" * 480))
    assert adapter.session.audio[0].mime_type == "audio/pcm;rate=16000"
    assert len(adapter.session.audio[0].data) == 320 * 2


def test_gemini_session_defaults_spoken_responses_to_english(monkeypatch):
    class Session:
        def receive(self):
            async def responses():
                if False:
                    yield None
            return responses()

    class Context:
        async def __aenter__(self):
            return Session()

        async def __aexit__(self, *_args):
            return None

    class Live:
        def connect(self, *, model, config):
            captured.update(model=model, config=config)
            return Context()

    captured = {}
    client = SimpleNamespace(aio=SimpleNamespace(live=Live()))
    monkeypatch.setattr(gemini_live, "build_gemini_client", lambda _settings: client)
    adapter = gemini_live.GeminiLiveAdapter(
        gemini_live.get_settings().model_copy(update={"gemini_api_key": "test-key"}),
        instructions="Stay calm and ask one question.",
        tools=[],
    )

    asyncio.run(adapter.connect())

    instruction = captured["config"]["system_instruction"]
    assert "Respond unmistakably in English by default" in instruction
    assert "only when the caller explicitly asks you to" in instruction
    assert captured["config"]["response_modalities"] == ["AUDIO"]
    assert "speech_config" not in captured["config"]


def test_gemini_context_update_sends_only_incremental_safety_state():
    class Session:
        def __init__(self):
            self.sent = []

        async def send_client_content(self, **kwargs):
            self.sent.append(kwargs)

    session = Session()
    adapter = gemini_live.GeminiLiveAdapter.__new__(gemini_live.GeminiLiveAdapter)
    adapter.session = session
    adapter._closed = False
    from emergency_dispatcher.gradbot_adapter import build_session_config
    config = build_session_config(first_aid_context={
        "scenario": "Road accident scene safety",
        "immediate_actions": ["Stay clear of traffic and smoke."],
        "actions_to_avoid": [],
    })

    asyncio.run(adapter.send_config(config))

    update = session.sent[0]["turns"]["parts"][0]["text"]
    assert "Current bystander-safety guidance" in update
    assert "Stay clear of traffic and smoke." in update
    assert "EMERGENCY RESPONSE DEMONSTRATION VOICE ASSISTANT" not in update
    assert "Respond unmistakably in English by default" in update
    assert session.sent[0]["turn_complete"] is False


def test_gemini_batches_all_function_responses_for_one_server_tool_call():
    class Session:
        def __init__(self):
            self.sent = []

        def receive(self):
            async def responses():
                yield SimpleNamespace(
                    server_content=None,
                    tool_call=SimpleNamespace(function_calls=[
                        SimpleNamespace(name="first_tool", id="call-1", args={"value": 1}),
                        SimpleNamespace(name="second_tool", id="call-2", args={"value": 2}),
                    ]),
                )
            return responses()

        async def send_tool_response(self, *, function_responses):
            self.sent.append(function_responses)

    session = Session()
    adapter = gemini_live.GeminiLiveAdapter.__new__(gemini_live.GeminiLiveAdapter)
    adapter.session = session
    adapter._responses = session.receive()
    adapter._closed = False
    adapter._turn_active = False
    adapter._pending = deque()

    async def reply_to_calls():
        await adapter.receive()  # llm_started event
        first = (await adapter.receive()).tool_call_handle
        second = (await adapter.receive()).tool_call_handle
        await second.send_json({"ok": 2})
        assert session.sent == []
        await first.send_json({"ok": 1})

    asyncio.run(reply_to_calls())

    assert len(session.sent) == 1
    assert [(item.id, item.name) for item in session.sent[0]] == [
        ("call-1", "first_tool"), ("call-2", "second_tool")
    ]
    assert [item.response for item in session.sent[0]] == [
        {"result": {"ok": 1}}, {"result": {"ok": 2}}
    ]


def test_gemini_receive_reopens_sdk_iterator_after_each_completed_interaction():
    class Session:
        def __init__(self):
            self.receive_calls = 0

        def receive(self):
            self.receive_calls += 1
            generation = self.receive_calls

            async def messages():
                if generation == 1:
                    yield SimpleNamespace(
                        server_content=SimpleNamespace(
                            input_transcription=SimpleNamespace(text="దయచేసి ఇంగ్లీష్‌లో మాట్లాడండి"),
                            output_transcription=None,
                            model_turn=None,
                            turn_complete=True,
                            interrupted=False,
                        ),
                        tool_call=None,
                    )
                else:
                    yield SimpleNamespace(
                        server_content=SimpleNamespace(
                            input_transcription=None,
                            output_transcription=SimpleNamespace(text="I will respond in English."),
                            model_turn=SimpleNamespace(parts=[
                                SimpleNamespace(inline_data=SimpleNamespace(data=b"\x01\x00" * 240))
                            ]),
                            turn_complete=True,
                            interrupted=False,
                        ),
                        tool_call=None,
                    )

            return messages()

    session = Session()
    adapter = gemini_live.GeminiLiveAdapter.__new__(gemini_live.GeminiLiveAdapter)
    adapter.session = session
    adapter._responses = session.receive()
    adapter._closed = False
    adapter._turn_active = False
    adapter._pending = deque()

    async def collect_messages():
        return [await adapter.receive() for _ in range(6)]

    messages = asyncio.run(collect_messages())
    assert session.receive_calls == 2
    assert all(message is not None for message in messages)
    assert [message.msg_type for message in messages] == [
        "event", "stt_text", "event", "tts_text", "audio", "event"
    ]
    assert messages[1].text == "దయచేసి ఇంగ్లీష్‌లో మాట్లాడండి"
    assert messages[3].text == "I will respond in English."


def test_gemini_live_control_events_are_surfaced_without_session_tokens():
    class Session:
        def receive(self):
            async def responses():
                yield SimpleNamespace(
                    server_content=None,
                    tool_call=None,
                    go_away=SimpleNamespace(time_left="12.5s"),
                    session_resumption_update=SimpleNamespace(
                        resumable=True,
                        new_handle="SENSITIVE_RESUMPTION_TOKEN",
                    ),
                    usage_metadata=SimpleNamespace(total_token_count=7),
                    voice_activity=SimpleNamespace(activity_type="START_OF_SPEECH"),
                    voice_activity_detection_signal=SimpleNamespace(),
                    tool_call_cancellation=SimpleNamespace(),
                )
            return responses()

    session = Session()
    adapter = gemini_live.GeminiLiveAdapter.__new__(gemini_live.GeminiLiveAdapter)
    adapter.session = session
    adapter._responses = session.receive()
    adapter._closed = False
    adapter._turn_active = False
    adapter._pending = deque()

    async def collect_events():
        return [await adapter.receive() for _ in range(6)]

    events = asyncio.run(collect_events())

    assert [item.event.event_type for item in events] == [
        "provider_go_away",
        "provider_session_resumption",
        "provider_usage_metadata",
        "provider_voice_activity",
        "provider_voice_activity_detection",
        "provider_tool_call_cancellation",
    ]
    assert events[0].event.time_left_seconds == 12.5
    assert events[1].event.resumable is True
    assert events[1].event.token_present is True
    assert "SENSITIVE_RESUMPTION_TOKEN" not in repr(events)


def test_gemini_voice_websocket_streams_audio_and_returns_text_and_pcm(monkeypatch, tmp_path):
    sent_audio = []
    recorder_updates = []

    class Adapter:
        def __init__(self, settings, *, instructions, tools):
            self.messages = [
                SimpleNamespace(msg_type="stt_text", text="There has been a crash", turn_idx=1, start_s=0.0, stop_s=1.0, interrupted=False, event=None, data=None),
                SimpleNamespace(msg_type="event", event=SimpleNamespace(event_type="llm_started"), text=None, data=None, turn_idx=1, start_s=None, stop_s=None, interrupted=False),
                SimpleNamespace(msg_type="tts_text", text="Are you safe?", turn_idx=1, start_s=1.0, stop_s=2.0, interrupted=False, event=None, data=None),
                SimpleNamespace(msg_type="audio", data=b"\x01\x00" * 1920, start_s=0.0, stop_s=0.08, turn_idx=1, interrupted=False, event=None, text=None),
                SimpleNamespace(msg_type="event", event=SimpleNamespace(event_type="end_of_turn"), text=None, data=None, turn_idx=1, start_s=None, stop_s=None, interrupted=False),
            ]

        async def connect(self):
            return None

        async def send_audio(self, data):
            sent_audio.append(data)

        async def send_config(self, config):
            return None

        async def close(self):
            return None

        async def receive(self):
            if self.messages:
                return self.messages.pop(0)
            await asyncio.sleep(0.05)
            return SimpleNamespace(msg_type="event", event=SimpleNamespace(event_type="unknown"), text=None, data=None, turn_idx=None, start_s=None, stop_s=None, interrupted=False)

    class Recorder:
        def __init__(self, session_id):
            self.session_id = session_id

        def patch_server(self, payload):
            recorder_updates.append(payload)

        def append_session_list(self, *args):
            return None

    monkeypatch.setattr(server, "GeminiLiveAdapter", Adapter)
    monkeypatch.setattr(server, "SessionRecorder", Recorder)
    monkeypatch.setattr(server, "finalize_audio_artifacts", _no_artifacts)
    monkeypatch.setattr(server, "clear_live_audio_window", lambda _session_id: None)
    monkeypatch.setattr(server, "append_live_audio_window", lambda *_args: None)
    monkeypatch.setattr(server, "get_settings", lambda: gemini_live.get_settings().model_copy(update={"gemini_api_key": "test-key"}))

    client = TestClient(server.app)
    with client.websocket_connect("/ws?voice_provider=gemini") as websocket:
        websocket.send_json({"type": "start", "session_id": "gemini-e2e", "triage_mode": "llm_only", "voice_provider": "gemini", "client": {}})
        assert websocket.receive_json() == {"type": "connection_status", "status": "connected"}
        websocket.send_bytes(b"\x02\x00" * 1920)

        received = []
        while len(received) < 5:
            frame = websocket.receive()
            if frame.get("type") == "websocket.send" and frame.get("bytes") is not None:
                received.append(("bytes", frame["bytes"]))
            elif frame.get("type") == "websocket.send" and frame.get("text"):
                received.append(("json", __import__("json").loads(frame["text"])))

        assert sent_audio == [b"\x02\x00" * 1920]
        audio_diagnostics = [
            update["provider_diagnostics"]
            for update in recorder_updates
            if "provider_diagnostics" in update
            and update["provider_diagnostics"].get("input_audio_chunks") == 1
        ]
        assert audio_diagnostics
        assert audio_diagnostics[0]["input_audio_bytes"] == 3840
        assert audio_diagnostics[0]["provider_audio_send_attempts"] == 1
        assert audio_diagnostics[0]["provider_audio_send_attempted_bytes"] == 3840
        assert audio_diagnostics[0]["provider_audio_send_successes"] == 1
        assert audio_diagnostics[0]["status"] == "receiving_audio"
        assert any(kind == "json" and value.get("type") == "user_text" for kind, value in received)
        assert any(kind == "json" and value.get("type") == "agent_text" for kind, value in received)
        assert any(kind == "bytes" and value == b"\x01\x00" * 1920 for kind, value in received)
        assert not any(
            update.get("enhancement", {}).get("status") == "degraded"
            and "NoneType" in update.get("enhancement", {}).get("fallback_reason", "")
            for update in recorder_updates
        )
        websocket.send_json({"type": "stop"})


def test_gemini_missing_key_rejects_live_socket_without_exposing_key(monkeypatch):
    monkeypatch.setattr(server, "get_settings", lambda: gemini_live.get_settings().model_copy(update={"gemini_api_key": None}))
    with pytest.raises(WebSocketDisconnect) as disconnected:
        with TestClient(server.app).websocket_connect("/ws?voice_provider=gemini"):
            pytest.fail("Gemini socket must not accept a session without its key")
    assert disconnected.value.code == 1008
    assert "test-key" not in str(disconnected.value)


def test_gemini_quota_failure_is_reported_and_session_closes(monkeypatch):
    class Recorder:
        def __init__(self, session_id):
            self.session_id = session_id

        def patch_server(self, payload):
            return None

        def append_session_list(self, *args):
            return None

    class QuotaError(Exception):
        status_code = 429
        body = {"error": {"type": "insufficient_quota", "code": "RESOURCE_EXHAUSTED"}}

    class Adapter:
        def __init__(self, *_args, **_kwargs):
            pass

        async def connect(self):
            raise QuotaError("private provider detail")

    monkeypatch.setattr(server, "get_settings", lambda: gemini_live.get_settings().model_copy(update={"gemini_api_key": "test-key"}))
    monkeypatch.setattr(server, "SessionRecorder", Recorder)
    monkeypatch.setattr(server, "GeminiLiveAdapter", Adapter)
    with TestClient(server.app).websocket_connect("/ws?voice_provider=gemini") as websocket:
        websocket.send_json({"type": "start", "session_id": "gemini-quota", "triage_mode": "llm_only", "voice_provider": "gemini", "client": {}})
        error = websocket.receive_json()
        close = websocket.receive()
    assert error == {
        "type": "error",
        "code": "gemini_quota_exhausted",
        "message": "Gemini Live quota is exhausted; no live response was generated.",
    }
    assert close["type"] == "websocket.close"
    assert close["code"] == 1011


def test_gemini_expected_upstream_close_during_stop_does_not_become_1011(monkeypatch):
    class Recorder:
        def __init__(self, session_id):
            self.session_id = session_id

        def patch_server(self, _payload):
            return None

        def append_session_list(self, *_args):
            return None

    class Adapter:
        def __init__(self, *_args, **_kwargs):
            self.closed = asyncio.Event()

        async def connect(self):
            return None

        async def send_audio(self, _data):
            return None

        async def send_config(self, _config):
            return None

        async def close(self):
            self.closed.set()

        async def receive(self):
            await self.closed.wait()
            raise APIError(1000, {"message": "None."})

    monkeypatch.setattr(server, "get_settings", lambda: gemini_live.get_settings().model_copy(update={"gemini_api_key": "test-key"}))
    monkeypatch.setattr(server, "SessionRecorder", Recorder)
    monkeypatch.setattr(server, "GeminiLiveAdapter", Adapter)
    monkeypatch.setattr(server, "finalize_audio_artifacts", _no_artifacts)
    monkeypatch.setattr(server, "clear_live_audio_window", lambda _session_id: None)

    with TestClient(server.app).websocket_connect("/ws?voice_provider=gemini") as websocket:
        websocket.send_json({"type": "start", "session_id": "gemini-clean-close", "triage_mode": "llm_only", "voice_provider": "gemini", "client": {}})
        assert websocket.receive_json() == {"type": "connection_status", "status": "connected"}
        websocket.send_json({"type": "stop"})
        close = websocket.receive()

    assert close["type"] == "websocket.close"
    assert close["code"] == 1000


def test_receive_task_cleanup_retrieves_expected_sdk_close_error():
    async def scenario():
        async def receive():
            raise APIError(1000, {"message": "None."})

        task = asyncio.create_task(receive())
        await asyncio.sleep(0)
        error = await server._cancel_and_drain_receive_task(task)
        assert isinstance(error, APIError)
        assert error.code == 1000

    asyncio.run(scenario())


def test_provider_close_diagnostics_classify_reason_without_logging_remote_text():
    secretish_details = "Unexpected function call: rejected content caller-private-fragment"
    diagnostics = server._provider_error_diagnostics(
        APIError(1007, secretish_details), provider_hint="gemini"
    )

    assert diagnostics["provider_close_code"] == 1007
    assert diagnostics["provider_close_reason"] == "unexpected_function_call"
    assert "caller-private-fragment" not in str(diagnostics)
    assert secretish_details not in str(diagnostics)


async def _no_artifacts(**_kwargs):
    return {}
