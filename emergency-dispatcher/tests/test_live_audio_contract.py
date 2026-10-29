import asyncio
from types import SimpleNamespace

from emergency_dispatcher.server import _forward_provider_message
from emergency_dispatcher.server import _browser_audio_diagnostics
from emergency_dispatcher.server import _provider_error_diagnostics
from emergency_dispatcher.server import _provider_event_completes_llm_turn


class RecordingWebSocket:
    def __init__(self):
        self.messages = []

    async def send_json(self, payload):
        self.messages.append(("json", payload))

    async def send_bytes(self, payload):
        self.messages.append(("bytes", payload))


def _message(msg_type, **values):
    defaults = {
        "text": None,
        "start_s": None,
        "stop_s": None,
        "turn_idx": None,
        "interrupted": False,
        "event": None,
        "data": None,
    }
    return SimpleNamespace(msg_type=msg_type, **(defaults | values))


def test_provider_caller_and_dispatcher_text_keep_distinct_wire_roles():
    websocket = RecordingWebSocket()

    asyncio.run(
        _forward_provider_message(
            websocket,
            _message("stt_text", text="Caller words", turn_idx=1),
            session_id="audio-contract-test",
        )
    )
    asyncio.run(
        _forward_provider_message(
            websocket,
            _message("tts_text", text="Dispatcher words", turn_idx=1),
            session_id="audio-contract-test",
        )
    )

    assert [kind for kind, _payload in websocket.messages] == ["json", "json"]
    assert [payload["type"] for _kind, payload in websocket.messages] == ["user_text", "agent_text"]


def test_provider_pcm_output_keeps_timing_json_before_binary_audio():
    websocket = RecordingWebSocket()
    audio = _message(
        "audio",
        start_s=0.0,
        stop_s=0.08,
        turn_idx=2,
        data=b"\x01\x00" * 1920,
    )

    asyncio.run(
        _forward_provider_message(
            websocket,
            audio,
            session_id="audio-contract-test",
        )
    )

    assert [kind for kind, _payload in websocket.messages] == ["json", "bytes"]
    assert websocket.messages[0][1]["type"] == "audio_timing"
    assert websocket.messages[0][1]["turn_idx"] == 2
    assert websocket.messages[1][1] == b"\x01\x00" * 1920


def test_end_of_turn_without_active_llm_cycle_is_not_assistant_completion():
    assert _provider_event_completes_llm_turn("end_of_turn", False) is False
    assert _provider_event_completes_llm_turn("end_of_turn", True) is True
    # Once consumed, a duplicate event cannot close a second assistant cycle.
    assert _provider_event_completes_llm_turn("end_of_turn", False) is False


def test_provider_diagnostics_classify_quota_without_logging_exception_text():
    class APIError(Exception):
        status_code = 429
        body = {
            "error": {
                "type": "insufficient_quota",
                "code": "credit_balance_exhausted",
                "message": "PRIVATE_ERROR_DETAIL_AND_CREDENTIAL_SENTINEL",
            }
        }

    diagnostic = _provider_error_diagnostics(APIError("PRIVATE_EXCEPTION_TEXT"))

    assert diagnostic == {
        "provider": "openai",
        "category": "llm_quota_exhausted",
        "exception_type": "APIError",
        "http_status": 429,
        "error_type": "insufficient_quota",
        "error_code": "credit_balance_exhausted",
    }
    assert "PRIVATE" not in repr(diagnostic)


def test_provider_diagnostics_distinguish_transport_errors_from_quota():
    diagnostic = _provider_error_diagnostics(TimeoutError("PRIVATE_TRANSCRIPT_OR_TOKEN"), provider_hint="gradbot")

    assert diagnostic["provider"] == "gradbot"
    assert diagnostic["category"] == "provider_transport_error"
    assert diagnostic["exception_type"] == "TimeoutError"
    assert diagnostic["http_status"] is None
    assert "PRIVATE" not in repr(diagnostic)


def test_provider_diagnostics_extract_wrapped_quota_metadata_without_exposing_message():
    error = RuntimeError(
        "Gradbot OpenAI request failed: Error code: 429; type=insufficient_quota; "
        "code=credit_balance_exhausted; private caller and credential sentinel"
    )

    diagnostic = _provider_error_diagnostics(error, provider_hint="gradbot")

    assert diagnostic["provider"] == "openai"
    assert diagnostic["category"] == "llm_quota_exhausted"
    assert diagnostic["http_status"] == 429
    assert diagnostic["error_type"] == "insufficient_quota"
    assert diagnostic["error_code"] == "credit_balance_exhausted"
    assert "private" not in repr(diagnostic).lower()


def test_browser_audio_diagnostics_keep_only_safe_categories_and_status_codes():
    diagnostic = _browser_audio_diagnostics(
        {
            "events": [
                {"text": "PRIVATE_TRANSCRIPT_SENTINEL"},
                {"text": "dispatcher audio bytes received from provider"},
                {"text": "audio stream or playback failed (Error)"},
                {"text": "websocket closed (1011)"},
                {"text": "Dispatcher audio playback started"},
            ]
        }
    )

    assert diagnostic == {
        "playback_error_count": 1,
        "playback_error_types": {"Error": 1},
        "audio_worklet_error_types": {},
        "provider_audio_byte_events": 1,
        "playback_diagnostic_events": 0,
        "received_audio_bytes": 0,
        "decoded_audio_samples": 0,
        "queued_audio_chunks": 0,
        "played_audio_samples": 0,
        "audio_context_states": [],
        "playback_started": True,
        "websocket_close_codes": [1011],
    }
    assert "PRIVATE" not in repr(diagnostic)


def test_browser_audio_playback_start_is_case_insensitive():
    diagnostic = _browser_audio_diagnostics({
        "events": [
            {"text": "audio_playback_diag stage=playing context=running received_chunks=1 received_bytes=3840 decoded_samples=1920 queued_chunks=1 played_s=0.040 played_samples=1920 buffer_ms=10.0 context_rate_hz=48000"},
            {"text": "dispatcher audio playback started"},
        ]
    })

    assert diagnostic["played_audio_samples"] == 1920
    assert diagnostic["playback_started"] is True


def test_browser_playback_diagnostics_keep_queue_and_sample_counts_only():
    diagnostic = _browser_audio_diagnostics(
        {
            "events": [{
                "text": (
                    "audio_playback_diag stage=playing context=running received_chunks=2 received_bytes=3840 "
                    "decoded_samples=1920 queued_chunks=2 played_s=0.080 played_samples=3840 buffer_ms=20.0 context_rate_hz=48000"
                )
            }, {"text": "caller said private address"}],
        }
    )

    assert diagnostic["playback_diagnostic_events"] == 1
    assert diagnostic["received_audio_bytes"] == 3840
    assert diagnostic["decoded_audio_samples"] == 1920
    assert diagnostic["queued_audio_chunks"] == 2
    assert diagnostic["played_audio_samples"] == 3840
    assert diagnostic["audio_context_states"] == ["running"]
    assert "private address" not in repr(diagnostic)


def test_browser_audio_diagnostics_capture_queue_failures_without_error_text():
    diagnostic = _browser_audio_diagnostics({
        "events": [{"text": "audio playback failed stage=queue error_type=DataError"},
                   {"text": "caller transcript and PRIVATE_KEY"}],
    })

    assert diagnostic["playback_error_types"] == {"DataError": 1}
    assert "PRIVATE" not in repr(diagnostic)


def test_abnormal_websocket_close_is_classified_without_retaining_remote_text():
    from types import SimpleNamespace

    diagnostic = _provider_error_diagnostics(
        SimpleNamespace(code=1006, details="1006 Abnormal closure; sensitive payload omitted"),
        provider_hint="gemini",
    )

    assert diagnostic["provider_close_code"] == 1006
    assert diagnostic["provider_close_reason"] == "abnormal_closure_no_close_frame"
    assert "sensitive payload" not in repr(diagnostic)
