"""Gemini Live API integration (google-genai SDK).

Design constraints:
- The API key is loaded from ``Settings`` (env var ``GEMINI_API_KEY``) and passed
  explicitly to ``genai.Client(api_key=...)``. It is never read from ambient env
  at call time, never returned in an API response, and never sent to the browser.
- Public helpers return only variable *names*, booleans, and safe labels. No
  function here returns the key value, and none of them log it.
- Import of ``google.genai`` is deferred so the rest of the app (and the whole
  test suite) keeps working when the optional SDK is not installed.
"""

from __future__ import annotations

from typing import Any
from types import SimpleNamespace
import asyncio
import json
import struct
import re
from collections import deque

from .settings import Settings
from .settings import get_settings


GEMINI_CLIENT_INPUT_RATE = 24000
GEMINI_INPUT_RATE = 16000
GEMINI_OUTPUT_RATE = 24000
CONSOLE_OUTPUT_RATE = 48000
ENGLISH_OUTPUT_POLICY = (
    "LANGUAGE POLICY: Respond unmistakably in English by default, even when the caller speaks another language. "
    "Keep the caller's recognized words as a faithful transcript; do not present a translation as if it were "
    "what the caller said. Switch to another spoken language only when the caller explicitly asks you to."
)


def _with_english_output_policy(instructions: str) -> str:
    return f"{instructions.rstrip()}\n\n{ENGLISH_OUTPUT_POLICY}"


def _runtime_context_only(instructions: str) -> str:
    """Keep Gemini context updates incremental; base instructions are set at connect."""
    markers = (
        "\nAdaptive turn-taking policy:",
        "\nLive triage packet:",
        "\nShared fact ledger packet:",
        "\nLive dispatch operations packet:",
        "\nHuman takeover packet:",
        "\nCurrent bystander-safety guidance from the local curated first-aid rules:",
    )
    starts = [instructions.find(marker) for marker in markers]
    starts = [index for index in starts if index >= 0]
    return instructions[min(starts):].strip() if starts else ""


def _resample_pcm16(data: bytes, source_rate: int, target_rate: int) -> bytes:
    """Linearly resample mono little-endian PCM16 without changing its format."""
    if source_rate == target_rate or len(data) < 4:
        return data
    usable = len(data) - len(data) % 2
    samples = struct.unpack(f"<{usable // 2}h", data[:usable])
    output_count = max(1, round(len(samples) * target_rate / source_rate))
    output = bytearray(output_count * 2)
    for index in range(output_count):
        position = index * source_rate / target_rate
        left = min(int(position), len(samples) - 1)
        right = min(left + 1, len(samples) - 1)
        fraction = position - left
        sample = round(samples[left] * (1 - fraction) + samples[right] * fraction)
        struct.pack_into("<h", output, index * 2, max(-32768, min(32767, sample)))
    return bytes(output)


class GeminiLiveAdapter:
    """Small server-side adapter for the Gemini Live audio session."""

    def __init__(self, settings: Settings, *, instructions: str, tools: list[Any]):
        self.settings = settings
        self.instructions = instructions
        self.tools = tools
        self.client: Any = None
        self._context: Any = None
        self.session: Any = None
        self._responses: Any = None
        self._closed = False
        self._turn_active = False
        self._pending: deque[Any] = deque()

    async def connect(self, timeout_s: float = 15.0) -> None:
        """Wait for the upstream Live handshake; never report ready before it."""
        client = build_gemini_client(self.settings)
        self.client = client
        declarations = []
        for tool in self.tools:
            parameters = json.loads(tool.parameters_json)
            _normalize_gemini_schema(parameters)
            declarations.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters,
            })
        config = {
            "response_modalities": ["AUDIO"],
            "system_instruction": _with_english_output_policy(self.instructions),
            "input_audio_transcription": {},
            "output_audio_transcription": {},
            "tools": [{"function_declarations": declarations}] if declarations else [],
        }
        self._context = client.aio.live.connect(model=self.settings.gemini_live_model, config=config)
        try:
            self.session = await asyncio.wait_for(self._context.__aenter__(), timeout=timeout_s)
        except Exception:
            await self._close_client()
            raise
        self._responses = self.session.receive()

    async def send_audio(self, data: bytes) -> None:
        if self._closed or not self.session:
            raise RuntimeError("Gemini Live session is closed")
        from google.genai import types
        gemini_audio = _resample_pcm16(data, GEMINI_CLIENT_INPUT_RATE, GEMINI_INPUT_RATE)
        await self.session.send_realtime_input(
            audio=types.Blob(data=gemini_audio, mime_type=f"audio/pcm;rate={GEMINI_INPUT_RATE}")
        )

    async def send_config(self, config: Any) -> None:
        # Live sessions cannot replace system instructions in place. Supply
        # operator state as a non-turn-completing context update so the same
        # conversation and tool state remain active.
        if self._closed or not self.session:
            return
        text = _runtime_context_only(str(getattr(config, "instructions", None) or ""))
        if text:
            await self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": f"[Dispatcher context update]\n{_with_english_output_policy(text)}"}]},
                turn_complete=False,
            )

    async def receive(self) -> Any | None:
        if self._closed or not self.session:
            return None
        if self._pending:
            return self._pending.popleft()
        while True:
            try:
                response = await self._responses.__anext__()
                break
            except StopAsyncIteration:
                if self._closed:
                    return None
                # google-genai's AsyncSession.receive() yields one complete
                # interaction, then ends that iterator while the Live
                # WebSocket session remains open for the next interaction.
                self._responses = self.session.receive()
        content = getattr(response, "server_content", None)
        if content:
            user = getattr(content, "input_transcription", None)
            if user and getattr(user, "text", None):
                self._pending.append(SimpleNamespace(msg_type="event", event=SimpleNamespace(event_type="push_to_llm"), text=None, data=None, turn_idx=None, start_s=None, stop_s=None, interrupted=False))
                self._pending.append(SimpleNamespace(msg_type="stt_text", text=user.text, turn_idx=None, start_s=None, stop_s=None, interrupted=False, event=None, data=None))
            assistant = getattr(content, "output_transcription", None)
            if assistant and getattr(assistant, "text", None):
                if not self._turn_active:
                    self._pending.append(SimpleNamespace(msg_type="event", event=SimpleNamespace(event_type="llm_started"), text=None, data=None, turn_idx=None, start_s=None, stop_s=None, interrupted=False))
                self._turn_active = True
                self._pending.append(SimpleNamespace(msg_type="tts_text", text=assistant.text, turn_idx=None, start_s=None, stop_s=None, interrupted=False, event=None, data=None))
            model_turn = getattr(content, "model_turn", None)
            if model_turn:
                for part in getattr(model_turn, "parts", []) or []:
                    inline = getattr(part, "inline_data", None)
                    if inline and inline.data:
                        audio = inline.data
                        if isinstance(audio, str):
                            import base64
                            audio = base64.b64decode(audio)
                        audio = _resample_pcm16(audio, GEMINI_OUTPUT_RATE, CONSOLE_OUTPUT_RATE)
                        if not self._turn_active:
                            self._pending.append(SimpleNamespace(msg_type="event", event=SimpleNamespace(event_type="llm_started"), text=None, data=None, turn_idx=None, start_s=None, stop_s=None, interrupted=False))
                        self._turn_active = True
                        self._pending.append(SimpleNamespace(msg_type="audio", data=audio, start_s=None, stop_s=None, turn_idx=None, interrupted=False, event=None, text=None))
            if getattr(content, "turn_complete", False):
                was_active = self._turn_active
                self._turn_active = False
                if was_active:
                    self._pending.append(SimpleNamespace(msg_type="event", event=SimpleNamespace(event_type="end_of_turn"), text=None, data=None, turn_idx=None, start_s=None, stop_s=None, interrupted=False))
            if getattr(content, "interrupted", False):
                self._turn_active = False
                self._pending.append(SimpleNamespace(msg_type="event", event=SimpleNamespace(event_type="previous_llm_gen"), text=None, data=None, turn_idx=None, start_s=None, stop_s=None, interrupted=True))
        tool_call = getattr(response, "tool_call", None)
        if tool_call and getattr(tool_call, "function_calls", None):
            calls = list(tool_call.function_calls)
            batch = GeminiToolResponseBatch(self.session, calls)
            if not self._turn_active:
                self._pending.append(SimpleNamespace(msg_type="event", event=SimpleNamespace(event_type="llm_started"), text=None, data=None, turn_idx=None, start_s=None, stop_s=None, interrupted=False))
            self._turn_active = True
            for call in calls:
                handle = GeminiToolCallHandle(self.session, call, batch=batch)
                self._pending.append(SimpleNamespace(msg_type="tool_call", tool_call=handle.info, tool_call_handle=handle, text=None, data=None, event=None, turn_idx=None, start_s=None, stop_s=None, interrupted=False))
        go_away = getattr(response, "go_away", None)
        if go_away is not None:
            time_left = str(getattr(go_away, "time_left", "") or "")
            match = re.fullmatch(r"(\d+(?:\.\d+)?)s", time_left)
            self._pending.append(SimpleNamespace(
                msg_type="event",
                event=SimpleNamespace(
                    event_type="provider_go_away",
                    time_left_seconds=float(match.group(1)) if match else None,
                ),
                text=None, data=None, turn_idx=None, start_s=None, stop_s=None, interrupted=False,
            ))
        resumption = getattr(response, "session_resumption_update", None)
        if resumption is not None:
            self._pending.append(SimpleNamespace(
                msg_type="event",
                event=SimpleNamespace(
                    event_type="provider_session_resumption",
                    resumable=getattr(resumption, "resumable", None),
                    token_present=bool(getattr(resumption, "new_handle", None)),
                ),
                text=None, data=None, turn_idx=None, start_s=None, stop_s=None, interrupted=False,
            ))
        if getattr(response, "usage_metadata", None) is not None:
            self._pending.append(SimpleNamespace(
                msg_type="event",
                event=SimpleNamespace(event_type="provider_usage_metadata"),
                text=None, data=None, turn_idx=None, start_s=None, stop_s=None, interrupted=False,
            ))
        for field, event_type in (
            ("voice_activity", "provider_voice_activity"),
            ("voice_activity_detection_signal", "provider_voice_activity_detection"),
            ("tool_call_cancellation", "provider_tool_call_cancellation"),
        ):
            if getattr(response, field, None) is not None:
                self._pending.append(SimpleNamespace(
                    msg_type="event",
                    event=SimpleNamespace(event_type=event_type),
                    text=None, data=None, turn_idx=None, start_s=None, stop_s=None, interrupted=False,
                ))
        if self._pending:
            return self._pending.popleft()
        return SimpleNamespace(msg_type="event", event=SimpleNamespace(event_type="unknown"), text=None, data=None, turn_idx=None, start_s=None, stop_s=None, interrupted=False)

    async def _close_client(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._context and self.session is not None:
                await self._context.__aexit__(None, None, None)
        finally:
            if self.client:
                close = getattr(self.client, "close", None)
                if close:
                    result = close()
                    if hasattr(result, "__await__"):
                        await result

    async def close(self) -> None:
        await self._close_client()


class GeminiToolCallHandle:
    def __init__(self, session: Any, call: Any, *, batch: Any = None):
        self.session = session
        self.call = call
        self.batch = batch
        self.name = call.name
        self.args = dict(call.args or {})
        self.info = SimpleNamespace(call_id=call.id, tool_name=self.name, args_json=json.dumps(self.args))

    async def send_json(self, result: Any) -> None:
        if self.batch is not None:
            await self.batch.send(self.call, result)
            return
        from google.genai import types
        await self.session.send_tool_response(function_responses=[
            types.FunctionResponse(name=self.name, id=self.call.id, response={"result": result})
        ])


class GeminiToolResponseBatch:
    """Send one protocol response for every function call in a server message."""

    def __init__(self, session: Any, calls: list[Any]):
        self.session = session
        self.call_ids = [call.id for call in calls]
        self.responses: dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def send(self, call: Any, result: Any) -> None:
        from google.genai import types
        async with self._lock:
            self.responses[call.id] = types.FunctionResponse(
                name=call.name,
                id=call.id,
                response={"result": result},
            )
            if all(call_id in self.responses for call_id in self.call_ids):
                ordered = [self.responses[call_id] for call_id in self.call_ids]
                await self.session.send_tool_response(function_responses=ordered)


def _normalize_gemini_schema(schema: Any) -> None:
    """Translate nullable JSON Schema unions into Gemini Schema fields."""
    if isinstance(schema, dict):
        if "oneOf" in schema and "anyOf" not in schema:
            schema["anyOf"] = schema.pop("oneOf")
        value_type = schema.get("type")
        if isinstance(value_type, list):
            non_null = [item for item in value_type if item != "null"]
            if len(non_null) == 1:
                schema["type"] = non_null[0]
                schema["nullable"] = True
        for value in schema.values():
            _normalize_gemini_schema(value)
    elif isinstance(schema, list):
        for value in schema:
            _normalize_gemini_schema(value)


# The exact env var this project uses for the Gemini Live API key. The
# google-genai SDK also honours GOOGLE_API_KEY, but we standardise on
# GEMINI_API_KEY to avoid colliding with GCP Application Default Credentials.
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_LIVE_MODEL_ENV = "GEMINI_LIVE_MODEL"

# Shown in UIs / logs instead of any key material.
REDACTED = "[redacted]"


class GeminiConfigError(RuntimeError):
    """Raised when the Gemini Live API is used without a configured API key."""


def gemini_live_missing(settings: Settings | None = None) -> list[str]:
    """Return the names of Gemini config that are absent (names only)."""
    settings = settings or get_settings()
    missing: list[str] = []
    if not (settings.gemini_api_key and settings.gemini_api_key.strip()):
        missing.append(GEMINI_API_KEY_ENV)
    return missing


def gemini_live_ready(settings: Settings | None = None) -> bool:
    """True when the Gemini Live API key is present and nonempty."""
    return not gemini_live_missing(settings)


def _require_api_key(settings: Settings) -> str:
    """Return the key or raise a clear, actionable error naming the variable."""
    key = (settings.gemini_api_key or "").strip()
    if not key:
        raise GeminiConfigError(
            "Gemini Live API is not configured: set the GEMINI_API_KEY "
            "environment variable (see .env.example). No Gemini request was made."
        )
    return key


def build_gemini_client(settings: Settings | None = None) -> Any:
    """Construct a ``google.genai.Client`` using the configured key.

    The key is supplied explicitly so the SDK does not fall back to ambient
    environment variables, keeping loading auditable in one place.
    """
    settings = settings or get_settings()
    api_key = _require_api_key(settings)
    try:
        from google import genai  # deferred optional import
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise GeminiConfigError(
            "google-genai is not installed. Install it with: pip install google-genai"
        ) from exc
    return genai.Client(api_key=api_key)


def gemini_live_status(settings: Settings | None = None) -> dict[str, Any]:
    """Safe, names-only status for logs and the frontend.

    Never includes the key value — only whether it is set, and which variable to
    configure if it is not.
    """
    settings = settings or get_settings()
    missing = gemini_live_missing(settings)
    return {
        "provider": "gemini-live",
        "sdk": "google-genai",
        "ready": not missing,
        "missing_config": missing,
        "model": settings.gemini_live_model,
        "api_key_env": GEMINI_API_KEY_ENV,
        "api_key_configured": not missing,
        # Explicitly never the key; useful if a caller ever formats this dict.
        "api_key_preview": REDACTED,
    }
