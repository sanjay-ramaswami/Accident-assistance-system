from __future__ import annotations

import asyncio
import json
import logging
import math
import re
from threading import Lock
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from datetime import UTC
from pathlib import Path
from typing import Any
from typing import Literal

import gradbot
import httpx
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi import WebSocket
from fastapi import WebSocketDisconnect
from fastapi import WebSocketException
from fastapi import status
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .gradbot_adapter import build_session_config
from .dispatch_tools import assess_searchable_location_query
from .dispatch_tools import dispatch_tool_call
from .dispatch_tools import extract_address_candidate
from .enhanced_audio import finalize_audio_artifacts
from .enhanced_audio import make_enhancer
from .enhanced_audio import PCM_SAMPLE_RATE
from .first_aid import analyze_first_aid
from .incident_assessment import build_incident_assessment
from .incident_classifier import classify_incident_text_async
from .incident_classifier import load_incident_text_model
from .incident_classifier import IncidentTextModelError
from .incident_classifier import make_input_id
from .demo_simulation import advance_demo_simulation
from .demo_simulation import create_demo_simulation
from .demo_simulation import update_demo_simulation_location
from .demo_simulation import reset_demo_simulation
from .session_store import SessionRecorder
from .session_store import SessionPersistenceError
from .session_store import get_session
from .session_store import list_sessions
from .session_store import utc_now
from .settings import get_settings
from .slm_fact_ledger import build_fact_priority_packet
from .slm_fact_ledger import build_ledger_prompt_context
from .slm_fact_ledger import classify_location_kind
from .slm_fact_ledger import empty_shared_ledger
from .slm_fact_ledger import fact_ledger_to_shared_hard_facts
from .slm_fact_ledger import FACT_LEDGER_SYSTEM_PROMPT
from .slm_fact_ledger import FactLedger
from .slm_fact_ledger import has_dispatchable_location_cue
from .slm_fact_ledger import is_immediate_fact_ledger_trigger
from .slm_fact_ledger import is_trivial_caller_turn
from .slm_fact_ledger import latest_substantive_caller_window
from .slm_fact_ledger import normalize_shared_ledger
from .slm_fact_ledger import shared_hard_facts_to_fact_ledger
from .stt_rescue import append_live_audio_window
from .stt_rescue import clear_live_audio_window
from .stt_rescue import maybe_run_stt_rescue
from .stt_rescue import stt_rescue_session_patch
from .triage import analyze_session_triage
from .triage import apply_rule_overrides
from .triage import enforce_rule_overrides
from .triage import get_triage_engine
from .triage import TriageResult
from .triage import triage_session_patch
from .triage_ml import ModelArtifactError
from .triage_ml import normalize_complaint_code
from .triage_ml import predict_complaint_chapter
from .gemini_live import GEMINI_API_KEY_ENV
from .gemini_live import gemini_live_missing
from .gemini_live import gemini_live_ready
from .gemini_live import gemini_live_status
from .gemini_live import GeminiLiveAdapter


logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Validate optional Gemini Live configuration at startup.

    The key is never printed — only whether it is present, and which variable to
    set if it is not. A missing Gemini key does not stop the app: the existing
    Gradium/OpenAI voice stack and all local features keep working. It is a
    clear, actionable startup error in the logs, not a fatal crash.
    """
    settings = get_settings()
    missing = gemini_live_missing(settings)
    if missing:
        logger.error(
            "Gemini Live API is not configured; the Gemini Live path is disabled. "
            "Missing: %s. Set %s in your environment or .env to enable it. "
            "The rest of the application is unaffected.",
            ", ".join(missing),
            GEMINI_API_KEY_ENV,
        )
    else:
        logger.info(
            "Gemini Live API configured (key present, value not logged); model=%s",
            settings.gemini_live_model,
        )
    try:
        artifact = load_incident_text_model()
        metadata = artifact["metadata"]
        logger.info(
            "incident_text_model status=ready model=%s version=%s training_examples=%s prototype_only=true",
            metadata["model_id"], metadata["model_version"], metadata["training_example_count"],
        )
    except FileNotFoundError:
        logger.warning("incident_text_model status=unavailable reason=artifact_missing")
    except IncidentTextModelError:
        logger.error("incident_text_model status=error reason=artifact_invalid")
    yield


app = FastAPI(title="Emergency Dispatcher", lifespan=_lifespan)

# Only one live WebSocket may own a session_id at a time. A duplicate or stale
# "start" must not interleave writes into the same persisted session file.
_ACTIVE_LIVE_SESSIONS: set[str] = set()
_DEMO_FLEET_LOCK = Lock()


def _acquire_live_session(session_id: str) -> bool:
    """Atomically claim a session_id for one live socket. Returns False if taken."""
    if session_id in _ACTIVE_LIVE_SESSIONS:
        return False
    _ACTIVE_LIVE_SESSIONS.add(session_id)
    return True


def _release_live_session(session_id: str) -> None:
    _ACTIVE_LIVE_SESSIONS.discard(session_id)


async def _cancel_and_drain_receive_task(task: asyncio.Future[Any] | None) -> Exception | None:
    """Cancel a pending provider receive and always retrieve its terminal error."""
    if task is None:
        return None
    if not task.done():
        task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        return None
    except Exception as exc:
        return exc
    return None


@app.exception_handler(SessionPersistenceError)
async def session_persistence_error_handler(_request: Request, _exc: SessionPersistenceError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": "Session storage is unavailable; the report could not be saved."},
    )
REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = REPO_ROOT / "frontend"
GRADBOT_JS_DIR = Path(gradbot.routes.__file__).resolve().parent / "js_audio"

app.mount("/demo-static", StaticFiles(directory=FRONTEND_DIR), name="demo_static")
app.mount("/static/js", StaticFiles(directory=GRADBOT_JS_DIR), name="gradbot_js")

AUTO_PIN_SAFE_LOCATION_REASONS = {
    "explicit_address",
    "street_number",
}
AUTO_LOCATION_HINT_PATTERN = re.compile(
    r"\b(?:outside|beside|opposite|near|close to|around|inside|on|at|in|by)\s+([A-Za-z0-9.,\-'’\s]{3,100}?)(?=,\s*(?:and|but|the|a|an|someone|person|there|he|she|they)\b|[.!?]|$)",
    re.IGNORECASE,
)
AUTO_CITY_COUNTRY_PATTERN = re.compile(
    r"\b(?:in|at|near|around)\s+([A-Za-z][A-Za-z.'’ -]{1,48}?)\s*,\s*([A-Za-z][A-Za-z.'’ -]{1,48}?)(?=\b(?:and|but|where|with)\b|[.!?]|$)",
    re.IGNORECASE,
)
AUTO_SUB_LOCATION_PATTERN = re.compile(
    r"\b(\d+(?:st|nd|rd|th)\s+floor|first floor|second floor|third floor|fourth floor|room\s+\w+|unit\s+\w+|stairwell|entrance|gate|lobby|basement)\b",
    re.IGNORECASE,
)
AUTO_EMERGENCY_CONTEXT_PATTERN = re.compile(
    r"\b("
    r"accident|crash|collision|hit|rolled over|upside down|"
    r"not responding|unresponsive|unconscious|not moving|"
    r"bleeding|blood|injured|injury|hurt|trapped|stuck|"
    r"fire|smoke|burning|gas leak|explosion|"
    r"gun|weapon|knife|hostage|armed|shooting|"
    r"breathing|not breathing|can't breathe|cannot breathe"
    r")\b",
    re.IGNORECASE,
)
SEMANTIC_NEGATION_VALUE_MAP = {
    "yes": "yes",
    "no": "no",
}
LEDGER_BINARY_FIELDS = {
    "child_present",
    "bleeding_status",
    "breathing_status",
    "consciousness_status",
    "trapped_status",
}
LEDGER_LOCATION_SAFE_KINDS = {"exact_address", "road_or_junction", "place_name"}
LOCATION_QUESTION_PATTERNS = {
    "ask_where": re.compile(r"\bwhere are you(?: right now)?\b", re.IGNORECASE),
    "ask_exact_address": re.compile(
        r"\b(?:exact address|what(?:'s| is)? the address|know the exact address|what address)\b",
        re.IGNORECASE,
    ),
    "confirm_candidate": re.compile(
        r"\b(?:you said|did you say|is that correct|is that right|correct\?)\b",
        re.IGNORECASE,
    ),
    "ask_spell": re.compile(r"\bspell\b", re.IGNORECASE),
    "ask_landmark": re.compile(
        r"\b(?:landmark|describe your surroundings|what do you see|what is around you|what's around you|road sign)\b",
        re.IGNORECASE,
    ),
}
LOCATION_DEAD_END_ATTEMPTS = 4
LOCATION_KIND_STRENGTH = {
    "none": 0,
    "vague": 0,
    "sub_location_only": 1,
    "place_name": 2,
    "road_or_junction": 3,
    "exact_address": 4,
}
LEDGER_JOB_REGISTRY: dict[str, dict[str, Any]] = {}


class TriageDemoRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    engine: Literal["auto", "slm", "heuristic"] = "auto"


class FirstAidRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class ComplaintCodeRequest(BaseModel):
    code: str = Field(min_length=1, max_length=24)


class SessionReportRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    session_id: str = Field(min_length=1, max_length=128)
    status: str = Field(default="live", min_length=1, max_length=32)
    triage_mode: str | None = Field(default=None, max_length=32)
    report_revision: int | None = Field(default=None, ge=0)
    transcripts: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    page: dict[str, Any] = Field(default_factory=dict)


class DemoIncidentRequest(BaseModel):
    scenario_id: str = Field(min_length=1, max_length=64)
    location_text: str | None = Field(default=None, max_length=240)
    use_demo_fallback: bool = False
    resolve_scenario_location: bool = False


class DemoMapLocationRequest(BaseModel):
    lat: float = Field(ge=-90, le=90)
    lon: float = Field(ge=-180, le=180)


@app.post("/api/first-aid")
def first_aid_guidance(request: FirstAidRequest) -> dict[str, Any]:
    """Return local deterministic guidance without calling an LLM or persisting input."""
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter a caller description.")
    return analyze_first_aid(text)


@app.post("/api/triage/complaint-code")
def complaint_code_triage(request: ComplaintCodeRequest) -> dict[str, Any]:
    """Run the independent research model on a coded complaint, never on voice text."""
    try:
        code = normalize_complaint_code(request.code)
        return predict_complaint_chapter(code)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ModelArtifactError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/triage")
def triage_demo(request: TriageDemoRequest) -> dict[str, Any]:
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Enter an incident description.")

    settings = get_settings()
    engine_name = request.engine
    if engine_name == "auto":
        engine_name = "slm" if settings.triage_api_key and settings.triage_decoder_model else "heuristic"
    if engine_name == "slm":
        if not settings.triage_api_key:
            raise HTTPException(status_code=503, detail="SLM triage needs TRIAGE_API_KEY.")
        if not settings.triage_decoder_model:
            raise HTTPException(status_code=503, detail="SLM triage needs TRIAGE_DECODER_MODEL.")
    try:
        configured = settings.model_copy(update={"triage_engine": engine_name})
        engine = get_triage_engine(configured)
        result = engine.analyze_utterance(text, TriageResult())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Triage inference failed: {exc}") from exc

    overrides = apply_rule_overrides(text, result)
    result = enforce_rule_overrides(result, overrides)
    return {
        "selected_engine": engine.selected_engine,
        "runtime_provider": engine.runtime_provider,
        "advisory_only": True,
        "notice": "Decision support only. Follow emergency protocols and operator judgment.",
        "triage": result.model_dump(),
        "model_artifacts": getattr(engine, "last_artifacts", {}) or None,
        "rule_overrides": [override.model_dump() for override in overrides],
    }


def _run_kwargs() -> dict[str, Any]:
    settings = get_settings()
    kwargs: dict[str, Any] = {
        "gradium_api_key": settings.gradium_api_key,
    }
    llm_base_url = settings.openai_base_url or (
        "https://api.openai.com/v1" if settings.openai_api_key else None
    )
    if llm_base_url:
        kwargs["llm_base_url"] = llm_base_url
    if settings.openai_model:
        kwargs["llm_model_name"] = settings.openai_model
    if settings.openai_api_key:
        kwargs["llm_api_key"] = settings.openai_api_key
    return kwargs


def _live_voice_missing(settings: Any) -> list[str]:
    missing = [
        name
        for name, value in (
            ("GRADIUM_API_KEY", settings.gradium_api_key),
            ("AI_COUSTICS_API_KEY", settings.ai_coustics_api_key),
        )
        if not value
    ]
    if not settings.openai_model:
        missing.append("OPENAI_MODEL")
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    return missing


def _valid_session_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}", value) is not None
        and not value.endswith(".")
    )


def _ensure_llm_ready() -> Any:
    settings = get_settings()
    missing = _live_voice_missing(settings)
    if missing:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason=(
                f"Live voice is unavailable; configure: {', '.join(missing)}. "
                "The text triage demo at /demo works without live voice credentials."
            ),
        )
    return settings


def _resolve_live_dispatch_mode(value: Any | None = None) -> str:
    settings = get_settings()
    normalized = str(value or settings.live_dispatch_mode or "slm").strip().lower().replace("-", "_")
    if normalized in {"off", "none", "disabled", "llm", "llm_only"}:
        return "llm_only"
    return "slm"


async def _forward_provider_message(websocket: WebSocket, msg: Any, *, session_id: str, provider: str = "gradbot") -> None:
    """Relay one Gradbot output using its JSON metadata and binary audio contract."""
    schema = gradbot.schemas.from_msg(msg)
    if schema is not None:
        if msg.msg_type in {"stt_text", "tts_text"}:
            logger.info(
                "live provider transcript event: session=%s role=%s turn=%s start_s=%s stop_s=%s",
                session_id,
                "caller" if msg.msg_type == "stt_text" else "assistant",
                msg.turn_idx,
                msg.start_s,
                msg.stop_s,
            )
        elif msg.msg_type == "event" and msg.event is not None:
            event_name = _safe_provider_token(getattr(msg.event, "event_type", None)) or "unknown"
            logger.info(
                "provider_event provider=%s session=%s event=%s",
                provider,
                session_id,
                event_name,
            )
        await websocket.send_json(schema.model_dump())
    if msg.msg_type == "audio":
        await websocket.send_bytes(msg.data)


_SAFE_PROVIDER_TOKEN = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_QUOTA_ERROR_TYPES = {"insufficient_quota", "credit_balance_exhausted"}
LLM_PREFLIGHT_TIMEOUT_S = 10.0
LLM_START_TIMEOUT_S = 8.0
LLM_NO_RESPONSE_LIMIT = 2


def _safe_provider_token(value: Any) -> str | None:
    if isinstance(value, str) and _SAFE_PROVIDER_TOKEN.fullmatch(value):
        return value
    return None


def _provider_error_diagnostics(error: Any, *, provider_hint: str | None = None) -> dict[str, Any]:
    """Extract only allowlisted provider metadata; never stringify errors."""
    exception_type = type(error).__name__
    error_module = type(error).__module__.split(".", 1)[0]
    provider = error_module if error_module in {"openai", "gradium", "gradbot", "httpx", "google"} else None
    provider = provider or _safe_provider_token(provider_hint) or "unknown"

    status_code = getattr(error, "status_code", None)
    if not isinstance(status_code, int):
        response = getattr(error, "response", None)
        status_code = getattr(response, "status_code", None)
    if not isinstance(status_code, int) or not 100 <= status_code <= 599:
        status_code = None

    bodies: list[dict[str, Any]] = []
    if isinstance(error, dict):
        bodies.append(error)
    for attr in ("body", "data"):
        body = getattr(error, attr, None)
        if isinstance(body, dict):
            bodies.append(body)
    # Some SDK adapters wrap the original HTTP error in a RuntimeError and
    # preserve its short status/type/code in the message. Extract only these
    # allowlisted tokens; the original message is never persisted or logged.
    wrapped_text = ""
    wrapped_text = ""
    if not bodies and not isinstance(error, dict):
        try:
            wrapped_text = str(error)
        except Exception:
            wrapped_text = ""
        if status_code is None:
            status_match = re.search(r"\b(?:HTTP(?:\s+status)?|error\s+code)\s*[:=]?\s*(\d{3})\b", wrapped_text, re.IGNORECASE)
            if status_match:
                parsed_status = int(status_match.group(1))
                status_code = parsed_status if 100 <= parsed_status <= 599 else None
    nested = [body.get("error") for body in bodies if isinstance(body.get("error"), dict)]
    bodies.extend(nested)
    reported_provider = next(
        (
            token
            for body in bodies
            if (token := _safe_provider_token(body.get("provider"))) in {"openai", "gradium", "gradbot"}
        ),
        None,
    )
    if reported_provider:
        provider = reported_provider
    error_type = next(
        (_safe_provider_token(body.get("type")) for body in bodies if _safe_provider_token(body.get("type"))),
        None,
    )
    error_code = next(
        (_safe_provider_token(body.get("code")) for body in bodies if _safe_provider_token(body.get("code"))),
        None,
    )
    if wrapped_text:
        if re.search(r"\bopenai\b", wrapped_text, re.IGNORECASE):
            provider = "openai"
        if re.search(r"\binsufficient_quota\b", wrapped_text, re.IGNORECASE):
            error_type = error_type or "insufficient_quota"
            error_code = error_code or "insufficient_quota"
        if re.search(r"\bcredit_balance_exhausted\b", wrapped_text, re.IGNORECASE):
            error_code = "credit_balance_exhausted"
            error_type = error_type or "credit_balance_exhausted"

    if provider in {"google", "gemini"} and (
        error_type in _QUOTA_ERROR_TYPES
        or error_code in _QUOTA_ERROR_TYPES
        or status_code == 429
        or error_code in {"RESOURCE_EXHAUSTED", "resource_exhausted"}
    ):
        category = "gemini_quota_exhausted"
    elif error_type in _QUOTA_ERROR_TYPES or error_code in _QUOTA_ERROR_TYPES:
        category = "llm_quota_exhausted"
        provider = "openai" if provider in {"openai", "unknown", "gradbot"} else provider
    elif provider in {"google", "gemini"} and (status_code == 429 or error_code in {"RESOURCE_EXHAUSTED", "resource_exhausted"}):
        category = "gemini_quota_exhausted"
    elif provider in {"google", "gemini"} and status_code in {401, 403}:
        category = "gemini_auth_failed"
    elif provider == "openai" and status_code == 429:
        category = "llm_rate_limited"
    elif provider == "openai":
        category = "llm_request_failed"
    elif isinstance(error, (TimeoutError, ConnectionError)) or exception_type in {
        "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "WebSocketDisconnect"
    }:
        category = "provider_transport_error"
    elif status_code is not None:
        category = "provider_http_error"
    else:
        category = "provider_error"

    diagnostic = {
        "provider": provider,
        "category": category,
        "exception_type": _safe_provider_token(exception_type) or "UnknownError",
        "http_status": status_code,
        "error_type": error_type,
        "error_code": error_code,
    }
    provider_close_code = getattr(error, "code", None)
    if isinstance(provider_close_code, int) and 1000 <= provider_close_code <= 4999:
        diagnostic["provider_close_code"] = provider_close_code
        close_reason = _safe_provider_close_reason(error)
        if close_reason:
            diagnostic["provider_close_reason"] = close_reason
    return diagnostic


def _safe_provider_close_reason(error: Any) -> str | None:
    """Classify provider close text without retaining arbitrary remote content."""
    details = getattr(error, "details", None)
    if not isinstance(details, str):
        details = getattr(error, "reason", None)
    if not isinstance(details, str):
        return None
    normalized = details.casefold()
    if "1006" in normalized or "abnormal closure" in normalized:
        return "abnormal_closure_no_close_frame"
    if "unexpected function call" in normalized:
        return "unexpected_function_call"
    if "invalid frame payload" in normalized:
        return "invalid_frame_payload"
    if "invalid argument" in normalized:
        return "invalid_argument"
    if "deadline expired" in normalized:
        return "deadline_expired"
    if "quota" in normalized or "resource_exhausted" in normalized:
        return "quota_exhausted"
    return "unclassified"


def _safe_traceback_summary(error: BaseException) -> str:
    """Return stack locations only; exception text and locals may contain secrets."""
    frames = traceback.extract_tb(error.__traceback__)[-8:]
    return " <- ".join(
        f"{Path(frame.filename).name}:{frame.lineno}:{_safe_provider_token(frame.name) or 'unknown'}"
        for frame in frames
    ) or "unavailable"


def _safe_provider_error_message(diagnostics: dict[str, Any]) -> str:
    if diagnostics.get("category") == "gemini_quota_exhausted":
        return "Gemini Live quota is exhausted; no live response was generated."
    if diagnostics.get("category") == "gemini_auth_failed":
        return "Gemini Live authentication failed; check GEMINI_API_KEY."
    if diagnostics.get("category") == "llm_quota_exhausted":
        return "The language model quota is exhausted; no live response was generated."
    if diagnostics.get("category") == "llm_rate_limited":
        return "The language model rate limit was reached; no live response was generated."
    if diagnostics.get("category") == "provider_transport_error":
        return "The live provider transport failed; no response was completed."
    return "The live provider failed; no response was completed."


async def _llm_preflight_check(settings: Any) -> dict[str, Any]:
    """Probe the configured LLM endpoint once with a one-token request.

    The gradbot native layer swallows upstream LLM failures (it only logs them),
    so a permanent failure such as exhausted account quota would otherwise
    surface as a silent session that later dies with WebSocket code 1011. This
    pre-flight converts that into an explicit, sanitized result. It runs at most
    once per session start and never retries known quota exhaustion.
    """
    if not settings.openai_api_key or not settings.openai_model:
        return {"status": "skipped", "reason": "llm_not_configured"}
    base_url = (settings.openai_base_url or "https://api.openai.com/v1").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=LLM_PREFLIGHT_TIMEOUT_S) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                json={
                    "model": settings.openai_model,
                    "max_tokens": 1,
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
    except httpx.HTTPError as exc:
        # Only the exception type name is retained; messages may embed URLs or tokens.
        return {"status": "unreachable", "error_type": type(exc).__name__}
    if response.status_code == 200:
        return {"status": "ok"}
    error_type: str | None = None
    error_code: str | None = None
    try:
        body = response.json()
        error = body.get("error") if isinstance(body, dict) else None
        if isinstance(error, dict):
            error_type = _safe_provider_token(error.get("type"))
            error_code = _safe_provider_token(error.get("code"))
    except Exception:
        pass
    quota_exhausted = response.status_code == 429 and (
        error_type in _QUOTA_ERROR_TYPES or error_code in _QUOTA_ERROR_TYPES
    )
    return {
        "status": "failed",
        "http_status": response.status_code,
        "error_type": error_type,
        "error_code": error_code,
        "quota_exhausted": quota_exhausted,
    }


def _browser_audio_diagnostics(report: dict[str, Any]) -> dict[str, Any]:
    """Summarize known UI audio/close events without retaining their text."""
    events = report.get("events")
    playback_error_types: dict[str, int] = {}
    socket_close_codes: list[int] = []
    audio_bytes_events = 0
    playback_started = False
    playback_diag_count = 0
    received_audio_bytes = 0
    decoded_audio_samples = 0
    queued_audio_chunks = 0
    played_samples = 0
    context_states: set[str] = set()
    worklet_error_types: dict[str, int] = {}
    for event in events[-1000:] if isinstance(events, list) else []:
        if not isinstance(event, dict) or not isinstance(event.get("text"), str):
            continue
        text = event["text"]
        match = re.fullmatch(r"audio stream or playback failed \(([A-Za-z0-9_.-]{1,64})\)", text, re.IGNORECASE)
        if match:
            error_type = _safe_provider_token(match.group(1)) or "UnknownError"
            playback_error_types[error_type] = playback_error_types.get(error_type, 0) + 1
        player_match = re.fullmatch(
            r"audio stream or playback failed stage=player error_type=([A-Za-z0-9_.-]{1,64}) context=([a-z]{1,16})",
            text,
        )
        if player_match:
            error_type = _safe_provider_token(player_match.group(1)) or "UnknownError"
            playback_error_types[error_type] = playback_error_types.get(error_type, 0) + 1
        worklet_match = re.fullmatch(r"audio playback failed stage=worklet error_type=([A-Za-z0-9_.-]{1,64}) context=([a-z]{1,16})", text)
        if worklet_match:
            error_type = _safe_provider_token(worklet_match.group(1)) or "UnknownError"
            worklet_error_types[error_type] = worklet_error_types.get(error_type, 0) + 1
        queue_match = re.fullmatch(r"audio playback failed stage=queue error_type=([A-Za-z0-9_.-]{1,64})", text)
        if queue_match:
            error_type = _safe_provider_token(queue_match.group(1)) or "UnknownError"
            playback_error_types[error_type] = playback_error_types.get(error_type, 0) + 1
        diag_match = re.fullmatch(
            r"audio_playback_diag stage=([a-z_]{1,32}) context=([a-z]{1,16}) received_chunks=(\d+) received_bytes=(\d+) decoded_samples=(\d+) queued_chunks=(\d+) played_s=(\d+(?:\.\d+)?) played_samples=(\d+) buffer_ms=(\d+(?:\.\d+)?) context_rate_hz=(\d+)",
            text,
        )
        if diag_match:
            playback_diag_count += 1
            context_states.add(diag_match.group(2))
            received_audio_bytes = max(received_audio_bytes, int(diag_match.group(4)))
            decoded_audio_samples = max(decoded_audio_samples, int(diag_match.group(5)))
            queued_audio_chunks = max(queued_audio_chunks, int(diag_match.group(6)))
            played_samples = max(played_samples, int(diag_match.group(8)))
        close_match = re.fullmatch(r"websocket closed \((\d{3,4})\)", text, re.IGNORECASE)
        if close_match:
            socket_close_codes.append(int(close_match.group(1)))
        if text == "dispatcher audio bytes received from provider":
            audio_bytes_events += 1
        if text.casefold() == "dispatcher audio playback started":
            playback_started = True
    return {
        "playback_error_count": sum(playback_error_types.values()),
        "playback_error_types": playback_error_types,
        "audio_worklet_error_types": worklet_error_types,
        "provider_audio_byte_events": audio_bytes_events,
        "playback_diagnostic_events": playback_diag_count,
        "received_audio_bytes": received_audio_bytes,
        "decoded_audio_samples": decoded_audio_samples,
        "queued_audio_chunks": queued_audio_chunks,
        "played_audio_samples": played_samples,
        "audio_context_states": sorted(context_states),
        "playback_started": playback_started,
        "websocket_close_codes": socket_close_codes[-10:],
    }


def _record_browser_audio_diagnostics(recorder: SessionRecorder, report: dict[str, Any]) -> None:
    diagnostics = _browser_audio_diagnostics(report)
    recorder.patch_server({"browser_audio_diagnostics": diagnostics})
    if any((diagnostics["playback_error_count"], diagnostics["provider_audio_byte_events"], diagnostics["playback_diagnostic_events"], diagnostics["websocket_close_codes"])):
        logger.info(
            "browser_audio_diagnostics session=%s playback_errors=%d playback_error_types=%s worklet_errors=%s provider_audio_byte_events=%d received_audio_bytes=%d decoded_samples=%d queued_chunks=%d played_samples=%d context_states=%s playback_started=%s websocket_close_codes=%s",
            recorder.session_id,
            diagnostics["playback_error_count"],
            sorted(diagnostics["playback_error_types"].keys()),
            sorted(diagnostics["audio_worklet_error_types"].keys()),
            diagnostics["provider_audio_byte_events"],
            diagnostics["received_audio_bytes"],
            diagnostics["decoded_audio_samples"],
            diagnostics["queued_audio_chunks"],
            diagnostics["played_audio_samples"],
            diagnostics["audio_context_states"],
            diagnostics["playback_started"],
            diagnostics["websocket_close_codes"],
        )


def _live_triage_settings() -> Any:
    return get_settings().model_copy(update={"triage_engine": "slm"})


def _fact_ledger_model_id(settings: Any) -> str | None:
    return settings.live_fact_ledger_model or settings.triage_decoder_model


def _live_triage_missing(settings) -> list[str]:
    missing: list[str] = []
    if not settings.triage_api_key:
        missing.append("TRIAGE_API_KEY")
    if not _fact_ledger_model_id(settings):
        missing.append("LIVE_FACT_LEDGER_MODEL")
    return missing


def _ensure_live_triage_ready_http():
    settings = _live_triage_settings()
    missing = _live_triage_missing(settings)
    if missing:
        raise HTTPException(
            status_code=503,
            detail=(
                "Live triage is pinned to the Pioneer SLM path. "
                f"Missing required config: {', '.join(missing)}."
            ),
        )
    return settings


def _ensure_live_triage_ready_ws():
    settings = _live_triage_settings()
    missing = _live_triage_missing(settings)
    if missing:
        raise WebSocketException(
            code=status.WS_1008_POLICY_VIOLATION,
            reason=(
                "Live triage is pinned to the Pioneer SLM path. "
                f"Missing required config: {', '.join(missing)}."
            ),
        )
    return settings


def _live_runtime_settings(mode: str) -> Any:
    if mode == "slm":
        return _live_triage_settings()
    return get_settings().model_copy(update={"triage_engine": "heuristic"})


def _ensure_live_runtime_ready_http(mode: str):
    if mode == "slm":
        return _ensure_live_triage_ready_http()
    return _live_runtime_settings(mode)


def _ensure_live_runtime_ready_ws(mode: str):
    if mode == "slm":
        return _ensure_live_triage_ready_ws()
    return _live_runtime_settings(mode)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "emergency-dispatcher",
        "status": "ready",
        "websocket": "/ws",
        "demo": "/demo",
        "review": "/review",
    }


@app.get("/api/audio-config")
def audio_config(triage_mode: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    mode = _resolve_live_dispatch_mode(triage_mode)
    voice_missing = _live_voice_missing(settings)
    triage_missing = _live_triage_missing(_live_triage_settings()) if mode == "slm" else []
    return {
        "default_dispatch_mode": _resolve_live_dispatch_mode(),
        "pcm": True,
        "pcm_input": True,
        "sample_rate": PCM_SAMPLE_RATE,
        "channels": 1,
        "chunk_samples": 1920,
        "live_voice_ready": not voice_missing,
        "live_voice_missing_config": voice_missing,
        "gemini_voice_missing_config": gemini_live_missing(settings),
        "triage_ready": not triage_missing,
        "triage_missing_config": triage_missing,
        # Gemini Live is an optional additive provider. Names/booleans only —
        # the API key value is never included in any response.
        "gemini_live_ready": gemini_live_ready(settings),
        "gemini_live_missing_config": gemini_live_missing(settings),
    }


@app.get("/api/gemini-live/status")
def gemini_live_status_endpoint() -> dict[str, Any]:
    """Names-only Gemini Live readiness. The API key value is never returned."""
    return gemini_live_status(get_settings())


@app.get("/demo")
def demo() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/review")
def review() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "review.html")


@app.post("/api/session-report")
async def session_report(request: SessionReportRequest) -> dict[str, Any]:
    report = request.model_dump(exclude_none=True)
    session_id = report.get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="Missing session_id")
    if not _valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    revision = report.get("report_revision")
    if revision is not None:
        current_report = get_session(session_id).get("client", {}).get("report_revision")
        if current_report is not None and int(revision) <= int(current_report):
            current = get_session(session_id)
            ledger = _shared_ledger(current)
            return {
                "status": "ok",
                "stale_report_ignored": True,
                "incident_assessment": current.get("incident_assessment"),
                "ledger_update": {
                    "version": ledger.get("version"),
                    "prompt_context": build_ledger_prompt_context(ledger),
                },
                "dispatch_update": None,
                "triage_update": None,
                "triage_meta": current.get("triage_meta") or {},
            }
    live_dispatch_mode = _resolve_live_dispatch_mode(report.get("triage_mode"))
    triage_unavailable: list[str] = []
    if live_dispatch_mode == "slm":
        try:
            triage_settings = _ensure_live_runtime_ready_http(live_dispatch_mode)
        except HTTPException as exc:
            if exc.status_code != 503:
                raise
            missing = _live_triage_missing(_live_triage_settings())
            if not missing:
                raise
            triage_unavailable = missing
            triage_settings = _live_triage_settings()
    else:
        triage_settings = _ensure_live_runtime_ready_http(live_dispatch_mode)
    recorder = SessionRecorder(session_id)
    _record_browser_audio_diagnostics(recorder, report)
    recorder.patch_client_report(report)
    recorder.patch_server(
        {
            "triage_engine": live_dispatch_mode,
            "live_dispatch_mode": live_dispatch_mode,
        }
    )
    session = get_session(session_id)
    stt_rescue_event = None if report.get("demo_mode") else await maybe_run_stt_rescue(
        session_id=session_id, session=session, settings=triage_settings,
    )
    if stt_rescue_event is not None:
        recorder.patch_session(stt_rescue_session_patch(session, stt_rescue_event))
        session = get_session(session_id)
    caller_turns = _latest_transcript_items(session, "user")
    latest_caller_turn = next(
        (
            turn for turn in reversed(caller_turns)
            if isinstance(turn, dict) and isinstance(turn.get("text"), str) and turn["text"].strip()
        ),
        None,
    )
    if latest_caller_turn is not None:
        caller_text = latest_caller_turn["text"]
        turn_at = str(latest_caller_turn.get("at") or "") or None
        input_id = make_input_id(session_id=session_id, turn_at=turn_at, text=caller_text)
        previous_classification = session.get("natural_language_classification") or {}
        if previous_classification.get("input_id") != input_id:
            classifier_started = time.monotonic()
            logger.info(
                "incident_text_classifier session=%s input_id=%s stage=request status=pending",
                session_id,
                input_id,
            )
            try:
                classification = await classify_incident_text_async(
                    caller_text,
                    session_id=session_id,
                    incident_id=session_id,
                    turn_at=turn_at,
                )
            except Exception as exc:
                logger.warning(
                    "incident_text_classification_failed session=%s input_id=%s error_type=%s",
                    session_id,
                    input_id,
                    type(exc).__name__,
                )
                classification = {
                    "classifier_id": "incident-text-classifier",
                    "taxonomy_version": "incident-text-v1-unavailable",
                    "status": "error",
                    "model_version": None,
                    "predicted_category": None,
                    "class_scores": {},
                    "scores_calibrated": False,
                    "training_coverage": [],
                    "out_of_distribution_status": "unknown",
                    "uncertainty_reason": "inference_execution_failed",
                    "input_id": input_id,
                    "session_id": session_id,
                    "incident_id": session_id,
                    "human_review_required": True,
                    "advisory_only": True,
                }
            logger.info(
                "incident_text_classifier session=%s input_id=%s stage=response status=%s elapsed_ms=%.1f predicted=%s",
                session_id,
                input_id,
                classification.get("status", "unknown"),
                max(0.0, time.monotonic() - classifier_started) * 1000,
                bool(classification.get("predicted_category")),
            )
            recorder.patch_session({"natural_language_classification": classification})
            session = get_session(session_id)
    if live_dispatch_mode == "llm_only":
        # The voice-only mode still receives a deterministic, local triage
        # update; it never calls the paid triage provider.
        local_triage_settings = get_settings().model_copy(update={"triage_engine": "heuristic"})
        triage_update = await asyncio.to_thread(analyze_session_triage, session, local_triage_settings)
        if triage_update is not None:
            recorder.patch_session(triage_session_patch(session, triage_update))
        if not report.get("demo_mode"):
            await asyncio.to_thread(_auto_follow_location_resolution, session, recorder)
        session = get_session(session_id)
    elif triage_unavailable:
        triage_update = None
        recorder.patch_server({
            "triage_status": "unavailable",
            "triage_unavailable_reason": f"Live triage unavailable; missing configuration: {', '.join(triage_unavailable)}.",
        })
    else:
        triage_update = None
    if not session.get("shared_ledger"):
        recorder.patch_session({"shared_ledger": empty_shared_ledger()})
        session = get_session(session_id)
    if report.get("demo_mode"):
        ledger = await asyncio.to_thread(
            _refresh_shared_ledger_runtime,
            session,
            recorder,
            allow_external_geocoding=False,
        )
    else:
        ledger = await asyncio.to_thread(_refresh_shared_ledger_runtime, session, recorder)
    session = get_session(session_id)
    if live_dispatch_mode == "slm" and not triage_unavailable:
        _schedule_shared_ledger_refresh(session_id, triage_settings)
    dispatch_update = None if triage_unavailable else await asyncio.to_thread(_auto_dispatch_services, session, recorder)
    session = get_session(session_id)
    caller_text = _combined_user_transcript(session)
    first_aid = analyze_first_aid(caller_text) if caller_text else None
    assessment = build_incident_assessment(session, first_aid=first_aid)
    natural = dict(assessment.get("natural_language_classification") or {})
    if natural.get("status") != "classified":
        rule_category = str((assessment.get("incident_type") or {}).get("value") or "Unknown")
        if rule_category != "Unknown":
            natural["result_source"] = "rule_based_demo"
            natural["rule_based_fallback"] = {
                "status": "available",
                "category": rule_category,
                "label": "Rule-based demo fallback — not an ML prediction",
            }
            assessment["natural_language_classification"] = natural
            recorder.patch_session({"natural_language_classification": natural})
    recorder.patch_session({"incident_assessment": assessment})
    return {
        "status": "saved" if triage_unavailable else "ok",
        "triage_mode": live_dispatch_mode,
        "triage_enabled": not bool(triage_unavailable),
        "triage_update": triage_update.model_dump(mode="json") if triage_update else None,
        "ledger_update": {
            "version": ledger.get("version"),
            "prompt_context": build_ledger_prompt_context(ledger),
        },
        "dispatch_update": dispatch_update,
        "incident_assessment": assessment,
        "triage_meta": {
            "selected_engine": "heuristic" if live_dispatch_mode == "llm_only" else "shared_ledger_slm",
            "runtime_provider": "local_rules" if live_dispatch_mode == "llm_only" else "unavailable" if triage_unavailable else "pioneer",
            "dispatchable": bool(session.get("dispatch_services", {}).get("dispatchable")),
            "autonomy_allowed": False,
            "human_review_required": True,
            "missing_or_unknown_fields": list((ledger.get("priority") or {}).get("missing_fields") or []),
            "missing_config": triage_unavailable,
        },
        "stt_rescue_meta": session.get("stt_rescue_meta") or {},
        "stt_rescue_event": stt_rescue_event.model_dump() if stt_rescue_event else None,
    }


def _load_emergency_scenarios() -> list[dict[str, Any]]:
    path = REPO_ROOT / "data" / "emergency_scenarios.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError("Emergency scenario data must be a list")
    return [item for item in payload if isinstance(item, dict) and item.get("id") and item.get("transcript")]


@app.get("/api/demo/scenarios")
def emergency_demo_scenarios() -> list[dict[str, str]]:
    """List bundled deterministic inputs without returning private session data."""
    return [{"id": str(item["id"]), "title": str(item.get("title") or item["id"])} for item in _load_emergency_scenarios()]


@app.post("/api/demo/incident")
async def run_emergency_demo(request: DemoIncidentRequest) -> dict[str, Any]:
    """Run a bundled transcript through the real local session/ledger/triage path.

    Demo inputs are synthetic and do not open a microphone, call a model API,
    or contact dispatch services. A caller-specified place may use the
    configured geocoder; the configured fallback requires explicit opt-in.
    """
    scenario = next((item for item in _load_emergency_scenarios() if item["id"] == request.scenario_id), None)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Unknown emergency demo scenario.")
    session_id = f"demo-{uuid.uuid4().hex}"
    recorder = SessionRecorder(session_id)
    recorder.patch_server({
        "status": "simulated",
        "live_dispatch_mode": "llm_only",
        "triage_engine": "heuristic",
        "provider": "deterministic_demo",
        "demo_mode": True,
    })
    report = SessionReportRequest.model_validate({
        "session_id": session_id,
        "status": "simulated",
        "triage_mode": "llm_only",
        "report_revision": 1,
        "demo_mode": True,
        "transcripts": {
            "user": [{"at": utc_now(), "text": str(scenario["transcript"]), "simulated": True}],
            "agent": [],
        },
        "events": [],
        "metrics": {},
        "page": {"href": "/demo", "demo_scenario": str(scenario["id"])},
    })
    result = await session_report(report)
    session_recorder = SessionRecorder(session_id)
    with _DEMO_FLEET_LOCK:
        unavailable_unit_ids: set[str] = set()
        for summary in list_sessions(limit=10000):
            existing_id = str(summary.get("session_id") or "")
            if not existing_id.startswith("demo-"):
                continue
            existing = get_session(existing_id)
            active_simulation = existing.get("demo_simulation")
            if not isinstance(active_simulation, dict):
                continue
            active_simulation = advance_demo_simulation(active_simulation)
            if active_simulation.get("stage") not in {"COMPLETED", "RESET", "NO_UNIT_AVAILABLE"}:
                selected_id = active_simulation.get("selected_unit_id")
                if selected_id:
                    unavailable_unit_ids.add(str(selected_id))
        explicit_location_phrase = _normalize_runtime_text(request.location_text)
        extracted, _, _ = _infer_location_bundle_from_text(str(scenario["transcript"]))
        location_phrase = explicit_location_phrase or extracted
        should_geocode = bool(explicit_location_phrase or request.resolve_scenario_location)
        incident_location = None
        location_error = None
        if location_phrase and re.fullmatch(r"marine\s+drive", location_phrase, flags=re.IGNORECASE):
            location_error = "ambiguous_location"
        elif location_phrase and should_geocode:
            geocoded = await asyncio.to_thread(
                dispatch_tool_call,
                "lookup_address",
                {"address_text": location_phrase, "allow_best_effort": True},
            )
            try:
                latitude, longitude = float(geocoded.get("lat")), float(geocoded.get("lon"))
                coordinates_valid = -90 <= latitude <= 90 and -180 <= longitude <= 180
            except (TypeError, ValueError):
                latitude = longitude = None
                coordinates_valid = False
            if geocoded.get("geocoded") and coordinates_valid and geocoded.get("normalized_address"):
                approximate = bool(geocoded.get("candidate_only") or geocoded.get("map_approximation") or geocoded.get("confidence") == "low")
                incident_location = {
                    "label": geocoded["normalized_address"], "lat": latitude, "lon": longitude,
                    "source": geocoded.get("source") or "configured_geocoder",
                    "original_phrase": location_phrase,
                    "confidence": "low" if approximate else str(geocoded.get("confidence") or "medium"),
                    "resolution_quality": "named_place_approximation" if approximate else "geocoded_address",
                    "verified_from_caller": False,
                    "display_label": "APPROXIMATE PLACE — CONFIRM EXACT INCIDENT POINT" if approximate else "GEOCODED PLACE — CONFIRM EXACT INCIDENT POINT",
                    "clarification": "This is an approximate place pin. What nearby entrance, road, hospital, or landmark is closest to the incident?" if approximate else "Please confirm the exact incident address or nearest entrance.",
                }
            else:
                location_error = str(geocoded.get("reason") or geocoded.get("error_category") or "geocoder_no_match")
        elif location_phrase:
            location_error = "geocoding_not_requested"
        use_fallback = bool(request.use_demo_fallback and incident_location is None)
        simulation = create_demo_simulation(
            session_id=session_id,
            unavailable_unit_ids=unavailable_unit_ids,
            incident_location=incident_location,
            use_demo_fallback=use_fallback,
        )
        if use_fallback and simulation.get("location"):
            simulation["location"]["original_phrase"] = location_phrase
            simulation["location"]["unresolved_caller_phrase"] = bool(location_phrase)
        # Persist the reservation while holding the fleet lock so concurrent
        # demo requests cannot both select the same currently-free unit.
        session_recorder.patch_session({"demo_simulation": simulation})
    session = get_session(session_id)
    assessment = dict(session.get("incident_assessment") or result.get("incident_assessment") or {})
    resolved_location = simulation.get("location")
    assessment["location"] = ({
        "candidate": resolved_location["label"],
        "original_phrase": resolved_location.get("original_phrase"),
        "status": "approximate" if resolved_location.get("confidence") == "low" else "needs_confirmation",
        "confidence": resolved_location.get("confidence"),
        "resolution_quality": resolved_location.get("resolution_quality") or "configured_demo_fallback",
        "source": resolved_location.get("source"),
        "latitude": resolved_location["lat"],
        "longitude": resolved_location["lon"],
        "caller_confirmed": False,
        "clarification": resolved_location.get("clarification") or resolved_location.get("display_label") or "Confirm a nearby landmark or exact address.",
    } if resolved_location else {
        "candidate": location_phrase,
        "status": "needs_confirmation" if location_error == "ambiguous_location" else "unresolved",
        "confidence": None,
        "resolution_quality": "unresolved",
        "source": "caller_transcript" if location_phrase else None,
        "latitude": None,
        "longitude": None,
        "caller_confirmed": False,
        "error_category": location_error or "no_location",
        "clarification": (
            "Which city is Marine Drive in, or what nearby landmark can you see?"
            if location_error == "ambiguous_location" and location_phrase and re.fullmatch(r"marine\s+drive", location_phrase, flags=re.IGNORECASE)
            else f"The place could not be uniquely verified ({location_error or 'no_location'}). Please provide its city and a nearby road, landmark, hospital, or entrance, or choose the configured demo fallback."
        ),
    })
    dispatch = dict(assessment.get("dispatch") or {})
    dispatch.update({"status": simulation["stage"], "simulation_only": True})
    assessment["dispatch"] = dispatch
    session_recorder.patch_session({"demo_simulation": simulation, "incident_assessment": assessment})
    return {
        **result,
        "incident_assessment": assessment,
        "demo_simulation": simulation,
        "demo": True,
        "scenario": {"id": str(scenario["id"]), "title": str(scenario.get("title") or scenario["id"])},
        "session_id": session_id,
        "demo_condition": str(scenario.get("demo_condition") or "deterministic_local_assessment"),
        "notice": str(scenario.get("notice") or "Synthetic transcript; local rules; live microphone and real dispatch are not used.") + (" Configured geocoder used for the supplied place." if incident_location else ""),
    }


@app.post("/api/demo/location/{session_id}")
def update_demo_location(session_id: str, request: DemoMapLocationRequest) -> dict[str, Any]:
    session = get_session(session_id)
    if not (session.get("server") or {}).get("demo_mode"):
        raise HTTPException(status_code=409, detail="Map correction is available only in a demo simulation.")
    simulation = session.get("demo_simulation")
    if not isinstance(simulation, dict):
        raise HTTPException(status_code=409, detail="No active demo simulation is available to update.")
    try:
        updated = update_demo_simulation_location(simulation, lat=request.lat, lon=request.lon)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    assessment = dict(session.get("incident_assessment") or {})
    assessment["location"] = {
        "candidate": updated["location"]["label"], "original_phrase": None,
        "status": "operator_selected", "confidence": "medium",
        "resolution_quality": "manual_map_selection", "source": "operator_map_selection",
        "latitude": request.lat, "longitude": request.lon,
        "caller_confirmed": False, "operator_confirmed": True,
        "clarification": "Operator-selected point for the simulation; confirm the caller's exact address when available.",
    }
    assessment_dispatch = dict(assessment.get("dispatch") or {})
    assessment_dispatch.update({"status": updated.get("stage"), "simulation_only": True})
    assessment["dispatch"] = assessment_dispatch
    SessionRecorder(session_id).patch_session({"demo_simulation": updated, "incident_assessment": assessment})
    return {"demo_simulation": updated, "incident_assessment": assessment}


@app.get("/api/sessions")
def api_sessions() -> list[dict[str, Any]]:
    return list_sessions()


@app.get("/api/sessions/{session_id}")
def api_session_detail(session_id: str) -> dict[str, Any]:
    session = get_session(session_id)
    if (session.get("server") or {}).get("demo_mode") and isinstance(session.get("demo_simulation"), dict):
        updated = advance_demo_simulation(session["demo_simulation"])
        if updated != session["demo_simulation"]:
            SessionRecorder(session_id).patch_session({"demo_simulation": updated})
            dispatch = dict((session.get("incident_assessment") or {}).get("dispatch") or {})
            dispatch.update({"status": updated["stage"], "simulation_only": True})
            assessment = dict(session.get("incident_assessment") or {})
            assessment["dispatch"] = dispatch
            SessionRecorder(session_id).patch_session({"incident_assessment": assessment})
            session = get_session(session_id)
    return session


@app.post("/api/demo/reset/{session_id}")
def reset_demo_incident(session_id: str) -> dict[str, Any]:
    if not _valid_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id")
    session = get_session(session_id)
    if not (session.get("server") or {}).get("demo_mode") or not isinstance(session.get("demo_simulation"), dict):
        raise HTTPException(status_code=404, detail="No active simulated demo exists for this session.")
    updated = reset_demo_simulation(session["demo_simulation"])
    assessment = dict(session.get("incident_assessment") or {})
    dispatch = dict(assessment.get("dispatch") or {})
    dispatch.update({"status": "RESET", "simulation_only": True})
    assessment["dispatch"] = dispatch
    SessionRecorder(session_id).patch_session({"demo_simulation": updated, "incident_assessment": assessment})
    return {"status": "reset", "simulation_only": True, "session_id": session_id, "demo_simulation": updated}


def _record_tool_result(
    recorder: SessionRecorder | None,
    *,
    name: str,
    args: dict[str, Any],
    result: dict[str, Any] | None = None,
    error: str | None = None,
    started_at: str | None = None,
    elapsed_ms: int | None = None,
) -> None:
    if recorder is None:
        return
    recorder.append_tool_call(
        name=name,
        args=args,
        result=result,
        error=error,
        started_at=started_at,
        elapsed_ms=elapsed_ms,
    )


def _normalize_runtime_text(value: Any | None) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _sanitize_location_candidate_text(value: Any | None) -> str | None:
    normalized = _normalize_runtime_text(value)
    if not normalized:
        return None
    normalized = re.sub(
        r"^(?:please\s+)?(?:find|search(?: for)?|look up|locate|pin|send help to|route to)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"^(?:near|close to|around|by|outside|inside|at|to|toward|towards)\s+",
        "",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = normalized.strip(" ,.")
    return normalized or None


def _looks_like_non_location_speech(value: Any | None) -> bool:
    candidate = _normalize_runtime_text(value)
    if not candidate:
        return True
    lowered = candidate.casefold()
    if "?" in candidate:
        return True
    if lowered in {
        "nearest landmark",
        "exact address",
        "the exact address",
        "a broken knee",
        "broken knee",
        "need help",
        "help me",
    }:
        return True
    if re.match(r"^(?:hello|hi|hey)\b", lowered):
        return True
    if re.match(r"^(?:can you|could you|will you|would you|do you|are you|please|thanks|thank you)\b", lowered):
        return True
    if re.match(r"^(?:have|has|had)\b", lowered):
        return True
    return False


def _best_effort_runtime_location_candidate(value: Any | None) -> str | None:
    candidate = _sanitize_location_candidate_text(value)
    if not candidate:
        return None
    if _looks_like_non_location_speech(candidate):
        return None
    lowered = candidate.casefold()
    if re.fullmatch(r"(?:the )?(?:junction|intersection)(?: (?:beside|near|opposite|by|at) the hospital)?", lowered):
        return None
    if len(lowered) < 4:
        return None
    generic_phrases = {
        "here",
        "at home",
        "home",
        "inside the building",
        "inside",
        "somewhere in berlin",
        "in the middle of the street",
        "middle of the street",
    }
    if lowered in generic_phrases:
        return None
    raw_tokens = [token for token in re.split(r"[\s,./-]+", lowered) if token]
    if len(raw_tokens) < 2:
        return None
    filtered_tokens = [
        token
        for token in raw_tokens
        if token not in {"me", "you", "we", "they", "there", "here", "around", "near", "close", "to", "at", "on", "in", "is", "am", "are", "the", "a", "an", "my", "your"}
        and token not in {"strasse", "straße", "allee", "platz", "ring", "ufer", "damm"}
    ]
    if len(filtered_tokens) < 2:
        return None
    place_hint_tokens = {
        "strasse",
        "straße",
        "allee",
        "platz",
        "ring",
        "ufer",
        "damm",
        "campus",
        "park",
        "gate",
        "bahnhof",
        "station",
        "feld",
        "theater",
        "theatre",
        "mall",
        "hospital",
        "school",
        "center",
        "centre",
        "tower",
        "tor",
    }
    if not any(token in place_hint_tokens for token in raw_tokens):
        return None
    if all(
        token in {
            "accident",
            "collision",
            "smoke",
            "fire",
            "bleeding",
            "blood",
            "trapped",
            "stuck",
            "hurt",
            "help",
            "daughter",
            "wife",
            "husband",
            "child",
            "people",
            "person",
        }
        for token in filtered_tokens
    ):
        return None
    return candidate


def _meaningful_runtime_location_candidate(value: Any | None) -> str | None:
    candidate = _sanitize_location_candidate_text(value)
    if not candidate:
        return None
    if _looks_like_non_location_speech(candidate):
        return None
    lowered = candidate.casefold()
    if re.fullmatch(r"(?:the )?(?:junction|intersection)(?: (?:beside|near|opposite|by|at) the hospital)?", lowered):
        return None
    if re.match(r"^(?:there is|there's|i am|i'm|hello|help)\b", lowered):
        return None
    noisy_tokens = {
        "there",
        "is",
        "smoke",
        "fire",
        "bleeding",
        "blood",
        "trapped",
        "stuck",
        "accident",
        "collision",
        "hurt",
        "daughter",
        "wife",
        "husband",
        "people",
        "involved",
        "around",
        "me",
    }
    tokens = [token for token in re.split(r"[\s,./-]+", lowered) if token]
    if len(tokens) > 3 and sum(token in noisy_tokens for token in tokens) >= 2:
        return None
    allowed, _reason = assess_searchable_location_query(candidate)
    if allowed:
        return candidate
    return None


def _runtime_location_candidate_kind(value: Any | None) -> str:
    candidate = _sanitize_location_candidate_text(value)
    if not candidate:
        return "none"
    kind = classify_location_kind(candidate)
    if kind != "none":
        return kind
    if _best_effort_runtime_location_candidate(candidate):
        lowered = candidate.casefold()
        if any(lowered.endswith(suffix) for suffix in ("strasse", "straße", "allee", "platz", "ring", "ufer", "damm")):
            return "road_or_junction"
        return "place_name"
    return "none"


def _has_emergency_context(transcript: str | None) -> bool:
    normalized = _normalize_runtime_text(transcript)
    if not normalized:
        return False
    if AUTO_EMERGENCY_CONTEXT_PATTERN.search(normalized):
        return True
    # Avoid triggering lookup/ticket automation on line checks or bare place mentions.
    return len(normalized.split()) >= 12 and bool(extract_address_candidate(normalized))


def _latest_tool_call(session: dict[str, Any], *names: str) -> dict[str, Any] | None:
    allowed = set(names)
    for tool_call in reversed(session.get("tool_calls") or []):
        if tool_call.get("name") in allowed:
            return tool_call
    return None


def _has_tool_call_for_query(session: dict[str, Any], names: tuple[str, ...], query: str) -> bool:
    normalized_query = _normalize_runtime_text(query).casefold()
    for tool_call in reversed(session.get("tool_calls") or []):
        if tool_call.get("name") not in names:
            continue
        args = tool_call.get("args") or {}
        candidate = args.get("address_text") or args.get("location_candidate") or ""
        if _normalize_runtime_text(candidate).casefold() == normalized_query:
            return True
    return False


def _has_tool_call_for_coords(session: dict[str, Any], name: str, lat: float | None, lon: float | None) -> bool:
    if lat is None or lon is None:
        return False
    target = (round(float(lat), 4), round(float(lon), 4))
    for tool_call in reversed(session.get("tool_calls") or []):
        if tool_call.get("name") != name:
            continue
        args = tool_call.get("args") or {}
        item_lat = args.get("lat")
        item_lon = args.get("lon")
        if item_lat is None or item_lon is None:
            continue
        candidate = (round(float(item_lat), 4), round(float(item_lon), 4))
        if candidate == target:
            return True
    return False


def _infer_location_bundle_from_text(text: str) -> tuple[str | None, str | None, str | None]:
    normalized = _normalize_runtime_text(text)
    if not normalized:
        return None, None, None

    sub_location_match = AUTO_SUB_LOCATION_PATTERN.search(normalized)
    sub_location = _normalize_runtime_text(sub_location_match.group(1)) if sub_location_match else None

    address = extract_address_candidate(normalized)
    if address:
        # Keep a location clause separate from a trailing explanation or
        # emergency fact. In particular, "near Marine Drive, but I don't
        # know which city" is still the ambiguous candidate "Marine Drive".
        address = re.split(r"\b(?:but|however)\b", address, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,.")
        return _sanitize_location_candidate_text(address), normalized, sub_location

    city_country_matches = list(AUTO_CITY_COUNTRY_PATTERN.finditer(normalized))
    for match in reversed(city_country_matches):
        city = _normalize_runtime_text(match.group(1)).strip(" ,.")
        country = _normalize_runtime_text(match.group(2)).strip(" ,.")
        if re.match(r"^(?:and|but|where|with)\b", country, flags=re.IGNORECASE):
            continue
        if city and country and city.casefold() not in {"the scene", "the accident", "the address"}:
            return f"{city}, {country}", normalized, sub_location

    matches = list(AUTO_LOCATION_HINT_PATTERN.finditer(normalized))
    for match in reversed(matches):
        candidate = _normalize_runtime_text(match.group(1))
        if not candidate:
            continue
        if re.match(r"^(?:me|you|we|they|here|there)\b", candidate, flags=re.IGNORECASE):
            continue
        if sub_location and sub_location.casefold() in candidate.casefold():
            candidate = _normalize_runtime_text(re.sub(re.escape(sub_location), "", candidate, flags=re.IGNORECASE))
        candidate = re.split(r"\b(?:and|but|only|except|with)\b", candidate, maxsplit=1, flags=re.IGNORECASE)[0].strip(" ,.")
        strong_candidate = _meaningful_runtime_location_candidate(candidate)
        if strong_candidate:
            return strong_candidate, normalized, sub_location
        best_effort_candidate = _best_effort_runtime_location_candidate(candidate)
        if best_effort_candidate:
            return best_effort_candidate, normalized, sub_location

    direct_candidate = _meaningful_runtime_location_candidate(normalized)
    if direct_candidate and len(normalized.split()) <= 6:
        note = normalized if direct_candidate.casefold() != normalized.casefold() else None
        return direct_candidate, note, sub_location
    direct_best_effort = _best_effort_runtime_location_candidate(normalized)
    if direct_best_effort and len(normalized.split()) <= 6:
        note = normalized if direct_best_effort.casefold() != normalized.casefold() else None
        return direct_best_effort, note, sub_location

    if has_dispatchable_location_cue(normalized) or sub_location:
        return None, normalized, sub_location
    return None, None, sub_location


def _infer_location_bundle_from_transcript(session: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    turns = session.get("client", {}).get("transcripts", {}).get("user", []) or []
    snippets = [_normalize_runtime_text(turn.get("text")) for turn in turns if _normalize_runtime_text(turn.get("text"))]
    if not snippets:
        return None, None, None
    combined = " ".join(snippets[-6:])
    sub_location_match = AUTO_SUB_LOCATION_PATTERN.search(combined)
    sub_location = _normalize_runtime_text(sub_location_match.group(1)) if sub_location_match else None
    address = extract_address_candidate(combined)
    if address:
        return _sanitize_location_candidate_text(address), combined, sub_location
    for text in reversed(snippets[-8:]):
        candidate, note, inferred_sub = _infer_location_bundle_from_text(text)
        if candidate or note or inferred_sub:
            return candidate, note, inferred_sub or sub_location
    return None, None, sub_location


def _latest_transcript_items(session: dict[str, Any], role: str) -> list[dict[str, Any]]:
    items = session.get("client", {}).get("transcripts", {}).get(role, []) or []
    return list(items) if isinstance(items, list) else []


def _location_question_kind(text: str | None) -> str | None:
    normalized = _normalize_runtime_text(text)
    if not normalized:
        return None
    for kind, pattern in LOCATION_QUESTION_PATTERNS.items():
        if pattern.search(normalized):
            return kind
    return None


def _location_attempt_state(session: dict[str, Any]) -> tuple[int, str | None]:
    attempts = 0
    last_kind = None
    for turn in _latest_transcript_items(session, "agent")[-12:]:
        kind = _location_question_kind(turn.get("text"))
        if kind:
            attempts += 1
            last_kind = kind
    return attempts, last_kind


def _last_user_response_text(session: dict[str, Any]) -> str | None:
    for turn in reversed(_latest_transcript_items(session, "user")):
        text = _normalize_runtime_text(turn.get("text"))
        if text:
            return text
    return None


def _latest_location_response_text(session: dict[str, Any]) -> str | None:
    last_kind = None
    for turn in reversed(_latest_transcript_items(session, "agent")):
        kind = _location_question_kind(turn.get("text"))
        if kind:
            last_kind = kind
            break
    if not last_kind:
        return None
    text = _last_user_response_text(session)
    if not text or is_trivial_caller_turn(text):
        return None
    return text


def _infer_shared_ledger_location_bundle(session: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    turns = _latest_transcript_items(session, "user")
    substantive_turns = [
        _normalize_runtime_text(turn.get("text"))
        for turn in turns
        if _normalize_runtime_text(turn.get("text"))
    ]
    combined = " ".join(substantive_turns[-18:])
    sub_location_match = AUTO_SUB_LOCATION_PATTERN.search(combined)
    sub_location = _normalize_runtime_text(sub_location_match.group(1)) if sub_location_match else None

    latest_location_response = _latest_location_response_text(session)
    if latest_location_response:
        candidate, note, latest_sub_location = _infer_location_bundle_from_text(latest_location_response)
        if candidate or note or latest_sub_location:
            return candidate, note or latest_location_response, latest_sub_location or sub_location

    for text in reversed(substantive_turns[-8:]):
        if not text or is_trivial_caller_turn(text):
            continue
        if has_dispatchable_location_cue(text):
            candidate, note, latest_sub_location = _infer_location_bundle_from_text(text)
            if candidate:
                return candidate, note or text, latest_sub_location or sub_location
            return None, text, sub_location
    candidate, note, inferred_sub = _infer_location_bundle_from_transcript(session)
    return candidate, note, inferred_sub or sub_location


def _location_followup_prompt(
    kind: str | None,
    candidate_text: str | None = None,
    *,
    candidate_source: str | None = None,
) -> str | None:
    if kind == "ask_where":
        return "Where are you right now?"
    if kind == "ask_exact_address":
        return "Do you know the exact address? If you do, say it slowly."
    if kind == "confirm_candidate" and candidate_text:
        if candidate_source == "maps_candidate":
            return f"I found {candidate_text}. Is that correct? Yes or no."
        return f"You said {candidate_text}. Is that correct? Yes or no."
    if kind == "ask_spell":
        return "Please spell the street or place name slowly for me."
    if kind == "ask_landmark":
        return "Tell me the nearest landmark you know and describe what you see around you."
    return None


def _build_location_followup_state(session: dict[str, Any], ledger: dict[str, Any]) -> dict[str, Any]:
    hard_facts = ledger.get("hard_facts") or {}
    location_gate = ledger.get("location_gate") or {}
    inferred_candidate, inferred_note, inferred_sub_location = _infer_shared_ledger_location_bundle(session)
    resolve_call = _latest_tool_call(session, "resolve_location_note")
    resolve_result = (resolve_call or {}).get("result") or {}
    lookup_call = _latest_tool_call(session, "lookup_address")
    lookup_result = (lookup_call or {}).get("result") or {}
    lookup_candidate_text = None
    lookup_candidate_kind = None
    candidate_source = None
    if lookup_result.get("candidate_only") and lookup_result.get("normalized_address"):
        lookup_candidate_text = _sanitize_location_candidate_text(lookup_result.get("normalized_address"))
        lookup_candidate_kind = "exact_address" if lookup_candidate_text else None
        candidate_source = "maps_candidate"

    candidate_text = (
        lookup_candidate_text
        or
        _meaningful_runtime_location_candidate(resolve_result.get("anchor_location"))
        or _meaningful_runtime_location_candidate(inferred_candidate)
        or _meaningful_runtime_location_candidate(hard_facts.get("location_candidate"))
        or None
    )
    note_text = (
        _normalize_runtime_text(hard_facts.get("location_note"))
        or _normalize_runtime_text(inferred_note)
        or _normalize_runtime_text(resolve_result.get("normalized_note"))
        or None
    )
    sub_location = (
        _normalize_runtime_text(hard_facts.get("sub_location"))
        or _normalize_runtime_text(inferred_sub_location)
        or _normalize_runtime_text(resolve_result.get("sub_location"))
        or None
    )
    candidate_kind = lookup_candidate_kind or _runtime_location_candidate_kind(candidate_text or "")
    candidate_is_meaningful = bool(candidate_text and candidate_kind in LEDGER_LOCATION_SAFE_KINDS)
    attempts, last_kind = _location_attempt_state(session)
    last_user_text = _last_user_response_text(session)
    last_user_confirmation = _bare_confirmation_value(last_user_text)
    confirmed_by_caller = bool(
        candidate_is_meaningful and last_kind == "confirm_candidate" and last_user_confirmation == "yes"
    )

    if location_gate.get("confirmed"):
        followup_kind = None
        dead_end = False
    elif confirmed_by_caller:
        followup_kind = None
        dead_end = False
    elif candidate_is_meaningful:
        if last_kind == "confirm_candidate" and last_user_confirmation == "no":
            followup_kind = "ask_spell"
        elif last_kind == "ask_spell" and attempts >= 3:
            followup_kind = "ask_landmark"
        else:
            followup_kind = "confirm_candidate"
        dead_end = False
    elif note_text or sub_location or candidate_text:
        if attempts >= LOCATION_DEAD_END_ATTEMPTS and last_kind == "ask_landmark":
            followup_kind = None
            dead_end = True
        elif last_kind in {"ask_spell", "ask_exact_address"} and attempts >= 2:
            followup_kind = "ask_landmark"
            dead_end = False
        else:
            followup_kind = "ask_exact_address"
            dead_end = False
    else:
        if attempts >= 2:
            followup_kind = "ask_landmark"
        else:
            followup_kind = "ask_where"
        dead_end = False

    return {
        "candidate_text": candidate_text,
        "candidate_kind": candidate_kind if candidate_text else None,
        "location_note": note_text,
        "sub_location": sub_location,
        "followup_kind": followup_kind,
        "followup_prompt": _location_followup_prompt(
            followup_kind,
            candidate_text,
            candidate_source=candidate_source,
        ),
        "attempts": attempts,
        "dead_end": dead_end,
        "needs_confirmation": bool(candidate_is_meaningful and not confirmed_by_caller and not location_gate.get("confirmed")),
        "confirmed_by_caller": confirmed_by_caller,
        "candidate_source": candidate_source,
    }


def _run_and_record_auto_tool(
    recorder: SessionRecorder | None,
    *,
    name: str,
    args: dict[str, Any],
    result_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = utc_now()
    started_perf = time.perf_counter()
    result = dispatch_tool_call(name, args)
    if result_override:
        applied_override = dict(result_override)
        if applied_override.get("map_approximation") and not (
            result.get("geocoded")
            and _valid_runtime_coordinates(result.get("lat"), result.get("lon"))
        ):
            applied_override.pop("map_approximation", None)
            applied_override.pop("confidence", None)
            applied_override.pop("reason", None)
        result = {**result, **applied_override}
    elapsed_ms = int((time.perf_counter() - started_perf) * 1000)
    _record_tool_result(
        recorder,
        name=name,
        args=args,
        result=result,
        started_at=started_at,
        elapsed_ms=elapsed_ms,
    )
    return result


def _valid_runtime_coordinates(latitude: Any, longitude: Any) -> bool:
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False
    return math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180


def _record_auto_tool_if_changed(
    session: dict[str, Any],
    recorder: SessionRecorder | None,
    *,
    name: str,
    args: dict[str, Any],
    result: dict[str, Any],
) -> None:
    latest = _latest_tool_call(session, name)
    if latest and (latest.get("args") or {}) == args and (latest.get("result") or {}) == result:
        return
    _record_tool_result(
        recorder,
        name=name,
        args=args,
        result=result,
        started_at=utc_now(),
        elapsed_ms=0,
    )


def _auto_follow_location_resolution(session: dict[str, Any], recorder: SessionRecorder | None) -> None:
    resolve_call = _latest_tool_call(session, "resolve_location_note")
    location_candidate = None
    location_note = None
    sub_location = None

    if resolve_call:
        resolve_args = resolve_call.get("args") or {}
        resolve_result = resolve_call.get("result") or {}
        location_candidate = _normalize_runtime_text(
            resolve_result.get("anchor_location") or resolve_args.get("location_candidate")
        ) or None
        location_note = _normalize_runtime_text(
            resolve_result.get("normalized_note") or resolve_args.get("location_note")
        ) or None
        sub_location = _normalize_runtime_text(
            resolve_result.get("sub_location") or resolve_args.get("sub_location")
        ) or None
    else:
        location_candidate, location_note, sub_location = _infer_location_bundle_from_transcript(session)
        if location_candidate and (location_note or sub_location):
            resolve_args = {
                "location_candidate": location_candidate,
                "location_note": location_note or sub_location,
                "sub_location": sub_location,
            }
            if not _has_tool_call_for_query(session, ("resolve_location_note",), location_candidate):
                _run_and_record_auto_tool(recorder, name="resolve_location_note", args=resolve_args)
                session = get_session(recorder.session_id) if recorder else session

    if not location_candidate:
        return
    geocode_query = location_candidate
    marine_drive = bool(re.fullmatch(r"marine\s+drive", location_candidate, flags=re.IGNORECASE))
    city_note = None
    if marine_drive and location_note and not AUTO_SUB_LOCATION_PATTERN.search(location_note):
        city_note = location_note.strip(" ,.")
        if city_note.casefold() == "kochi":
            geocode_query = "Marine Drive, Kochi, Kerala, India"
        elif city_note:
            geocode_query = f"Marine Drive, {city_note}"
    elif marine_drive:
        resolve_result = {**(resolve_call.get("result") or {})} if resolve_call else {}
        resolve_result.update(
            {
                "requires_city_clarification": True,
                "clarification_question": "Which city is Marine Drive in, or what nearby landmark can you see?",
                "reason": "ambiguous_location_without_city",
            }
        )
        resolve_args = resolve_call.get("args") or {} if resolve_call else {}
        _record_auto_tool_if_changed(
            session,
            recorder,
            name="resolve_location_note",
            args=resolve_args,
            result=resolve_result,
        )
        return
    transcript = _combined_user_transcript(session)
    if not _has_emergency_context(transcript) and not (marine_drive and city_note):
        return

    allowed, reason = assess_searchable_location_query(geocode_query)
    if not allowed:
        return
    if _has_tool_call_for_query(session, ("validate_address", "lookup_address"), geocode_query):
        return

    approximate_city_qualified_place = marine_drive and bool(city_note)
    result_override = {
        "location_query_reason": "named_place" if approximate_city_qualified_place else reason,
        "requires_confirmation": approximate_city_qualified_place or reason not in AUTO_PIN_SAFE_LOCATION_REASONS,
        "candidate_only": approximate_city_qualified_place or reason not in AUTO_PIN_SAFE_LOCATION_REASONS,
    }
    if approximate_city_qualified_place:
        result_override.update(
            {
                "confidence": "low",
                "reason": "approximate_place_match",
                "map_approximation": True,
            }
        )
    lookup_result = _run_and_record_auto_tool(
        recorder,
        name="lookup_address",
        args={"address_text": geocode_query, "allow_best_effort": True},
        result_override=result_override,
    )
    if reason not in AUTO_PIN_SAFE_LOCATION_REASONS and not approximate_city_qualified_place:
        return
    normalized_address = _normalize_runtime_text(lookup_result.get("normalized_address"))
    if not normalized_address:
        return


def _combined_user_transcript(session: dict[str, Any], *, count: int = 18) -> str:
    turns = session.get("client", {}).get("transcripts", {}).get("user", []) or []
    snippets = [_normalize_runtime_text(turn.get("text")) for turn in turns[-count:]]
    return _normalize_runtime_text(" ".join(item for item in snippets if item))


def _voice_first_aid_context(session: dict[str, Any]) -> dict[str, Any] | None:
    """Return bounded curated guidance for the live prompt, without caller text."""
    caller_text = _combined_user_transcript(session)
    if not caller_text:
        return None
    guidance = analyze_first_aid(caller_text)
    assessment = session.get("incident_assessment") or {}
    conflicts = [
        fact.get("field")
        for fact in (assessment.get("facts") or [])
        if fact.get("evidence_status") == "ambiguous"
    ]
    if conflicts:
        return {
            "scenario": "caller reports conflict; immediate status needs confirmation",
            "immediate_actions": ["Keep yourself safe and follow any real emergency call-taker's instructions."],
            "actions_to_avoid": [],
            "conflicting_facts": conflicts[:4],
        }
    scenario = (guidance.get("detected_scenario") or {}).get("label")
    return {
        "scenario": str(scenario or "unclear emergency")[:120],
        "immediate_actions": [str(item)[:280] for item in (guidance.get("immediate_actions") or [])[:2]],
        "actions_to_avoid": [str(item)[:240] for item in (guidance.get("actions_to_avoid") or [])[:2]],
        "conflicting_facts": [],
    }


def _best_runtime_fact_text(session: dict[str, Any]) -> str:
    server = session.get("server") or {}
    enhancement = server.get("enhancement") or {}
    candidates = [
        _normalize_runtime_text(enhancement.get("clean_reconstruction")),
        _normalize_runtime_text(enhancement.get("raw_reconstruction")),
        _combined_user_transcript(session),
    ]
    best = max(candidates, key=lambda item: len(item or ""))
    return best or ""


def _shared_ledger(session: dict[str, Any]) -> dict[str, Any]:
    return normalize_shared_ledger(session.get("shared_ledger"))


def _patch_shared_ledger(recorder: SessionRecorder | None, ledger: dict[str, Any]) -> None:
    if recorder is None:
        return
    normalized = normalize_shared_ledger(ledger)
    recorder.patch_session(
        {
            "shared_ledger": normalized,
            "ledger_prompt_context": build_ledger_prompt_context(normalized),
        }
    )


def _apply_soft_ledger_update(
    session: dict[str, Any],
    recorder: SessionRecorder | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    ledger = _shared_ledger(session)
    soft_state = dict((result or {}).get("soft_state") or {})
    ledger["soft_state"]["people_count_best_guess"] = soft_state.get("people_count_best_guess")
    ledger["soft_state"]["caller_role"] = str(soft_state.get("caller_role") or "unknown")
    ledger["soft_state"]["notes"] = list(soft_state.get("notes") or [])
    ledger["version"] = int(ledger.get("version") or 0) + 1
    _patch_shared_ledger(recorder, ledger)
    return ledger


def _shared_ledger_tool_args(shared_ledger: dict[str, Any]) -> dict[str, Any]:
    hard_facts = normalize_shared_ledger(shared_ledger)["hard_facts"]
    return {
        "location_candidate": hard_facts.get("location_candidate"),
        "location_kind": hard_facts.get("location_kind"),
        "address_in_utterance": bool(hard_facts.get("address_in_utterance")),
        "location_note": hard_facts.get("location_note"),
        "sub_location": hard_facts.get("sub_location"),
        "inside_building": hard_facts.get("inside_building"),
        "issue_cues": list(hard_facts.get("issue_cues") or []),
        "victim_count": hard_facts.get("victim_count_confirmed", "unknown"),
        "child_present": hard_facts.get("child_present"),
        "bleeding_status": hard_facts.get("bleeding_status"),
        "breathing_status": hard_facts.get("breathing_status"),
        "consciousness_status": hard_facts.get("consciousness_status"),
        "trapped_status": hard_facts.get("trapped_status"),
    }


def _meaningful_location_kind(kind: str | None) -> bool:
    return str(kind or "none") in LEDGER_LOCATION_SAFE_KINDS


def _merge_shared_hard_facts(existing: dict[str, Any], incoming: dict[str, Any], *, lock_location: bool = False) -> dict[str, Any]:
    merged = dict(existing or {})
    existing_location = _normalize_runtime_text((existing or {}).get("location_candidate")) or None
    incoming_location = _normalize_runtime_text((incoming or {}).get("location_candidate")) or None
    existing_kind = str((existing or {}).get("location_kind") or "none")
    incoming_kind = str((incoming or {}).get("location_kind") or "none")
    existing_strength = LOCATION_KIND_STRENGTH.get(existing_kind, 0)
    incoming_strength = LOCATION_KIND_STRENGTH.get(incoming_kind, 0)

    if lock_location and existing_location:
        merged["location_candidate"] = existing_location
        merged["location_kind"] = existing_kind
        merged["address_in_utterance"] = bool(existing.get("address_in_utterance"))
    elif incoming_location and (
        not existing_location
        or incoming_strength >= existing_strength
        or incoming_location.casefold() == existing_location.casefold()
    ):
        merged["location_candidate"] = incoming_location
        merged["location_kind"] = incoming_kind
        merged["address_in_utterance"] = bool(incoming.get("address_in_utterance"))
    elif existing_location and (not incoming_location or incoming_kind in {"none", "vague"}):
        merged["location_candidate"] = existing_location
        merged["location_kind"] = existing_kind
        merged["address_in_utterance"] = bool(existing.get("address_in_utterance"))
    else:
        merged["location_candidate"] = incoming_location
        merged["location_kind"] = incoming_kind
        merged["address_in_utterance"] = bool(incoming.get("address_in_utterance"))

    for field_name in ("location_note", "sub_location"):
        if lock_location and existing_location:
            candidate = _normalize_runtime_text(existing.get(field_name)) or None
        else:
            candidate = _normalize_runtime_text(incoming.get(field_name)) or _normalize_runtime_text(existing.get(field_name)) or None
        merged[field_name] = candidate

    merged["inside_building"] = (
        incoming.get("inside_building")
        if incoming.get("inside_building") not in (None, "", "unknown")
        else existing.get("inside_building", "unknown")
    )
    merged["issue_cues"] = list(
        dict.fromkeys(list(existing.get("issue_cues") or []) + list(incoming.get("issue_cues") or []))
    )
    merged["victim_count_confirmed"] = (
        incoming.get("victim_count_confirmed")
        if incoming.get("victim_count_confirmed") not in (None, "", "unknown")
        else existing.get("victim_count_confirmed", "unknown")
    )
    for field_name in ("child_present", "bleeding_status", "breathing_status", "consciousness_status", "trapped_status"):
        merged[field_name] = (
            incoming.get(field_name)
            if incoming.get(field_name) not in (None, "", "unknown")
            else existing.get(field_name, "unknown")
        )
    return normalize_shared_ledger({"hard_facts": merged})["hard_facts"]


def _runtime_fallback_hard_facts(session: dict[str, Any]) -> dict[str, Any]:
    text = _best_runtime_fact_text(session)
    resolved: dict[str, Any] = {
        "location_candidate": None,
        "location_kind": "none",
        "address_in_utterance": False,
        "location_note": None,
        "sub_location": None,
        "inside_building": "unknown",
        "issue_cues": [],
        "victim_count_confirmed": "unknown",
        "child_present": "unknown",
        "bleeding_status": "unknown",
        "breathing_status": "unknown",
        "consciousness_status": "unknown",
        "trapped_status": "unknown",
    }
    if not text:
        return resolved

    lowered = text.casefold()
    candidate, note, sub_location = _infer_shared_ledger_location_bundle(session)
    text_candidate, text_note, text_sub_location = _infer_location_bundle_from_text(text)
    candidate = text_candidate or candidate
    note = text_note or note
    sub_location = text_sub_location or sub_location
    resolve_call = _latest_tool_call(session, "resolve_location_note")
    resolve_result = (resolve_call or {}).get("result") or {}
    anchor_location = _meaningful_runtime_location_candidate(resolve_result.get("anchor_location")) or candidate
    if anchor_location:
        resolved["location_candidate"] = anchor_location
        resolved["location_kind"] = _runtime_location_candidate_kind(anchor_location)
        resolved["address_in_utterance"] = resolved["location_kind"] in LEDGER_LOCATION_SAFE_KINDS
    resolved["location_note"] = (
        _normalize_runtime_text(resolve_result.get("normalized_note"))
        or _normalize_runtime_text(note)
        or None
    )
    resolved["sub_location"] = (
        _normalize_runtime_text(resolve_result.get("sub_location"))
        or _normalize_runtime_text(sub_location)
        or None
    )
    if any(word in lowered for word in ("building", "room", "shop", "inside", "entrance", "lobby", "floor")):
        resolved["inside_building"] = "yes"

    if _has_emergency_context(text):
        service_plan = dispatch_tool_call(
            "plan_response_services",
            {
                "transcript": text,
                "issue_type": None,
                "priority": None,
                "issue_cues": [],
            },
        )
        resolved["issue_cues"] = list(service_plan.get("issue_cues") or [])

    if any(word in lowered for word in ("daughter", "son", "child", "kid", "kids")):
        resolved["child_present"] = "yes"
        if "child_present" not in resolved["issue_cues"]:
            resolved["issue_cues"].append("child_present")

    if re.search(r"\b(?:no|not)\s+bleeding\b|\bno blood\b", lowered):
        resolved["bleeding_status"] = "no"
    elif any(word in lowered for word in ("bleeding", "blood", "nose is bleeding", "my nose is bleeding")):
        resolved["bleeding_status"] = "yes"

    if re.search(r"\b(?:not breathing|can't breathe|cannot breathe)\b", lowered):
        resolved["breathing_status"] = "no"
    elif re.search(r"\b(?:can breathe|breathing properly|able to breathe)\b", lowered):
        resolved["breathing_status"] = "yes"

    if re.search(r"\b(?:unconscious|not responding|unresponsive)\b", lowered):
        resolved["consciousness_status"] = "no"
    elif re.search(r"\b(?:awake|conscious|responsive)\b", lowered):
        resolved["consciousness_status"] = "yes"

    if re.search(r"\b(?:not trapped)\b", lowered):
        resolved["trapped_status"] = "no"
    elif re.search(r"\b(?:trapped|stuck|can't get out|cannot get out)\b", lowered):
        resolved["trapped_status"] = "yes"

    # Local live triage is turn based. Prefer its latest explicit binary
    # statement over an earlier long-form transcript reconstruction.
    merged_triage = session.get("merged_triage_state") or {}
    for triage_field, ledger_field in (
        ("victim_breathing", "breathing_status"),
        ("victim_bleeding", "bleeding_status"),
        ("victim_conscious", "consciousness_status"),
    ):
        value = merged_triage.get(triage_field)
        if value in {"yes", "no"}:
            resolved[ledger_field] = value

    return normalize_shared_ledger({"hard_facts": resolved})["hard_facts"]


def _location_lock_active(ledger: dict[str, Any], location_followup: dict[str, Any] | None) -> bool:
    priority = ledger.get("priority") or {}
    gate = ledger.get("location_gate") or {}
    followup = location_followup or {}
    if gate.get("search_allowed") or gate.get("confirmed"):
        return False
    if followup.get("dead_end"):
        return False
    return bool(
        followup.get("followup_kind")
        or priority.get("location_followup_prompt")
        or gate.get("candidate_text")
        or (ledger.get("hard_facts") or {}).get("location_note")
    )


def _bare_confirmation_value(text: str | None) -> str | None:
    normalized = _normalize_runtime_text(text).strip(" .,!?:;").casefold()
    if normalized in {"yes", "yeah", "yep", "correct", "that's right", "thats right"}:
        return "yes"
    if normalized in {"no", "nope"}:
        return "no"
    return None


def _apply_runtime_confirmation_to_ledger(
    session: dict[str, Any],
    recorder: SessionRecorder | None,
) -> dict[str, Any]:
    ledger = _shared_ledger(session)
    user_turns = session.get("client", {}).get("transcripts", {}).get("user", []) or []
    last_runtime_turn_index = int((ledger.get("provenance") or {}).get("last_runtime_turn_index") or 0)
    next_question_field = (ledger.get("priority") or {}).get("next_question_field")
    if next_question_field == "location_candidate":
        changed = False
        followup = _build_location_followup_state(session, ledger)
        candidate_text = _normalize_runtime_text(followup.get("candidate_text")) or _normalize_runtime_text(
            (ledger.get("location_gate") or {}).get("candidate_text")
        )
        candidate_kind = str(
            followup.get("candidate_kind")
            or (ledger.get("location_gate") or {}).get("candidate_kind")
            or _runtime_location_candidate_kind(candidate_text)
            or "none"
        )
        for turn in user_turns[last_runtime_turn_index:]:
            value = _bare_confirmation_value(turn.get("text"))
            if value is None:
                continue
            if value == "yes" and candidate_text:
                ledger["hard_facts"]["location_candidate"] = candidate_text
                ledger["hard_facts"]["location_kind"] = candidate_kind
                ledger["hard_facts"]["address_in_utterance"] = candidate_kind in LEDGER_LOCATION_SAFE_KINDS
                ledger["location_gate"]["candidate_text"] = candidate_text
                ledger["location_gate"]["candidate_kind"] = candidate_kind
                ledger["location_gate"]["confirmed"] = True
                ledger["location_gate"]["search_allowed"] = True
                ledger["location_gate"]["needs_confirmation"] = False
                changed = True
            elif value == "no":
                ledger["location_gate"]["confirmed"] = False
                ledger["location_gate"]["needs_confirmation"] = False
                changed = True
        ledger["provenance"]["last_runtime_turn_index"] = len(user_turns)
        if changed:
            facts = shared_hard_facts_to_fact_ledger(ledger["hard_facts"])
            priority_flags = build_fact_priority_packet(facts, facts)["priority_flags"]
            ledger["priority"] = {
                **ledger["priority"],
                "missing_fields": list(priority_flags.get("missing_fields") or []),
                "next_question_field": priority_flags.get("next_question_field"),
                "next_question_goal": priority_flags.get("next_question_goal"),
                "question_style": priority_flags.get("question_style") or "short_open",
                "move_on_allowed": bool(priority_flags.get("move_on_allowed")),
            }
            ledger["version"] = int(ledger.get("version") or 0) + 1
            _patch_shared_ledger(recorder, ledger)
        return ledger
    if next_question_field not in LEDGER_BINARY_FIELDS:
        ledger["provenance"]["last_runtime_turn_index"] = len(user_turns)
        return ledger
    changed = False
    for turn in user_turns[last_runtime_turn_index:]:
        value = _bare_confirmation_value(turn.get("text"))
        if value is None:
            continue
        if ledger["hard_facts"].get(next_question_field) != value:
            ledger["hard_facts"][next_question_field] = value
            changed = True
    ledger["provenance"]["last_runtime_turn_index"] = len(user_turns)
    if changed:
        facts = shared_hard_facts_to_fact_ledger(ledger["hard_facts"])
        priority_flags = build_fact_priority_packet(facts, facts)["priority_flags"]
        ledger["priority"] = {
            **ledger["priority"],
            "missing_fields": list(priority_flags.get("missing_fields") or []),
            "next_question_field": priority_flags.get("next_question_field"),
            "next_question_goal": priority_flags.get("next_question_goal"),
            "question_style": priority_flags.get("question_style") or "short_open",
            "move_on_allowed": bool(priority_flags.get("move_on_allowed")),
        }
        ledger["version"] = int(ledger.get("version") or 0) + 1
        _patch_shared_ledger(recorder, ledger)
    return ledger


def _ledger_location_gate(
    shared_ledger: dict[str, Any],
    session: dict[str, Any],
    *,
    location_followup: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ledger = normalize_shared_ledger(shared_ledger)
    existing_gate = ledger.get("location_gate") or {}
    hard_facts = ledger["hard_facts"]
    facts = shared_hard_facts_to_fact_ledger(hard_facts)
    priority_flags = build_fact_priority_packet(facts, facts)["priority_flags"]
    lookup_call = _latest_tool_call(session, "lookup_address")
    validate_call = _latest_tool_call(session, "validate_address")
    nearby_call = _latest_tool_call(session, "nearby_context")
    lookup_result = lookup_call.get("result") if lookup_call else {}
    validate_result = validate_call.get("result") if validate_call else {}
    nearby_result = nearby_call.get("result") if nearby_call else {}
    followup = location_followup or {}
    lookup_reason = str((lookup_result or {}).get("location_query_reason") or "")
    validate_reason = str((validate_result or {}).get("location_query_reason") or "")
    exact_tool_match = bool(
        (
            lookup_result
            and not lookup_result.get("candidate_only")
            and lookup_result.get("lat") is not None
            and lookup_reason in {"exact_address", "explicit_address", "street_number"}
        )
        or (
            validate_result
            and not validate_result.get("candidate_only")
            and validate_result.get("lat") is not None
            and validate_reason in {"exact_address", "explicit_address", "street_number"}
        )
    )
    candidate_only = bool(
        (lookup_result or {}).get("candidate_only") or (validate_result or {}).get("candidate_only")
    )
    confirmed = bool(
        existing_gate.get("confirmed")
        or (
            followup.get("confirmed_by_caller")
            and followup.get("candidate_source") == "maps_candidate"
        )
        or exact_tool_match
    )
    candidate_text = _normalize_runtime_text(
        followup.get("candidate_text") or hard_facts.get("location_candidate")
    ) or None
    if _normalize_runtime_text(followup.get("candidate_text")):
        candidate_kind = followup.get("candidate_kind")
    elif _normalize_runtime_text(hard_facts.get("location_candidate")):
        candidate_kind = hard_facts.get("location_kind")
    else:
        candidate_kind = followup.get("candidate_kind")
    runtime_candidate_allowed = bool(
        candidate_text
        and candidate_kind in LEDGER_LOCATION_SAFE_KINDS
        and (followup.get("confirmed_by_caller") or existing_gate.get("confirmed"))
    )
    search_allowed = bool(priority_flags.get("location_search_allowed") or runtime_candidate_allowed)
    needs_confirmation = bool(
        followup.get("needs_confirmation")
        and not confirmed
        and candidate_text
    )

    if confirmed:
        search_reason = "confirmed_by_tools" if exact_tool_match else "caller_confirmed_runtime_candidate"
    elif priority_flags.get("location_search_allowed"):
        search_reason = "slm_meaningful_location"
    elif runtime_candidate_allowed:
        search_reason = "caller_confirmed_runtime_candidate"
    elif needs_confirmation and candidate_text:
        search_reason = "candidate_needs_confirmation"
    elif hard_facts.get("location_note") or hard_facts.get("sub_location"):
        search_reason = "best_effort_note_only"
    elif followup.get("location_note") or followup.get("sub_location"):
        search_reason = "best_effort_note_only"
    else:
        search_reason = "needs_more_location_detail"
    return {
        "search_allowed": search_allowed,
        "candidate_only": bool(
            candidate_only
            or (candidate_text and candidate_kind in LEDGER_LOCATION_SAFE_KINDS and not search_allowed)
        ),
        "confirmed": confirmed,
        "search_reason": search_reason,
        "candidate_text": candidate_text,
        "candidate_kind": candidate_kind,
        "needs_confirmation": needs_confirmation,
    }


def _refresh_shared_ledger_runtime(
    session: dict[str, Any],
    recorder: SessionRecorder | None,
    *,
    allow_external_geocoding: bool = True,
) -> dict[str, Any]:
    ledger = _apply_runtime_confirmation_to_ledger(session, recorder)
    ledger = _shared_ledger(get_session(recorder.session_id) if recorder else session)
    runtime_fallback = _runtime_fallback_hard_facts(session)
    ledger["hard_facts"] = _merge_shared_hard_facts(
        ledger.get("hard_facts") or {},
        runtime_fallback,
        lock_location=bool((ledger.get("location_gate") or {}).get("confirmed")),
    )
    facts = shared_hard_facts_to_fact_ledger(ledger["hard_facts"])
    priority_flags = build_fact_priority_packet(facts, facts)["priority_flags"]
    ledger["priority"] = {
        "missing_fields": list(priority_flags.get("missing_fields") or []),
        "next_question_field": priority_flags.get("next_question_field"),
        "next_question_goal": priority_flags.get("next_question_goal"),
        "question_style": priority_flags.get("question_style") or "short_open",
        "move_on_allowed": bool(priority_flags.get("move_on_allowed")),
        "location_followup_kind": None,
        "location_followup_prompt": None,
        "location_dead_end": False,
        "location_attempts": 0,
        "location_lock_active": False,
    }

    tool_args = _shared_ledger_tool_args(ledger)
    _record_auto_tool_if_changed(
        session,
        recorder,
        name="checklist_by_incident",
        args=tool_args,
        result=dispatch_tool_call("checklist_by_incident", tool_args),
    )
    handoff_args = {
        **tool_args,
        "caller_summary": _combined_user_transcript(session) or None,
    }
    _record_auto_tool_if_changed(
        session,
        recorder,
        name="build_handoff_brief",
        args=handoff_args,
        result=dispatch_tool_call("build_handoff_brief", handoff_args),
    )

    inferred_candidate, inferred_note, inferred_sub_location = _infer_shared_ledger_location_bundle(session)
    resolve_candidate = _meaningful_runtime_location_candidate(tool_args.get("location_candidate")) or inferred_candidate
    resolve_note = (
        tool_args.get("location_note")
        or inferred_note
        or tool_args.get("sub_location")
        or inferred_sub_location
    )
    resolve_sub_location = tool_args.get("sub_location") or inferred_sub_location
    resolve_args = {
        "location_candidate": resolve_candidate,
        "location_note": resolve_note,
        "sub_location": resolve_sub_location,
    }
    if resolve_args.get("location_note"):
        latest_resolve = _latest_tool_call(session, "resolve_location_note")
        if (latest_resolve or {}).get("args") != resolve_args:
            _run_and_record_auto_tool(recorder, name="resolve_location_note", args=resolve_args)
            session = get_session(recorder.session_id) if recorder else session

    location_followup = _build_location_followup_state(session, ledger)
    ledger["location_gate"] = _ledger_location_gate(ledger, session, location_followup=location_followup)
    location_candidate = (
        _normalize_runtime_text(ledger["hard_facts"].get("location_candidate"))
        or _normalize_runtime_text((ledger.get("location_gate") or {}).get("candidate_text"))
        or None
    )
    def _apply_location_followup_priority(location_followup_state: dict[str, Any]) -> None:
        if location_followup_state.get("followup_kind") and not ledger["location_gate"]["search_allowed"]:
            location_lock_active = _location_lock_active(ledger, location_followup_state)
            ledger["priority"].update(
                {
                    "next_question_field": "location_candidate",
                    "next_question_goal": {
                        "ask_where": "ask where they are",
                        "ask_exact_address": "get the exact address",
                        "confirm_candidate": "confirm the location clue",
                        "ask_spell": "spell the location clue",
                        "ask_landmark": "get the nearest landmark and surroundings",
                    }.get(location_followup_state.get("followup_kind"), "pinpoint where they are"),
                    "question_style": "yes_no"
                    if location_followup_state.get("followup_kind") == "confirm_candidate"
                    else "short_open",
                    "move_on_allowed": bool(location_followup_state.get("dead_end") or priority_flags.get("move_on_allowed")),
                    "location_followup_kind": location_followup_state.get("followup_kind"),
                    "location_followup_prompt": location_followup_state.get("followup_prompt"),
                    "location_dead_end": bool(location_followup_state.get("dead_end")),
                    "location_attempts": int(location_followup_state.get("attempts") or 0),
                    "location_lock_active": location_lock_active,
                }
            )
        else:
            ledger["priority"].update(
                {
                    "location_followup_kind": location_followup_state.get("followup_kind"),
                    "location_followup_prompt": location_followup_state.get("followup_prompt"),
                    "location_dead_end": bool(location_followup_state.get("dead_end")),
                    "location_attempts": int(location_followup_state.get("attempts") or 0),
                    "location_lock_active": _location_lock_active(ledger, location_followup_state),
                }
            )

    _apply_location_followup_priority(location_followup)
    preflight_candidate = _meaningful_runtime_location_candidate(location_followup.get("candidate_text"))
    if allow_external_geocoding and (
        preflight_candidate
        and location_followup.get("needs_confirmation")
        and _has_emergency_context(_combined_user_transcript(session))
        and not _has_tool_call_for_query(session, ("lookup_address",), preflight_candidate)
    ):
        preflight_kind = classify_location_kind(preflight_candidate)
        _run_and_record_auto_tool(
            recorder,
            name="lookup_address",
            args={"address_text": preflight_candidate, "allow_best_effort": True},
            result_override={
                "location_query_reason": preflight_kind,
                "requires_confirmation": True,
                "candidate_only": True,
            },
        )
        session = get_session(recorder.session_id) if recorder else session
        location_followup = _build_location_followup_state(session, ledger)
        ledger["location_gate"] = _ledger_location_gate(ledger, session, location_followup=location_followup)
        _apply_location_followup_priority(location_followup)
    if allow_external_geocoding and ledger["location_gate"]["search_allowed"] and location_candidate and not location_followup.get("needs_confirmation"):
        latest_lookup = _latest_tool_call(session, "lookup_address")
        if not latest_lookup or _normalize_runtime_text((latest_lookup.get("args") or {}).get("address_text")) != location_candidate:
            location_kind = (
                ledger["hard_facts"].get("location_kind")
                if _normalize_runtime_text(ledger["hard_facts"].get("location_candidate"))
                else (ledger.get("location_gate") or {}).get("candidate_kind")
            )
            exact_like = str(location_kind or "none") == "exact_address"
            caller_confirmed = bool((ledger.get("location_gate") or {}).get("confirmed"))
            result_override = {
                "location_query_reason": location_kind,
                "requires_confirmation": not caller_confirmed and not exact_like,
                "candidate_only": not caller_confirmed and not exact_like,
            }
            _run_and_record_auto_tool(
                recorder,
                name="lookup_address",
                args={"address_text": location_candidate, "allow_best_effort": True},
                result_override=result_override,
            )
            session = get_session(recorder.session_id) if recorder else session
            location_followup = _build_location_followup_state(session, ledger)
            ledger["location_gate"] = _ledger_location_gate(ledger, session, location_followup=location_followup)

    previous = normalize_shared_ledger(session.get("shared_ledger"))
    next_ledger = normalize_shared_ledger({**ledger, "version": previous.get("version", 0)})
    if next_ledger != previous:
        next_ledger["version"] = int(previous.get("version") or 0) + 1
    else:
        next_ledger["version"] = int(previous.get("version") or 0)
    _patch_shared_ledger(recorder, next_ledger)
    return next_ledger


def _should_schedule_shared_ledger(session: dict[str, Any]) -> bool:
    ledger = _shared_ledger(session)
    user_turns = session.get("client", {}).get("transcripts", {}).get("user", []) or []
    start_index = int((ledger.get("provenance") or {}).get("last_slm_turn_index") or 0)
    if start_index >= len(user_turns):
        return False
    new_turns = user_turns[start_index:]
    substantive_count = 0
    for turn in new_turns:
        text = _normalize_runtime_text(turn.get("text"))
        if not text:
            continue
        if is_immediate_fact_ledger_trigger(text):
            return True
        if not is_trivial_caller_turn(text):
            substantive_count += 1
    return substantive_count >= 2


def _fact_ledger_endpoint(settings: Any) -> str:
    return f"{settings.triage_base_url.rstrip('/')}/chat/completions"


def _parse_fact_ledger_response(payload: dict[str, Any]) -> FactLedger:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("No choices returned from fact ledger model")
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content if part.get("type") == "text")
    content = _normalize_runtime_text(content)
    content = re.sub(r"^```json\s*|\s*```$", "", content, flags=re.IGNORECASE)
    parsed = json.loads(content)
    return FactLedger.model_validate((parsed or {}).get("merged_facts") or {})


def _run_fact_ledger_inference(settings: Any, caller_turn: str, prior_facts: FactLedger) -> FactLedger:
    response = httpx.post(
        _fact_ledger_endpoint(settings),
        headers={
            "Authorization": f"Bearer {settings.triage_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": _fact_ledger_model_id(settings),
            "messages": [
                {
                    "role": "system",
                    "content": FACT_LEDGER_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "caller_turn": caller_turn,
                            "prior_facts": prior_facts.model_dump(),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
        },
        timeout=settings.triage_request_timeout_s,
    )
    response.raise_for_status()
    return _parse_fact_ledger_response(response.json())


async def _run_shared_ledger_job(session_id: str, settings: Any) -> None:
    recorder = SessionRecorder(session_id)
    registry_state = LEDGER_JOB_REGISTRY.setdefault(
        session_id,
        {"running": False, "dirty": False, "task": None, "last_user_turn_count": 0},
    )
    try:
        while True:
            session = get_session(session_id)
            ledger = _shared_ledger(session)
            ledger["provenance"]["slm_job_state"] = "analyzing"
            _patch_shared_ledger(recorder, ledger)
            user_turns = session.get("client", {}).get("transcripts", {}).get("user", []) or []
            caller_turn, substantive_turn_index = latest_substantive_caller_window(user_turns, limit=3)
            if not caller_turn:
                ledger["provenance"]["slm_job_state"] = "idle"
                _patch_shared_ledger(recorder, ledger)
                break
            prior_facts = shared_hard_facts_to_fact_ledger(ledger["hard_facts"])
            merged_facts = await asyncio.to_thread(_run_fact_ledger_inference, settings, caller_turn, prior_facts)
            session = get_session(session_id)
            next_ledger = _shared_ledger(session)
            incoming_hard_facts = fact_ledger_to_shared_hard_facts(merged_facts)
            next_ledger["hard_facts"] = _merge_shared_hard_facts(
                next_ledger["hard_facts"],
                incoming_hard_facts,
                lock_location=bool((next_ledger.get("location_gate") or {}).get("confirmed")),
            )
            next_ledger["provenance"]["last_slm_turn_index"] = max(substantive_turn_index, len(user_turns))
            next_ledger["provenance"]["last_slm_run_at"] = utc_now()
            next_ledger["provenance"]["slm_job_state"] = "idle"
            next_ledger["version"] = int(next_ledger.get("version") or 0) + 1
            _patch_shared_ledger(recorder, next_ledger)
            session = get_session(session_id)
            _refresh_shared_ledger_runtime(session, recorder)
            registry_state = LEDGER_JOB_REGISTRY.get(session_id) or registry_state
            if registry_state.get("dirty"):
                registry_state["dirty"] = False
                continue
            break
    except Exception as exc:
        session = get_session(session_id)
        ledger = _shared_ledger(session)
        ledger["hard_facts"] = _merge_shared_hard_facts(
            ledger.get("hard_facts") or {},
            _runtime_fallback_hard_facts(session),
            lock_location=bool((ledger.get("location_gate") or {}).get("confirmed")),
        )
        ledger["provenance"]["slm_job_state"] = "error"
        ledger["version"] = int(ledger.get("version") or 0) + 1
        _patch_shared_ledger(recorder, ledger)
        recorder.patch_server({"shared_ledger_error": str(exc)})
        session = get_session(session_id)
        _refresh_shared_ledger_runtime(session, recorder)
    finally:
        registry_state = LEDGER_JOB_REGISTRY.setdefault(
            session_id,
            {"running": False, "dirty": False, "task": None, "last_user_turn_count": 0},
        )
        registry_state["running"] = False
        registry_state["task"] = None


def _schedule_shared_ledger_refresh(session_id: str, settings: Any) -> None:
    session = get_session(session_id)
    if not _should_schedule_shared_ledger(session):
        return
    user_turn_count = len(session.get("client", {}).get("transcripts", {}).get("user", []) or [])
    registry_state = LEDGER_JOB_REGISTRY.setdefault(
        session_id,
        {"running": False, "dirty": False, "task": None, "last_user_turn_count": 0},
    )
    registry_state["last_user_turn_count"] = user_turn_count
    if registry_state.get("running"):
        registry_state["dirty"] = True
        return
    registry_state["running"] = True
    registry_state["dirty"] = False
    registry_state["task"] = asyncio.create_task(_run_shared_ledger_job(session_id, settings))


def _derive_dispatch_inputs(session: dict[str, Any]) -> dict[str, Any]:
    tool_calls = session.get("tool_calls") or []
    live_dispatch_mode = _resolve_live_dispatch_mode(
        (session.get("server") or {}).get("live_dispatch_mode") or (session.get("server") or {}).get("triage_engine")
    )
    shared_ledger = _shared_ledger(session)
    hard_facts = shared_ledger.get("hard_facts") or {}
    location_gate = shared_ledger.get("location_gate") or {}
    checklist_call = _latest_tool_call(session, "checklist_by_incident")
    ticket_call = _latest_tool_call(session, "create_incident_ticket")
    lookup_call = _latest_tool_call(session, "lookup_address")
    validate_call = _latest_tool_call(session, "validate_address")
    nearby_call = _latest_tool_call(session, "nearby_context")
    resolve_call = _latest_tool_call(session, "resolve_location_note")
    transcript = _combined_user_transcript(session)
    ticket_result = ticket_call.get("result") if ticket_call else {}
    checklist_args = checklist_call.get("args") if checklist_call else {}
    lookup_result = lookup_call.get("result") if lookup_call else {}
    validate_result = validate_call.get("result") if validate_call else {}
    nearby_result = nearby_call.get("result") if nearby_call else {}
    resolve_result = resolve_call.get("result") if resolve_call else {}
    ticket_address = _normalize_runtime_text(ticket_result.get("address"))
    if live_dispatch_mode == "slm" and not location_gate.get("confirmed"):
        ticket_address = None
    gated_hard_location = (
        _normalize_runtime_text(hard_facts.get("location_candidate"))
        if location_gate.get("search_allowed") or location_gate.get("confirmed")
        else None
    )
    gated_candidate_text = (
        _normalize_runtime_text(location_gate.get("candidate_text"))
        if location_gate.get("search_allowed") or location_gate.get("confirmed")
        else None
    )

    location_text = (
        ticket_address
        or _normalize_runtime_text(nearby_result.get("normalized_address"))
        or _normalize_runtime_text(
            None if validate_result.get("candidate_only") else validate_result.get("normalized_address")
        )
        or _normalize_runtime_text(
            None if lookup_result.get("candidate_only") else lookup_result.get("normalized_address")
        )
        or gated_hard_location
        or gated_candidate_text
        or _normalize_runtime_text(hard_facts.get("location_note"))
        or _normalize_runtime_text(resolve_result.get("anchor_location"))
        or _normalize_runtime_text(lookup_result.get("normalized_address"))
        or _normalize_runtime_text(validate_result.get("normalized_address"))
    )
    lat = (
        nearby_result.get("lat")
        or (None if validate_result.get("candidate_only") else validate_result.get("lat"))
        or (None if lookup_result.get("candidate_only") else lookup_result.get("lat"))
    )
    lon = (
        nearby_result.get("lon")
        or (None if validate_result.get("candidate_only") else validate_result.get("lon"))
        or (None if lookup_result.get("candidate_only") else lookup_result.get("lon"))
    )
    location_confirmed = bool(location_gate.get("confirmed"))
    dispatch_location_text = (
        _normalize_runtime_text(nearby_result.get("normalized_address"))
        or _normalize_runtime_text(
            None if validate_result.get("candidate_only") else validate_result.get("normalized_address")
        )
        or _normalize_runtime_text(
            None if lookup_result.get("candidate_only") else lookup_result.get("normalized_address")
        )
        or (
            gated_hard_location
            if location_gate.get("confirmed")
            else None
        )
        or (
            gated_candidate_text
            if location_gate.get("confirmed")
            else None
        )
    )
    return {
        "transcript": transcript,
        "issue_type": ticket_result.get("issue_type"),
        "priority": ticket_result.get("priority"),
        "issue_cues": list(hard_facts.get("issue_cues") or checklist_args.get("issue_cues") or []),
        "location_text": location_text or None,
        "dispatch_location_text": dispatch_location_text or None,
        "lat": lat,
        "lon": lon,
        "location_confirmed": location_confirmed,
        "tool_calls": tool_calls,
    }


def _dispatch_state_changed(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    if not previous:
        return True
    previous_services = previous.get("services") or {}
    current_services = current.get("services") or {}
    if previous.get("human_monitoring") != current.get("human_monitoring"):
        return True
    if previous.get("human_review_recommended") != current.get("human_review_recommended"):
        return True
    if previous.get("summary") != current.get("summary"):
        return True
    if previous.get("announcement_line") != current.get("announcement_line"):
        return True
    if set(previous_services) != set(current_services):
        return True
    for name, current_item in current_services.items():
        previous_item = previous_services.get(name) or {}
        if (
            previous_item.get("status") != current_item.get("status")
            or previous_item.get("eta_text") != current_item.get("eta_text")
            or previous_item.get("location_confirmed") != current_item.get("location_confirmed")
            or previous_item.get("base_name") != current_item.get("base_name")
            or previous_item.get("base_address") != current_item.get("base_address")
            or previous_item.get("distance_km") != current_item.get("distance_km")
        ):
            return True
    return False


def _dispatch_simulation_due(
    latest_call: dict[str, Any] | None,
    previous_state: dict[str, Any],
    current_args: dict[str, Any],
    *,
    now: datetime | None = None,
) -> bool:
    """Run the simulator only for changed inputs or a due status milestone."""
    if not latest_call or not previous_state.get("services"):
        return True
    prior_args = latest_call.get("args") or {}
    stable_keys = ("session_id", "plan", "location_text", "location_confirmed", "response_bases")
    if any(prior_args.get(key) != current_args.get(key) for key in stable_keys):
        return True
    now_dt = now or datetime.now(UTC)
    location_ready = bool(str(current_args.get("location_text") or "").strip())
    for item in previous_state.get("services", {}).values():
        try:
            activated_dt = datetime.fromisoformat(str(item.get("activated_at")).replace("Z", "+00:00"))
            elapsed = max(0.0, (now_dt - activated_dt).total_seconds())
        except (TypeError, ValueError):
            return True
        expected = "locating" if not location_ready or elapsed < 1.4 else "assigned" if elapsed < 3.0 else "dispatched"
        if item.get("status") != expected:
            return True
    return False


def _provider_event_completes_llm_turn(event_name: str, llm_response_active: bool) -> bool:
    """An end-of-turn is an assistant completion only inside an LLM turn."""
    return event_name == "end_of_turn" and llm_response_active


def _dispatch_prompt_context(dispatch_state: dict[str, Any], version: int) -> dict[str, Any]:
    services = dispatch_state.get("services") or {}
    summary_parts = [
        f"{service.get('label')} {service.get('status_label', '').lower()}"
        + (f" ({service.get('eta_text')})" if service.get("eta_text") else "")
        for service in services.values()
    ]
    return {
        "version": version,
        "human_monitoring": bool(dispatch_state.get("human_monitoring")),
        "human_review_recommended": bool(dispatch_state.get("human_review_recommended")),
        "monitor_name": None,
        "serious_emergency": bool(dispatch_state.get("serious_emergency")),
        "services_summary": " · ".join(summary_parts) if summary_parts else "No active service movement.",
        "announcement_line": dispatch_state.get("announcement_line"),
    }


def _maybe_auto_create_ticket(
    session: dict[str, Any],
    recorder: SessionRecorder | None,
    *,
    plan: dict[str, Any],
    dispatch_state: dict[str, Any],
    dispatch_inputs: dict[str, Any],
) -> None:
    live_dispatch_mode = _resolve_live_dispatch_mode(
        (session.get("server") or {}).get("live_dispatch_mode") or (session.get("server") or {}).get("triage_engine")
    )
    if _latest_tool_call(session, "create_incident_ticket"):
        return
    if not _has_emergency_context(dispatch_inputs.get("transcript")):
        return
    location_text = _normalize_runtime_text(
        dispatch_inputs.get("dispatch_location_text")
        if live_dispatch_mode == "slm"
        else dispatch_inputs.get("location_text")
    )
    if not location_text:
        return
    if live_dispatch_mode == "slm" and not dispatch_inputs.get("location_confirmed"):
        return
    if not dispatch_state.get("dispatchable"):
        return
    caller_summary = None
    handoff_call = _latest_tool_call(session, "build_handoff_brief")
    if handoff_call:
        handoff_result = handoff_call.get("result") or {}
        caller_summary = handoff_result.get("caller_summary") or handoff_result.get("one_line")
    caller_summary = caller_summary or dispatch_inputs.get("transcript") or "Emergency call in progress."
    service_names = plan.get("needed_services") or []
    notes = [f"service={name}" for name in service_names]
    if dispatch_state.get("human_review_recommended"):
        notes.append("human_review_recommended")
    _run_and_record_auto_tool(
        recorder,
        name="create_incident_ticket",
        args={
            "caller_summary": caller_summary,
            "address": location_text,
            "issue_type": plan.get("issue_type") or "GENERAL",
            "priority": plan.get("priority") or "MEDIUM",
            "notes": notes,
        },
    )


def _auto_dispatch_services(session: dict[str, Any], recorder: SessionRecorder | None) -> dict[str, Any] | None:
    dispatch_inputs = _derive_dispatch_inputs(session)
    live_dispatch_mode = _resolve_live_dispatch_mode(
        (session.get("server") or {}).get("live_dispatch_mode") or (session.get("server") or {}).get("triage_engine")
    )
    if not dispatch_inputs.get("transcript"):
        return None
    if not (_has_emergency_context(dispatch_inputs.get("transcript")) or (dispatch_inputs.get("issue_cues") or [])):
        return None
    plan_args = {
        "transcript": dispatch_inputs.get("transcript"),
        "issue_type": dispatch_inputs.get("issue_type"),
        "priority": dispatch_inputs.get("priority"),
        "issue_cues": dispatch_inputs.get("issue_cues") or [],
    }
    plan = dispatch_tool_call("plan_response_services", plan_args)
    response_bases = dict((session.get("dispatch_services") or {}).get("response_bases") or {})
    lat = dispatch_inputs.get("lat")
    lon = dispatch_inputs.get("lon")
    if dispatch_inputs.get("location_confirmed") and lat is not None and lon is not None:
        if not response_bases or not _has_tool_call_for_coords(session, "lookup_response_bases", lat, lon):
            response_bases = _run_and_record_auto_tool(
                recorder,
                name="lookup_response_bases",
                args={
                    "lat": lat,
                    "lon": lon,
                    "location_text": dispatch_inputs.get("location_text"),
                },
            ).get("services", {})
            session = get_session(recorder.session_id) if recorder else session
    previous_state = session.get("dispatch_services") or {}
    dispatch_args = {
        "session_id": session.get("session_id") or "",
        "plan": plan,
        "location_text": dispatch_inputs.get("dispatch_location_text")
        if live_dispatch_mode == "slm"
        else dispatch_inputs.get("location_text"),
        "location_confirmed": bool(dispatch_inputs.get("location_confirmed")),
        "response_bases": response_bases,
        "prior_state": previous_state,
        "now_iso": utc_now(),
    }
    latest_simulation = _latest_tool_call(session, "simulate_dispatch_services")
    if _dispatch_simulation_due(latest_simulation, previous_state, dispatch_args):
        dispatch_state = dispatch_tool_call("simulate_dispatch_services", dispatch_args)
    else:
        # Preserve the last authoritative simulation result between milestones.
        dispatch_state = previous_state
    _record_auto_tool_if_changed(session, recorder, name="plan_response_services", args=plan_args, result=plan)
    _record_auto_tool_if_changed(session, recorder, name="simulate_dispatch_services", args=dispatch_args, result=dispatch_state)

    _maybe_auto_create_ticket(session, recorder, plan=plan, dispatch_state=dispatch_state, dispatch_inputs=dispatch_inputs)
    session = get_session(recorder.session_id) if recorder else session
    previous_state = session.get("dispatch_services") or {}
    changed = _dispatch_state_changed(previous_state, dispatch_state)
    version = int(previous_state.get("version") or 0)
    if changed:
        version += 1
    dispatch_state["version"] = version
    prompt_context = _dispatch_prompt_context(dispatch_state, version)
    if changed and recorder is not None:
        recorder.patch_session(
            {
                "dispatch_services": dispatch_state,
                "dispatch_prompt_context": prompt_context,
            }
        )
    elif not session.get("dispatch_prompt_context"):
        if recorder is not None:
            recorder.patch_session(
                {
                    "dispatch_services": dispatch_state,
                    "dispatch_prompt_context": prompt_context,
                }
            )
    return {
        "version": version,
        "prompt_context": prompt_context,
    }


async def _websocket_session_ai_coustics(websocket: WebSocket) -> None:
    settings = get_settings()
    enhancer = None
    requested_provider = str(websocket.query_params.get("voice_provider") or "gradbot").strip().lower()
    if requested_provider not in {"gradbot", "gemini"}:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Unknown voice provider")
    if requested_provider == "gradbot":
        settings = _ensure_llm_ready()
        enhancer = make_enhancer(settings)
    elif gemini_live_missing(settings):
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION, reason="Gemini Live API key is not configured")
    recorder: SessionRecorder | None = None
    interrupt_count = 0
    latest_triage_context: dict[str, Any] | None = None
    latest_ledger_context: dict[str, Any] | None = None
    latest_dispatch_context: dict[str, Any] | None = None
    raw_pcm = bytearray()
    clean_pcm = bytearray()
    provider_audio_chunks = 0
    provider_audio_bytes = 0
    provider_output_audio_chunks = 0
    provider_diagnostics: dict[str, Any] = {
        "provider": requested_provider,
        "input_audio_format": "s16le_mono_24000hz",
        "provider_input_audio_format": "s16le_mono_16000hz" if requested_provider == "gemini" else "s16le_mono_24000hz",
        "input_audio_chunks": 0,
        "input_audio_bytes": 0,
        "provider_audio_send_attempts": 0,
        "provider_audio_send_attempted_bytes": 0,
        "provider_audio_send_successes": 0,
        "provider_audio_send_total_ms": 0.0,
        "provider_audio_send_max_ms": 0.0,
        "caller_transcript_events": 0,
        "assistant_text_events": 0,
        "assistant_audio_frames_forwarded": 0,
        "assistant_audio_bytes_forwarded": 0,
        "llm_started_events": 0,
        "end_of_turn_events": 0,
        "completed_llm_responses": 0,
        "llm_preflight": None,
        "llm_watchdog": {"silent_push_cycles": 0, "last_silent_cycle_at": None},
        "provider_event_counts": {},
        "stream_close": None,
        "last_failure": None,
    }
    enhancement_mode = "raw" if requested_provider == "gemini" else "enhanced"
    enhancement_fallback_reason: str | None = None
    human_takeover_active = False
    handoff_recovery_prompt: str | None = None

    async def on_tool_call(tool_handle) -> None:
        nonlocal latest_ledger_context, latest_dispatch_context
        started_at = utc_now()
        started_perf = time.perf_counter()
        try:
            if live_dispatch_mode == "slm" and tool_handle.name in {
                "validate_address",
                "lookup_address",
                "lookup_response_bases",
            }:
                current_session = get_session(recorder.session_id) if recorder else {}
                location_gate = (_shared_ledger(current_session).get("location_gate") or {})
                if not location_gate.get("search_allowed"):
                    result = {
                        "blocked": True,
                        "reason": "location_search_not_allowed",
                        "message": "Location search is blocked until the shared ledger confirms a meaningful clue.",
                    }
                else:
                    result = dispatch_tool_call(tool_handle.name, tool_handle.args)
            else:
                result = dispatch_tool_call(tool_handle.name, tool_handle.args)
            elapsed_ms = int((time.perf_counter() - started_perf) * 1000)
            _record_tool_result(
                recorder,
                name=tool_handle.name,
                args=tool_handle.args,
                result=result,
                started_at=started_at,
                elapsed_ms=elapsed_ms,
            )
            if live_dispatch_mode == "slm" and tool_handle.name == "update_soft_ledger":
                current_session = get_session(recorder.session_id) if recorder else {}
                _apply_soft_ledger_update(current_session, recorder, result)
            if live_dispatch_mode == "slm" and tool_handle.name in {
                "update_soft_ledger",
                "resolve_location_note",
                "checklist_by_incident",
                "build_handoff_brief",
                "validate_address",
                "lookup_address",
                "lookup_response_bases",
            }:
                current_session = get_session(recorder.session_id) if recorder else {}
                _refresh_shared_ledger_runtime(current_session, recorder)
                current_session = get_session(recorder.session_id) if recorder else current_session
                _auto_dispatch_services(current_session, recorder)
                current_session = get_session(recorder.session_id) if recorder else current_session
                latest_ledger_context = current_session.get("ledger_prompt_context") or latest_ledger_context
                latest_dispatch_context = current_session.get("dispatch_prompt_context") or latest_dispatch_context
                if latest_ledger_context or latest_dispatch_context:
                    config = build_session_config(
                        live_dispatch_mode=live_dispatch_mode,
                        interrupt_count=interrupt_count,
                        interruption_recovery=False,
                        triage_context=None,
                        ledger_context=latest_ledger_context if live_dispatch_mode == "slm" else None,
                        dispatch_context=latest_dispatch_context,
                        human_takeover_active=human_takeover_active,
                        handoff_recovery_prompt=handoff_recovery_prompt,
                        first_aid_context=_voice_first_aid_context(current_session),
                    )
                    # Gemini has an outstanding function call here. Sending
                    # client content before its FunctionResponse can make the
                    # upstream tool exchange invalid; the result itself carries
                    # the updated facts, and the next client context update can
                    # add the refreshed ledger safely after this tool turn.
                    if voice_provider != "gemini":
                        await input_handle.send_config(config)
                        recorder.append_session_list(
                            "llm_prompt_snapshots",
                            {
                                "at": utc_now(),
                                "reason": f"tool:{tool_handle.name}",
                                "interrupt_count": interrupt_count,
                                "interruption_recovery": False,
                                "flush_duration_s": config.flush_duration_s,
                                "silence_timeout_s": config.silence_timeout_s,
                                "triage_version": None,
                                "ledger_version": latest_ledger_context.get("version") if latest_ledger_context else None,
                                "dispatch_version": latest_dispatch_context.get("version") if latest_dispatch_context else None,
                                "selected_engine": live_dispatch_mode,
                                "runtime_provider": "pioneer" if live_dispatch_mode == "slm" else "gradium",
                                "instructions": config.instructions,
                                "live_dispatch_mode": live_dispatch_mode,
                            },
                        )
            await tool_handle.send_json(result)
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - started_perf) * 1000)
            # Keep operational evidence without logging tool arguments,
            # results, exception text, or caller content.
            logger.error(
                "live_tool_call_failed session=%s provider=%s stage=tool_callback exception_type=%s elapsed_ms=%s",
                recorder.session_id if recorder else "unavailable",
                voice_provider,
                type(exc).__name__,
                elapsed_ms,
            )
            _record_tool_result(
                recorder,
                name=tool_handle.name,
                args=tool_handle.args,
                error=str(exc),
                started_at=started_at,
                elapsed_ms=elapsed_ms,
            )
            raise

    await websocket.accept()
    try:
        start_msg = await websocket.receive_json()
    except ValueError:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Expected a JSON start message")
        return
    if not isinstance(start_msg, dict) or start_msg.get("type") != "start":
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Expected a JSON start message")
        return
    client_metadata = start_msg.get("client")
    if client_metadata is not None and not isinstance(client_metadata, dict):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Client metadata must be an object")
        return
    requested_session_id = start_msg.get("session_id")
    if requested_session_id is not None and not _valid_session_id(requested_session_id):
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Invalid session_id")
        return

    live_dispatch_mode = _resolve_live_dispatch_mode(
        start_msg.get("triage_mode") or (start_msg.get("client") or {}).get("triage_mode")
    )
    voice_provider = str(start_msg.get("voice_provider") or requested_provider).strip().lower()
    if voice_provider != requested_provider:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Voice provider does not match the WebSocket request")
        return
    if voice_provider == "gemini" and live_dispatch_mode == "slm":
        _ensure_live_triage_ready_ws()
    live_triage_settings = _ensure_live_runtime_ready_ws(live_dispatch_mode)
    session_id = requested_session_id or f"session-{utc_now().replace(':', '-')}"
    # Reject a duplicate or stale "start" that reuses a session_id already owned
    # by a live socket; interleaved writes would corrupt the active session.
    if not _acquire_live_session(session_id):
        logger.warning("session_rejected_duplicate_start session=%s", session_id)
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION,
            reason="Session already active",
        )
        return
    try:
        recorder = SessionRecorder(session_id)
        clear_live_audio_window(session_id)
        initial_config = build_session_config(live_dispatch_mode=live_dispatch_mode)
        recorder.patch_server(
            {
                "status": "started",
                "started_at": utc_now(),
                "llm_model": settings.openai_model,
                "client_metadata": start_msg.get("client", {}),
                "mode": "live_gemini" if voice_provider == "gemini" else "live_ai_coustics",
                "voice_provider": voice_provider,
                "triage_engine": live_dispatch_mode,
                "live_dispatch_mode": live_dispatch_mode,
                "adaptive_policy": {
                    "interrupt_count": 0,
                    "flush_duration_s": initial_config.flush_duration_s,
                    "silence_timeout_s": initial_config.silence_timeout_s,
                },
                "enhancement": {
                    "status": "live",
                    "ai_coustics_model_id": settings.ai_coustics_model_id,
                    "live_input_source": "raw" if voice_provider == "gemini" else "enhanced",
                    "parallel_enhancement": voice_provider == "gradbot",
                    "fallback_to_raw": voice_provider == "gemini",
                },
            }
        )
        recorder.append_session_list(
            "llm_prompt_snapshots",
            {
                "at": utc_now(),
                "reason": "session_start",
                "interrupt_count": 0,
                "interruption_recovery": False,
                "flush_duration_s": initial_config.flush_duration_s,
                "silence_timeout_s": initial_config.silence_timeout_s,
                "triage_version": None,
                "ledger_version": None,
                "selected_engine": None,
                "runtime_provider": None,
                "live_dispatch_mode": live_dispatch_mode,
                "instructions": initial_config.instructions,
            },
        )

        # The gradbot native layer logs upstream LLM failures without forwarding
        # them, so a dead LLM (for example exhausted account quota) would otherwise
        # produce a silent session that later closes with code 1011. Probe once and
        # refuse explicitly instead of starting a session that cannot answer.
        llm_preflight = await _llm_preflight_check(settings) if voice_provider == "gradbot" else {"status": "not_required_for_gemini"}
        provider_diagnostics["llm_preflight"] = llm_preflight
        recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
        if voice_provider == "gradbot" and llm_preflight.get("quota_exhausted"):
            provider_diagnostics["stream_close"] = {
                "reason": "llm_quota_exhausted_preflight",
                "websocket_code": 1011,
                "response_completed": False,
            }
            recorder.patch_server(
                {
                    "status": "error",
                    "ended_at": utc_now(),
                    "provider_diagnostics": dict(provider_diagnostics),
                }
            )
            logger.error(
                "provider_failure session=%s stage=llm_preflight provider=openai category=llm_quota_exhausted http_status=%s",
                session_id,
                llm_preflight.get("http_status"),
            )
            try:
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "llm_quota_exhausted",
                        "message": (
                            "Live voice is unavailable: the assistant language model quota is exhausted. "
                            "No call was started. Local first-aid guidance and the text demo still work."
                        ),
                    }
                )
                await websocket.close(code=1011, reason="LLM quota exhausted")
            except Exception:
                pass
            _release_live_session(session_id)
            return
    except BaseException:
        # Setup failed before the live loops started; free the session_id so a
        # retry is not permanently blocked, then surface the error.
        _release_live_session(session_id)
        raise

    try:
        if voice_provider == "gemini":
            gemini_adapter = GeminiLiveAdapter(settings, instructions=initial_config.instructions, tools=initial_config.tools)
            await gemini_adapter.connect()
            input_handle = output_handle = gemini_adapter
        else:
            input_handle, output_handle = await gradbot.run(
                **_run_kwargs(),
                session_config=initial_config,
                input_format=gradbot.AudioFormat.Pcm,
                output_format=gradbot.AudioFormat.Pcm,
            )
    except Exception as exc:
        provider_failure = _provider_error_diagnostics(exc, provider_hint=voice_provider)
        provider_failure["stage"] = "provider_initialization"
        provider_error_label = provider_failure["exception_type"]
        if provider_failure["http_status"] is not None:
            provider_error_label += f" (HTTP {provider_failure['http_status']})"
        recorder.patch_server(
            {
                "status": "error",
                "ended_at": utc_now(),
                "provider_initialization_error": provider_failure["exception_type"],
                "provider_diagnostics": {
                    **provider_diagnostics,
                    "status": "initialization_failed",
                    "last_failure": provider_failure,
                    "stream_close": {"reason": "provider_initialization_failed", "websocket_code": 1011},
                },
            }
        )
        logger.error(
            "provider_failure session=%s stage=provider_initialization provider=%s category=%s exception_type=%s http_status=%s error_type=%s error_code=%s",
            session_id,
            provider_failure["provider"],
            provider_failure["category"],
            provider_failure["exception_type"],
            provider_failure["http_status"],
            provider_failure["error_type"],
            provider_failure["error_code"],
        )
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": provider_failure["category"],
                    "message": (
                        _safe_provider_error_message(provider_failure)
                        if provider_failure["category"] in {"gemini_quota_exhausted", "gemini_auth_failed", "provider_transport_error"}
                        else f"Live provider initialization failed ({provider_error_label})."
                    ),
                }
            )
            await websocket.close(code=1011, reason="Live provider initialization failed")
        except Exception:
            pass
        _release_live_session(session_id)
        return

    session_started_perf = time.monotonic()
    first_provider_response_logged = False
    first_provider_audio_logged = False
    first_client_audio_logged = False
    await websocket.send_json({"type": "connection_status", "status": "connected"})
    logger.info(
        "provider_session_initialized provider=%s session=%s client_input_format=s16le_mono_24000hz provider_input_format=%s output_format=s16le_mono_24000hz",
        voice_provider,
        session_id,
        provider_diagnostics["provider_input_audio_format"],
    )
    provider_diagnostics["status"] = "initialized"
    provider_diagnostics["provider_session_ready_at"] = utc_now()
    recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})

    stop_event = asyncio.Event()
    pending_tool_tasks: set[asyncio.Task] = set()
    # Owned by output_loop; hoisted so a pending native receive can be cancelled
    # after both loops finish instead of leaking a dangling task.
    output_receive_task: asyncio.Future | None = None

    async def input_loop() -> None:
        nonlocal interrupt_count, latest_triage_context, latest_ledger_context, latest_dispatch_context, enhancement_mode, enhancement_fallback_reason, human_takeover_active, handoff_recovery_prompt, provider_audio_chunks, provider_audio_bytes, first_client_audio_logged
        while not stop_event.is_set():
            try:
                raw = await websocket.receive()
                if "bytes" in raw:
                    pcm_bytes = raw["bytes"]
                    if not pcm_bytes:
                        continue
                    if not first_client_audio_logged:
                        first_client_audio_logged = True
                        logger.info("voice_latency provider=%s session=%s stage=server_first_client_audio elapsed_s=%.3f", voice_provider, session_id, time.monotonic() - session_started_perf)
                    raw_pcm.extend(pcm_bytes)
                    live_bytes = pcm_bytes
                    clean_bytes = b""
                    try:
                        if enhancer is not None:
                            clean_bytes = enhancer.process_bytes(pcm_bytes)
                            if clean_bytes:
                                clean_pcm.extend(clean_bytes)
                                if enhancement_mode == "enhanced":
                                    live_bytes = clean_bytes
                    except Exception as exc:
                        enhancement_mode = "raw_fallback"
                        enhancement_fallback_reason = str(exc)
                        recorder.patch_server(
                            {
                                "enhancement": {
                                    "status": "degraded",
                                    "live_input_source": "raw_fallback",
                                    "fallback_to_raw": True,
                                    "fallback_reason": enhancement_fallback_reason,
                                }
                            }
                        )
                    append_live_audio_window(session_id, pcm_bytes, clean_bytes)
                    send_started = time.perf_counter()
                    provider_diagnostics["provider_audio_send_attempts"] += 1
                    provider_diagnostics["provider_audio_send_attempted_bytes"] += len(live_bytes)
                    if provider_diagnostics["provider_audio_send_attempts"] == 1 or provider_diagnostics["provider_audio_send_attempts"] % 100 == 0:
                        recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
                        logger.info(
                            "provider_audio_send session=%s stage=attempt at=%s attempts=%d attempted_bytes=%d elapsed_s=%.3f format=%s",
                            session_id,
                            utc_now(),
                            provider_diagnostics["provider_audio_send_attempts"],
                            provider_diagnostics["provider_audio_send_attempted_bytes"],
                            time.monotonic() - session_started_perf,
                            provider_diagnostics["provider_input_audio_format"],
                        )
                    await input_handle.send_audio(live_bytes)
                    send_elapsed_ms = (time.perf_counter() - send_started) * 1000
                    provider_audio_chunks += 1
                    provider_audio_bytes += len(live_bytes)
                    provider_diagnostics["provider_audio_send_successes"] = provider_audio_chunks
                    provider_diagnostics["provider_audio_send_total_ms"] = round(
                        provider_diagnostics["provider_audio_send_total_ms"] + send_elapsed_ms, 2
                    )
                    provider_diagnostics["provider_audio_send_max_ms"] = round(
                        max(provider_diagnostics["provider_audio_send_max_ms"], send_elapsed_ms), 2
                    )
                    provider_diagnostics["input_audio_chunks"] = provider_audio_chunks
                    provider_diagnostics["input_audio_bytes"] = provider_audio_bytes
                    if provider_diagnostics.get("status") == "initialized":
                        provider_diagnostics["status"] = "receiving_audio"
                    if provider_audio_chunks == 1 or provider_audio_chunks % 100 == 0:
                        recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
                        logger.info(
                            "provider_audio_send session=%s stage=completed at=%s attempts=%d successes=%d chunks=%d bytes=%d duration_s=%.2f format=s16le_mono_24000hz enhancement=%s send_elapsed_ms=%.2f",
                            session_id,
                            utc_now(),
                            provider_diagnostics["provider_audio_send_attempts"],
                            provider_diagnostics["provider_audio_send_successes"],
                            provider_audio_chunks,
                            provider_audio_bytes,
                            provider_audio_bytes / (PCM_SAMPLE_RATE * 2),
                            enhancement_mode,
                            send_elapsed_ms,
                        )
                    continue

                if "text" not in raw:
                    continue

                data = json.loads(raw["text"])
                msg_type = data.get("type")
                if msg_type == "config":
                    reason = data.get("reason")
                    recovery = bool(data.get("interruption_recovery"))
                    human_takeover_active = bool(data.get("human_takeover_active"))
                    handoff_recovery_prompt = data.get("handoff_recovery_prompt")
                    if live_dispatch_mode == "slm" and isinstance(data.get("triage_context"), dict):
                        latest_triage_context = data.get("triage_context")
                    if live_dispatch_mode == "slm" and isinstance(data.get("ledger_context"), dict):
                        latest_ledger_context = data.get("ledger_context")
                    if isinstance(data.get("dispatch_context"), dict):
                        latest_dispatch_context = data.get("dispatch_context")
                    interrupt_count = max(0, int(data.get("interrupt_count") or 0))
                    config = build_session_config(
                        live_dispatch_mode=live_dispatch_mode,
                        interrupt_count=interrupt_count,
                        interruption_recovery=recovery,
                        triage_context=latest_triage_context if live_dispatch_mode == "slm" else None,
                        ledger_context=latest_ledger_context if live_dispatch_mode == "slm" else None,
                        dispatch_context=latest_dispatch_context,
                        human_takeover_active=human_takeover_active,
                        handoff_recovery_prompt=handoff_recovery_prompt,
                        first_aid_context=_voice_first_aid_context(get_session(session_id)),
                    )
                    await input_handle.send_config(config)
                    recorder.patch_server(
                        {
                            "adaptive_policy": {
                                "interrupt_count": interrupt_count,
                                "last_reason": reason,
                                "interruption_recovery": recovery,
                                "flush_duration_s": config.flush_duration_s,
                                "silence_timeout_s": config.silence_timeout_s,
                            }
                        },
                    )
                    recorder.append_session_list(
                        "llm_prompt_snapshots",
                        {
                            "at": utc_now(),
                            "reason": reason,
                            "interrupt_count": interrupt_count,
                            "interruption_recovery": recovery,
                            "flush_duration_s": config.flush_duration_s,
                            "silence_timeout_s": config.silence_timeout_s,
                            "triage_version": latest_triage_context.get("version") if latest_triage_context else None,
                            "ledger_version": latest_ledger_context.get("version") if latest_ledger_context else None,
                            "dispatch_version": latest_dispatch_context.get("version") if latest_dispatch_context else None,
                            "selected_engine": latest_ledger_context.get("selected_engine")
                            if latest_ledger_context
                            else (latest_triage_context.get("selected_engine") if latest_triage_context else live_dispatch_mode),
                            "runtime_provider": latest_ledger_context.get("runtime_provider")
                            if latest_ledger_context
                            else (latest_triage_context.get("runtime_provider") if latest_triage_context else ("gradium" if live_dispatch_mode == "llm_only" else None)),
                            "instructions": config.instructions,
                            "live_dispatch_mode": live_dispatch_mode,
                            "human_takeover_active": human_takeover_active,
                            "handoff_recovery_prompt": handoff_recovery_prompt,
                        },
                    )
                    if latest_ledger_context is not None:
                        recorder.patch_server(
                            {
                                "ledger_prompt": {
                                    "version": latest_ledger_context.get("version"),
                                    "missing_fields": latest_ledger_context.get("missing_fields"),
                                    "next_question_goal": latest_ledger_context.get("next_question_goal"),
                                    "location_search_allowed": latest_ledger_context.get("location_search_allowed"),
                                }
                            }
                        )
                    if latest_triage_context is not None:
                        recorder.patch_server(
                            {
                                "triage_prompt": {
                                    "version": latest_triage_context.get("version"),
                                    "selected_engine": latest_triage_context.get("selected_engine"),
                                    "runtime_provider": latest_triage_context.get("runtime_provider"),
                                    "missing_or_unknown_fields": latest_triage_context.get("missing_or_unknown_fields"),
                                    "dispatchable": latest_triage_context.get("dispatchable"),
                                    "autonomy_allowed": latest_triage_context.get("autonomy_allowed"),
                                }
                            }
                        )
                    continue
                if msg_type == "stop":
                    provider_diagnostics["stream_close"] = {
                        "reason": "client_stop",
                        "websocket_code": 1000,
                        "response_completed": provider_diagnostics["completed_llm_responses"] > 0,
                    }
                    tail_bytes = b""
                    if enhancement_mode == "enhanced":
                        try:
                            tail_bytes = enhancer.flush()
                        except Exception as exc:
                            enhancement_mode = "raw_fallback"
                            enhancement_fallback_reason = str(exc)
                            recorder.patch_server(
                                {
                                    "enhancement": {
                                        "status": "degraded",
                                        "live_input_source": "raw_fallback",
                                        "fallback_to_raw": True,
                                        "fallback_reason": enhancement_fallback_reason,
                                    }
                                }
                            )
                    if tail_bytes:
                        clean_pcm.extend(tail_bytes)
                        append_live_audio_window(session_id, b"", tail_bytes)
                        await input_handle.send_audio(tail_bytes)
                    stop_event.set()
                    await input_handle.close()
                    break
            except WebSocketDisconnect as exc:
                if provider_diagnostics["stream_close"] is None:
                    provider_diagnostics["stream_close"] = {
                        "reason": "browser_disconnect",
                        "websocket_code": exc.code,
                        "response_completed": provider_diagnostics["completed_llm_responses"] > 0,
                    }
                provider_diagnostics["input_audio_chunks"] = provider_audio_chunks
                provider_diagnostics["input_audio_bytes"] = provider_audio_bytes
                recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
                logger.info(
                    "provider_stream_closed session=%s provider=%s reason=browser_disconnect websocket_code=%s input_chunks=%d input_bytes=%d assistant_text_events=%d assistant_audio_chunks=%d",
                    session_id,
                    voice_provider,
                    exc.code,
                    provider_audio_chunks,
                    provider_audio_bytes,
                    provider_diagnostics["assistant_text_events"],
                    provider_output_audio_chunks,
                )
                stop_event.set()
                await input_handle.close()
                break
            except Exception as exc:
                prior_close = provider_diagnostics.get("stream_close") or {}
                if stop_event.is_set() and prior_close.get("reason") in {
                    "provider_output_eof",
                    "provider_output_exception",
                }:
                    logger.warning(
                        "provider_secondary_failure_ignored session=%s provider=%s stage=provider_input exception_type=%s elapsed_s=%.3f websocket_close_code=%s traceback=%s root_stage=%s",
                        session_id,
                        voice_provider,
                        _safe_provider_token(type(exc).__name__) or "ProviderError",
                        time.monotonic() - session_started_perf,
                        prior_close.get("websocket_code"),
                        _safe_traceback_summary(exc),
                        prior_close.get("reason"),
                    )
                    break
                failure = _provider_error_diagnostics(exc, provider_hint=voice_provider)
                failure["stage"] = "provider_input"
                provider_diagnostics["last_failure"] = failure
                provider_diagnostics["input_audio_chunks"] = provider_audio_chunks
                provider_diagnostics["input_audio_bytes"] = provider_audio_bytes
                provider_diagnostics["stream_close"] = {
                    "reason": "provider_input_exception",
                    "websocket_code": 1011,
                    "response_completed": provider_diagnostics["completed_llm_responses"] > 0,
                }
                recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
                logger.error(
                    "provider_failure session=%s stage=provider_input provider=%s category=%s exception_type=%s http_status=%s error_type=%s error_code=%s elapsed_s=%.3f websocket_close_code=%s provider_close_code=%s provider_close_reason=%s traceback=%s",
                    session_id,
                    failure["provider"],
                    failure["category"],
                    failure["exception_type"],
                    failure["http_status"],
                    failure["error_type"],
                    failure["error_code"],
                    time.monotonic() - session_started_perf,
                    provider_diagnostics["stream_close"]["websocket_code"],
                    failure.get("provider_close_code"),
                    failure.get("provider_close_reason"),
                    _safe_traceback_summary(exc),
                )
                stop_event.set()
                await input_handle.close()
                break

    async def output_loop() -> None:
        nonlocal provider_output_audio_chunks, output_receive_task, first_provider_response_logged, first_provider_audio_logged
        llm_response_active = False
        llm_push_pending_since: float | None = None
        silent_llm_push_cycles = 0
        terminal_notified = False
        watchdog_tick = object()

        async def notify_terminal(code: str, message: str, reason: str) -> None:
            """Surface a runtime LLM failure explicitly instead of a silent stall."""
            nonlocal terminal_notified
            if terminal_notified or stop_event.is_set():
                return
            terminal_notified = True
            provider_diagnostics["input_audio_chunks"] = provider_audio_chunks
            provider_diagnostics["input_audio_bytes"] = provider_audio_bytes
            provider_diagnostics["stream_close"] = {
                "reason": reason,
                "websocket_code": 1011,
                "response_completed": provider_diagnostics["completed_llm_responses"] > 0,
            }
            recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
            logger.error(
                "provider_failure session=%s stage=llm_runtime category=%s silent_push_cycles=%d llm_started_events=%d",
                session_id,
                code,
                silent_llm_push_cycles,
                provider_diagnostics["llm_started_events"],
            )
            stop_event.set()
            try:
                await input_handle.close()
            except Exception:
                pass
            try:
                await websocket.send_json({"type": "error", "code": code, "message": message})
                await websocket.close(code=1011, reason=reason[:120])
            except Exception:
                pass

        while not stop_event.is_set():
            try:
                # Poll without cancelling the native receive call: cancellation
                # of a pending native await could drop provider output.
                if output_receive_task is None:
                    output_receive_task = asyncio.ensure_future(output_handle.receive())
                done, _ = await asyncio.wait({output_receive_task}, timeout=1.0)
                if output_receive_task in done:
                    completed_receive_task = output_receive_task
                    output_receive_task = None
                    msg = completed_receive_task.result()
                else:
                    msg = watchdog_tick
                if msg is watchdog_tick:
                    # The gradbot native layer forwards push_to_llm but drops
                    # upstream LLM failures (they are only logged natively), so
                    # an assistant that never starts must be detected here.
                    if (
                        llm_push_pending_since is not None
                        and not llm_response_active
                        and provider_diagnostics["assistant_text_events"] == 0
                        and time.monotonic() - llm_push_pending_since >= LLM_START_TIMEOUT_S
                    ):
                        llm_push_pending_since = None
                        silent_llm_push_cycles += 1
                        provider_diagnostics["llm_watchdog"] = {
                            "silent_push_cycles": silent_llm_push_cycles,
                            "last_silent_cycle_at": utc_now(),
                        }
                        recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
                    if silent_llm_push_cycles >= LLM_NO_RESPONSE_LIMIT and not human_takeover_active:
                        await notify_terminal(
                            "llm_no_response",
                            (
                                "The assistant language model did not answer after repeated attempts; "
                                "live voice is degraded for this session. Local first-aid guidance and "
                                "the text triage demo still work."
                            ),
                            "llm_watchdog_no_response",
                        )
                        break
                    continue
                if msg is None:
                    if not stop_event.is_set():
                        provider_diagnostics["input_audio_chunks"] = provider_audio_chunks
                        provider_diagnostics["input_audio_bytes"] = provider_audio_bytes
                        provider_diagnostics["assistant_audio_frames_forwarded"] = provider_output_audio_chunks
                        provider_diagnostics["stream_close"] = {
                            "reason": "provider_output_eof",
                            "websocket_code": 1011,
                            "response_completed": provider_diagnostics["completed_llm_responses"] > 0,
                        }
                        provider_diagnostics["last_failure"] = {
                            "provider": voice_provider,
                            "category": "provider_output_eof",
                            "exception_type": "ReceiveStreamEnded",
                            "http_status": None,
                            "error_type": None,
                            "error_code": None,
                            "stage": "provider_output",
                        }
                        recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
                        logger.warning(
                            "provider_failure session=%s provider=%s stage=provider_output category=provider_output_eof exception_type=ReceiveStreamEnded elapsed_s=%.3f websocket_close_code=1011 input_chunks=%d input_bytes=%d caller_transcripts=%d assistant_text_events=%d assistant_audio_chunks=%d assistant_audio_bytes=%d end_of_turn_events=%d",
                            session_id,
                            voice_provider,
                            time.monotonic() - session_started_perf,
                            provider_audio_chunks,
                            provider_audio_bytes,
                            provider_diagnostics["caller_transcript_events"],
                            provider_diagnostics["assistant_text_events"],
                            provider_output_audio_chunks,
                            provider_diagnostics["assistant_audio_bytes_forwarded"],
                            provider_diagnostics["end_of_turn_events"],
                        )
                        stop_event.set()
                        await input_handle.close()
                        await websocket.send_json({
                            "type": "error",
                            "code": "provider_output_eof",
                            "message": "Provider output ended before the live response completed.",
                        })
                        await websocket.close(code=1011, reason="Live provider output ended")
                    break
                if not first_provider_response_logged:
                    first_provider_response_logged = True
                    provider_diagnostics["first_provider_event_at"] = utc_now()
                    logger.info("voice_latency provider=%s session=%s stage=server_first_provider_event elapsed_s=%.3f msg_type=%s", voice_provider, session_id, time.monotonic() - session_started_perf, _safe_provider_token(msg.msg_type) or "unknown")
                if msg.msg_type == "error":
                    failure = _provider_error_diagnostics(getattr(msg, "data", None) or msg, provider_hint=voice_provider)
                    failure["stage"] = "provider_output"
                    provider_diagnostics["last_failure"] = failure
                    provider_diagnostics["input_audio_chunks"] = provider_audio_chunks
                    provider_diagnostics["input_audio_bytes"] = provider_audio_bytes
                    provider_diagnostics["stream_close"] = {"reason": "provider_error_event", "websocket_code": 1011}
                    recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
                    logger.error(
                        "provider_failure session=%s stage=provider_output provider=%s category=%s exception_type=%s http_status=%s error_type=%s error_code=%s elapsed_s=%.3f websocket_close_code=1011 provider_close_code=%s traceback=unavailable source=provider_error_event",
                        session_id,
                        failure["provider"],
                        failure["category"],
                        failure["exception_type"],
                        failure["http_status"],
                        failure["error_type"],
                        failure["error_code"],
                        time.monotonic() - session_started_perf,
                        failure.get("provider_close_code"),
                    )
                    stop_event.set()
                    await input_handle.close()
                    await websocket.send_json({
                        "type": "error",
                        "code": failure["category"],
                        "message": _safe_provider_error_message(failure),
                    })
                    await websocket.close(code=1011, reason="Live provider output failed")
                    break
                provider_diagnostics["input_audio_chunks"] = provider_audio_chunks
                provider_diagnostics["input_audio_bytes"] = provider_audio_bytes
                if msg.msg_type == "stt_text":
                    provider_diagnostics["caller_transcript_events"] += 1
                    recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
                elif msg.msg_type == "tts_text":
                    provider_diagnostics["assistant_text_events"] += 1
                    recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
                elif msg.msg_type == "event" and msg.event is not None:
                    event_name = _safe_provider_token(getattr(msg.event, "event_type", None)) or "unknown"
                    event_counts = provider_diagnostics["provider_event_counts"]
                    event_counts[event_name] = event_counts.get(event_name, 0) + 1
                    if event_name == "llm_started":
                        provider_diagnostics["llm_started_events"] += 1
                        llm_response_active = True
                        llm_push_pending_since = None
                        silent_llm_push_cycles = 0
                    elif event_name == "push_to_llm":
                        if llm_push_pending_since is None:
                            llm_push_pending_since = time.monotonic()
                    elif event_name == "end_of_turn":
                        provider_diagnostics["end_of_turn_events"] += 1
                        # Gradbot emits end_of_turn for speech recognition as
                        # well as completed assistant turns. Count completion
                        # only when it closes an observed llm_started cycle.
                        if _provider_event_completes_llm_turn(event_name, llm_response_active):
                            provider_diagnostics["completed_llm_responses"] += 1
                            llm_response_active = False
                    elif event_name == "provider_go_away":
                        provider_diagnostics["provider_go_away"] = {
                            "at": utc_now(),
                            "time_left_seconds": getattr(msg.event, "time_left_seconds", None),
                        }
                    elif event_name == "provider_session_resumption":
                        provider_diagnostics["provider_session_resumption"] = {
                            "at": utc_now(),
                            "resumable": getattr(msg.event, "resumable", None),
                            "token_present": bool(getattr(msg.event, "token_present", False)),
                        }
                    logger.info(
                        "provider_event provider=%s session=%s event=%s elapsed_s=%.3f llm_started_events=%d end_of_turn_events=%d",
                        voice_provider,
                        session_id,
                        event_name,
                        time.monotonic() - session_started_perf,
                        provider_diagnostics["llm_started_events"],
                        provider_diagnostics["end_of_turn_events"],
                    )
                    if event_name in {"llm_started", "end_of_turn"} or event_name.startswith("provider_"):
                        recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
                if msg.msg_type == "tool_call":
                    handle = msg.tool_call_handle if voice_provider == "gemini" else gradbot.websocket.ToolHandle(msg.tool_call_handle, msg.tool_call)

                    async def _safe_tool_call(tool_handle=handle):
                        await on_tool_call(tool_handle)

                    task = asyncio.create_task(_safe_tool_call())
                    pending_tool_tasks.add(task)
                    task.add_done_callback(pending_tool_tasks.discard)
                    continue

                if msg.msg_type == "audio" and human_takeover_active:
                    continue
                if msg.msg_type == "audio" and not first_provider_audio_logged:
                    first_provider_audio_logged = True
                    logger.info("voice_latency provider=%s session=%s stage=server_first_provider_audio elapsed_s=%.3f", voice_provider, session_id, time.monotonic() - session_started_perf)
                audio_size = len(msg.data or b"") if msg.msg_type == "audio" else 0
                await _forward_provider_message(websocket, msg, session_id=session_id, provider=voice_provider)
                if msg.msg_type == "audio":
                    provider_output_audio_chunks += 1
                    provider_diagnostics["assistant_audio_frames_forwarded"] = provider_output_audio_chunks
                    provider_diagnostics["assistant_audio_bytes_forwarded"] += audio_size
                    if provider_output_audio_chunks == 1 or provider_output_audio_chunks % 100 == 0:
                        recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
                        logger.info(
                            "provider_assistant_audio provider=%s session=%s frames_forwarded=%d bytes_forwarded=%d format=s16le_mono_48000hz",
                            voice_provider,
                            session_id,
                            provider_output_audio_chunks,
                            provider_diagnostics["assistant_audio_bytes_forwarded"],
                        )
            except Exception as exc:
                if stop_event.is_set() and (provider_diagnostics.get("stream_close") or {}).get("reason") in {
                    "client_stop",
                    "browser_disconnect",
                }:
                    # google-genai reports the expected close of a pending
                    # session.receive() as APIError("1000 None.") when the
                    # input loop closes its context during normal cleanup.
                    # Do not turn an operator stop or browser disconnect into
                    # a provider 1011 failure.
                    logger.info(
                        "provider_receive_closed_during_cleanup session=%s provider=%s exception_type=%s reason=%s",
                        session_id,
                        voice_provider,
                        _safe_provider_token(type(exc).__name__) or "ProviderError",
                        (provider_diagnostics.get("stream_close") or {}).get("reason"),
                    )
                    break
                failure = _provider_error_diagnostics(exc, provider_hint=voice_provider)
                failure["stage"] = "provider_output"
                provider_diagnostics["last_failure"] = failure
                provider_diagnostics["input_audio_chunks"] = provider_audio_chunks
                provider_diagnostics["input_audio_bytes"] = provider_audio_bytes
                provider_diagnostics["stream_close"] = {"reason": "provider_output_exception", "websocket_code": 1011}
                recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
                logger.error(
                    "provider_failure session=%s stage=provider_output provider=%s category=%s exception_type=%s http_status=%s error_type=%s error_code=%s elapsed_s=%.3f websocket_close_code=%s provider_close_code=%s provider_close_reason=%s traceback=%s",
                    session_id,
                    failure["provider"],
                    failure["category"],
                    failure["exception_type"],
                    failure["http_status"],
                    failure["error_type"],
                    failure["error_code"],
                    time.monotonic() - session_started_perf,
                    (provider_diagnostics.get("stream_close") or {}).get("websocket_code"),
                    failure.get("provider_close_code"),
                    failure.get("provider_close_reason"),
                    _safe_traceback_summary(exc),
                )
                stop_event.set()
                try:
                    await input_handle.close()
                    await websocket.send_json({
                        "type": "error",
                        "code": failure["category"],
                        "message": _safe_provider_error_message(failure),
                    })
                    await websocket.close(code=1011, reason="Live provider output failed")
                except Exception:
                    pass
                break

    try:
        await asyncio.gather(output_loop(), input_loop(), return_exceptions=True)
        receive_cleanup_error = await _cancel_and_drain_receive_task(output_receive_task)
        output_receive_task = None
        if receive_cleanup_error is not None:
            close_reason = (provider_diagnostics.get("stream_close") or {}).get("reason")
            if close_reason in {"client_stop", "browser_disconnect"}:
                logger.info(
                    "provider_receive_closed_during_cleanup session=%s provider=%s exception_type=%s reason=%s",
                    session_id,
                    voice_provider,
                    _safe_provider_token(type(receive_cleanup_error).__name__) or "ProviderError",
                    close_reason,
                )
            else:
                failure = _provider_error_diagnostics(receive_cleanup_error, provider_hint=voice_provider)
                failure["stage"] = "provider_receive_cleanup"
                provider_diagnostics["last_failure"] = failure
                provider_diagnostics["stream_close"] = {
                    "reason": "provider_receive_cleanup_exception",
                    "websocket_code": 1011,
                }
                recorder.patch_server({"provider_diagnostics": dict(provider_diagnostics)})
                logger.error(
                    "provider_failure session=%s stage=provider_receive_cleanup provider=%s category=%s exception_type=%s elapsed_s=%.3f websocket_close_code=1011",
                    session_id,
                    voice_provider,
                    failure["category"],
                    failure["exception_type"],
                    time.monotonic() - session_started_perf,
                )
        if pending_tool_tasks:
            await asyncio.gather(*pending_tool_tasks, return_exceptions=True)

        provider_diagnostics["elapsed_session_s"] = round(time.monotonic() - session_started_perf, 3)
        recorder.patch_server(
            {
                "status": "processing_artifacts",
                "ended_at": utc_now(),
                "provider_diagnostics": {
                    **provider_diagnostics,
                    "input_audio_chunks": provider_audio_chunks,
                    "input_audio_bytes": provider_audio_bytes,
                    "assistant_audio_frames_forwarded": provider_output_audio_chunks,
                },
                "enhancement": {
                    "status": "processing_reconstructions",
                    "live_input_source": enhancement_mode,
                    "fallback_to_raw": enhancement_mode != "enhanced",
                    "fallback_reason": enhancement_fallback_reason,
                },
            }
        )
        artifacts = await finalize_audio_artifacts(
            session_id=session_id,
            raw_pcm=bytes(raw_pcm),
            clean_pcm=bytes(clean_pcm),
            settings=settings,
        )
        recorder.patch_server(
            {
                "status": "completed",
                "ended_at": utc_now(),
                "enhancement": {
                    "status": "completed",
                    "live_input_source": enhancement_mode,
                    "fallback_to_raw": enhancement_mode != "enhanced",
                    "fallback_reason": enhancement_fallback_reason,
                    **artifacts,
                },
            }
        )
    except Exception as exc:
        recorder.patch_server(
            {
                "status": "error",
                "ended_at": utc_now(),
                "error": str(exc),
                "enhancement": {
                    "status": "error",
                },
            }
        )
        raise
    finally:
        _release_live_session(session_id)
        clear_live_audio_window(session_id)
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/ws")
async def websocket_session(websocket: WebSocket) -> None:
    await _websocket_session_ai_coustics(websocket)


@app.websocket("/ws-enhanced")
async def websocket_session_enhanced(websocket: WebSocket) -> None:
    await _websocket_session_ai_coustics(websocket)
