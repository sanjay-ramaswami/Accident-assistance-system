from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

from .settings import REPO_ROOT
from .slm_fact_ledger import empty_shared_ledger


SESSION_DIR = REPO_ROOT / "runs" / "sessions"
_LOCKS_GUARD = threading.Lock()
_SESSION_LOCKS: dict[str, threading.RLock] = {}
logger = logging.getLogger(__name__)


def _session_lock(session_id: str) -> threading.RLock:
    # Session updates can come from both the event loop and asyncio.to_thread.
    # Keep the full read/merge/write transaction serialized per session.
    with _LOCKS_GUARD:
        return _SESSION_LOCKS.setdefault(session_id, threading.RLock())


class SessionPersistenceError(RuntimeError):
    """Raised when a persisted session cannot be read or written safely."""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def session_path(session_id: str) -> Path:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return SESSION_DIR / f"{session_id}.json"


def _read_session(session_id: str) -> dict[str, Any]:
    try:
        path = session_path(session_id)
        if not path.exists():
            return {
                "session_id": session_id,
                "created_at": utc_now(),
                "updated_at": utc_now(),
                "server": {},
                "client": {},
                "tool_calls": [],
                "triage_turns": [],
                "merged_triage_state": {},
                "rule_overrides": [],
                "handoff_packet": None,
                "triage_meta": {},
                "stt_rescue_events": [],
                "stt_rescue_overrides": {},
                "stt_rescue_meta": {},
                "llm_prompt_snapshots": [],
                "dispatch_services": {},
                "dispatch_prompt_context": {},
                "shared_ledger": empty_shared_ledger(),
                "ledger_prompt_context": {},
            }
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SessionPersistenceError("Session storage could not be read") from exc


def _write_session(session_id: str, payload: dict[str, Any]) -> None:
    payload["updated_at"] = utc_now()
    temporary_path: Path | None = None
    try:
        target = session_path(session_id)
        # Write in the same directory and atomically replace so readers never
        # observe a truncated JSON document if a process exits mid-write.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, target)
    except OSError as exc:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise SessionPersistenceError("Session storage could not be written") from exc


def _merge_dict(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _json_safe(value: Any, active: set[int] | None = None) -> Any:
    """Convert runtime tool payloads to JSON values without failing on cycles."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if active is None:
        active = set()
    if isinstance(value, (dict, list, tuple)):
        identity = id(value)
        if identity in active:
            return {"_serialization_note": "circular_reference_omitted"}
        active.add(identity)
        try:
            if isinstance(value, dict):
                return {str(key): _json_safe(item, active) for key, item in value.items()}
            return [_json_safe(item, active) for item in value]
        finally:
            active.remove(identity)
    # Do not serialize object reprs; SDK reprs can include credentials or
    # caller data. Preserve the fact that a value was omitted without content.
    return {
        "_serialization_note": "unsupported_value_omitted",
        "value_type": type(value).__name__,
    }


class SessionRecorder:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.ensure_exists()

    def ensure_exists(self) -> None:
        with _session_lock(self.session_id):
            payload = _read_session(self.session_id)
            _write_session(self.session_id, payload)

    def patch_server(self, patch: dict[str, Any]) -> None:
        with _session_lock(self.session_id):
            payload = _read_session(self.session_id)
            payload["server"] = _merge_dict(payload.get("server", {}), patch)
            _write_session(self.session_id, payload)

    def patch_session(self, patch: dict[str, Any]) -> None:
        with _session_lock(self.session_id):
            payload = _read_session(self.session_id)
            for key, value in patch.items():
                if isinstance(value, dict) and isinstance(payload.get(key), dict):
                    payload[key] = _merge_dict(payload.get(key, {}), value)
                else:
                    payload[key] = value
            _write_session(self.session_id, payload)

    def append_tool_call(
        self,
        *,
        name: str,
        args: dict[str, Any],
        result: dict[str, Any] | None = None,
        error: str | None = None,
        started_at: str | None = None,
        elapsed_ms: int | None = None,
    ) -> None:
        with _session_lock(self.session_id):
            payload = _read_session(self.session_id)
            payload.setdefault("tool_calls", []).append(
                {
                    "at": utc_now(),
                    "started_at": started_at,
                    "name": name,
                    "args": args,
                    "result": result,
                    "error": error,
                    "elapsed_ms": elapsed_ms,
                }
            )
            _write_session(self.session_id, payload)

    def append_session_list(self, key: str, entry: dict[str, Any]) -> None:
        with _session_lock(self.session_id):
            payload = _read_session(self.session_id)
            payload.setdefault(key, []).append(entry)
            _write_session(self.session_id, payload)

    def patch_client_report(self, report: dict[str, Any]) -> None:
        with _session_lock(self.session_id):
            payload = _read_session(self.session_id)
            payload["client"] = _merge_dict(payload.get("client", {}), report)
            _write_session(self.session_id, payload)


def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        SESSION_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    sessions: list[dict[str, Any]] = []
    for path in files[:limit]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            # Session filenames and contents can contain sensitive identifiers
            # and transcripts. Keep the failure visible without logging either.
            logger.warning("session_index_record_skipped error_type=%s", type(exc).__name__)
            continue
        if not isinstance(payload, dict):
            logger.warning("session_index_record_skipped error_type=InvalidDocument")
            continue
        sessions.append(
            {
                "session_id": payload.get("session_id"),
                "updated_at": payload.get("updated_at"),
                "created_at": payload.get("created_at"),
                "server_status": payload.get("server", {}).get("status"),
                "client_status": payload.get("client", {}).get("status"),
                "user_turns": len(payload.get("client", {}).get("transcripts", {}).get("user", [])),
                "agent_turns": len(payload.get("client", {}).get("transcripts", {}).get("agent", [])),
                "tool_call_count": len(payload.get("tool_calls", [])),
                "triage_turn_count": len(payload.get("triage_turns", [])),
            }
        )
    return sessions


def get_session(session_id: str) -> dict[str, Any]:
    with _session_lock(session_id):
        return _read_session(session_id)
