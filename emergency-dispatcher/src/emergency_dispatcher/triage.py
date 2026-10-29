from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Protocol
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from openai import OpenAI
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator

from .dispatch_tools import extract_address_candidate
from .dispatch_tools import infer_issue_type
from .dispatch_tools import infer_safety_risk
from .dispatch_tools import is_searchable_location_query
from .settings import REPO_ROOT
from .settings import Settings
from .session_store import get_session
from .slm_guidance import CONSTRAINT_MODES
from .slm_guidance import DECODER_SYSTEM_PROMPT
from .slm_guidance import GLINER_ENTITY_LABELS
from .slm_guidance import GUIDANCE_ISSUE_TYPES
from .slm_guidance import GUIDANCE_PRIORITIES
from .slm_guidance import GuidanceRecord
from .slm_guidance import YES_NO_UNKNOWN
from .slm_guidance import constraint_mode_for_severity
from .slm_guidance import has_explicit_location_cue
from .slm_guidance import priority_for_severity


TRIAGE_ENGINE_VALUES = ("heuristic", "llm", "slm")
TRIAGE_ISSUE_TYPES = GUIDANCE_ISSUE_TYPES
TRIAGE_PRIORITIES = GUIDANCE_PRIORITIES
YN_UNKNOWN = YES_NO_UNKNOWN
UNKNOWN_COUNT = "unknown"
SERIOUS_TRIAGE_TYPES = {
    "BUILDING_FIRE",
    "BREATHING_DISTRESS",
    "ACTIVE_THREAT_OR_WEAPON",
    "GAS_LEAK_OR_HAZMAT",
    "TRAPPED_PERSON",
    "VEHICLE_ACCIDENT_WITH_INJURY",
    "VEHICLE_FIRE",
    "PERSON_UNCONSCIOUS_OR_UNRESPONSIVE",
}
ALWAYS_ESCALATE_CODES = {
    "WEAPON_REPORTED",
    "EXPLOSION_REPORTED",
    "PERSON_IN_WATER",
    "CHILD_LOCKED_IN_HEAT",
    "ACTIVE_SELF_DANGER",
    "LOCATION_UNUSABLE",
    "MULTIPLE_CASUALTIES",
    "CHILD_CALLING_ALONE",
    "EVOLVING_DANGER",
}
HAZARD_WEAPON_PATTERN = re.compile(r"\b(weapon|gun|knife|shooting|shot|rifle|armed)\b", re.IGNORECASE)
EXPLOSION_PATTERN = re.compile(r"\b(explosion|exploded|blast|detonation)\b", re.IGNORECASE)
WATER_PATTERN = re.compile(r"\b(water|river|lake|canal|drowning|submerged|sinking)\b", re.IGNORECASE)
CHILD_PATTERN = re.compile(r"\b(child|kid|toddler|baby|infant)\b", re.IGNORECASE)
HEAT_PATTERN = re.compile(r"\b(hot car|heat|overheating car|sun)\b", re.IGNORECASE)
EVOLVING_PATTERN = re.compile(
    r"\b(getting worse|worsening|spreading|more smoke|more fire|still shooting|active shooter|spreading fire)\b",
    re.IGNORECASE,
)
SELF_DANGER_PATTERN = re.compile(r"\b(i am not safe|i am trapped|i cannot get out|i am going to die|help me now)\b", re.IGNORECASE)
MULTIPLE_PATTERN = re.compile(r"\b(multiple|several|many people|people are shot|casualties)\b", re.IGNORECASE)
CALLER_CHILD_PATTERN = re.compile(r"\b(i am a child|i'm a child|i am alone and i am \d{1,2})\b", re.IGNORECASE)
LOCATION_HINT_PATTERN = re.compile(
    r"\b(?:at|near|close to|on|around|inside|in)\s+([A-Za-z0-9.\-'\s]{3,80}?)(?=[,.!?]|$)",
    re.IGNORECASE,
)
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}
QUESTION_PRIORITY_ORDER: tuple[tuple[str, str], ...] = (
    ("issue_type", "what_is_the_emergency"),
    ("caller_safe", "are_you_safe"),
    ("location", "where_are_you"),
    ("victim_count", "how_many_people"),
    ("victim_conscious", "is_anyone_awake"),
    ("victim_breathing", "is_anyone_breathing"),
    ("victim_bleeding", "is_there_bleeding"),
)


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(needle in lowered for needle in needles)


def _clean_location_candidate(value: str | None) -> str | None:
    candidate = _normalize_text(value)
    if not candidate:
        return None
    for splitter in (
        " and ",
        " but ",
        " because ",
        " while ",
        " where ",
        " with ",
        " my ",
        " there ",
    ):
        if splitter in candidate.casefold():
            parts = re.split(splitter, candidate, maxsplit=1, flags=re.IGNORECASE)
            candidate = _normalize_text(parts[0])
            break
    candidate = re.sub(r"^(the|a)\s+", "", candidate, flags=re.IGNORECASE)
    return candidate or None


def _infer_breathing(text: str) -> Literal["yes", "no", "unknown"]:
    lowered = text.casefold()
    uncertain_breathing = re.search(
        r"\b(?:don't know|do not know|not sure|unsure|can't tell|cannot tell|unable to tell|not able to tell|maybe)\b"
        r"[^.!?]{0,40}\b(?:breath|breathing)\b|"
        r"\b(?:breath|breathing)\b[^.!?]{0,25}\b(?:or not|maybe|not sure)\b",
        lowered,
    )
    if uncertain_breathing:
        return "unknown"
    if _contains_any(lowered, ("not breathing", "isn't breathing", "isnt breathing", "stopped breathing")):
        return "no"
    if "breath" not in lowered:
        return "unknown"
    if _contains_any(lowered, ("breathing", "breathing normally", "is breathing")):
        return "yes"
    return "unknown"


def _infer_bleeding(text: str) -> Literal["yes", "no", "unknown"]:
    lowered = text.casefold()
    if _contains_any(lowered, ("no bleeding", "not bleeding", "isn't bleeding", "isnt bleeding")):
        return "no"
    if _contains_any(lowered, ("bleeding", "blood", "heavy bleeding", "bleeding badly")):
        return "yes"
    return "unknown"


def _infer_conscious(text: str) -> Literal["yes", "no", "unknown"]:
    lowered = text.casefold()
    if _contains_any(lowered, ("unconscious", "unresponsive", "passed out")):
        return "no"
    if "conscious" not in lowered and not _contains_any(lowered, ("awake", "responding", "responsive")):
        return "unknown"
    if _contains_any(lowered, ("don't know", "do not know", "not sure", "unsure", "maybe")):
        return "unknown"
    if _contains_any(lowered, ("awake", "conscious", "responding", "responsive")):
        return "yes"
    return "unknown"


def _infer_location(text: str) -> str | None:
    for landmark in ("Delta Campus", "Mullerstrasse", "Donaustraße", "Berliner Straße"):
        if landmark.casefold() in text.casefold():
            return landmark
    candidate = extract_address_candidate(text)
    if candidate:
        return candidate
    match = LOCATION_HINT_PATTERN.search(text)
    if match:
        location = _clean_location_candidate(match.group(1))
        if location and len(location) >= 3:
            return location
    return None


def _infer_triage_issue_type(text: str) -> str | None:
    lowered = text.casefold()
    if HAZARD_WEAPON_PATTERN.search(text):
        return "ACTIVE_THREAT_OR_WEAPON"
    if _contains_any(lowered, ("gas leak", "chemical", "hazmat", "fumes", "toxic")):
        return "GAS_LEAK_OR_HAZMAT"
    if _contains_any(lowered, ("child missing", "can't find my child", "cannot find my child", "kid is missing", "my son is missing", "my daughter is missing")):
        return "CHILD_MISSING_OR_UNACCOUNTED"
    if "fire" in lowered or "burning" in lowered:
        if _contains_any(lowered, ("apartment", "building", "house", "kitchen", "bedroom", "inside")):
            return "BUILDING_FIRE"
        return "VEHICLE_FIRE"
    if "smoke" in lowered:
        if _contains_any(lowered, ("car", "vehicle", "engine", "under the hood", "bonnet")):
            return "VEHICLE_FIRE"
        return "SMOKE_INVESTIGATION"
    if _contains_any(lowered, ("can't breathe", "cannot breathe", "breathing trouble", "trouble breathing", "short of breath", "asthma")):
        return "BREATHING_DISTRESS"
    if "not breathing" in lowered or "unconscious" in lowered or "unresponsive" in lowered:
        return "PERSON_UNCONSCIOUS_OR_UNRESPONSIVE"
    if any(token in lowered for token in ("car accident", "collision", "crash", "hit", "bleeding", "injured")):
        return "VEHICLE_ACCIDENT_WITH_INJURY"
    if any(token in lowered for token in ("fender bender", "rear ended", "rear-ended", "bumped", "minor crash")):
        return "VEHICLE_ACCIDENT"
    if "flat tyre" in lowered or "flat tire" in lowered or "puncture" in lowered:
        return "VEHICLE_BREAKDOWN"
    if "battery" in lowered and any(token in lowered for token in ("dead", "won't start", "wont start", "not start")):
        return "VEHICLE_BREAKDOWN"
    if "out of fuel" in lowered or "empty tank" in lowered or "ran out of gas" in lowered:
        return "VEHICLE_BREAKDOWN"
    if "wrong fuel" in lowered or "diesel in petrol" in lowered or "petrol in diesel" in lowered:
        return "VEHICLE_BREAKDOWN"
    if "locked out" in lowered or "keys in car" in lowered:
        return "VEHICLE_BREAKDOWN"
    if "debris" in lowered or "pothole" in lowered or "obstruction" in lowered or "hazard" in lowered:
        return "ROAD_HAZARD"
    if "overheating" in lowered or "steam" in lowered or "temperature warning" in lowered:
        return "VEHICLE_BREAKDOWN"
    if "breakdown" in lowered or "broke down" in lowered or "won't move" in lowered:
        return "VEHICLE_BREAKDOWN"
    if _contains_any(lowered, ("trapped", "can't get out", "cannot get out", "pinned")):
        return "TRAPPED_PERSON"
    if _contains_any(lowered, ("medical emergency", "passed out", "seizure", "chest pain", "collapse", "collapsed")):
        return "MEDICAL_EMERGENCY"
    if "lost" in lowered or "stranded" in lowered:
        return "LOST_OR_STRANDED"
    mapped = infer_issue_type(text)
    if mapped == "FIRE":
        if _contains_any(lowered, ("apartment", "building", "house", "inside")):
            return "BUILDING_FIRE"
        return "VEHICLE_FIRE"
    if mapped == "MEDICAL":
        return "MEDICAL_EMERGENCY"
    if mapped == "TRAFFIC":
        return "VEHICLE_ACCIDENT"
    if mapped == "HAZMAT":
        return "GAS_LEAK_OR_HAZMAT"
    if mapped == "POLICE":
        return "ACTIVE_THREAT_OR_WEAPON"
    return None


def _infer_priority(issue_type: str | None, text: str) -> str | None:
    lowered = text.casefold()
    if issue_type in SERIOUS_TRIAGE_TYPES:
        return "CRITICAL"
    if issue_type in {"MEDICAL_EMERGENCY", "VEHICLE_ACCIDENT", "CHILD_MISSING_OR_UNACCOUNTED"}:
        return "HIGH"
    if any(token in lowered for token in ("heavy bleeding", "not breathing", "trapped", "fire", "shooting")):
        return "CRITICAL"
    if any(token in lowered for token in ("injured", "unsafe", "urgent")):
        return "HIGH"
    if any(token in lowered for token in ("stranded", "breakdown", "flat tyre", "flat tire", "road hazard", "smoke")):
        return "MEDIUM"
    if issue_type:
        return "LOW"
    return None


def _infer_victim_count(text: str) -> int | Literal["unknown"]:
    lowered = text.casefold()
    digits = re.search(r"\b(\d{1,2})\s+(?:people|victims|casualties|persons)\b", lowered)
    if digits:
        return int(digits.group(1))
    for word, number in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\s+(?:people|victims|casualties|persons)\b", lowered):
            return number
    if MULTIPLE_PATTERN.search(text):
        return 2
    return UNKNOWN_COUNT


def _infer_caller_safe(text: str) -> Literal["yes", "no", "unknown"]:
    lowered = text.casefold()
    if any(token in lowered for token in ("i am safe", "we are safe", "safe location", "out of danger")):
        return "yes"
    if any(token in lowered for token in ("not safe", "unsafe", "trapped", "inside with fire", "active danger")):
        return "no"
    return "unknown"


def _heuristic_severity_drivers(result: "TriageResult", raw_text: str = "") -> list[str]:
    drivers: list[str] = []
    lowered = raw_text.casefold()
    if result.issue_type == "BUILDING_FIRE":
        drivers.append("indoor_fire_immediate")
    elif result.issue_type == "VEHICLE_FIRE":
        drivers.append("vehicle_fire_adjacent")
    elif result.issue_type == "ACTIVE_THREAT_OR_WEAPON":
        drivers.append("weapon_shooting_explosion")
    elif result.issue_type == "BREATHING_DISTRESS":
        drivers.append("breathing_difficulty_with_smoke")
    elif result.issue_type == "TRAPPED_PERSON":
        drivers.append("trapped_or_cannot_exit")
    elif result.issue_type == "CHILD_MISSING_OR_UNACCOUNTED":
        drivers.append("children_unaccounted_for")
    elif result.issue_type == "VEHICLE_ACCIDENT_WITH_INJURY":
        drivers.append("injury_reported")
    if result.victim_breathing == "no":
        drivers.append("unconscious_or_unresponsive")
    if result.victim_bleeding == "yes":
        drivers.append("heavy_bleeding" if "heavy bleeding" in lowered else "bleeding_any")
    if result.victim_conscious == "no":
        drivers.append("unconscious_or_unresponsive")
    if result.caller_safe == "no":
        drivers.append("caller_cannot_leave")
    if isinstance(result.victim_count, int) and result.victim_count > 1:
        drivers.append("multiple_casualties")
    if _contains_any(lowered, ("worse", "worsening", "spreading", "more smoke", "more fire")):
        drivers.append("situation_worsening")
    if _contains_any(lowered, ("everyone is safe", "nobody is hurt", "no one is hurt")):
        drivers.append("everyone_explicitly_safe")
    return drivers


def _heuristic_severity_from_result(result: "TriageResult", raw_text: str = "") -> int:
    score = {
        None: 0,
        "LOW": 15,
        "MEDIUM": 40,
        "HIGH": 65,
        "CRITICAL": 90,
    }.get(result.priority, 0)
    if result.caller_safe == "no":
        score = max(score, 80)
    if result.victim_breathing == "no":
        score = max(score, 95)
    if result.victim_bleeding == "yes":
        score = max(score, 70)
    if result.victim_conscious == "no":
        score = max(score, 90)
    if isinstance(result.victim_count, int) and result.victim_count > 1:
        score = min(100, score + 10)
    if _contains_any(raw_text.casefold(), ("everyone is safe", "nobody is hurt", "no one is hurt")):
        score = max(0, score - 15)
    return max(0, min(100, score))


def _enrich_triage_guidance(result: "TriageResult", *, previous_severity: int, raw_text: str = "") -> "TriageResult":
    severity_score = result.severity_score
    if severity_score is None:
        severity_score = _heuristic_severity_from_result(result, raw_text)
    severity_delta = result.severity_delta
    if severity_delta is None:
        severity_delta = severity_score - previous_severity
    severity_drivers = list(result.severity_drivers or _heuristic_severity_drivers(result, raw_text))
    human_required = bool(result.human_required or result.escalate_to_human or severity_score >= 85 or severity_delta >= 30)
    human_recommended = bool(result.human_recommended or human_required or severity_score >= 60 or severity_delta >= 20)
    constraint_mode = result.constraint_mode if result.constraint_mode in CONSTRAINT_MODES else constraint_mode_for_severity(severity_score)
    priority = result.priority or priority_for_severity(severity_score)
    escalation_reason = result.escalation_reason
    if human_required and not escalation_reason:
        escalation_reason = "human_required"
    return result.model_copy(
        update={
            "priority": priority,
            "severity_score": severity_score,
            "severity_delta": severity_delta,
            "severity_drivers": severity_drivers,
            "human_recommended": human_recommended,
            "human_required": human_required,
            "agent_should_continue": True,
            "constraint_mode": constraint_mode,
            "escalate_to_human": bool(result.escalate_to_human or human_required),
            "escalation_reason": escalation_reason,
        }
    )


class TriageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str | None = None
    issue_type: str | None = None
    priority: str | None = None
    victim_breathing: str = "unknown"
    victim_bleeding: str = "unknown"
    victim_conscious: str = "unknown"
    victim_count: int | Literal["unknown"] = UNKNOWN_COUNT
    caller_safe: str = "unknown"
    escalate_to_human: bool = False
    escalation_reason: str | None = None
    severity_score: int | None = None
    severity_delta: int | None = None
    severity_drivers: list[str] = Field(default_factory=list)
    human_recommended: bool = False
    human_required: bool = False
    agent_should_continue: bool = True
    constraint_mode: str | None = None
    address_in_utterance: bool = False

    @field_validator("issue_type")
    @classmethod
    def _validate_issue_type(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if value not in TRIAGE_ISSUE_TYPES:
            raise ValueError(f"Unsupported issue_type: {value}")
        return value

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if value not in TRIAGE_PRIORITIES:
            raise ValueError(f"Unsupported priority: {value}")
        return value

    @field_validator("victim_breathing", "victim_bleeding", "victim_conscious", "caller_safe")
    @classmethod
    def _validate_yes_no_unknown(cls, value: str) -> str:
        if value not in YN_UNKNOWN:
            raise ValueError(f"Unsupported yes/no/unknown value: {value}")
        return value

    @field_validator("victim_count", mode="before")
    @classmethod
    def _coerce_victim_count(cls, value: Any) -> int | str:
        if value in (None, "", "unknown"):
            return UNKNOWN_COUNT
        if isinstance(value, bool):
            raise ValueError("victim_count cannot be boolean")
        return max(int(value), 0)

    @field_validator("severity_score", mode="before")
    @classmethod
    def _coerce_severity_score(cls, value: Any) -> int | None:
        if value in (None, ""):
            return None
        return max(0, min(100, int(value)))

    @field_validator("severity_delta", mode="before")
    @classmethod
    def _coerce_severity_delta(cls, value: Any) -> int | None:
        if value in (None, ""):
            return None
        return max(-100, min(100, int(value)))

    @field_validator("severity_drivers", mode="before")
    @classmethod
    def _coerce_severity_drivers(cls, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if not isinstance(value, list):
            raise ValueError("severity_drivers must be a list")
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("constraint_mode")
    @classmethod
    def _validate_constraint_mode(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if value not in CONSTRAINT_MODES:
            raise ValueError(f"Unsupported constraint_mode: {value}")
        return value


def _unknown_triage_fields(result: TriageResult) -> list[str]:
    unknown = []
    for field_name in ("location", "issue_type", "priority"):
        if getattr(result, field_name) in (None, ""):
            unknown.append(field_name)
    for field_name in ("victim_breathing", "victim_bleeding", "victim_conscious", "caller_safe"):
        if getattr(result, field_name) == "unknown":
            unknown.append(field_name)
    if result.victim_count == UNKNOWN_COUNT:
        unknown.append("victim_count")
    return unknown


class RuleOverride(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    reason: str


class TriageUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    selected_engine: str
    runtime_provider: str
    version: int
    raw_utterance: str
    latest_triage_delta: TriageResult
    merged_triage_state: TriageResult
    missing_or_unknown_fields: list[str]
    rule_overrides: list[RuleOverride]
    handoff_packet: dict[str, Any] | None = None
    dispatchable: bool = False
    autonomy_allowed: bool = False
    stt_rescue: dict[str, Any] | None = None
    gliner_entities: list[dict[str, Any]] = Field(default_factory=list)
    model_artifacts: dict[str, Any] | None = None
    prompt_context: dict[str, Any]


class TriageEngine(Protocol):
    selected_engine: str
    runtime_provider: str
    last_artifacts: dict[str, Any]

    def analyze_utterance(
        self,
        raw_text: str,
        current_state: TriageResult,
    ) -> TriageResult: ...


class HeuristicTriageEngine:
    selected_engine = "heuristic"
    runtime_provider = "heuristic"

    def __init__(self):
        self.last_artifacts: dict[str, Any] = {}

    def analyze_utterance(self, raw_text: str, current_state: TriageResult) -> TriageResult:
        text = _normalize_text(raw_text)
        issue_type = _infer_triage_issue_type(text) or current_state.issue_type
        priority = _infer_priority(issue_type, text) or current_state.priority
        breathing = _infer_breathing(text)
        bleeding = _infer_bleeding(text)
        conscious = _infer_conscious(text)
        caller_safe = _infer_caller_safe(text)
        result = TriageResult(
            location=_infer_location(text),
            issue_type=issue_type,
            priority=priority,
            victim_breathing=breathing,
            victim_bleeding=bleeding,
            victim_conscious=conscious,
            victim_count=_infer_victim_count(text),
            caller_safe=caller_safe,
            escalate_to_human=False,
            escalation_reason=None,
        )
        if result.priority == "CRITICAL":
            result.escalate_to_human = True
            result.escalation_reason = "priority_critical"
        if result.issue_type in SERIOUS_TRIAGE_TYPES:
            result.escalate_to_human = True
            result.escalation_reason = f"serious_incident:{result.issue_type}"
        result = _enrich_triage_guidance(result, previous_severity=int(current_state.severity_score or 0), raw_text=text)
        self.last_artifacts = {
            "model_id": "heuristic-demo-v1",
            "model_kind": "deterministic_rules",
            "inference_status": "demo_only",
            "demo_only": True,
            "input_features": {
                "location": result.location,
                "issue_type": result.issue_type,
                "victim_breathing": result.victim_breathing,
                "victim_bleeding": result.victim_bleeding,
                "victim_conscious": result.victim_conscious,
                "victim_count": result.victim_count,
                "caller_safe": result.caller_safe,
            },
            "uncertainty": {
                "unknown_fields": _unknown_triage_fields(result),
                "calibrated_confidence": None,
            },
            "limitations": "Rule-based demo fallback; not a trained or clinically validated model.",
        }
        return result


def _effective_triage_api_key(settings: Settings) -> str | None:
    return settings.triage_api_key or settings.openai_api_key


def _chat_completion_url(base_url: str) -> str:
    normalized = (base_url or "https://api.pioneer.ai/v1").rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def _post_json(*, url: str, api_key: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    request = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(request, timeout=timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} calling triage inference endpoint: {detail}") from exc
    except urllib_error.URLError as exc:
        raise RuntimeError(f"Failed to reach triage inference endpoint: {exc.reason}") from exc


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("No choices returned from model")
    message = (choices[0] or {}).get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            text = item.get("text") or item.get("content")
            if text:
                text_parts.append(str(text))
        if text_parts:
            return "".join(text_parts)
    raise ValueError("Model response did not contain message content")


def _parse_json_content(payload: dict[str, Any]) -> dict[str, Any]:
    content = _extract_message_content(payload).strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(content[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model returned non-object JSON")
    return parsed


def _normalize_gliner_entities(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_entities = payload.get("entities")
    if raw_entities is None:
        parsed = _parse_json_content(payload)
        raw_entities = parsed.get("entities", [])
    entities: list[dict[str, Any]] = []
    for item in raw_entities or []:
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("entity") or item.get("type") or "").strip()
            text = str(item.get("text") or item.get("span") or item.get("value") or "").strip()
            if not label or not text:
                continue
            entity: dict[str, Any] = {
                "label": label,
                "text": text,
            }
            if item.get("confidence") is not None:
                entity["confidence"] = item.get("confidence")
            elif item.get("score") is not None:
                entity["confidence"] = item.get("score")
            if item.get("start") is not None:
                entity["start"] = item.get("start")
            if item.get("end") is not None:
                entity["end"] = item.get("end")
            entities.append(entity)
            continue
        if isinstance(item, list) and len(item) >= 2:
            text = str(item[0]).strip()
            label = str(item[1]).strip()
            if not label or not text:
                continue
            entity = {"label": label, "text": text}
            if len(item) >= 3 and item[2] is not None:
                entity["confidence"] = item[2]
            entities.append(entity)
    return entities


def _coerce_gliner_victim_count(text: str) -> int | Literal["unknown"]:
    lowered = text.casefold()
    if lowered in NUMBER_WORDS:
        return NUMBER_WORDS[lowered]
    digits = re.search(r"\d{1,2}", lowered)
    if digits:
        return int(digits.group(0))
    return UNKNOWN_COUNT


def _apply_gliner_hints(raw_text: str, result: TriageResult, entities: list[dict[str, Any]]) -> TriageResult:
    updates: dict[str, Any] = {}
    if result.location is None:
        for entity in entities:
            if entity.get("label") == "LOCATION":
                candidate = _clean_location_candidate(entity.get("text"))
                if candidate and (has_explicit_location_cue(raw_text) or candidate.casefold() in raw_text.casefold()):
                    updates["location"] = candidate
                    break
    if result.victim_count == UNKNOWN_COUNT:
        for entity in entities:
            if entity.get("label") == "VICTIM_COUNT":
                count = _coerce_gliner_victim_count(str(entity.get("text") or ""))
                if count != UNKNOWN_COUNT:
                    updates["victim_count"] = count
                    break
    if not updates:
        return result
    return result.model_copy(update=updates)


class LLMTriageEngine:
    selected_engine = "llm"
    runtime_provider = "openai"

    def __init__(self, settings: Settings):
        self._settings = settings
        self.last_artifacts: dict[str, Any] = {}
        base_url = (
            settings.openai_base_url
            if settings.openai_base_url and settings.openai_base_url.startswith(("http://", "https://"))
            else "https://api.openai.com/v1"
        )
        self._client = OpenAI(api_key=settings.openai_api_key, base_url=base_url)

    def analyze_utterance(self, raw_text: str, current_state: TriageResult) -> TriageResult:
        schema = {
            "location": None,
            "issue_type": None,
            "priority": None,
            "victim_breathing": "unknown",
            "victim_bleeding": "unknown",
            "victim_conscious": "unknown",
            "victim_count": "unknown",
            "caller_safe": "unknown",
            "escalate_to_human": False,
            "escalation_reason": None,
        }
        prompt = (
            "Return only JSON. Extract the emergency triage schema from the caller utterance.\n"
            f"Current merged state: {current_state.model_dump_json()}\n"
            f"Allowed issue_type values: {', '.join(TRIAGE_ISSUE_TYPES)}.\n"
            f"Allowed priority values: {', '.join(TRIAGE_PRIORITIES)}.\n"
            f"Allowed yes/no/unknown fields: {', '.join(YN_UNKNOWN)}.\n"
            f"Default unknown object: {json.dumps(schema)}\n"
            f"Caller utterance: {raw_text}"
        )
        try:
            response = self._client.chat.completions.create(
                model=self._settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You extract structured emergency triage JSON. Respond with JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            result = TriageResult.model_validate(parsed)
            result = _enrich_triage_guidance(result, previous_severity=int(current_state.severity_score or 0), raw_text=raw_text)
            self.runtime_provider = "openai"
            self.last_artifacts = {
                "model_id": getattr(response, "model", None) or self._settings.openai_model,
                "model_kind": "remote_llm",
                "inference_status": "success",
                "demo_only": False,
                "uncertainty": {
                    "unknown_fields": _unknown_triage_fields(result),
                    "calibrated_confidence": None,
                },
            }
            return result
        except Exception as exc:
            fallback = HeuristicTriageEngine()
            result = fallback.analyze_utterance(raw_text, current_state)
            self.runtime_provider = "heuristic_fallback"
            self.last_artifacts = {
                **fallback.last_artifacts,
                "requested_model_id": self._settings.openai_model,
                "inference_status": "failed_demo_fallback",
                "fallback_error_type": type(exc).__name__,
            }
            return result


class SLMTriageEngine:
    selected_engine = "slm"
    runtime_provider = "pioneer"

    def __init__(self, settings: Settings):
        self._settings = settings
        self._api_key = _effective_triage_api_key(settings)
        self._endpoint = _chat_completion_url(settings.triage_base_url)
        self._timeout_s = float(settings.triage_request_timeout_s or 15.0)
        self._gliner_live_enabled = bool(getattr(settings, "triage_gliner_live_enabled", True))
        self._gliner_timeout_s = float(getattr(settings, "triage_gliner_timeout_s", self._timeout_s) or self._timeout_s)
        self.last_artifacts: dict[str, Any] = {}

    def _decoder_payload(self, *, raw_text: str, previous_severity: int) -> dict[str, Any]:
        return {
            "model": self._settings.triage_decoder_model,
            "messages": [
                {"role": "system", "content": DECODER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "previous_severity": previous_severity,
                            "raw_utterance": raw_text,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0,
        }

    def _gliner_payload(self, *, raw_text: str) -> dict[str, Any]:
        return {
            "model": self._settings.triage_gliner_model,
            "messages": [{"role": "user", "content": raw_text}],
            "schema": {
                "entities": list(GLINER_ENTITY_LABELS),
                # Pioneer currently rejects empty schema sections.
                "relations": ["related_to"],
            },
            "include_confidence": True,
            "include_spans": True,
        }

    def analyze_utterance(self, raw_text: str, current_state: TriageResult) -> TriageResult:
        self.last_artifacts = {}
        if not self._api_key or not self._settings.triage_decoder_model:
            self.runtime_provider = "heuristic_fallback"
            return HeuristicTriageEngine().analyze_utterance(raw_text, current_state)

        text = _normalize_text(raw_text)
        previous_severity = int(current_state.severity_score or 0)
        started_at = time.perf_counter()
        decoder_elapsed_s: float | None = None
        gliner_elapsed_s: float | None = None
        try:
            decoder_started_at = time.perf_counter()
            decoder_response = _post_json(
                url=self._endpoint,
                api_key=self._api_key,
                payload=self._decoder_payload(raw_text=text, previous_severity=previous_severity),
                timeout_s=self._timeout_s,
            )
            decoder_elapsed_s = time.perf_counter() - decoder_started_at
            decoder_payload = _parse_json_content(decoder_response)
            guidance = GuidanceRecord.model_validate(
                {
                    "raw_utterance": text,
                    "previous_severity": previous_severity,
                    **decoder_payload,
                }
            )
            result = TriageResult.model_validate(
                {
                    "location": guidance.extracted_fields.location,
                    "issue_type": guidance.extracted_fields.issue_type,
                    "priority": guidance.extracted_fields.priority,
                    "victim_breathing": guidance.extracted_fields.victim_breathing,
                    "victim_bleeding": guidance.extracted_fields.victim_bleeding,
                    "victim_conscious": guidance.extracted_fields.victim_conscious,
                    "victim_count": guidance.extracted_fields.victim_count,
                    "caller_safe": guidance.extracted_fields.caller_safe,
                    "escalate_to_human": guidance.human_required,
                    "escalation_reason": "human_required" if guidance.human_required else None,
                    "severity_score": guidance.severity_score,
                    "severity_delta": guidance.severity_delta,
                    "severity_drivers": guidance.severity_drivers,
                    "human_recommended": guidance.human_recommended,
                    "human_required": guidance.human_required,
                    "agent_should_continue": guidance.agent_should_continue,
                    "constraint_mode": guidance.constraint_mode,
                    "address_in_utterance": guidance.address_in_utterance,
                }
            )
            gliner_entities: list[dict[str, Any]] = []
            if self._settings.triage_gliner_model and self._gliner_live_enabled:
                try:
                    gliner_started_at = time.perf_counter()
                    gliner_response = _post_json(
                        url=self._endpoint,
                        api_key=self._api_key,
                        payload=self._gliner_payload(raw_text=text),
                        timeout_s=min(self._timeout_s, self._gliner_timeout_s),
                    )
                    gliner_elapsed_s = time.perf_counter() - gliner_started_at
                    gliner_entities = _normalize_gliner_entities(gliner_response)
                    result = _apply_gliner_hints(text, result, gliner_entities)
                except Exception:
                    gliner_entities = []
            self.runtime_provider = "pioneer"
            result = _enrich_triage_guidance(result, previous_severity=previous_severity, raw_text=text)
            self.last_artifacts = {
                "decoder_model": self._settings.triage_decoder_model,
                "decoder_guidance": guidance.model_dump(),
                "gliner_model": self._settings.triage_gliner_model,
                "gliner_live_enabled": self._gliner_live_enabled,
                "gliner_entities": gliner_entities,
                "model_id": self._settings.triage_decoder_model,
                "model_kind": "remote_slm",
                "inference_status": "success",
                "demo_only": False,
                "input_features": guidance.extracted_fields.model_dump(),
                "uncertainty": {
                    "unknown_fields": _unknown_triage_fields(result),
                    "calibrated_confidence": None,
                },
                "limitations": "Advisory information extraction only; emergency protocols and human authorization remain authoritative.",
                "timings_s": {
                    "total": round(time.perf_counter() - started_at, 4),
                    "decoder": round(decoder_elapsed_s or 0.0, 4),
                    "gliner": round(gliner_elapsed_s or 0.0, 4) if gliner_elapsed_s is not None else None,
                },
            }
            return result
        except Exception as exc:
            self.runtime_provider = "pioneer_error"
            self.last_artifacts = {
                "decoder_model": self._settings.triage_decoder_model,
                "gliner_model": self._settings.triage_gliner_model,
                "gliner_live_enabled": self._gliner_live_enabled,
                "timings_s": {
                    "total": round(time.perf_counter() - started_at, 4),
                    "decoder": round(decoder_elapsed_s or 0.0, 4),
                    "gliner": round(gliner_elapsed_s or 0.0, 4) if gliner_elapsed_s is not None else None,
                },
                "error": str(exc),
            }
            raise RuntimeError(f"SLM triage failed: {exc}") from exc


def get_triage_engine(settings: Settings) -> TriageEngine:
    selected = (settings.triage_engine or "heuristic").strip().lower()
    if selected == "llm" and settings.openai_api_key and settings.openai_model:
        return LLMTriageEngine(settings)
    if selected == "slm":
        if not _effective_triage_api_key(settings):
            raise RuntimeError("TRIAGE_ENGINE=slm requires TRIAGE_API_KEY or OPENAI_API_KEY.")
        if not settings.triage_decoder_model:
            raise RuntimeError("TRIAGE_ENGINE=slm requires TRIAGE_DECODER_MODEL.")
        return SLMTriageEngine(settings)
    return HeuristicTriageEngine()


def merge_triage_state(current_state: TriageResult, delta: TriageResult) -> TriageResult:
    merged = current_state.model_dump()
    incoming = delta.model_dump()
    for key, value in incoming.items():
        if value is None:
            continue
        if key in {"victim_breathing", "victim_bleeding", "victim_conscious", "caller_safe"} and value == "unknown":
            continue
        if key == "victim_count" and value == UNKNOWN_COUNT:
            continue
        if key == "escalate_to_human":
            merged[key] = bool(merged.get(key) or value)
            continue
        if key == "human_recommended":
            merged[key] = bool(merged.get(key) or value)
            continue
        if key == "human_required":
            merged[key] = bool(merged.get(key) or value)
            continue
        if key == "escalation_reason":
            if value:
                merged[key] = value
            continue
        if key == "severity_drivers":
            existing = list(merged.get(key) or [])
            merged[key] = list(dict.fromkeys(existing + list(value or [])))
            continue
        if key == "severity_score":
            existing = merged.get(key)
            merged[key] = max(int(existing or 0), int(value))
            continue
        if key == "severity_delta":
            merged[key] = int(value)
            continue
        if key == "constraint_mode":
            existing = merged.get(key)
            existing_rank = CONSTRAINT_MODES.index(existing) if existing in CONSTRAINT_MODES else 0
            incoming_rank = CONSTRAINT_MODES.index(value) if value in CONSTRAINT_MODES else 0
            merged[key] = value if incoming_rank >= existing_rank else existing
            continue
        merged[key] = value
    return TriageResult.model_validate(merged)


def compute_missing_or_unknown_fields(state: TriageResult) -> list[str]:
    missing: list[str] = []
    for field_name in ("location", "issue_type", "priority"):
        if getattr(state, field_name) in {None, ""}:
            missing.append(field_name)
    for field_name in ("victim_breathing", "victim_bleeding", "victim_conscious", "caller_safe"):
        if getattr(state, field_name) == "unknown":
            missing.append(field_name)
    if state.victim_count == UNKNOWN_COUNT:
        missing.append("victim_count")
    return missing


def _field_is_missing(state: TriageResult, field_name: str) -> bool:
    value = getattr(state, field_name)
    if field_name in {"issue_type", "priority", "location"}:
        return value in {None, ""}
    if field_name in {"victim_breathing", "victim_bleeding", "victim_conscious", "caller_safe"}:
        return value == "unknown"
    if field_name == "victim_count":
        return value == UNKNOWN_COUNT
    return value in {None, ""}


def _conversation_focus(
    *,
    raw_utterance: str,
    latest_delta: TriageResult,
    merged_state: TriageResult,
) -> dict[str, Any]:
    missing_fields = [field for field, _goal in QUESTION_PRIORITY_ORDER if _field_is_missing(merged_state, field)]
    next_question_field = missing_fields[0] if missing_fields else None
    next_question_goal = next(
        (goal for field, goal in QUESTION_PRIORITY_ORDER if field == next_question_field),
        None,
    )
    location_candidate = latest_delta.location or merged_state.location or extract_address_candidate(raw_utterance)
    location_has_searchable_clue = bool(location_candidate and is_searchable_location_query(location_candidate))
    location_search_allowed = bool(
        location_has_searchable_clue and next_question_field not in {"issue_type", "caller_safe"}
    )
    if location_search_allowed:
        location_search_reason = "searchable_location_clue_present"
    elif location_has_searchable_clue:
        location_search_reason = "higher_priority_field_still_missing"
    else:
        location_search_reason = "needs_more_location_detail"
    return {
        "question_priority_order": [field for field, _goal in QUESTION_PRIORITY_ORDER],
        "missing_priority_fields": missing_fields,
        "next_question_field": next_question_field,
        "next_question_goal": next_question_goal,
        "location_candidate": location_candidate,
        "location_search_allowed": location_search_allowed,
        "location_search_reason": location_search_reason,
    }


def apply_rule_overrides(raw_text: str, triage_state: TriageResult) -> list[RuleOverride]:
    overrides: list[RuleOverride] = []
    lowered = raw_text.casefold()
    if HAZARD_WEAPON_PATTERN.search(raw_text):
        overrides.append(RuleOverride(code="WEAPON_REPORTED", reason="Weapons or shooting mentioned"))
    if EXPLOSION_PATTERN.search(raw_text):
        overrides.append(RuleOverride(code="EXPLOSION_REPORTED", reason="Explosion or blast mentioned"))
    if WATER_PATTERN.search(raw_text):
        overrides.append(RuleOverride(code="PERSON_IN_WATER", reason="Person in water or sinking incident"))
    if CHILD_PATTERN.search(raw_text) and HEAT_PATTERN.search(raw_text):
        overrides.append(RuleOverride(code="CHILD_LOCKED_IN_HEAT", reason="Child in vehicle heat danger"))
    if SELF_DANGER_PATTERN.search(raw_text):
        overrides.append(RuleOverride(code="ACTIVE_SELF_DANGER", reason="Caller reports active self danger"))
    if triage_state.location is None and ("don't know where" in lowered or "do not know where" in lowered):
        overrides.append(RuleOverride(code="LOCATION_UNUSABLE", reason="Caller incoherent and location unusable"))
    if isinstance(triage_state.victim_count, int) and triage_state.victim_count > 1:
        overrides.append(RuleOverride(code="MULTIPLE_CASUALTIES", reason="Multiple casualties reported"))
    if MULTIPLE_PATTERN.search(raw_text):
        overrides.append(RuleOverride(code="MULTIPLE_CASUALTIES", reason="Multiple casualties language detected"))
    if CALLER_CHILD_PATTERN.search(raw_text):
        overrides.append(RuleOverride(code="CHILD_CALLING_ALONE", reason="Child caller alone"))
    if EVOLVING_PATTERN.search(raw_text) or triage_state.caller_safe == "no":
        if "fire" in lowered or "shoot" in lowered or "trap" in lowered:
            overrides.append(RuleOverride(code="EVOLVING_DANGER", reason="Situation is actively evolving or worsening"))
    return overrides


def enforce_rule_overrides(state: TriageResult, overrides: list[RuleOverride]) -> TriageResult:
    if not overrides:
        return state
    reason = overrides[0].code.lower()
    return state.model_copy(
        update={
            "escalate_to_human": True,
            "escalation_reason": reason,
            "priority": state.priority or "CRITICAL",
            "severity_score": max(int(state.severity_score or 0), 86),
            "human_recommended": True,
            "human_required": True,
            "constraint_mode": "holding_pattern_only",
        }
    )


def is_dispatchable(state: TriageResult) -> bool:
    if not state.location or not state.issue_type or not state.priority:
        return False
    if state.escalate_to_human:
        return True
    return state.caller_safe in {"yes", "no"}


def autonomy_allowed(state: TriageResult, overrides: list[RuleOverride]) -> bool:
    return is_dispatchable(state) and not state.escalate_to_human and not overrides


def build_handoff_packet(state: TriageResult, overrides: list[RuleOverride]) -> dict[str, Any] | None:
    if not state.escalate_to_human:
        return None
    return {
        "status": "ready_for_human_handoff",
        "issue_type": state.issue_type,
        "priority": state.priority,
        "location": state.location,
        "victim_breathing": state.victim_breathing,
        "victim_bleeding": state.victim_bleeding,
        "victim_conscious": state.victim_conscious,
        "victim_count": state.victim_count,
        "caller_safe": state.caller_safe,
        "escalation_reason": state.escalation_reason,
        "severity_score": state.severity_score,
        "severity_delta": state.severity_delta,
        "severity_drivers": state.severity_drivers,
        "constraint_mode": state.constraint_mode,
        "rule_overrides": [item.model_dump() for item in overrides],
    }


def build_prompt_context(
    *,
    update: TriageUpdate,
) -> dict[str, Any]:
    merged = update.merged_triage_state
    conversation_focus = _conversation_focus(
        raw_utterance=update.raw_utterance,
        latest_delta=update.latest_triage_delta,
        merged_state=merged,
    )
    return {
        "version": update.version,
        "raw_utterance": update.raw_utterance,
        "selected_engine": update.selected_engine,
        "runtime_provider": update.runtime_provider,
        "latest_triage_delta": update.latest_triage_delta.model_dump(),
        "merged_triage_state": merged.model_dump(),
        "missing_or_unknown_fields": update.missing_or_unknown_fields,
        "rule_overrides": [item.model_dump() for item in update.rule_overrides],
        "handoff_packet": update.handoff_packet,
        "dispatchable": update.dispatchable,
        "autonomy_allowed": update.autonomy_allowed,
        "stt_rescue": update.stt_rescue,
        "gliner_entities": update.gliner_entities,
        "conversation_focus": conversation_focus,
    }


def _join_new_user_texts(
    user_items: list[dict[str, Any]],
    start_index: int,
    rescue_overrides: dict[str, Any] | None = None,
) -> str:
    override_map = rescue_overrides or {}
    pieces: list[str] = []
    for idx in range(start_index, len(user_items)):
        override_text = override_map.get(str(idx))
        pieces.append(override_text or user_items[idx].get("text", ""))
    return _normalize_text(" ".join(pieces))


def analyze_session_triage(
    session: dict[str, Any],
    settings: Settings,
) -> TriageUpdate | None:
    user_items = session.get("client", {}).get("transcripts", {}).get("user", []) or []
    meta = session.get("triage_meta", {}) or {}
    start_index = int(meta.get("user_index", 0))
    if start_index >= len(user_items):
        return None
    raw_utterance = _join_new_user_texts(
        user_items,
        start_index,
        rescue_overrides=session.get("stt_rescue_overrides") or {},
    )
    if not raw_utterance:
        return None

    selected_engine = (settings.triage_engine or "heuristic").strip().lower()
    engine = get_triage_engine(settings)
    actual_engine = getattr(engine, "selected_engine", selected_engine)
    current_state = TriageResult.model_validate(session.get("merged_triage_state") or {})
    latest_delta = engine.analyze_utterance(raw_utterance, current_state)
    overrides = apply_rule_overrides(raw_utterance, latest_delta)
    latest_delta = enforce_rule_overrides(latest_delta, overrides)
    merged_state = merge_triage_state(current_state, latest_delta)
    if overrides:
        merged_state = enforce_rule_overrides(merged_state, overrides)
    version = int(meta.get("version", 0)) + 1
    missing = compute_missing_or_unknown_fields(merged_state)
    dispatchable = is_dispatchable(merged_state)
    handoff_packet = build_handoff_packet(merged_state, overrides)
    rescue_events = session.get("stt_rescue_events", []) or []
    latest_rescue = rescue_events[-1] if rescue_events else None
    if latest_rescue and int(latest_rescue.get("user_index", -1)) < start_index:
        latest_rescue = None
    model_artifacts = getattr(engine, "last_artifacts", {}) or {}
    update = TriageUpdate(
        selected_engine=actual_engine,
        runtime_provider=getattr(engine, "runtime_provider", selected_engine),
        version=version,
        raw_utterance=raw_utterance,
        latest_triage_delta=latest_delta,
        merged_triage_state=merged_state,
        missing_or_unknown_fields=missing,
        rule_overrides=overrides,
        handoff_packet=handoff_packet,
        dispatchable=dispatchable,
        autonomy_allowed=autonomy_allowed(merged_state, overrides),
        stt_rescue=latest_rescue,
        gliner_entities=list(model_artifacts.get("gliner_entities") or []),
        model_artifacts=model_artifacts or None,
        prompt_context={},
    )
    update.prompt_context = build_prompt_context(update=update)
    return update


def triage_session_patch(
    session: dict[str, Any],
    update: TriageUpdate,
) -> dict[str, Any]:
    triage_turns = list(session.get("triage_turns", []) or [])
    triage_turns.append(
        {
            "at": session.get("updated_at"),
            "version": update.version,
            "selected_engine": update.selected_engine,
            "runtime_provider": update.runtime_provider,
            "raw_utterance": update.raw_utterance,
            "latest_triage_delta": update.latest_triage_delta.model_dump(),
            "merged_triage_state": update.merged_triage_state.model_dump(),
            "missing_or_unknown_fields": update.missing_or_unknown_fields,
            "rule_overrides": [item.model_dump() for item in update.rule_overrides],
            "handoff_packet": update.handoff_packet,
            "dispatchable": update.dispatchable,
            "autonomy_allowed": update.autonomy_allowed,
            "stt_rescue": update.stt_rescue,
            "gliner_entities": update.gliner_entities,
            "model_artifacts": update.model_artifacts,
        }
    )
    user_count = len(session.get("client", {}).get("transcripts", {}).get("user", []) or [])
    return {
        "triage_turns": triage_turns,
        "merged_triage_state": update.merged_triage_state.model_dump(),
        "rule_overrides": [item.model_dump() for item in update.rule_overrides],
        "handoff_packet": update.handoff_packet,
        "model_artifacts": update.model_artifacts,
        "triage_meta": {
            "selected_engine": update.selected_engine,
            "runtime_provider": update.runtime_provider,
            "version": update.version,
            "user_index": user_count,
            "missing_or_unknown_fields": update.missing_or_unknown_fields,
            "dispatchable": update.dispatchable,
            "autonomy_allowed": update.autonomy_allowed,
        },
    }


def replay_triage_session(
    session: dict[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    running_state = TriageResult()
    engine = get_triage_engine(settings)
    turns: list[dict[str, Any]] = []
    for idx, item in enumerate(session.get("client", {}).get("transcripts", {}).get("user", []) or [], start=1):
        raw_text = _normalize_text(item.get("text", ""))
        if not raw_text:
            continue
        delta = engine.analyze_utterance(raw_text, running_state)
        overrides = apply_rule_overrides(raw_text, delta)
        delta = enforce_rule_overrides(delta, overrides)
        running_state = merge_triage_state(running_state, delta)
        if overrides:
            running_state = enforce_rule_overrides(running_state, overrides)
        turns.append(
            {
                "turn_index": idx,
                "raw_utterance": raw_text,
                "triage_delta": delta.model_dump(),
                "merged_state": running_state.model_dump(),
                "rule_overrides": [item.model_dump() for item in overrides],
                "dispatchable": is_dispatchable(running_state),
                "gliner_entities": list((getattr(engine, "last_artifacts", {}) or {}).get("gliner_entities") or []),
                "model_artifacts": (getattr(engine, "last_artifacts", {}) or {}) or None,
            }
        )
    return {
        "session_id": session.get("session_id"),
        "selected_engine": getattr(engine, "selected_engine", settings.triage_engine),
        "runtime_provider": getattr(engine, "runtime_provider", settings.triage_engine),
        "turns": turns,
        "final_state": running_state.model_dump(),
    }


def replay_triage_from_session_id(session_id: str, settings: Settings) -> dict[str, Any]:
    session = get_session(session_id)
    return replay_triage_session(session, settings)


def replay_triage_to_file(
    *,
    session_id: str,
    settings: Settings,
    output_path: Path | None = None,
) -> Path:
    payload = replay_triage_from_session_id(session_id, settings)
    target = output_path or (REPO_ROOT / "runs" / "replay" / f"{session_id}_triage.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
