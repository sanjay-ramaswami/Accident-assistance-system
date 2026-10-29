from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, Field, field_validator


SLM_GUIDANCE_SCHEMA_VERSION = "2026-04-25.v1"
GUIDANCE_ISSUE_TYPES = (
    "VEHICLE_BREAKDOWN",
    "VEHICLE_ACCIDENT",
    "VEHICLE_ACCIDENT_WITH_INJURY",
    "VEHICLE_FIRE",
    "BUILDING_FIRE",
    "SMOKE_INVESTIGATION",
    "MEDICAL_EMERGENCY",
    "PERSON_UNCONSCIOUS_OR_UNRESPONSIVE",
    "BREATHING_DISTRESS",
    "ACTIVE_THREAT_OR_WEAPON",
    "ROAD_HAZARD",
    "CHILD_MISSING_OR_UNACCOUNTED",
    "GAS_LEAK_OR_HAZMAT",
    "TRAPPED_PERSON",
    "LOST_OR_STRANDED",
)
GUIDANCE_PRIORITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")
YES_NO_UNKNOWN = ("yes", "no", "unknown")
CONSTRAINT_MODES = (
    "normal",
    "single_question_only",
    "pre_arrival_priority",
    "holding_pattern_only",
)
GLINER_ENTITY_LABELS = (
    "BLEEDING_STATUS",
    "BREATHING_STATUS",
    "CONSCIOUSNESS_STATUS",
    "ISSUE_CUE",
    "LOCATION",
    "SEVERITY_DRIVER",
    "VICTIM_COUNT",
    "CALLER_SAFETY",
)
DECODER_SYSTEM_PROMPT = """You convert one caller utterance plus previous severity into structured dispatch-guidance JSON.
Return JSON only.
Do not include markdown.
Use exactly these keys:
- extracted_fields
- severity_score
- severity_delta
- severity_drivers
- human_recommended
- human_required
- agent_should_continue
- constraint_mode
- address_in_utterance

Within extracted_fields use exactly these keys:
- location
- issue_type
- priority
- victim_breathing
- victim_bleeding
- victim_conscious
- victim_count
- caller_safe

Allowed values:
- issue_type: one of %s
- priority: one of %s
- victim_breathing, victim_bleeding, victim_conscious, caller_safe: one of %s
- victim_count: integer or "unknown"
- constraint_mode: one of %s
- severity_score: integer from 0 to 100
- severity_delta: integer from -100 to 100
- severity_drivers: array of short snake_case strings
- human_recommended, human_required, agent_should_continue, address_in_utterance: boolean
- location: string or null
""" % (
    ", ".join(GUIDANCE_ISSUE_TYPES),
    ", ".join(GUIDANCE_PRIORITIES),
    ", ".join(YES_NO_UNKNOWN),
    ", ".join(CONSTRAINT_MODES),
)

LOCATION_HINT_PATTERN = re.compile(
    r"\b("
    r"\d{1,4}\s+[A-Za-zÀ-ÿ0-9.'-]+(?:\s+[A-Za-zÀ-ÿ0-9.'-]+){0,3}\s+"
    r"(?:street|st\.?|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|lane|ln\.?|drive|dr\.?|"
    r"strasse|straße|platz|allee|way)"
    r"|a\d{1,3}"
    r"|u-?bahn"
    r"|s-?bahn"
    r"|station"
    r"|bahnhof"
    r"|highway"
    r"|autobahn"
    r"|exit"
    r"|junction"
    r"|intersection"
    r"|corner"
    r"|bridge"
    r"|tunnel"
    r"|near\b"
    r")\b",
    flags=re.IGNORECASE,
)


class GuidanceSignals(BaseModel):
    unconscious_or_unresponsive: bool = False
    indoor_fire_immediate: bool = False
    vehicle_fire_adjacent: bool = False
    heavy_bleeding: bool = False
    breathing_difficulty_with_smoke: bool = False
    trapped_or_cannot_exit: bool = False
    weapon_shooting_explosion: bool = False
    bleeding_any: bool = False
    injury_reported: bool = False
    children_unaccounted_for: bool = False
    child_in_danger: bool = False
    child_unreachable: bool = False
    vulnerable_adult_involved: bool = False
    situation_worsening: bool = False
    caller_cannot_leave: bool = False
    caller_alone_with_victim: bool = False
    multiple_casualties: bool = False
    caller_calm_coherent: bool = False
    situation_stable_static: bool = False
    everyone_explicitly_safe: bool = False
    suspicious_transcript_streak: int = 0
    unresolved_address_attempts: int = 0

    @field_validator("suspicious_transcript_streak", "unresolved_address_attempts", mode="before")
    @classmethod
    def _coerce_non_negative_int(cls, value: Any) -> int:
        if value in (None, ""):
            return 0
        return max(int(value), 0)


class SeedExtractedFields(BaseModel):
    location: str | None = None
    issue_type: str
    victim_breathing: str = "unknown"
    victim_bleeding: str = "unknown"
    victim_conscious: str = "unknown"
    victim_count: int | str = "unknown"
    caller_safe: str = "unknown"

    @field_validator("issue_type")
    @classmethod
    def _validate_issue_type(cls, value: str) -> str:
        if value not in GUIDANCE_ISSUE_TYPES:
            raise ValueError(f"Unsupported issue_type: {value}")
        return value

    @field_validator("victim_breathing", "victim_bleeding", "victim_conscious", "caller_safe")
    @classmethod
    def _validate_yes_no_unknown(cls, value: str) -> str:
        if value not in YES_NO_UNKNOWN:
            raise ValueError(f"Unsupported yes/no/unknown value: {value}")
        return value

    @field_validator("victim_count", mode="before")
    @classmethod
    def _coerce_victim_count(cls, value: Any) -> int | str:
        if value in (None, "", "unknown"):
            return "unknown"
        if isinstance(value, bool):
            raise ValueError("victim_count cannot be boolean")
        return max(int(value), 0)


class GuidanceSeed(BaseModel):
    raw_utterance: str = Field(min_length=8)
    extracted_fields: SeedExtractedFields
    address_in_utterance: bool = False
    scoring_signals: GuidanceSignals


class FinalExtractedFields(SeedExtractedFields):
    priority: str

    @field_validator("priority")
    @classmethod
    def _validate_priority(cls, value: str) -> str:
        if value not in GUIDANCE_PRIORITIES:
            raise ValueError(f"Unsupported priority: {value}")
        return value


class GuidanceRecord(BaseModel):
    raw_utterance: str = Field(min_length=8)
    previous_severity: int = Field(ge=0, le=100)
    extracted_fields: FinalExtractedFields
    severity_score: int = Field(ge=0, le=100)
    severity_delta: int = Field(ge=-100, le=100)
    severity_drivers: list[str]
    human_recommended: bool
    human_required: bool
    agent_should_continue: bool
    constraint_mode: str
    address_in_utterance: bool

    @field_validator("constraint_mode")
    @classmethod
    def _validate_constraint_mode(cls, value: str) -> str:
        if value not in CONSTRAINT_MODES:
            raise ValueError(f"Unsupported constraint_mode: {value}")
        return value


def has_explicit_location_cue(text: str) -> bool:
    if not text:
        return False
    return bool(LOCATION_HINT_PATTERN.search(text))


def priority_for_severity(severity_score: int) -> str:
    if severity_score >= 86:
        return "CRITICAL"
    if severity_score >= 61:
        return "HIGH"
    if severity_score >= 31:
        return "MEDIUM"
    return "LOW"


def constraint_mode_for_severity(severity_score: int) -> str:
    if severity_score >= 86:
        return "holding_pattern_only"
    if severity_score >= 61:
        return "pre_arrival_priority"
    if severity_score >= 31:
        return "single_question_only"
    return "normal"


def _iter_weighted_drivers(signals: GuidanceSignals, victim_count: int | str) -> Iterable[tuple[str, int]]:
    if signals.unconscious_or_unresponsive:
        yield ("unconscious_or_unresponsive", 40)
    if signals.indoor_fire_immediate:
        yield ("indoor_fire_immediate", 35)
    if signals.vehicle_fire_adjacent:
        yield ("vehicle_fire_adjacent", 30)
    if signals.heavy_bleeding:
        yield ("heavy_bleeding", 30)
    if signals.breathing_difficulty_with_smoke:
        yield ("breathing_difficulty_with_smoke", 25)
    if signals.trapped_or_cannot_exit:
        yield ("trapped_or_cannot_exit", 25)
    if signals.weapon_shooting_explosion:
        yield ("weapon_shooting_explosion", 20)
    if signals.bleeding_any:
        yield ("bleeding_any", 15)
    if signals.injury_reported:
        yield ("injury_reported", 10)

    if signals.children_unaccounted_for:
        yield ("children_unaccounted_for", 25)
    if signals.child_in_danger:
        yield ("child_in_danger", 20)
    if signals.child_unreachable:
        yield ("child_unreachable", 15)
    if signals.vulnerable_adult_involved:
        yield ("vulnerable_adult_involved", 10)

    if isinstance(victim_count, int) and victim_count > 1:
        extra_victims = min((victim_count - 1) * 5, 20)
        if extra_victims:
            yield ("additional_victims", extra_victims)

    if signals.situation_worsening:
        yield ("situation_worsening", 15)
    if signals.caller_cannot_leave:
        yield ("caller_cannot_leave", 15)
    if signals.caller_alone_with_victim:
        yield ("caller_alone_with_victim", 10)
    if signals.multiple_casualties:
        yield ("multiple_casualties", 10)

    if signals.suspicious_transcript_streak >= 2:
        yield ("suspicious_transcript_streak", 10)
    if signals.unresolved_address_attempts >= 3:
        yield ("unresolved_address_attempts", 15)

    if signals.caller_calm_coherent:
        yield ("caller_calm_coherent", -10)
    if signals.situation_stable_static:
        yield ("situation_stable_static", -15)
    if signals.everyone_explicitly_safe:
        yield ("everyone_explicitly_safe", -20)


def severity_from_signals(signals: GuidanceSignals, victim_count: int | str) -> tuple[int, list[str]]:
    total = 0
    drivers: list[str] = []
    for name, weight in _iter_weighted_drivers(signals, victim_count):
        total += weight
        drivers.append(name)
    return max(0, min(100, total)), drivers


def build_guidance_record(seed: GuidanceSeed, previous_severity: int) -> dict[str, Any]:
    explicit_location = bool(seed.address_in_utterance and has_explicit_location_cue(seed.raw_utterance))
    final_location = seed.extracted_fields.location if explicit_location else None

    severity_score, severity_drivers = severity_from_signals(
        seed.scoring_signals,
        seed.extracted_fields.victim_count,
    )
    severity_delta = severity_score - previous_severity
    human_recommended = severity_score >= 60 or severity_delta >= 20
    human_required = severity_score >= 85 or severity_delta >= 30

    extracted_fields = FinalExtractedFields(
        location=final_location,
        issue_type=seed.extracted_fields.issue_type,
        priority=priority_for_severity(severity_score),
        victim_breathing=seed.extracted_fields.victim_breathing,
        victim_bleeding=seed.extracted_fields.victim_bleeding,
        victim_conscious=seed.extracted_fields.victim_conscious,
        victim_count=seed.extracted_fields.victim_count,
        caller_safe=seed.extracted_fields.caller_safe,
    )

    record = GuidanceRecord(
        raw_utterance=seed.raw_utterance.strip(),
        previous_severity=int(previous_severity),
        extracted_fields=extracted_fields,
        severity_score=severity_score,
        severity_delta=severity_delta,
        severity_drivers=severity_drivers,
        human_recommended=human_recommended,
        human_required=human_required,
        agent_should_continue=True,
        constraint_mode=constraint_mode_for_severity(severity_score),
        address_in_utterance=explicit_location,
    )
    return record.model_dump()


def schema_snapshots() -> dict[str, dict[str, Any]]:
    return {
        "seed": GuidanceSeed.model_json_schema(),
        "record": GuidanceRecord.model_json_schema(),
    }


def _first_matching_span(text: str, candidates: Iterable[str]) -> tuple[str, str] | None:
    lowered = text.lower()
    for candidate in candidates:
        if not candidate:
            continue
        idx = lowered.find(candidate.lower())
        if idx >= 0:
            return text[idx : idx + len(candidate)], candidate
    return None


def build_gliner_sidecar_record(record: dict[str, Any]) -> dict[str, Any] | None:
    text = record["raw_utterance"]
    fields = record["extracted_fields"]
    entities: list[list[str]] = []

    location = fields.get("location")
    if location and location.lower() in text.lower():
        entities.append([location, "LOCATION"])

    phrase_hints = {
        "ISSUE_CUE": [
            "won't start",
            "flat tire",
            "flat tyre",
            "smoke",
            "fire",
            "burning",
            "not breathing",
            "unresponsive",
            "collapsed",
            "gun",
            "knife",
            "gas leak",
            "chemical smell",
            "trapped",
            "can't get out",
            "kid is missing",
            "child is missing",
        ],
        "BREATHING_STATUS": ["breathing", "not breathing", "can't breathe", "wheezing", "unresponsive"],
        "BLEEDING_STATUS": ["bleeding", "bleeding heavily", "heavy bleeding", "blood everywhere"],
        "CONSCIOUSNESS_STATUS": ["awake", "conscious", "unconscious", "passed out", "not responding"],
        "CALLER_SAFETY": ["i am safe", "we are safe", "unsafe", "i am trapped", "i'm trapped"],
        "SEVERITY_DRIVER": [
            "smoke",
            "fire",
            "not breathing",
            "unresponsive",
            "trapped",
            "bleeding",
            "gun",
            "explosion",
            "child",
        ],
    }
    for label, hints in phrase_hints.items():
        match = _first_matching_span(text, hints)
        if match:
            entities.append([match[0], label])

    victim_count = fields.get("victim_count")
    if isinstance(victim_count, int):
        count_match = re.search(rf"\b{victim_count}\b|\b(one|two|three|four|five)\b", text, flags=re.IGNORECASE)
        if count_match:
            entities.append([count_match.group(0), "VICTIM_COUNT"])

    if not entities:
        return None
    return {"text": text, "entities": entities}
