from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


FACT_LEDGER_SCHEMA_VERSION = "2026-04-26.v1"
FACT_YES_NO_UNKNOWN = ("yes", "no", "unknown")
FACT_LOCATION_KINDS = (
    "exact_address",
    "road_or_junction",
    "place_name",
    "sub_location_only",
    "vague",
    "none",
)
FACT_ISSUE_CUES = (
    "vehicle",
    "collision",
    "injury",
    "bleeding",
    "breathing_problem",
    "unconscious",
    "smoke",
    "fire",
    "trapped",
    "weapon",
    "explosion",
    "gas_smell",
    "child_present",
)
FACT_PRIORITY_FIELDS = (
    "issue_cues",
    "location_candidate",
    "victim_count",
    "breathing_status",
    "bleeding_status",
    "consciousness_status",
    "trapped_status",
)
FACT_SHARED_BINARY_FIELDS = (
    "child_present",
    "bleeding_status",
    "breathing_status",
    "consciousness_status",
    "trapped_status",
)
FACT_GOAL_FIELD_MAP = {
    "clarify the emergency": "issue_cues",
    "pinpoint where they are": "location_candidate",
    "confirm how many people are involved": "victim_count",
    "confirm whether a child is involved": "child_present",
    "confirm bleeding": "bleeding_status",
    "confirm breathing": "breathing_status",
    "confirm consciousness": "consciousness_status",
    "confirm whether anyone is trapped": "trapped_status",
}
_TRIVIAL_TURN_PATTERN = re.compile(
    r"^(?:yes|no|okay|ok|hello|hi|yeah|yep|nope|correct|right|sure|mm+|uh+|um+|hmm+|sorry)\W*$",
    flags=re.IGNORECASE,
)
_SEMANTIC_CORRECTION_PATTERN = re.compile(
    r"\b(?:actually|sorry|no\b(?!\s*$)|not\b(?!\s*$)|wait|correction)\b",
    flags=re.IGNORECASE,
)
_HIGH_SIGNAL_PATTERN = re.compile(
    r"\b(?:gun|weapon|bleeding|blood|not breathing|can't breathe|cannot breathe|trapped|stuck|smoke|fire)\b",
    flags=re.IGNORECASE,
)

FACT_LEDGER_SYSTEM_PROMPT = """You update a strict evidence-only fact ledger for a live emergency dispatch assistant.
Return JSON only.
Do not include markdown.

Use exactly this top-level shape:
{
  "merged_facts": {
    "location_candidate": "string or null",
    "location_kind": "one of exact_address, road_or_junction, place_name, sub_location_only, vague, none",
    "address_in_utterance": true,
    "location_note": "string or null",
    "sub_location": "string or null",
    "inside_building": "yes|no|unknown",
    "issue_cues": ["zero or more from vehicle, collision, injury, bleeding, breathing_problem, unconscious, smoke, fire, trapped, weapon, explosion, gas_smell, child_present"],
    "victim_count": "integer or unknown",
    "child_present": "yes|no|unknown",
    "bleeding_status": "yes|no|unknown",
    "breathing_status": "yes|no|unknown",
    "consciousness_status": "yes|no|unknown",
    "trapped_status": "yes|no|unknown"
  }
}

Rules:
- Use only hard facts directly stated in caller_turn or already present in prior_facts.
- Carry forward prior_facts unless caller_turn explicitly changes them.
- Do not infer caller safety, severity, threat level, dispatch priority, or escalation.
- Do not invent a location.
- address_in_utterance may be true only if caller_turn itself contains a dispatch-usable location clue.
- location_candidate may stay populated from prior_facts even when address_in_utterance is false.
- location_note may capture a short literal landmark or relative-location clue from caller_turn even when location_candidate is null.
- Generic phrases like "at home", "inside the building", "somewhere in Berlin", and "in the middle of the street" are not dispatch-usable locations.
- location_kind=sub_location_only is for floor, unit, stairwell, entrance, room, or similar sub-location without a building/address clue in this turn.
- issue_cues must be explicit, not inferred.
- If caller_turn is just a confirmation like "yes" or "that's correct", keep facts unless the turn explicitly changes them.
- If caller_turn is a negation like "no bleeding" or "not trapped", update only the directly negated field.
"""

_GENERIC_LOCATION_PATTERN = re.compile(
    r"\b("
    r"at home|home|my house|my apartment|inside the building|inside the house|inside my house|inside my apartment|"
    r"somewhere in berlin|berlin|middle of the street|in the middle of the street|on the street|in the street|"
    r"inside|upstairs|downstairs|third floor|second floor|first floor"
    r")\b",
    flags=re.IGNORECASE,
)
_ADDRESS_PATTERN = re.compile(
    r"\d{1,4}\s+[A-Za-zÀ-ÿ0-9.'-]+(?:\s+[A-Za-zÀ-ÿ0-9.'-]+){0,3}\s+"
    r"(?:street|st\.?|road|rd\.?|avenue|ave\.?|boulevard|blvd\.?|lane|ln\.?|drive|dr\.?|"
    r"strasse|straße|platz|allee|way)\b",
    flags=re.IGNORECASE,
)
_ADDRESS_PATTERN_STREET_FIRST = re.compile(
    r"[A-Za-zÀ-ÿ0-9.'-]+(?:\s+[A-Za-zÀ-ÿ0-9.'-]+){0,3}\s+\d{1,4}\b",
    flags=re.IGNORECASE,
)
_STREET_NAME_PATTERN = re.compile(
    r"\b(?:[A-Za-zÀ-ÿ0-9.'-]+\s+){0,2}[A-Za-zÀ-ÿ0-9.'-]*(?:straße|strasse|allee|platz|ring|ufer|damm)\b",
    flags=re.IGNORECASE,
)
_ROAD_OR_JUNCTION_PATTERN = re.compile(
    r"\b(?:A|B|U|S)\s?-?\d{1,4}\b|\b(?:junction|intersection|corner|exit|bridge|tunnel|station|bahnhof)\b",
    flags=re.IGNORECASE,
)
_PLACE_NAME_PATTERN = re.compile(
    r"\b(?:"
    r"delta campus|zoo station|ostkreuz|alexanderplatz|berlin hauptbahnhof|"
    r"hermannplatz|tempelhofer feld|charite campus mitte|mall of berlin|"
    r"neukoelln arcaden|gesundbrunnen station|suedkreuz station|"
    r"(?:[A-Za-zÀ-ÿ0-9.'-]+\s+){1,3}(?:campus|station|bahnhof|hospital|clinic|park|mall|arcaden|center|zentrum|university|schule|school)"
    r")\b",
    flags=re.IGNORECASE,
)
_SUB_LOCATION_PATTERN = re.compile(
    r"\b(?:floor|unit|apartment|flat|room|stairwell|entrance|wing|level)\b",
    flags=re.IGNORECASE,
)
_BARE_STREET_SUFFIXES = {"straße", "strasse", "platz", "allee", "ring", "ufer", "damm"}
_BARE_PLACE_DESCRIPTORS = {
    "campus",
    "station",
    "bahnhof",
    "hospital",
    "clinic",
    "park",
    "mall",
    "center",
    "zentrum",
    "school",
    "university",
}


def _meaningful_street_name(text: str) -> bool:
    match = _STREET_NAME_PATTERN.search(text or "")
    if not match:
        return False
    token = match.group(0).strip(" ,.!?").casefold()
    return token not in _BARE_STREET_SUFFIXES


def _meaningful_place_name(text: str) -> bool:
    match = _PLACE_NAME_PATTERN.search(text or "")
    if not match:
        return False
    token = match.group(0).strip(" ,.!?").casefold()
    return token not in _BARE_PLACE_DESCRIPTORS


class FactLedger(BaseModel):
    location_candidate: str | None = None
    location_kind: str = "none"
    address_in_utterance: bool = False
    location_note: str | None = None
    sub_location: str | None = None
    inside_building: str = "unknown"
    issue_cues: list[str] = []
    victim_count: int | str = "unknown"
    child_present: str = "unknown"
    bleeding_status: str = "unknown"
    breathing_status: str = "unknown"
    consciousness_status: str = "unknown"
    trapped_status: str = "unknown"

    @field_validator("location_kind")
    @classmethod
    def _validate_location_kind(cls, value: str) -> str:
        if value not in FACT_LOCATION_KINDS:
            raise ValueError(f"Unsupported location_kind: {value}")
        return value

    @field_validator(
        "inside_building",
        "child_present",
        "bleeding_status",
        "breathing_status",
        "consciousness_status",
        "trapped_status",
    )
    @classmethod
    def _validate_yes_no_unknown(cls, value: str) -> str:
        if value not in FACT_YES_NO_UNKNOWN:
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

    @field_validator("issue_cues")
    @classmethod
    def _validate_issue_cues(cls, value: list[str]) -> list[str]:
        cleaned = [str(item) for item in value or []]
        unknown = [item for item in cleaned if item not in FACT_ISSUE_CUES]
        if unknown:
            raise ValueError(f"Unsupported issue_cues: {', '.join(unknown)}")
        return list(dict.fromkeys(cleaned))

    @field_validator("location_candidate")
    @classmethod
    def _clean_location_candidate(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = re.sub(r"\s+", " ", str(value)).strip()
        return cleaned or None

    @field_validator("sub_location")
    @classmethod
    def _clean_sub_location(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = re.sub(r"\s+", " ", str(value)).strip()
        return cleaned or None

    @field_validator("location_note")
    @classmethod
    def _clean_location_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = re.sub(r"\s+", " ", str(value)).strip()
        return cleaned or None

    def model_post_init(self, __context: Any) -> None:
        if self.location_kind == "none":
            self.location_candidate = None
        if self.location_kind == "vague":
            self.location_candidate = None
            self.address_in_utterance = False
        if self.location_kind == "sub_location_only" and not self.sub_location:
            raise ValueError("sub_location_only requires sub_location")
        if self.location_kind in {"none", "vague"} and self.address_in_utterance:
            raise ValueError("address_in_utterance cannot be true for none/vague location kinds")


class FactLedgerTrainingExample(BaseModel):
    caller_turn: str = Field(min_length=1, max_length=280)
    prior_facts: FactLedger
    merged_facts: FactLedger


def has_dispatchable_location_cue(text: str) -> bool:
    if not text:
        return False
    has_specific = bool(
        _ADDRESS_PATTERN.search(text)
        or _ADDRESS_PATTERN_STREET_FIRST.search(text)
        or _meaningful_street_name(text)
        or _ROAD_OR_JUNCTION_PATTERN.search(text)
        or _meaningful_place_name(text)
    )
    if has_specific:
        return True
    if _GENERIC_LOCATION_PATTERN.search(text):
        return False
    return False


def classify_location_kind(text: str) -> str:
    if not text:
        return "none"
    if _ADDRESS_PATTERN.search(text):
        return "exact_address"
    if _ADDRESS_PATTERN_STREET_FIRST.search(text):
        return "exact_address"
    if _meaningful_street_name(text):
        return "road_or_junction"
    if _ROAD_OR_JUNCTION_PATTERN.search(text):
        return "road_or_junction"
    if _meaningful_place_name(text):
        return "place_name"
    if _SUB_LOCATION_PATTERN.search(text):
        return "sub_location_only"
    if _GENERIC_LOCATION_PATTERN.search(text):
        return "vague"
    return "none"


def build_fact_delta(prior_facts: FactLedger, merged_facts: FactLedger) -> dict[str, Any]:
    prior = prior_facts.model_dump()
    merged = merged_facts.model_dump()
    delta: dict[str, Any] = {}
    for key, value in merged.items():
        if value != prior.get(key):
            delta[key] = value
    return delta


def build_fact_priority_packet(prior_facts: FactLedger, merged_facts: FactLedger) -> dict[str, Any]:
    delta = build_fact_delta(prior_facts, merged_facts)
    missing_fields: list[str] = []
    if not merged_facts.issue_cues:
        missing_fields.append("issue_cues")
    if merged_facts.location_candidate is None or merged_facts.location_kind in {"none", "vague"}:
        missing_fields.append("location_candidate")
    if merged_facts.victim_count == "unknown":
        missing_fields.append("victim_count")
    if merged_facts.breathing_status == "unknown":
        missing_fields.append("breathing_status")
    if merged_facts.bleeding_status == "unknown":
        missing_fields.append("bleeding_status")
    if merged_facts.consciousness_status == "unknown":
        missing_fields.append("consciousness_status")
    if merged_facts.trapped_status == "unknown":
        missing_fields.append("trapped_status")
    next_question_field = missing_fields[0] if missing_fields else None
    next_question_goal = {
        "issue_cues": "clarify the emergency",
        "location_candidate": "pinpoint where they are",
        "victim_count": "confirm how many people are involved",
        "child_present": "confirm whether a child is involved",
        "bleeding_status": "confirm bleeding",
        "breathing_status": "confirm breathing",
        "consciousness_status": "confirm consciousness",
        "trapped_status": "confirm whether anyone is trapped",
    }.get(next_question_field)
    question_style = "yes_no" if next_question_field in FACT_SHARED_BINARY_FIELDS else "short_open"
    return {
        "facts_delta": delta,
        "priority_flags": {
            "missing_fields": missing_fields,
            "next_question_field": next_question_field,
            "next_question_goal": next_question_goal,
            "question_style": question_style,
            "location_search_allowed": bool(
                merged_facts.address_in_utterance
                and merged_facts.location_kind in {"exact_address", "road_or_junction", "place_name"}
                and merged_facts.location_candidate
            ),
            "best_effort_location_note": merged_facts.location_note,
            "inside_building": merged_facts.inside_building,
            "has_sub_location": bool(merged_facts.sub_location),
            "move_on_allowed": bool(
                merged_facts.location_candidate or merged_facts.location_note or merged_facts.sub_location
            ),
        },
    }


def build_pioneer_fact_ledger_record(example: dict[str, Any]) -> dict[str, Any]:
    payload = FactLedgerTrainingExample.model_validate(example)
    return {
        "schema_version": FACT_LEDGER_SCHEMA_VERSION,
        "messages": [
            {"role": "system", "content": FACT_LEDGER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "caller_turn": payload.caller_turn,
                        "prior_facts": payload.prior_facts.model_dump(),
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "merged_facts": payload.merged_facts.model_dump(),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }


def empty_shared_ledger() -> dict[str, Any]:
    return {
        "version": 0,
        "hard_facts": {
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
        },
        "soft_state": {
            "people_count_best_guess": None,
            "caller_role": "unknown",
            "notes": [],
        },
        "priority": {
            "missing_fields": [],
            "next_question_field": None,
            "next_question_goal": None,
            "question_style": "short_open",
            "move_on_allowed": False,
            "location_followup_kind": None,
            "location_followup_prompt": None,
            "location_dead_end": False,
            "location_attempts": 0,
            "location_lock_active": False,
        },
        "location_gate": {
            "search_allowed": False,
            "candidate_only": False,
            "confirmed": False,
            "search_reason": None,
            "candidate_text": None,
            "candidate_kind": None,
            "needs_confirmation": False,
        },
        "provenance": {
            "last_slm_turn_index": 0,
            "last_runtime_turn_index": 0,
            "last_slm_run_at": None,
            "slm_job_state": "idle",
        },
    }


def shared_hard_facts_to_fact_ledger(hard_facts: dict[str, Any] | None) -> FactLedger:
    payload = dict(hard_facts or {})
    payload["victim_count"] = payload.pop("victim_count_confirmed", "unknown")
    return FactLedger.model_validate(payload)


def fact_ledger_to_shared_hard_facts(facts: FactLedger) -> dict[str, Any]:
    payload = facts.model_dump()
    payload["victim_count_confirmed"] = payload.pop("victim_count")
    return payload


def normalize_shared_ledger(payload: dict[str, Any] | None) -> dict[str, Any]:
    normalized = empty_shared_ledger()
    if not isinstance(payload, dict):
        return normalized
    hard_facts = payload.get("hard_facts") or {}
    soft_state = payload.get("soft_state") or {}
    priority = payload.get("priority") or {}
    location_gate = payload.get("location_gate") or {}
    provenance = payload.get("provenance") or {}
    normalized["version"] = int(payload.get("version") or 0)
    normalized["hard_facts"] = fact_ledger_to_shared_hard_facts(shared_hard_facts_to_fact_ledger(hard_facts))
    normalized["soft_state"]["people_count_best_guess"] = soft_state.get("people_count_best_guess")
    normalized["soft_state"]["caller_role"] = str(soft_state.get("caller_role") or "unknown")
    normalized["soft_state"]["notes"] = [
        re.sub(r"\s+", " ", str(item)).strip()
        for item in list(soft_state.get("notes") or [])
        if re.sub(r"\s+", " ", str(item)).strip()
    ]
    normalized["priority"] = {
        **normalized["priority"],
        "missing_fields": list(priority.get("missing_fields") or []),
        "next_question_field": priority.get("next_question_field"),
        "next_question_goal": priority.get("next_question_goal"),
        "question_style": priority.get("question_style") or "short_open",
        "move_on_allowed": bool(priority.get("move_on_allowed")),
        "location_followup_kind": priority.get("location_followup_kind"),
        "location_followup_prompt": re.sub(r"\s+", " ", str(priority.get("location_followup_prompt") or "")).strip() or None,
        "location_dead_end": bool(priority.get("location_dead_end")),
        "location_attempts": int(priority.get("location_attempts") or 0),
        "location_lock_active": bool(priority.get("location_lock_active")),
    }
    normalized["location_gate"] = {
        "search_allowed": bool(location_gate.get("search_allowed")),
        "candidate_only": bool(location_gate.get("candidate_only")),
        "confirmed": bool(location_gate.get("confirmed")),
        "search_reason": location_gate.get("search_reason"),
        "candidate_text": re.sub(r"\s+", " ", str(location_gate.get("candidate_text") or "")).strip() or None,
        "candidate_kind": location_gate.get("candidate_kind"),
        "needs_confirmation": bool(location_gate.get("needs_confirmation")),
    }
    normalized["provenance"] = {
        "last_slm_turn_index": int(provenance.get("last_slm_turn_index") or 0),
        "last_runtime_turn_index": int(provenance.get("last_runtime_turn_index") or 0),
        "last_slm_run_at": provenance.get("last_slm_run_at"),
        "slm_job_state": provenance.get("slm_job_state") or "idle",
    }
    return normalized


def build_ledger_prompt_context(shared_ledger: dict[str, Any]) -> dict[str, Any]:
    ledger = normalize_shared_ledger(shared_ledger)
    hard_facts = ledger["hard_facts"]
    priority = ledger["priority"]
    location_gate = ledger["location_gate"]
    known_location = location_gate.get("candidate_text") or hard_facts.get("location_candidate")
    known_sub_location = hard_facts.get("sub_location")
    blocked_actions = [
        "do_not_repeat_resolved_facts",
        "do_not_geocode_vague_location",
    ]
    if priority.get("location_followup_prompt") and not location_gate.get("search_allowed"):
        blocked_actions.append("do_not_drop_location_until_meaningful_or_dead_end")
    if priority.get("location_lock_active"):
        blocked_actions.append("do_not_switch_away_from_location_until_meaningful_or_dead_end")
    return {
        "version": ledger["version"],
        "known": {
            "issue_cues": list(hard_facts.get("issue_cues") or []),
            "location_candidate": known_location,
            "sub_location": known_sub_location,
            "victim_count_confirmed": hard_facts.get("victim_count_confirmed"),
        },
        "missing_fields": list(priority.get("missing_fields") or []),
        "next_question_field": priority.get("next_question_field"),
        "next_question_goal": priority.get("next_question_goal"),
        "question_style": priority.get("question_style") or "short_open",
        "location_search_allowed": bool(location_gate.get("search_allowed")),
        "location_followup_kind": priority.get("location_followup_kind"),
        "location_followup_prompt": priority.get("location_followup_prompt"),
        "location_dead_end": bool(priority.get("location_dead_end")),
        "location_attempts": int(priority.get("location_attempts") or 0),
        "location_lock_active": bool(priority.get("location_lock_active")),
        "candidate_needs_confirmation": bool(location_gate.get("needs_confirmation")),
        "must_say_next": priority.get("location_followup_prompt")
        if bool(location_gate.get("needs_confirmation")) and priority.get("location_followup_prompt")
        else None,
        "blocked_actions": blocked_actions,
    }


def is_trivial_caller_turn(text: str | None) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return True
    if len(normalized) <= 2:
        return True
    return bool(_TRIVIAL_TURN_PATTERN.fullmatch(normalized))


def is_immediate_fact_ledger_trigger(text: str | None) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized or is_trivial_caller_turn(normalized):
        return False
    if has_dispatchable_location_cue(normalized):
        return True
    if _SEMANTIC_CORRECTION_PATTERN.search(normalized):
        return True
    if _HIGH_SIGNAL_PATTERN.search(normalized):
        return True
    return False


def latest_substantive_caller_window(turns: list[dict[str, Any]] | None, *, limit: int = 3) -> tuple[str, int]:
    kept: list[str] = []
    max_index = 0
    for index, turn in enumerate(turns or [], start=1):
        text = re.sub(r"\s+", " ", str((turn or {}).get("text") or "")).strip()
        if not text or is_trivial_caller_turn(text):
            continue
        kept.append(text)
        max_index = index
    if not kept:
        return "", max_index
    return " ".join(kept[-limit:]), max_index
