"""Validated operator-facing assessment assembled from persisted session evidence.

This module does not call a model or geocoder. It presents the existing
triage/ledger/location/dispatch outputs with their evidence source and limits.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvidenceFact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    value: str
    evidence_status: Literal["caller_reported", "rule_derived", "unknown", "unverified", "ambiguous"]
    source_utterance: str | None = None
    source_at: str | None = None
    conflicting_reports: list[dict[str, str | None]] = Field(default_factory=list)


class IncidentLocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate: str | None = None
    status: Literal["unresolved", "needs_confirmation", "approximate", "resolved"]
    confidence: str | None = None
    source: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    caller_confirmed: bool = False
    clarification: str | None = None

    @model_validator(mode="after")
    def validate_coordinate_pair(self) -> "IncidentLocation":
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("location coordinates must be present as a pair")
        if self.latitude is not None and not (-90 <= self.latitude <= 90 and -180 <= self.longitude <= 180):
            raise ValueError("location coordinates are outside valid ranges")
        return self


class EmergencyAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    incident_id: str
    generated_at: str
    status: str
    extraction_status: Literal["structured", "partial", "awaiting_speech"]
    extraction_source: str
    incident_type: EvidenceFact
    urgency_recommendation: EvidenceFact
    facts: list[EvidenceFact] = Field(default_factory=list)
    symptoms: list[EvidenceFact] = Field(default_factory=list)
    location: IncidentLocation
    missing_information: list[str] = Field(default_factory=list)
    next_questions: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    first_aid: dict[str, Any] | None = None
    dispatch: dict[str, Any]
    human_review_required: bool = True
    natural_language_classification: dict[str, Any] | None = None
    complaint_code_ml: dict[str, Any] | None = None


_ISSUE_CUE_LABELS = {
    "vehicle": "vehicle involved",
    "collision": "road collision",
    "injury": "injury",
    "bleeding": "bleeding",
    "breathing_problem": "breathing difficulty",
    "unconscious": "unconscious or unresponsive person",
    "smoke": "smoke",
    "fire": "fire or burning",
    "trapped": "trapped person",
    "weapon": "weapon or threat",
    "explosion": "explosion",
    "gas_smell": "gas or fumes",
    "child_present": "child present",
}
_CUE_PATTERNS = {
    "vehicle": r"\b(?:car|vehicle|truck|van|motorcycle|bus)\b",
    "collision": r"\b(?:crash|accident|collision|hit by a car|vehicle hit)\b",
    "injury": r"\b(?:injur(?:y|ed)|hurt|fracture|broken bone)\b",
    "bleeding": r"\b(?:bleed(?:ing)?|blood|hemorrhage)\b",
    "breathing_problem": r"\b(?:can't breathe|cannot breathe|trouble breathing|difficulty breathing|breathing problem|choking|asthma)\b",
    "unconscious": r"\b(?:unconscious|unresponsive|not responding|collapsed|passed out)\b",
    "smoke": r"\bsmoke\b",
    "fire": r"\b(?:fire|burn(?:ing|ed)?)\b",
    "trapped": r"\b(?:trapped|stuck|can't get out|cannot get out)\b",
    "weapon": r"\b(?:weapon|gun|knife|armed|hostage|shooting)\b",
    "explosion": r"\b(?:explosion|exploded|blast)\b",
    "gas_smell": r"\b(?:gas leak|gas smell|fumes)\b",
    "child_present": r"\b(?:child|kid|baby|infant|daughter|son)\b",
}
_FACT_PATTERNS = {
    "breathing_status": r"\b(?:breath(?:ing)?|gasping|choking|can't breathe|cannot breathe)\b",
    "consciousness_status": r"\b(?:unconscious|unresponsive|not responding|awake|conscious|responsive|collapsed)\b",
    "bleeding_status": r"\b(?:bleed(?:ing)?|blood|hemorrhage)\b",
    "trapped_status": r"\b(?:trapped|stuck|can't get out|cannot get out|not trapped)\b",
    "victim_count": r"\b(?:\d+|one|two|three|four|five|six|several|multiple)\s+(?:people|persons|victims|casualties|patients|injured)\b|\b(?:multiple|several) casualties\b",
}
_BINARY_STATEMENT_PATTERNS = {
    "breathing_status": (r"\b(?:not breathing|no normal breathing|only gasping|can't breathe|cannot breathe)\b", r"\b(?:breathing normally|breathing okay|breathing properly|can breathe|able to breathe)\b"),
    "consciousness_status": (r"\b(?:unconscious|unresponsive|not responding|can't wake|cannot wake|passed out)\b", r"\b(?:awake|conscious|responsive|responding)\b"),
    "bleeding_status": (r"\b(?:no|not)\s+(?:any\s+)?bleeding\b|\bno blood\b", r"\b(?:bleeding|blood|hemorrhage)\b"),
    "trapped_status": (r"\bnot trapped\b|\bcan(?:not|'t) get out\b", r"\b(?:trapped|stuck)\b"),
}


def _turn_source(turns: list[dict[str, Any]], pattern: str) -> tuple[str | None, str | None]:
    compiled = re.compile(pattern, re.IGNORECASE)
    for turn in reversed(turns):
        text = str((turn or {}).get("text") or "")
        if compiled.search(text):
            return text, str(turn.get("at") or "") or None
    return None, None


def _fact(field: str, value: Any, turns: list[dict[str, Any]], pattern: str | None = None) -> EvidenceFact:
    if value in (None, "", "unknown", "none", []):
        return EvidenceFact(field=field, value="Unknown", evidence_status="unknown")
    source, at = _turn_source(turns, pattern or r"(?!)")
    return EvidenceFact(
        field=field,
        value=str(value),
        evidence_status="caller_reported" if source else "unverified",
        source_utterance=source,
        source_at=at,
    )


def _explicit_binary_value(field: str, text: str) -> str | None:
    negative_pattern, positive_pattern = _BINARY_STATEMENT_PATTERNS[field]
    negatives = list(re.finditer(negative_pattern, text, flags=re.IGNORECASE))
    matches = [(match.start(), "no") for match in negatives]
    for match in re.finditer(positive_pattern, text, flags=re.IGNORECASE):
        if any(match.start() < negative.end() and match.end() > negative.start() for negative in negatives):
            continue
        matches.append((match.start(), "yes"))
    if not matches:
        return None
    return max(matches, key=lambda item: item[0])[1]


def _binary_fact(key: str, label: str, value: Any, turns: list[dict[str, Any]]) -> EvidenceFact:
    reports = []
    for turn in turns:
        text = str((turn or {}).get("text") or "")
        reported = _explicit_binary_value(key, text)
        if reported is not None:
            reports.append({"value": reported, "source_utterance": text, "source_at": str(turn.get("at") or "") or None})
    distinct = {item["value"] for item in reports}
    result = _fact(label, value, turns, _FACT_PATTERNS[key])
    if len(distinct) > 1:
        result.evidence_status = "ambiguous"
        result.conflicting_reports = reports
    return result


def build_incident_assessment(
    session: dict[str, Any],
    *,
    first_aid: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = session.get("client") or {}
    turns = client.get("transcripts", {}).get("user", []) or []
    ledger = session.get("shared_ledger") or {}
    hard_facts = ledger.get("hard_facts") or {}
    triage = session.get("merged_triage_state") or {}
    priority = ledger.get("priority") or {}
    gate = ledger.get("location_gate") or {}
    tool_calls = session.get("tool_calls") or []

    cues = [str(item) for item in (hard_facts.get("issue_cues") or []) if item in _ISSUE_CUE_LABELS]
    symptoms = []
    if hard_facts.get("bleeding_status") == "no":
        cues = [cue for cue in cues if cue != "bleeding"]
    if hard_facts.get("consciousness_status") == "yes":
        cues = [cue for cue in cues if cue != "unconscious"]
    if hard_facts.get("trapped_status") == "no":
        cues = [cue for cue in cues if cue != "trapped"]
    for cue in cues:
        source, at = _turn_source(turns, _CUE_PATTERNS[cue])
        symptoms.append(EvidenceFact(
            field="caller-reported concern",
            value=_ISSUE_CUE_LABELS[cue],
            evidence_status="caller_reported" if source else "unverified",
            source_utterance=source,
            source_at=at,
        ))

    breathing = hard_facts.get("breathing_status", triage.get("victim_breathing", "unknown"))
    consciousness = hard_facts.get("consciousness_status", triage.get("victim_conscious", "unknown"))
    bleeding = hard_facts.get("bleeding_status", triage.get("victim_bleeding", "unknown"))
    victim_count = hard_facts.get("victim_count_confirmed", triage.get("victim_count", "unknown"))
    facts = [
        _binary_fact("breathing_status", "Breathing status", breathing, turns),
        _binary_fact("consciousness_status", "Consciousness status", consciousness, turns),
        _binary_fact("bleeding_status", "Bleeding status", bleeding, turns),
        _binary_fact("trapped_status", "Trapped status", hard_facts.get("trapped_status", "unknown"), turns),
        _fact("Number of injured people", victim_count, turns, _FACT_PATTERNS["victim_count"]),
    ]

    location_call = next((item for item in reversed(tool_calls) if item.get("name") in {"lookup_address", "validate_address"}), None)
    location_result = (location_call or {}).get("result") or {}
    nearby_call = next((item for item in reversed(tool_calls) if item.get("name") == "nearby_context"), None)
    nearby = (nearby_call or {}).get("result") or {}
    lat = nearby.get("lat") if nearby else location_result.get("lat")
    lon = nearby.get("lon") if nearby else location_result.get("lon")
    try:
        lat_num = float(lat) if lat is not None else None
        lon_num = float(lon) if lon is not None else None
        valid_pair = lat_num is not None and lon_num is not None and -90 <= lat_num <= 90 and -180 <= lon_num <= 180
    except (TypeError, ValueError):
        lat_num = lon_num = None
        valid_pair = False
    if not valid_pair:
        lat_num = lon_num = None
    candidate = (
        hard_facts.get("location_candidate")
        or gate.get("candidate_text")
        or location_result.get("normalized_address")
        or nearby.get("normalized_address")
    )
    confirmed = bool(gate.get("confirmed"))
    confidence = location_result.get("confidence") or nearby.get("confidence")
    if not valid_pair:
        location_status = "needs_confirmation" if candidate and gate.get("needs_confirmation") else "unresolved"
    elif confidence == "low":
        location_status = "approximate"
    else:
        location_status = "resolved" if confirmed else "needs_confirmation"
    source_text, source_at = _turn_source(turns, re.escape(str(candidate))) if candidate else (None, None)
    location = IncidentLocation(
        candidate=str(candidate) if candidate else None,
        status=location_status,
        confidence=str(confidence) if confidence else None,
        source=str(location_result.get("source") or nearby.get("source") or "caller transcript") if candidate else None,
        latitude=lat_num,
        longitude=lon_num,
        caller_confirmed=confirmed,
        clarification=(priority.get("location_followup_prompt") or "Please provide a city, nearby landmark, road, or exact address.") if location_status in {"unresolved", "needs_confirmation", "approximate"} else None,
    )
    if source_text:
        facts.append(EvidenceFact(field="Incident location", value=str(candidate), evidence_status="caller_reported", source_utterance=source_text, source_at=source_at))

    issue_type = triage.get("issue_type") or (" · ".join(_ISSUE_CUE_LABELS[cue] for cue in cues) if cues else None)
    incident_type = _fact("Incident type", issue_type, turns, r"\b(?:accident|crash|collision|cardiac|chest pain|breath(?:ing)?|unconscious|bleed(?:ing)?|burn|fall|poison|fire|smoke)\b")
    priority_value = triage.get("priority") or "Unknown"
    urgency = EvidenceFact(
        field="Advisory urgency recommendation",
        value=str(priority_value),
        evidence_status="rule_derived" if priority_value not in (None, "", "unknown", "Unknown") else "unknown",
    )
    conflict_fields = [fact.field for fact in facts if fact.evidence_status == "ambiguous"]
    missing = list(dict.fromkeys(list(priority.get("missing_fields") or []) + list(session.get("triage_meta", {}).get("missing_or_unknown_fields") or []) + [f"confirm {field.lower()} due to conflicting reports" for field in conflict_fields]))
    questions = []
    next_goal = priority.get("next_question_goal")
    if next_goal:
        questions.append(str(next_goal))
    questions.extend(f"Please confirm {field.lower()}; caller statements conflict." for field in conflict_fields)
    if location.clarification:
        questions.append(location.clarification)
    services_state = session.get("dispatch_services") or {}
    dispatch = {
        "simulation_only": True,
        "status": str(services_state.get("summary") or "No simulated service plan yet."),
        "services": services_state.get("services") or {},
        "human_confirmation_required": True,
        "human_confirmation_status": "not recorded",
        "notice": "No real emergency service is contacted by this application.",
    }
    if not turns:
        extraction_status = "awaiting_speech"
    elif issue_type or cues or candidate:
        extraction_status = "structured"
    else:
        extraction_status = "partial"
    source = "shared fact ledger" if ledger.get("provenance", {}).get("last_slm_run_at") else "local rules and validated session state"
    rationale = []
    if triage.get("escalation_reason"):
        rationale.append(f"Escalation rationale: {triage['escalation_reason']}")
    if triage.get("severity_drivers"):
        rationale.extend(str(item) for item in triage["severity_drivers"])
    if not rationale:
        rationale.append("No rule-based escalation rationale is available yet.")

    assessment = EmergencyAssessment(
        session_id=str(session.get("session_id") or "unknown"),
        incident_id=str(session.get("session_id") or "unknown"),
        generated_at=datetime.now(UTC).isoformat(),
        status=str(client.get("status") or (session.get("server") or {}).get("status") or "active"),
        extraction_status=extraction_status,
        extraction_source=source,
        incident_type=incident_type,
        urgency_recommendation=urgency,
        facts=facts,
        symptoms=symptoms,
        location=location,
        missing_information=missing,
        next_questions=list(dict.fromkeys(questions)),
        rationale=rationale,
        first_aid=first_aid,
        dispatch=dispatch,
        human_review_required=True,
        natural_language_classification=session.get("natural_language_classification"),
        complaint_code_ml=session.get("live_complaint_classification"),
    )
    return assessment.model_dump(mode="json")
