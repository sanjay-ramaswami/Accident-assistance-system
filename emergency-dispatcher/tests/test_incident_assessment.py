from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from emergency_dispatcher import server, session_store
from emergency_dispatcher.first_aid import analyze_first_aid
from emergency_dispatcher.incident_assessment import IncidentLocation, build_incident_assessment
from emergency_dispatcher.server import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_session_storage(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")


def test_assessment_keeps_caller_sources_unknown_location_and_first_aid_provenance():
    source = "One person is unconscious and not breathing."
    session = {
        "session_id": "assessment-test",
        "client": {"status": "live", "transcripts": {"user": [
            {"at": "2026-10-29T10:00:00Z", "text": "There was a road crash."},
            {"at": "2026-10-29T10:00:02Z", "text": source},
        ]}},
        "shared_ledger": {
            "hard_facts": {
                "location_candidate": None, "issue_cues": ["collision", "unconscious", "breathing_problem"],
                "breathing_status": "no", "consciousness_status": "no", "bleeding_status": "unknown",
                "trapped_status": "unknown", "victim_count_confirmed": "unknown",
            },
            "priority": {"missing_fields": ["location_candidate"], "next_question_goal": "pinpoint where they are"},
            "location_gate": {},
            "provenance": {},
        },
        "merged_triage_state": {"issue_type": "PERSON_UNCONSCIOUS_OR_UNRESPONSIVE", "priority": "CRITICAL"},
        "tool_calls": [],
        "dispatch_services": {},
        "natural_language_classification": {
            "classifier_id": "incident-text-classifier",
            "taxonomy_version": "incident-text-v1-unavailable",
            "status": "unavailable",
            "predicted_category": None,
            "class_scores": {},
            "scores_calibrated": False,
            "training_coverage": [],
            "out_of_distribution_status": "unknown",
            "uncertainty_reason": "no_suitable_licensed_text_training_corpus_or_trained_model",
            "input_id": "opaque-correlation-id",
            "session_id": "assessment-test",
            "incident_id": "assessment-test",
            "human_review_required": True,
            "advisory_only": True,
        },
    }
    assessment = build_incident_assessment(session, first_aid=analyze_first_aid(source))

    assert assessment["incident_type"]["value"] == "PERSON_UNCONSCIOUS_OR_UNRESPONSIVE"
    assert assessment["urgency_recommendation"]["value"] == "CRITICAL"
    breathing = next(item for item in assessment["facts"] if item["field"] == "Breathing status")
    assert breathing["value"] == "no"
    assert breathing["source_utterance"] == source
    assert assessment["location"]["status"] == "unresolved"
    assert assessment["location"]["latitude"] is None
    assert assessment["first_aid"]["human_review_required"] is True
    assert assessment["dispatch"]["simulation_only"] is True
    assert assessment["natural_language_classification"]["status"] == "unavailable"
    assert assessment["natural_language_classification"]["incident_id"] == assessment["incident_id"]


def test_assessment_marks_contradictory_life_status_and_asks_for_confirmation():
    session = {
        "session_id": "conflict-test",
        "client": {"transcripts": {"user": [
            {"at": "1", "text": "He is breathing normally."},
            {"at": "2", "text": "Actually, he is not breathing."},
        ]}},
        "shared_ledger": {
            "hard_facts": {"breathing_status": "no"},
            "priority": {}, "location_gate": {}, "provenance": {},
        },
        "merged_triage_state": {"victim_breathing": "no"},
        "tool_calls": [],
        "dispatch_services": {},
    }
    assessment = build_incident_assessment(session)
    breathing = next(item for item in assessment["facts"] if item["field"] == "Breathing status")
    assert breathing["evidence_status"] == "ambiguous"
    assert [item["value"] for item in breathing["conflicting_reports"]] == ["yes", "no"]
    assert any("statements conflict" in item for item in assessment["next_questions"])


def test_incident_location_rejects_invalid_or_partial_coordinates():
    with pytest.raises(ValueError, match="present as a pair"):
        IncidentLocation(candidate="unknown", status="resolved", latitude=12.0)
    with pytest.raises(ValueError, match="outside valid ranges"):
        IncidentLocation(candidate="unknown", status="resolved", latitude=91, longitude=2)


def test_live_voice_safety_context_uses_rules_without_copying_raw_caller_text():
    context = server._voice_first_aid_context({
        "client": {"transcripts": {"user": [{"text": "There has been a road accident near traffic."}]}}
    })

    assert context["scenario"] == "Road accident scene safety"
    assert any("safe to approach" in action for action in context["immediate_actions"])
    assert "There has been a road accident" not in repr(context)


def test_live_voice_safety_context_withholds_stale_specific_instructions_on_conflict():
    context = server._voice_first_aid_context({
        "client": {"transcripts": {"user": [{"text": "He is breathing normally. Actually, he is not breathing normally."}]}},
        "incident_assessment": {"facts": [{"field": "Breathing status", "evidence_status": "ambiguous"}]},
    })

    assert context["conflicting_facts"] == ["Breathing status"]
    assert "not breathing" not in repr(context).lower()


def test_llm_only_session_report_runs_local_triage_and_persists_assessment(monkeypatch):
    async def no_rescue(**_kwargs):
        return None

    monkeypatch.setattr(server, "maybe_run_stt_rescue", no_rescue)
    monkeypatch.setattr(server, "_auto_follow_location_resolution", lambda *_args: None)
    transcript = "There was a road accident. Two people are injured; one is unconscious and not breathing."
    session_id = f"assessment-{uuid.uuid4().hex}"
    response = client.post("/api/session-report", json={
        "session_id": session_id,
        "report_revision": 1,
        "status": "live",
        "triage_mode": "llm_only",
        "transcripts": {"user": [{"at": "2026-10-29T10:00:00Z", "text": transcript}], "agent": []},
    })

    assert response.status_code == 200
    payload = response.json()
    assert payload["triage_meta"]["runtime_provider"] == "local_rules"
    assert payload["triage_update"]["merged_triage_state"]["victim_breathing"] == "no"
    assert payload["triage_update"]["merged_triage_state"]["victim_count"] == 2
    saved = client.get(f"/api/sessions/{session_id}").json()
    assert saved["client"]["transcripts"]["user"][0]["text"] == transcript
    assert saved["incident_assessment"]["dispatch"]["simulation_only"] is True
    assert saved["incident_assessment"]["location"]["latitude"] is None


def test_session_report_ignores_stale_revision_without_replacing_caller_evidence(monkeypatch):
    async def no_rescue(**_kwargs):
        return None

    monkeypatch.setattr(server, "maybe_run_stt_rescue", no_rescue)
    session_id = f"revision-{uuid.uuid4().hex}"
    newer = client.post("/api/session-report", json={
        "session_id": session_id,
        "report_revision": 2,
        "status": "live",
        "triage_mode": "llm_only",
        "transcripts": {"user": [{"at": "2", "text": "Caller corrected: one injured person."}], "agent": []},
    })
    stale = client.post("/api/session-report", json={
        "session_id": session_id,
        "report_revision": 1,
        "status": "live",
        "triage_mode": "llm_only",
        "transcripts": {"user": [{"at": "1", "text": "Two injured people."}], "agent": []},
    })

    assert newer.status_code == 200
    assert stale.status_code == 200
    assert stale.json()["stale_report_ignored"] is True
    saved = client.get(f"/api/sessions/{session_id}").json()
    assert saved["client"]["transcripts"]["user"][0]["text"] == "Caller corrected: one injured person."


def test_empty_session_report_marks_assessment_as_awaiting_speech(monkeypatch):
    async def no_rescue(**_kwargs):
        return None

    monkeypatch.setattr(server, "maybe_run_stt_rescue", no_rescue)
    async def classifier_must_wait_for_a_finalized_utterance(*_args, **_kwargs):
        raise AssertionError("classifier must not run before a caller transcript is finalized")

    monkeypatch.setattr(server, "classify_incident_text_async", classifier_must_wait_for_a_finalized_utterance)
    session_id = f"empty-{uuid.uuid4().hex}"
    response = client.post("/api/session-report", json={
        "session_id": session_id,
        "report_revision": 1,
        "status": "live",
        "triage_mode": "llm_only",
        "transcripts": {"user": [], "agent": []},
    })

    assert response.status_code == 200
    payload = response.json()
    assert {"incident_assessment", "ledger_update", "triage_meta", "status"} <= set(payload)
    assessment = payload["incident_assessment"]
    assert assessment["extraction_status"] == "awaiting_speech"
    assert assessment["natural_language_classification"] is None
    assert assessment["incident_type"]["evidence_status"] == "unknown"
    assert assessment["location"]["latitude"] is None


def test_deterministic_demo_uses_bundled_input_without_voice_models_or_geocoder(monkeypatch):
    async def forbidden_rescue(**_kwargs):
        raise AssertionError("demo mode must not invoke transcript rescue")

    original_dispatch = server.dispatch_tool_call

    def forbid_external_location(name, args):
        if name in {"lookup_address", "validate_address", "lookup_response_bases"}:
            raise AssertionError("demo mode must not geocode or fetch nearby routing data")
        return original_dispatch(name, args)

    monkeypatch.setattr(server, "maybe_run_stt_rescue", forbidden_rescue)
    monkeypatch.setattr(server, "dispatch_tool_call", forbid_external_location)
    scenarios = client.get("/api/demo/scenarios")
    assert scenarios.status_code == 200
    assert len(scenarios.json()) >= 10

    for scenario in scenarios.json():
        response = client.post("/api/demo/incident", json={"scenario_id": scenario["id"], "use_demo_fallback": True})
        assert response.status_code == 200
        payload = response.json()
        assert payload["demo"] is True
        assert payload["triage_meta"]["runtime_provider"] == "local_rules"
        saved = client.get(f"/api/sessions/{payload['session_id']}").json()
        assert saved["server"]["demo_mode"] is True
        assert saved["client"]["transcripts"]["user"][0]["simulated"] is True
        assert saved["incident_assessment"]["dispatch"]["simulation_only"] is True
        demo_location = saved["incident_assessment"]["location"]
        assert demo_location["latitude"] == 9.9784782
        assert demo_location["longitude"] == 76.2757727
        assert demo_location["source"] == "demo_configuration"
        assert demo_location["confidence"] == "low"
        assert demo_location["caller_confirmed"] is False


def test_demo_scenario_request_rejects_unknown_id():
    assert client.post("/api/demo/incident", json={"scenario_id": "not-a-real-scenario"}).status_code == 404
