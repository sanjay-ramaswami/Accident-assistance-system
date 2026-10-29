import pytest
from fastapi.testclient import TestClient

from emergency_dispatcher import server
from emergency_dispatcher.first_aid import analyze_first_aid
from emergency_dispatcher.server import app


client = TestClient(app)


@pytest.mark.parametrize(
    ("text", "scenario"),
    [
        ("A person is unconscious but breathing normally.", "unconsciousness"),
        ("They are unresponsive and only gasping.", "cardiac_arrest"),
        ("There is severe bleeding that will not stop.", "severe_bleeding"),
        ("A child was burned by hot water.", "burns"),
        ("Someone is choking and cannot cough.", "choking"),
        ("I can't breathe and am gasping.", "breathing_difficulty"),
        ("Their face is drooping and speech is slurred.", "stroke"),
        ("They are having a seizure.", "seizure"),
        ("Possible medication overdose.", "poisoning"),
        ("Possible broken arm after a fall.", "serious_injury_or_fracture"),
    ],
)
def test_supported_first_aid_scenarios_have_structured_guidance(text, scenario):
    result = analyze_first_aid(text)

    assert result["detected_scenario"]["id"] == scenario
    assert result["engine"] == {
        "id": "first-aid-rules",
        "version": "1.0.0",
        "source_type": "deterministic_rules",
    }
    assert result["immediate_actions"]
    assert result["actions_to_avoid"]
    assert result["critical_follow_up_questions"]
    assert result["sources"]


def test_road_accident_guidance_prioritizes_scene_safety_and_is_sourced():
    result = analyze_first_aid("There has been a road accident near a smoking vehicle.")

    assert result["detected_scenario"]["id"] == "road_accident_scene_safety"
    assert any("safe to approach" in action for action in result["immediate_actions"])
    assert any("cannot place the call" in action for action in result["immediate_actions"])
    assert any(source["organization"].startswith("UK Government") for source in result["sources"])
    assert all(source["verified_on"] == "2026-10-29" for source in result["sources"])
    assert result["limitations"].find("not clinically validated") >= 0


def test_unresponsive_alone_does_not_assume_cardiac_arrest_and_escalates():
    result = analyze_first_aid("They are unconscious, but I can see normal breathing.")

    assert result["detected_scenario"]["id"] == "unconsciousness"
    assert result["severity"] == "critical"
    assert result["emergency_escalation"]["recommended"] is True
    assert any("Do not assume cardiac arrest" in item for item in result["actions_to_avoid"])


def test_unknown_description_gives_immediate_safe_actions_and_human_review():
    result = analyze_first_aid("Something happened, I am not sure what.")

    assert result["detected_scenario"]["id"] == "unknown_or_ambiguous"
    assert result["ambiguous_or_unsupported"] is True
    assert result["human_review_required"] is True
    assert result["emergency_escalation"]["recommended"] is True
    assert result["immediate_actions"]


def test_multiple_serious_findings_do_not_lower_critical_safety_override():
    result = analyze_first_aid("The person is unconscious, not breathing normally, and bleeding severely.")

    assert result["severity"] == "critical"
    assert result["emergency_escalation"]["recommended"] is True
    assert result["human_review_required"] is True
    assert "life_threatening_indicator" in result["safety_flags"]
    assert any(
        "direct pressure" in action.lower()
        for related in result["related_guidance"]
        for action in related["immediate_actions"]
    )


def test_first_aid_http_route_is_local_and_does_not_require_live_llm(monkeypatch):
    def llm_settings_must_not_be_read():
        raise AssertionError("first-aid rules must not depend on LLM configuration")

    monkeypatch.setattr(server, "get_settings", llm_settings_must_not_be_read)
    response = client.post("/api/first-aid", json={"text": "They are unconscious."})

    assert response.status_code == 200
    assert response.json()["engine"]["source_type"] == "deterministic_rules"
    assert response.json()["detected_scenario"]["id"] == "unconsciousness"


@pytest.mark.parametrize(
    "payload",
    [{"text": "   "}, {"text": "x" * 2001}, {"description": "unconscious"}, None],
)
def test_first_aid_route_rejects_malformed_empty_or_oversized_payloads(payload):
    response = client.post("/api/first-aid", json=payload)

    assert response.status_code == 422

