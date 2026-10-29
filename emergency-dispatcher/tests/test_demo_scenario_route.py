from datetime import UTC, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from emergency_dispatcher import server, session_store
from emergency_dispatcher.server import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def isolate_sessions(monkeypatch, tmp_path):
    monkeypatch.setattr(session_store, "SESSION_DIR", tmp_path / "sessions")
    real_dispatch_tool = server.dispatch_tool_call

    def deterministic_lookup(name, args):
        if name == "lookup_address":
            query = args.get("address_text") or ""
            return {
                "geocoded": True, "normalized_address": query, "lat": 9.95, "lon": 76.25,
                "source": "nominatim", "confidence": "low", "reason": "approximate_place_match",
            }
        return real_dispatch_tool(name, args)

    monkeypatch.setattr(server, "dispatch_tool_call", deterministic_lookup)


def test_demo_scenario_returns_model_location_fleet_route_and_simulation_state():
    response = client.post("/api/demo/incident", json={"scenario_id": "chest-pain-breathing", "use_demo_fallback": True})
    assert response.status_code == 200
    payload = response.json()
    assessment = payload["incident_assessment"]
    simulation = payload["demo_simulation"]
    assert payload["demo"] is True
    assert "synthetic" in payload["notice"].lower()
    assert assessment["natural_language_classification"]["status"] == "classified"
    assert assessment["natural_language_classification"]["prototype_only"] is True
    assert assessment["location"]["source"] == "demo_configuration"
    assert assessment["location"]["caller_confirmed"] is False
    assert assessment["location"]["latitude"] == simulation["location"]["lat"]
    assert assessment["dispatch"]["simulation_only"] is True
    assert simulation["selected_unit_id"]
    assert simulation["route"]["method"] == "simulated_straight_line"


def test_demo_detail_advances_state_and_reset_releases_unit(monkeypatch):
    result = client.post("/api/demo/incident", json={"scenario_id": "factory-collapse", "resolve_scenario_location": True}).json()
    session_id = result["session_id"]
    started_at = datetime.fromisoformat(result["demo_simulation"]["started_at"])
    real_advance = server.advance_demo_simulation
    monkeypatch.setattr(
        server,
        "advance_demo_simulation",
        lambda state: real_advance(state, now=started_at + timedelta(seconds=6)),
    )
    detail = client.get(f"/api/sessions/{session_id}").json()
    assert detail["demo_simulation"]["stage"] == "EN_ROUTE"
    assert detail["incident_assessment"]["dispatch"]["status"] == "EN_ROUTE"
    assert detail["demo_simulation"]["ambulance_position"]

    reset = client.post(f"/api/demo/reset/{session_id}")
    assert reset.status_code == 200
    assert reset.json()["demo_simulation"]["stage"] == "RESET"
    assert client.get(f"/api/sessions/{session_id}").json()["incident_assessment"]["dispatch"]["status"] == "RESET"


def test_a_second_active_demo_gets_another_unit_and_no_real_dispatch():
    first = client.post("/api/demo/incident", json={"scenario_id": "road-accident-unconscious", "use_demo_fallback": True}).json()
    second = client.post("/api/demo/incident", json={"scenario_id": "factory-collapse", "use_demo_fallback": True}).json()
    assert first["demo_simulation"]["selected_unit_id"] != second["demo_simulation"]["selected_unit_id"]
    assert first["demo_simulation"]["simulation_only"] is True
    assert second["demo_simulation"]["simulation_only"] is True
    assert "real dispatch" in first["notice"].lower() or "no emergency services" in first["notice"].lower()


def test_demo_scenario_skips_corrupt_prior_session_without_leaking_its_content(caplog):
    session_store.SESSION_DIR.mkdir(parents=True, exist_ok=True)
    marker = "SENSITIVE_PRIOR_CALL_TEXT"
    (session_store.SESSION_DIR / "damaged-session.json").write_text(
        '{"transcript":"' + marker + '"}}', encoding="utf-8"
    )
    with caplog.at_level("WARNING", logger=session_store.__name__):
        response = client.post("/api/demo/incident", json={"scenario_id": "factory-collapse", "use_demo_fallback": True})

    assert response.status_code == 200
    assert response.json()["demo_simulation"]["simulation_only"] is True
    assert "session_index_record_skipped error_type=JSONDecodeError" in caplog.text
    assert marker not in caplog.text


def test_concurrent_demo_scenarios_reserve_distinct_units():
    def run(scenario_id):
        with TestClient(app) as concurrent_client:
            response = concurrent_client.post("/api/demo/incident", json={"scenario_id": scenario_id, "use_demo_fallback": True})
            assert response.status_code == 200
            return response.json()["demo_simulation"]["selected_unit_id"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        assignments = list(pool.map(run, ["factory-collapse", "road-accident-unconscious"]))
    assert len(set(assignments)) == 2


def test_named_place_demo_uses_geocoder_coordinates_for_location_and_route():
    response = client.post("/api/demo/incident", json={
        "scenario_id": "factory-collapse", "location_text": "Lulu Mall, Kochi",
    })
    assert response.status_code == 200
    payload = response.json()
    location = payload["demo_simulation"]["location"]
    assert location["original_phrase"] == "Lulu Mall, Kochi"
    assert location["source"] == "nominatim"
    assert [location["lat"], location["lon"]] == [9.95, 76.25]
    assert location["resolution_quality"] == "named_place_approximation"
    assert "nearby entrance" in payload["incident_assessment"]["location"]["clarification"]
    assert payload["demo_simulation"]["route"]["coordinates"][-1] == [9.95, 76.25]
    assert payload["incident_assessment"]["location"]["latitude"] == 9.95


def test_ambiguous_marine_drive_stays_unresolved_without_demo_fallback():
    response = client.post("/api/demo/incident", json={
        "scenario_id": "factory-collapse", "location_text": "Marine Drive",
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["demo_simulation"]["location"] is None
    assert payload["demo_simulation"]["selected_unit_id"] is None
    assert payload["demo_simulation"]["route"]["coordinates"] == []
    assert payload["incident_assessment"]["location"]["status"] == "needs_confirmation"
    assert payload["incident_assessment"]["location"]["error_category"] == "ambiguous_location"
    assert "Which city" in payload["incident_assessment"]["location"]["clarification"]


def test_geocoder_ambiguity_for_other_place_gets_place_specific_clarification(monkeypatch):
    monkeypatch.setattr(server, "dispatch_tool_call", lambda _name, _args: {
        "found": False, "geocoded": False, "lat": None, "lon": None,
        "source": "nominatim", "confidence": None, "reason": "ambiguous_location",
    })
    response = client.post("/api/demo/incident", json={
        "scenario_id": "factory-collapse", "location_text": "Lulu Mall, Kochi",
    })
    location = response.json()["incident_assessment"]["location"]
    assert location["error_category"] == "ambiguous_location"
    assert "Which city is Marine Drive" not in location["clarification"]
    assert "nearby road, landmark, hospital, or entrance" in location["clarification"]


def test_explicit_fallback_after_ambiguous_location_remains_labeled_and_unconfirmed():
    response = client.post("/api/demo/incident", json={
        "scenario_id": "factory-collapse", "location_text": "Marine Drive", "use_demo_fallback": True,
    })
    payload = response.json()
    location = payload["demo_simulation"]["location"]
    assert location["source"] == "demo_configuration"
    assert location["original_phrase"] == "Marine Drive"
    assert location["unresolved_caller_phrase"] is True
    assert payload["incident_assessment"]["location"]["caller_confirmed"] is False
    assert "NOT VERIFIED FROM CALLER" in location["display_label"]


def test_map_correction_persists_location_and_rebuilds_simulated_route():
    result = client.post("/api/demo/incident", json={
        "scenario_id": "factory-collapse", "use_demo_fallback": True,
    }).json()
    session_id = result["session_id"]
    corrected = client.post(f"/api/demo/location/{session_id}", json={"lat": 10.1, "lon": 76.3})
    assert corrected.status_code == 200
    payload = corrected.json()
    location = payload["demo_simulation"]["location"]
    route = payload["demo_simulation"]["route"]["coordinates"]
    assert location["source"] == "operator_map_selection"
    assert location["operator_confirmed"] is True
    assert route[-1] == [10.1, 76.3]
    assert payload["incident_assessment"]["location"]["latitude"] == 10.1
