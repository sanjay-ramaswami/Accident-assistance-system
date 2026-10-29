from datetime import UTC, datetime, timedelta

from emergency_dispatcher.demo_simulation import advance_demo_simulation
from emergency_dispatcher.demo_simulation import create_demo_simulation
from emergency_dispatcher.demo_simulation import reset_demo_simulation


def test_demo_simulation_uses_configured_location_nearest_unit_and_simulated_route():
    started = datetime(2026, 10, 30, tzinfo=UTC)
    simulation = create_demo_simulation(session_id="demo-a", now=started)

    assert simulation["simulation_only"] is True
    assert simulation["location"]["display_label"] == "DEMO LOCATION — NOT VERIFIED FROM CALLER"
    assert simulation["location"]["verified_from_caller"] is False
    assert -90 <= simulation["location"]["lat"] <= 90
    assert -180 <= simulation["location"]["lon"] <= 180
    assert simulation["selected_unit_id"] == "SIM-AMB-01"
    assert simulation["route"]["method"] == "simulated_straight_line"
    assert "NOT ROAD NAVIGATION" in simulation["route"]["label"]
    assert simulation["route"]["coordinates"][-1] == [simulation["location"]["lat"], simulation["location"]["lon"]]
    selected = next(unit for unit in simulation["fleet"] if unit["id"] == simulation["selected_unit_id"])
    assert selected["available"] is False
    assert selected["assigned_incident_id"] == "demo-a"


def test_demo_simulation_advances_persists_movement_and_completes_idempotently():
    started = datetime(2026, 10, 30, tzinfo=UTC)
    initial = create_demo_simulation(session_id="demo-b", now=started)
    en_route = advance_demo_simulation(initial, now=started + timedelta(seconds=6))
    assert en_route["stage"] == "EN_ROUTE"
    assert 0 < en_route["route_progress"] < 1
    assert en_route["ambulance_status"] == "en_route"
    assert [event["stage"] for event in en_route["timeline"]] == [
        "RECEIVED", "ASSESSING", "LOCATION_RESOLVED", "AMBULANCE_ASSIGNED", "EN_ROUTE"
    ]
    done = advance_demo_simulation(en_route, now=started + timedelta(seconds=20))
    assert done["stage"] == "COMPLETED"
    assert done["route_progress"] == 1
    assert [event["stage"] for event in done["timeline"]][-2:] == ["ARRIVED", "COMPLETED"]
    selected = next(unit for unit in done["fleet"] if unit["id"] == done["selected_unit_id"])
    assert selected["available"] is True
    assert selected["assigned_incident_id"] is None
    assert advance_demo_simulation(done, now=started + timedelta(seconds=30)) == done


def test_demo_fleet_exhaustion_and_reset_release_assignment():
    started = datetime(2026, 10, 30, tzinfo=UTC)
    exhausted = create_demo_simulation(
        session_id="demo-c",
        now=started,
        unavailable_unit_ids={"SIM-AMB-01", "SIM-AMB-02"},
    )
    assert exhausted["stage"] == "NO_UNIT_AVAILABLE"
    assert exhausted["selected_unit_id"] is None
    assert exhausted["route"]["coordinates"] == []

    active = create_demo_simulation(session_id="demo-d", now=started)
    reset = reset_demo_simulation(active, now=started + timedelta(seconds=2))
    assert reset["stage"] == "RESET"
    assert reset["timeline"][-1]["stage"] == "RESET"
    assert reset["ambulance_position"] is None
    selected = next(unit for unit in reset["fleet"] if unit["id"] == reset["selected_unit_id"])
    assert selected["available"] is True
    assert selected["assigned_incident_id"] is None

