"""Deterministic, local-only ambulance demo simulation.

All positions and movement here are illustrative configuration. They are not
caller GPS, live fleet data, road navigation, or a real dispatch connection.
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
import math
from typing import Any


DEMO_LOCATION = {
    "label": "Marine Drive, Kochi, Kerala, India",
    "lat": 9.9784782,
    "lon": 76.2757727,
    "source": "demo_configuration",
    "confidence": "low",
    "resolution_quality": "configured_demo_fallback",
    "verified_from_caller": False,
    "display_label": "DEMO LOCATION — NOT VERIFIED FROM CALLER",
}
DEMO_FLEET = (
    {"id": "SIM-AMB-01", "call_sign": "Demo Ambulance 1", "lat": 9.9630, "lon": 76.2880},
    {"id": "SIM-AMB-02", "call_sign": "Demo Ambulance 2", "lat": 9.9940, "lon": 76.2960},
)
SIMULATION_STAGES = (
    (0.0, "RECEIVED"),
    (1.0, "ASSESSING"),
    (2.0, "LOCATION_RESOLVED"),
    (3.0, "AMBULANCE_ASSIGNED"),
    (4.0, "EN_ROUTE"),
    (12.0, "ARRIVED"),
    (15.0, "COMPLETED"),
)


def _distance_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    radius_km = 6371.0088
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dlat, dlon = math.radians(b_lat - a_lat), math.radians(b_lon - a_lon)
    h = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * radius_km * math.asin(math.sqrt(h))


def create_demo_simulation(
    *, session_id: str, now: datetime | None = None, unavailable_unit_ids: set[str] | None = None,
    incident_location: dict[str, Any] | None = None, use_demo_fallback: bool = True,
) -> dict[str, Any]:
    started = now or datetime.now(UTC)
    location = dict(incident_location or DEMO_LOCATION) if (incident_location or use_demo_fallback) else None
    if location is None:
        return {
            "simulation_only": True, "session_id": session_id, "started_at": started.isoformat(),
            "stage": "LOCATION_UNRESOLVED", "location": None,
            "fleet": [dict(unit, available=True, assigned_incident_id=None, current_lat=unit["lat"], current_lon=unit["lon"]) for unit in DEMO_FLEET],
            "selected_unit_id": None, "selected_call_sign": None, "assignment_status": "location_unresolved",
            "route": {"method": "simulated_straight_line", "label": "No route: incident location unresolved", "coordinates": [], "distance_km": None, "eta_minutes": None},
            "timeline": [{"stage": "LOCATION_UNRESOLVED", "at": started.isoformat(), "simulation_only": True}],
        }
    try:
        lat, lon = float(location["lat"]), float(location["lon"])
    except (KeyError, TypeError, ValueError):
        raise ValueError("A verified incident location requires numeric coordinates.")
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Incident coordinates are outside valid latitude/longitude ranges.")
    location["lat"], location["lon"] = lat, lon
    units = [dict(unit, available=True, assigned_incident_id=None, current_lat=unit["lat"], current_lon=unit["lon"]) for unit in DEMO_FLEET]
    available = [unit for unit in units if unit["id"] not in (unavailable_unit_ids or set())]
    selected = min(available, key=lambda unit: _distance_km(unit["lat"], unit["lon"], lat, lon)) if available else None
    if selected:
        selected["available"] = False
        selected["assigned_incident_id"] = session_id
        distance = _distance_km(selected["lat"], selected["lon"], lat, lon)
    else:
        distance = None
    return {
        "simulation_only": True,
        "session_id": session_id,
        "started_at": started.isoformat(),
        "stage": "RECEIVED" if selected else "NO_UNIT_AVAILABLE",
        "location": location,
        "fleet": units,
        "selected_unit_id": selected["id"] if selected else None,
        "selected_call_sign": selected["call_sign"] if selected else None,
        "assignment_status": "assigned" if selected else "no_unit_available",
        "route": {
            "method": "simulated_straight_line",
            "label": "SIMULATED ROUTE — NOT ROAD NAVIGATION",
            "coordinates": [[selected["lat"], selected["lon"]], [lat, lon]] if selected else [],
            "distance_km": round(distance, 2) if distance is not None else None,
            "eta_minutes": max(1, round(distance / 0.4)) if distance is not None else None,
        },
        "timeline": [{"stage": "RECEIVED" if selected else "NO_UNIT_AVAILABLE", "at": started.isoformat(), "simulation_only": True}],
    }


def advance_demo_simulation(state: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Advance idempotently from wall-clock simulation time and interpolate unit position."""
    if state.get("stage") in {"RESET", "NO_UNIT_AVAILABLE", "LOCATION_UNRESOLVED"}:
        return state
    current = now or datetime.now(UTC)
    try:
        started = datetime.fromisoformat(str(state["started_at"]))
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
    except (KeyError, TypeError, ValueError):
        return state
    elapsed = max(0.0, (current - started).total_seconds())
    stage = "RECEIVED"
    for threshold, candidate in SIMULATION_STAGES:
        if elapsed >= threshold:
            stage = candidate
    result = copy.deepcopy(state)
    timeline = list(state.get("timeline") or [])
    if stage != state.get("stage"):
        prior_index = next((i for i, (_, item) in enumerate(SIMULATION_STAGES) if item == state.get("stage")), -1)
        next_index = next((i for i, (_, item) in enumerate(SIMULATION_STAGES) if item == stage), prior_index)
        for _, next_stage in SIMULATION_STAGES[prior_index + 1 : next_index + 1]:
            timeline.append({"stage": next_stage, "at": current.isoformat(), "simulation_only": True})
    result["stage"] = stage
    result["timeline"] = timeline
    route = result.get("route") or {}
    coords = route.get("coordinates") or []
    if len(coords) >= 2:
        fraction = min(1.0, max(0.0, (elapsed - 4.0) / 8.0))
        start, end = coords[0], coords[-1]
        position = [start[0] + (end[0] - start[0]) * fraction, start[1] + (end[1] - start[1]) * fraction]
        result["ambulance_position"] = position
        result["route_progress"] = round(fraction, 4)
        result["ambulance_status"] = "available" if stage == "COMPLETED" else "arrived" if stage == "ARRIVED" else "en_route" if stage == "EN_ROUTE" else "assigned"
        for unit in result.get("fleet") or []:
            if unit.get("id") == result.get("selected_unit_id"):
                unit.update({"current_lat": position[0], "current_lon": position[1], "available": stage == "COMPLETED", "assigned_incident_id": None if stage == "COMPLETED" else result.get("session_id")})
    return result


def update_demo_simulation_location(state: dict[str, Any], *, lat: float, lon: float) -> dict[str, Any]:
    """Persist an operator-confirmed map point and rebuild the simulated route."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Incident coordinates are outside valid latitude/longitude ranges.")
    result = copy.deepcopy(state)
    location = {
        "label": "Operator-selected map point", "lat": float(lat), "lon": float(lon),
        "source": "operator_map_selection", "original_phrase": None,
        "confidence": "medium", "resolution_quality": "manual_map_selection",
        "verified_from_caller": False, "operator_confirmed": True,
        "display_label": "OPERATOR-SELECTED POINT — SIMULATION ONLY",
    }
    result["location"] = location
    selected = next((unit for unit in result.get("fleet", []) if unit.get("id") == result.get("selected_unit_id")), None)
    if selected is None:
        available = [unit for unit in result.get("fleet", []) if unit.get("available", True)]
        selected = min(available, key=lambda unit: _distance_km(unit["lat"], unit["lon"], lat, lon)) if available else None
        if selected:
            selected["available"] = False
            selected["assigned_incident_id"] = result.get("session_id")
            result["selected_unit_id"] = selected["id"]
            result["selected_call_sign"] = selected["call_sign"]
            result["assignment_status"] = "assigned"
            result["stage"] = "RECEIVED"
            result["started_at"] = datetime.now(UTC).isoformat()
    if selected:
        start_lat = float(selected.get("current_lat", selected["lat"]))
        start_lon = float(selected.get("current_lon", selected["lon"]))
        distance = _distance_km(start_lat, start_lon, lat, lon)
        result["route"] = {
            "method": "simulated_straight_line", "label": "SIMULATED ROUTE — NOT ROAD NAVIGATION",
            "coordinates": [[start_lat, start_lon], [float(lat), float(lon)]],
            "distance_km": round(distance, 2), "eta_minutes": max(1, round(distance / 0.4)),
        }
    result.setdefault("timeline", []).append({"stage": "LOCATION_CORRECTED", "at": datetime.now(UTC).isoformat(), "simulation_only": True})
    return result


def reset_demo_simulation(state: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    """Release the unit and record reset without deleting the session audit."""
    result = copy.deepcopy(state)
    current = now or datetime.now(UTC)
    result["stage"] = "RESET"
    result["assignment_status"] = "reset"
    result["timeline"] = list(state.get("timeline") or []) + [{"stage": "RESET", "at": current.isoformat(), "simulation_only": True}]
    for unit in result.get("fleet") or []:
        if unit.get("id") == result.get("selected_unit_id"):
            unit.update({"available": True, "assigned_incident_id": None, "current_lat": unit.get("lat"), "current_lon": unit.get("lon")})
    result["ambulance_position"] = None
    result["route_progress"] = 0
    return result
