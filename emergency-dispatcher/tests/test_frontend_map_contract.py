from pathlib import Path


APP_JS = Path(__file__).resolve().parents[1] / "frontend" / "app.js"
LOCATION_UTILS = Path(__file__).resolve().parents[1] / "frontend" / "location-utils.js"


def test_map_defaults_to_india_and_partial_updates_retain_the_current_session_pin():
    source = APP_JS.read_text(encoding="utf-8")

    assert "const INDIA_MAP_CENTER = [22.5, 79.0];" in source
    assert "const INDIA_MAP_ZOOM = 5;" in source
    assert "}).setView(INDIA_MAP_CENTER, INDIA_MAP_ZOOM);" in source
    assert source.count("map.setView(INDIA_MAP_CENTER, INDIA_MAP_ZOOM);") >= 2
    location_utils = LOCATION_UTILS.read_text(encoding="utf-8")
    assert "!Number.isFinite(latitude) || !Number.isFinite(longitude)" in location_utils
    assert "latitude < -90 || latitude > 90" in location_utils
    assert "longitude < -180 || longitude > 180" in location_utils
    update_map = source.split("function updateMap(state)", 1)[1].split("function flushPcmPending", 1)[0]
    assert "retainMapLocation" in update_map
    assert "map.removeLayer(mapMarker);" not in update_map.split("if (!lastValidMapLocation || !map)", 1)[1].split("const retained", 1)[0]
    assert "current update omitted coordinates; the prior pin is retained" in update_map
    assert "mapSessionId !== sessionId" in update_map
    assert "approximateMapLocation?.lat" in source
    assert "approximateMapLocation?.lon" in source
    assert "mapMarker.setLatLng(coordinates);" in source
    assert "renderDemoSimulationMap(state.demoSimulation)" in source
    assert "window.L.polyline(routeCoordinates" in source
    assert "demoAmbulanceMarker.setLatLng(position)" in source
    assert "SIMULATED ROUTE — NOT ROAD NAVIGATION" in source
    assert "map.flyTo(coordinates, mapPresentation.zoom" in source
    assert "[52.52, 13.405]" not in source
    assert "mapCorrectionMode = !mapCorrectionMode" in source
    assert "api/demo/location/" in source
    assert "routeKey" in source
    assert "Original phrase:" in source
    assert "resolution_quality" in source
    assert 'use_demo_fallback: Boolean(incidentDemoFallback?.checked)' in source
    assert 'body: JSON.stringify({ lat: event.latlng.lat, lon: event.latlng.lng })' in source
    assert 'Geocoder: ${geocoderError}' in source
    assert "state.incidentAssessmentLocation?.error_category" in source
    assert '"Human Required Now"' not in source
