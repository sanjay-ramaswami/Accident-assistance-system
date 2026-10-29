from pathlib import Path


FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


def test_simulated_dispatch_and_local_ticket_are_visibly_labeled():
    app = (FRONTEND / "app.js").read_text(encoding="utf-8")
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")

    assert 'title: "Dispatch Simulation"' in app
    assert "SIMULATED DISPATCH ONLY" in app
    assert "No emergency services are contacted." in app
    assert "LOCAL DEMO TICKET ONLY" in app
    assert "km straight-line" in app
    assert "<h2>Demo Ticket</h2>" in html
    assert "not sent to a dispatch service" in html


def test_first_aid_panel_is_explicitly_local_and_never_claims_live_ai():
    app = (FRONTEND / "app.js").read_text(encoding="utf-8")
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")

    assert 'id="first-aid-form"' in html
    assert 'id="first-aid-result"' in html
    assert 'fetch("/api/first-aid"' in app
    assert "deterministic first-aid rules" in app
    assert "not live AI" in app
    assert "CALL LOCAL EMERGENCY SERVICES NOW — THIS APP DOES NOT PLACE THE CALL" in app


def test_demo_state_is_not_rendered_as_caller_verified_or_live_operator_activity():
    app = (FRONTEND / "app.js").read_text(encoding="utf-8")

    assert "SIMULATION ONLY · ${state.issueType" in app
    assert "if (!state.demoSimulation && state.location" in app
    assert "if (!state.demoSimulation && state.hasPin)" in app
    assert "if (!state.demoSimulation && state.humanMonitoring" in app
    assert "humanMonitoring: demoSimulation ? false" in app
    assert "monitorName: demoSimulation ? null" in app
    assert "configured demo coordinates" in app
    assert "not caller verified" in app
    assert "Configured demo location and simulated fleet only" in app


def test_demo_refresh_updates_stage_and_assessment_uses_simulated_unit():
    app = (FRONTEND / "app.js").read_text(encoding="utf-8")

    assert "emergencyAssessmentStatus = `Simulation · ${session.demo_simulation.stage" in app
    assert "const dispatchPlan = simulation" in app
    assert "simulation.selected_call_sign || \"No demo unit assigned\"" in app
    assert "SIMULATED ROUTE — NOT ROAD NAVIGATION" in app


def test_live_assessment_displays_ml_unavailability_and_actual_live_result_prominently():
    app = (FRONTEND / "app.js").read_text(encoding="utf-8")
    html = (FRONTEND / "index.html").read_text(encoding="utf-8")

    assert 'id="emergency-assessment-panel"' in html
    assert html.index('id="emergency-assessment-panel"') < html.index('class="map-shell map-shell-prominent"')
    assert 'aria-label="Live complaint-code classification"' in app
    assert 'aria-label="Natural-language incident classifier"' in app
    assert "Synthetic-data prototype only" in app
    assert "authored synthetic demo scenarios only" in app
    assert "not clinically validated" in app
    assert "Rule-based demo fallback — not an ML prediction" in app
    assert "ML PREDICTION COMPLETE" in app
    assert 'id="incident-demo-run"' in html
    assert 'id="incident-demo-reset"' in html
    assert "SIMULATION MODE" in app
    assert "Heuristic advisory signal only; not clinically validated severity." in app
    assert "DEMO LOCATION — NOT VERIFIED FROM CALLER" in app
    assert "SIMULATED ROUTE — NOT ROAD NAVIGATION" in app
    assert "demo_simulation" in app
    assert "Rule-based assessment remains separate" in app
    assert "Not applicable — complaint code unavailable" in app
    assert "prediction.chapter" in app
    assert "incident_assessment" in app
    assert 'url: "/api/session-report"' in app
    assert "DispatcherSessionReportUtils.postSessionReport" in app
