from types import SimpleNamespace

from fastapi.testclient import TestClient

from emergency_dispatcher import server
from emergency_dispatcher.server import app
from emergency_dispatcher.settings import get_settings
from emergency_dispatcher.triage import TriageResult


client = TestClient(app)


def test_text_triage_uses_labelled_local_fallback_and_advisory_overrides(monkeypatch):
    monkeypatch.setattr(server, "get_settings", lambda: get_settings().model_copy(update={"triage_api_key": None, "triage_decoder_model": None}))

    response = client.post(
        "/api/triage",
        json={"engine": "auto", "text": "There is a fire and someone is trapped near Alexanderplatz. They are not breathing."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_engine"] == "heuristic"
    assert payload["advisory_only"] is True
    assert payload["model_artifacts"]["demo_only"] is True
    assert payload["model_artifacts"]["model_id"] == "heuristic-demo-v1"
    assert payload["model_artifacts"]["uncertainty"]["calibrated_confidence"] is None
    assert payload["triage"]["escalate_to_human"] is True
    assert payload["rule_overrides"]


def test_text_triage_uses_configured_slm_when_available(monkeypatch):
    settings = get_settings().model_copy(
        update={"triage_api_key": "test-token", "triage_decoder_model": "decoder-model"}
    )
    monkeypatch.setattr(server, "get_settings", lambda: settings)

    def fake_engine(configured):
        assert configured.triage_engine == "slm"
        return SimpleNamespace(
            selected_engine="slm",
            runtime_provider="pioneer",
            analyze_utterance=lambda _text, _state: TriageResult(
                issue_type="BUILDING_FIRE",
                priority="CRITICAL",
                severity_score=90,
            ),
        )

    monkeypatch.setattr(server, "get_triage_engine", fake_engine)
    response = client.post("/api/triage", json={"engine": "auto", "text": "Smoke in a building."})

    assert response.status_code == 200
    assert response.json()["selected_engine"] == "slm"
    assert response.json()["runtime_provider"] == "pioneer"
    assert response.json()["triage"]["issue_type"] == "BUILDING_FIRE"


def test_text_triage_reports_missing_slm_configuration(monkeypatch):
    settings = get_settings().model_copy(update={"triage_api_key": None, "triage_decoder_model": None})
    monkeypatch.setattr(server, "get_settings", lambda: settings)

    response = client.post("/api/triage", json={"engine": "slm", "text": "A person has collapsed."})

    assert response.status_code == 503
    assert response.json()["detail"] == "SLM triage needs TRIAGE_API_KEY."


def test_text_triage_validates_empty_and_oversized_inputs():
    empty = client.post("/api/triage", json={"text": "   "})
    oversized = client.post("/api/triage", json={"text": "x" * 2001})

    assert empty.status_code == 422
    assert oversized.status_code == 422
