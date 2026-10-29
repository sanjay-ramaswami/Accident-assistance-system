from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from emergency_dispatcher.first_aid import analyze_first_aid
from emergency_dispatcher.server import app
from emergency_dispatcher.triage_ml import CHAPTERS
from emergency_dispatcher.triage_ml import MODEL_PATH
from emergency_dispatcher.triage_ml import ModelArtifactError
from emergency_dispatcher.triage_ml import load_complaint_chapter_model
from emergency_dispatcher.triage_ml import normalize_complaint_code
from emergency_dispatcher.triage_ml import predict_complaint_chapter


client = TestClient(app)


def test_saved_model_loads_and_predicts_with_explicit_advisory_limits():
    artifact = load_complaint_chapter_model()
    result = predict_complaint_chapter(" r07.9 ", artifact=artifact)

    assert set(artifact["label_mapping"]) == set(CHAPTERS)
    assert result["input_code"] == "R07.9"
    assert result["prediction"]["chapter_id"] in CHAPTERS
    assert result["uncalibrated_model_score"] >= 0
    assert result["advisory_only"] is True
    assert result["human_review_required"] is True
    assert result["dispatch_authorized"] is False


@pytest.mark.parametrize("value", [None, 123, "", "   ", "R07.9/../../secret", "x" * 25])
def test_complaint_code_rejects_missing_or_malformed_values(value):
    with pytest.raises(ValueError):
        normalize_complaint_code(value)


def test_unseen_valid_code_is_explicitly_marked_for_human_review():
    result = predict_complaint_chapter("ZZZ.999")
    assert result["training_coverage"] == "unseen_code"
    assert result["human_review_required"] is True
    assert result["dispatch_authorized"] is False


def test_unknown_classes_in_model_artifact_are_rejected(tmp_path: Path):
    artifact = copy.copy(load_complaint_chapter_model())
    artifact["label_mapping"] = {**artifact["label_mapping"], "9": "Unknown chapter"}
    path = tmp_path / "unknown-class.joblib"
    joblib.dump(artifact, path)

    with pytest.raises(ModelArtifactError, match="unknown chapter classes"):
        load_complaint_chapter_model(path)


def test_missing_model_artifact_is_reported_as_unavailable(tmp_path: Path):
    with pytest.raises(ModelArtifactError, match="unavailable"):
        load_complaint_chapter_model(tmp_path / "missing.joblib")


def test_complaint_code_api_runs_without_voice_provider_and_validates_inputs():
    response = client.post("/api/triage/complaint-code", json={"code": "R07.9"})
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {
        "task", "input_code", "prediction", "uncalibrated_model_score", "training_coverage",
        "model", "dataset", "advisory_only", "human_review_required", "dispatch_authorized", "notice",
    }
    assert set(payload["prediction"]) == {"chapter_id", "chapter"}
    assert payload["task"] == "complaint_code_to_clinical_functional_chapter"
    assert payload["input_code"] == "R07.9"
    assert payload["prediction"]["chapter"] == "Cardiorespiratory"
    assert payload["training_coverage"] == "seen_in_training"
    assert isinstance(payload["uncalibrated_model_score"], float)
    assert payload["advisory_only"] is True
    assert payload["human_review_required"] is True
    assert payload["dispatch_authorized"] is False

    assert client.post("/api/triage/complaint-code", json={}).status_code == 422
    assert client.post("/api/triage/complaint-code", json={"code": "   "}).status_code == 422
    malformed = client.post("/api/triage/complaint-code", json={"code": "???"})
    assert malformed.status_code == 422
    assert "alphanumeric complaint code" in malformed.json()["detail"]


def test_complaint_code_api_marks_valid_unseen_code_and_still_requires_review():
    response = client.post("/api/triage/complaint-code", json={"code": "zzz.999"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["input_code"] == "ZZZ.999"
    assert payload["training_coverage"] == "unseen_code"
    assert payload["prediction"]["chapter"] in CHAPTERS.values()
    assert payload["uncalibrated_model_score"] >= 0
    assert payload["human_review_required"] is True
    assert payload["dispatch_authorized"] is False


def test_model_output_cannot_lower_independent_critical_first_aid_override():
    ml_result = predict_complaint_chapter("R07.9")
    safety_result = analyze_first_aid("unconscious and not breathing normally")

    assert ml_result["dispatch_authorized"] is False
    assert safety_result["severity"] == "critical"
    assert safety_result["emergency_escalation"]["level"] == "call_local_emergency_services_now"
    assert safety_result["human_review_required"] is True


def test_dashboard_labels_the_model_task_and_its_limitations():
    root = Path(__file__).resolve().parents[1]
    html = (root / "frontend/index.html").read_text(encoding="utf-8")
    app_js = (root / "frontend/app.js").read_text(encoding="utf-8")
    assert 'id="complaint-code-form"' in html
    assert "ML Complaint Classification" in html
    assert "Run ML Classification" in html
    assert "diagnose, authorize dispatch" in html
    assert 'id="live-ml-status"' in html
    assert "Not applicable — complaint code unavailable" in html
    assert "Enter a complaint code before running classification." in app_js
    assert "does not interpret arbitrary symptoms" in html
    assert 'fetch("/api/triage/complaint-code"' in app_js
    assert "DispatcherComplaintCodeUtils.presentComplaintCodeResult(payload)" in app_js
    assert "liveMlClassifier.finalize(text, sessionId)" in app_js
    assert "training_coverage !== \"seen_in_training\"" in (root / "frontend/live-ml-utils.js").read_text(encoding="utf-8")
    assert "recordVoiceLatency(\"microphone_capture\")" in app_js
    assert "recordVoiceLatency(\"outbound_audio_send\")" in app_js
    assert "recordVoiceLatency(\"first_returned_audio_chunk\")" in app_js
    assert "recordVoiceLatency(\"frontend_audio_enqueue\")" in app_js
    assert "recordVoiceLatency(\"actual_playback_start\")" in app_js
    assert "recordCallerTurnLatency(\"first_provider_response\")" in app_js
    assert "recordCallerTurnLatency(\"first_returned_audio_chunk\")" in app_js
    assert "recordCallerTurnLatency(\"frontend_audio_enqueue\")" in app_js
    assert "recordCallerTurnLatency(\"actual_playback_start\")" in app_js


def test_default_model_path_points_to_saved_research_artifact():
    assert MODEL_PATH.is_file()


def test_evaluation_record_proves_code_disjoint_holdout():
    root = Path(__file__).resolve().parents[1]
    report = json.loads((root / "data/triage_models/eden_ricapps_code_chapter_evaluation.json").read_text(encoding="utf-8"))
    assert report["split"]["train_count"] + report["split"]["test_count"] == report["dataset"]["unique_codes"]
    assert report["split"]["code_overlap"] == 0
    assert set(report["dataset"]["class_counts"]) == set(CHAPTERS)
    assert report["metrics"]["model"]["macro_f1"] > report["metrics"]["majority_class_baseline"]["macro_f1"]
