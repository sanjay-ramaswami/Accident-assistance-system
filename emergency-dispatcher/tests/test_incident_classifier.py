import asyncio

from emergency_dispatcher.incident_classifier import classify_incident_text
from emergency_dispatcher.incident_classifier import classify_incident_text_async
from emergency_dispatcher.incident_classifier import load_incident_text_model
from emergency_dispatcher.incident_classifier import make_input_id


def test_free_text_classifier_uses_labeled_synthetic_prototype_separately_from_code_model():
    result = classify_incident_text(
        "Someone is having difficulty breathing.",
        session_id="session-a",
        incident_id="incident-a",
        turn_at="2026-10-29T10:00:00Z",
    )

    assert result["status"] == "classified"
    assert result["predicted_category"]
    assert result["model_id"] == "incident-text-synthetic-prototype"
    assert result["model_type"] == "synthetic_data_prototype"
    assert result["prototype_only"] is True
    assert result["training_example_count"] == 14
    assert result["class_scores"][result["predicted_category"]] > 0
    assert result["scores_calibrated"] is False
    assert result["out_of_distribution_status"] == "unknown"
    assert result["human_review_required"] is True
    assert result["advisory_only"] is True
    assert result["session_id"] == "session-a"
    assert result["incident_id"] == "incident-a"
    assert result["input_id"] == make_input_id(
        session_id="session-a",
        turn_at="2026-10-29T10:00:00Z",
        text="Someone is having difficulty breathing.",
    )
    assert "difficulty breathing" not in str(result)


def test_empty_text_waits_for_a_finalized_description():
    result = classify_incident_text("  ", session_id="s", incident_id="s")
    assert result["status"] == "awaiting"
    assert result["input_id"] is None
    assert result["predicted_category"] is None


def test_greeting_or_connection_check_is_insufficient_input_not_a_synthetic_prediction():
    result = classify_incident_text("Hello, can you hear me?", session_id="s", incident_id="s")
    assert result["status"] == "insufficient_input"
    assert result["predicted_category"] is None
    assert result["class_scores"] == {}
    assert result["uncertainty_reason"] == "no_incident_content_detected"


def test_async_inference_interface_runs_prototype_off_loop():
    result = asyncio.run(
        classify_incident_text_async(
            "A person fell.",
            session_id="s",
            incident_id="s",
        )
    )
    assert result["status"] == "classified"
    assert result["advisory_only"] is True


def test_prototype_prediction_is_reproducible_and_scores_are_uncalibrated():
    args = dict(text="There has been a road accident. One person is unconscious and only gasping.", session_id="s", incident_id="s")
    first = classify_incident_text(**args)
    second = classify_incident_text(**args)
    assert first["predicted_category"] == second["predicted_category"] == "road accident"
    assert first["class_scores"] == second["class_scores"]
    assert first["scores_calibrated"] is False


def test_checked_in_prototype_artifact_loads_with_expected_training_coverage():
    artifact = load_incident_text_model()
    assert artifact["metadata"]["model_id"] == "incident-text-synthetic-prototype"
    assert artifact["metadata"]["training_example_count"] == 14
    assert artifact["metadata"]["training_label_count"] == 12
    assert artifact["metadata"]["metrics"] is None
    assert artifact["metadata"]["clinical_validation"] is False


def test_artifact_resolution_is_independent_of_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert load_incident_text_model()["metadata"]["model_id"] == "incident-text-synthetic-prototype"


def test_missing_and_corrupt_artifacts_are_explicit_and_do_not_echo_caller_text(tmp_path):
    missing = classify_incident_text("Caller secret phrase", session_id="s", incident_id="s", model_path=tmp_path / "missing.joblib")
    assert missing["status"] == "unavailable"
    corrupt_path = tmp_path / "corrupt.joblib"
    corrupt_path.write_bytes(b"not a joblib artifact")
    corrupt = classify_incident_text("Caller secret phrase", session_id="s", incident_id="s", model_path=corrupt_path)
    assert corrupt["status"] == "error"
    assert "Caller secret phrase" not in str(missing) + str(corrupt)
