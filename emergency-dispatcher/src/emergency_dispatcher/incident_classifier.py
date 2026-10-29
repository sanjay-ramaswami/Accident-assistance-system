"""Separate advisory classifier for caller descriptions.

The checked-in artifact is explicitly a synthetic-data prototype trained only
on this application's authored demo scenarios. It is not clinically validated
and must never be described as a diagnosis, severity estimate, or dispatch
recommendation.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "data" / "triage_models" / "incident_text_scenario_prototype_v1.joblib"
TAXONOMY_VERSION = "incident-text-synthetic-scenario-v1"
MODEL_ID = "incident-text-synthetic-prototype"
MODEL_VERSION = "incident-text-scenario-prototype-1.0.0"


class IncidentTextModelError(RuntimeError):
    """The local synthetic prototype artifact is missing or invalid."""


class IncidentClassificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classifier_id: str = "incident-text-classifier"
    taxonomy_version: str = TAXONOMY_VERSION
    status: Literal["awaiting", "pending", "insufficient_input", "unavailable", "classified", "abstained", "error"]
    model_id: str | None = None
    model_version: str | None = None
    model_type: Literal["synthetic_data_prototype"] | None = None
    result_source: Literal["trained_model", "rule_based_demo", "none"] = "none"
    rule_based_fallback: dict[str, Any] | None = None
    prototype_only: bool = True
    predicted_category: str | None = None
    class_scores: dict[str, float] = Field(default_factory=dict)
    scores_calibrated: bool = False
    training_coverage: list[str] = Field(default_factory=list)
    training_example_count: int | None = None
    out_of_distribution_status: Literal["unknown", "in_distribution", "out_of_distribution"] = "unknown"
    uncertainty_reason: str | None = None
    input_id: str | None = None
    session_id: str
    incident_id: str
    human_review_required: bool = True
    advisory_only: bool = True


def make_input_id(*, session_id: str, turn_at: str | None, text: str) -> str:
    """Return a stable opaque correlation id without retaining the speech text."""
    payload = "\x1f".join((session_id, turn_at or "", text.strip()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _has_incident_content(text: str) -> bool:
    """Avoid assigning a scenario label to greetings and connection checks."""
    import re

    return bool(re.search(
        r"\b(?:accident|crash|collision|injur\w*|bleed\w*|blood|breath\w*|gasp\w*|chest pain|"
        r"unconscious|unresponsive|collapsed|fall|fell|fire|smoke|burn\w*|poison\w*|seizure|chok\w*|"
        r"overdose|trapped|explosion|not waking|not responding)\b",
        text.casefold(),
    ))


def load_incident_text_model(path: Path | str = MODEL_PATH) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise FileNotFoundError("The synthetic incident-text prototype artifact is unavailable.")
    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        raise IncidentTextModelError("The synthetic incident-text prototype artifact could not be loaded.") from exc
    if not isinstance(artifact, dict) or artifact.get("artifact_format") != 1:
        raise IncidentTextModelError("The synthetic incident-text prototype artifact has an invalid structure.")
    metadata = artifact.get("metadata")
    pipeline = artifact.get("pipeline")
    labels = artifact.get("labels")
    if not isinstance(metadata, dict) or metadata.get("model_id") != MODEL_ID:
        raise IncidentTextModelError("The synthetic incident-text prototype metadata is invalid.")
    if metadata.get("model_version") != MODEL_VERSION or metadata.get("model_type") != "synthetic_data_prototype":
        raise IncidentTextModelError("The synthetic incident-text prototype version is invalid.")
    if metadata.get("clinical_validation") is not False or metadata.get("metrics") is not None:
        raise IncidentTextModelError("The synthetic incident-text prototype metadata is unsafe.")
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) and label for label in labels):
        raise IncidentTextModelError("The synthetic incident-text prototype labels are invalid.")
    if not callable(getattr(pipeline, "predict", None)) or not callable(getattr(pipeline, "predict_proba", None)):
        raise IncidentTextModelError("The synthetic incident-text prototype pipeline is invalid.")
    classifier = getattr(pipeline, "named_steps", {}).get("classifier")
    classes = list(map(str, getattr(classifier, "classes_", ())))
    if sorted(classes) != sorted(labels):
        raise IncidentTextModelError("The synthetic incident-text prototype classes do not match its metadata.")
    if int(metadata.get("training_example_count") or 0) < len(labels):
        raise IncidentTextModelError("The synthetic incident-text prototype example count is invalid.")
    return artifact


def classify_incident_text(
    text: str,
    *,
    session_id: str,
    incident_id: str,
    turn_at: str | None = None,
    model_path: Path | str = MODEL_PATH,
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Classify one finalized caller description with the separate demo prototype."""
    input_id = make_input_id(session_id=session_id, turn_at=turn_at, text=text) if text.strip() else None
    base = {
        "classifier_id": "incident-text-classifier",
        "taxonomy_version": TAXONOMY_VERSION,
        "model_id": None,
        "model_version": None,
        "model_type": None,
        "result_source": "none",
        "rule_based_fallback": None,
        "prototype_only": True,
        "predicted_category": None,
        "class_scores": {},
        "scores_calibrated": False,
        "training_coverage": [],
        "training_example_count": None,
        "out_of_distribution_status": "unknown",
        "input_id": input_id,
        "session_id": session_id,
        "incident_id": incident_id,
        "human_review_required": True,
        "advisory_only": True,
    }
    if not text.strip():
        return IncidentClassificationResult(
            **base,
            status="awaiting",
            uncertainty_reason="awaiting_finalized_caller_description",
        ).model_dump(mode="json")
    try:
        loaded = artifact if artifact is not None else load_incident_text_model(model_path)
        # Validate injected artifacts too; test hooks cannot bypass the runtime checks.
        if not isinstance(loaded, dict) or loaded.get("artifact_format") != 1:
            raise IncidentTextModelError("The synthetic incident-text prototype artifact has an invalid structure.")
        metadata = loaded.get("metadata") or {}
        pipeline = loaded.get("pipeline")
        if metadata.get("model_id") != MODEL_ID or metadata.get("model_version") != MODEL_VERSION:
            raise IncidentTextModelError("The synthetic incident-text prototype version is invalid.")
        if not _has_incident_content(text):
            return IncidentClassificationResult(
                **base,
                status="insufficient_input",
                uncertainty_reason="no_incident_content_detected",
            ).model_dump(mode="json")
        predicted = str(pipeline.predict([text.strip()])[0])
        classes = list(map(str, pipeline.named_steps["classifier"].classes_))
        probabilities = np.asarray(pipeline.predict_proba([text.strip()])[0], dtype=float)
        if predicted not in classes or len(classes) != len(probabilities) or not np.isfinite(probabilities).all():
            raise IncidentTextModelError("The synthetic incident-text prototype returned an invalid result.")
        base.update({
            "model_id": MODEL_ID,
            "model_version": MODEL_VERSION,
            "model_type": "synthetic_data_prototype",
            "result_source": "trained_model",
            "predicted_category": predicted,
            "class_scores": {label: float(probabilities[index]) for index, label in enumerate(classes)},
            "training_coverage": ["authored_synthetic_demo_scenarios"],
            "training_example_count": int(metadata["training_example_count"]),
        })
        return IncidentClassificationResult(
            **base,
            status="classified",
            uncertainty_reason="synthetic_data_prototype_not_clinically_validated",
        ).model_dump(mode="json")
    except FileNotFoundError:
        return IncidentClassificationResult(
            **base,
            status="unavailable",
            uncertainty_reason="prototype_model_artifact_unavailable",
        ).model_dump(mode="json")
    except IncidentTextModelError:
        return IncidentClassificationResult(
            **base,
            status="error",
            uncertainty_reason="prototype_model_artifact_invalid",
        ).model_dump(mode="json")
    except Exception:
        return IncidentClassificationResult(
            **base,
            status="error",
            uncertainty_reason="prototype_inference_failed",
        ).model_dump(mode="json")


async def classify_incident_text_async(
    text: str,
    *,
    session_id: str,
    incident_id: str,
    turn_at: str | None = None,
    model_path: Path | str = MODEL_PATH,
) -> dict[str, Any]:
    """Run CPU inference off the asyncio event loop used by live voice sockets."""
    return await asyncio.to_thread(
        classify_incident_text,
        text,
        session_id=session_id,
        incident_id=incident_id,
        turn_at=turn_at,
        model_path=model_path,
    )
