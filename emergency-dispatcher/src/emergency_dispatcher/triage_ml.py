"""Safe loader and inference boundary for the research complaint-code model."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib


MODEL_VERSION = "eden-ricapps-code-chapter-1.0.0"
MODEL_PATH = Path(__file__).resolve().parents[2] / "data/triage_models/eden_ricapps_code_chapter_v1.joblib"
DATASET_URL = "https://zenodo.org/records/19606710"
CHAPTERS = {
    "1": "Cardiorespiratory",
    "2": "Acute neurological",
    "3": "Sepsis / potential infection",
    "4": "Medical-surgical abdominal",
    "5": "Major trauma",
    "6": "Debility and fatigue",
    "7": "Initial low vital risk",
    "8": "Other complaints",
}
CODE_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,23}$")


class ModelArtifactError(RuntimeError):
    """Raised when the local trained artifact is missing or inconsistent."""


def normalize_complaint_code(code: Any) -> str:
    if not isinstance(code, str):
        raise ValueError("Enter a complaint code as text.")
    normalized = code.strip().upper()
    if not normalized:
        raise ValueError("Enter a complaint code.")
    if not CODE_PATTERN.fullmatch(normalized):
        raise ValueError("Use a short alphanumeric complaint code with optional dots or hyphens.")
    return normalized


def load_complaint_chapter_model(path: Path | str = MODEL_PATH) -> dict[str, Any]:
    artifact_path = Path(path)
    if not artifact_path.is_file():
        raise ModelArtifactError("The local complaint-code model artifact is unavailable.")
    try:
        artifact = joblib.load(artifact_path)
    except Exception as exc:
        raise ModelArtifactError("The local complaint-code model artifact could not be loaded.") from exc

    if not isinstance(artifact, dict) or not isinstance(artifact.get("label_mapping"), dict):
        raise ModelArtifactError("The local model artifact has an invalid structure.")
    if artifact["label_mapping"] != CHAPTERS:
        raise ModelArtifactError("The local model artifact contains unknown chapter classes.")
    pipeline = artifact.get("pipeline")
    known_codes = artifact.get("known_training_codes")
    metadata = artifact.get("metadata")
    if not callable(getattr(pipeline, "predict", None)) or not callable(getattr(pipeline, "predict_proba", None)):
        raise ModelArtifactError("The local model artifact does not contain a probability-capable pipeline.")
    classes = getattr(getattr(pipeline, "named_steps", {}).get("classifier"), "classes_", ())
    if set(map(str, classes)) != set(CHAPTERS):
        raise ModelArtifactError("The fitted model contains unknown or missing chapter classes.")
    if not isinstance(known_codes, list) or not all(isinstance(item, str) for item in known_codes):
        raise ModelArtifactError("The model training-code index is invalid.")
    if not isinstance(metadata, dict) or metadata.get("version") != MODEL_VERSION:
        raise ModelArtifactError("The model version metadata is invalid.")
    return artifact


def predict_complaint_chapter(code: Any, *, artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_complaint_code(code)
    loaded = artifact if artifact is not None else load_complaint_chapter_model()
    # Call the same checks for injected artifacts, which keeps tests and API callers
    # from bypassing the class/metadata validation boundary.
    if not isinstance(loaded, dict) or loaded.get("label_mapping") != CHAPTERS:
        raise ModelArtifactError("The local model artifact contains unknown chapter classes.")
    pipeline = loaded.get("pipeline")
    if not callable(getattr(pipeline, "predict_proba", None)):
        raise ModelArtifactError("The local model artifact does not contain a probability-capable pipeline.")
    chapter_id = str(pipeline.predict([normalized])[0])
    if chapter_id not in CHAPTERS:
        raise ModelArtifactError("The model produced an unknown chapter class.")
    probabilities = pipeline.predict_proba([normalized])[0]
    classes = list(map(str, pipeline.named_steps["classifier"].classes_))
    score = float(probabilities[classes.index(chapter_id)])
    known = normalized in set(loaded.get("known_training_codes", []))
    return {
        "task": "complaint_code_to_clinical_functional_chapter",
        "input_code": normalized,
        "prediction": {"chapter_id": chapter_id, "chapter": CHAPTERS[chapter_id]},
        "uncalibrated_model_score": score,
        "training_coverage": "seen_in_training" if known else "unseen_code",
        "model": {"version": MODEL_VERSION, "source": "scikit-learn character n-gram classifier"},
        "dataset": {"doi": "10.5281/zenodo.19606710", "license": "CC BY 4.0", "url": DATASET_URL},
        "advisory_only": True,
        "human_review_required": True,
        "dispatch_authorized": False,
        "notice": (
            "Research/demo code-to-chapter categorization only. This is not a free-text symptom classifier, "
            "Manchester acuity level, diagnosis, clinical triage decision, or dispatch instruction. "
            "The model score is uncalibrated; always require human review."
        ),
    }
