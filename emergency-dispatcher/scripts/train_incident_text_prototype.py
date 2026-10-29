"""Train the explicitly synthetic, demo-only incident-text prototype.

The only corpus used here is ``data/emergency_scenarios.json``. Those rows are
authored deterministic demo scenarios, not real emergency-call data. This
script intentionally reports no accuracy or clinical metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "emergency_scenarios.json"
DEFAULT_OUTPUT = ROOT / "data" / "triage_models" / "incident_text_scenario_prototype_v1.joblib"
MODEL_VERSION = "incident-text-scenario-prototype-1.0.0"
RANDOM_STATE = 20261030


def load_examples(source: Path) -> tuple[list[str], list[str], str]:
    raw = source.read_bytes()
    payload: Any = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError("Synthetic demo dataset must contain at least two scenario rows.")
    texts: list[str] = []
    labels: list[str] = []
    scenario_ids: set[str] = set()
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError("Synthetic demo dataset rows must be objects.")
        scenario_id = row.get("id")
        text = row.get("transcript")
        label = row.get("expected_issue_type")
        if not all(isinstance(value, str) and value.strip() for value in (scenario_id, text, label)):
            raise ValueError("Each synthetic scenario needs an id, transcript, and expected_issue_type.")
        if scenario_id in scenario_ids:
            raise ValueError("Synthetic demo scenario ids must be unique.")
        scenario_ids.add(scenario_id)
        texts.append(text.strip())
        labels.append(label.strip())
    return texts, labels, hashlib.sha256(raw).hexdigest()


def train_prototype(source: Path = DEFAULT_SOURCE, output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    texts, labels, source_sha256 = load_examples(source)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1, sublinear_tf=True)),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )),
    ])
    pipeline.fit(texts, labels)
    artifact = {
        "artifact_format": 1,
        "pipeline": pipeline,
        "labels": sorted(set(labels)),
        "metadata": {
            "model_id": "incident-text-synthetic-prototype",
            "model_version": MODEL_VERSION,
            "model_type": "synthetic_data_prototype",
            "source_dataset": "data/emergency_scenarios.json",
            "source_dataset_type": "authored_deterministic_demo_scenarios",
            "source_sha256": source_sha256,
            "training_example_count": len(texts),
            "training_label_count": len(set(labels)),
            "random_state": RANDOM_STATE,
            "metrics": None,
            "clinical_validation": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, suffix=".joblib.tmp", delete=False) as stream:
            temp_path = Path(stream.name)
        joblib.dump(artifact, temp_path, compress=3)
        os.replace(temp_path, output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    artifact = train_prototype(args.source, args.output)
    metadata = artifact["metadata"]
    print(json.dumps({
        "artifact": str(args.output.resolve()),
        "model_version": metadata["model_version"],
        "training_example_count": metadata["training_example_count"],
        "training_label_count": metadata["training_label_count"],
        "metrics": None,
        "clinical_validation": False,
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()
