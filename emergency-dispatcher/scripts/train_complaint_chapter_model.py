"""Train and evaluate the reproducible complaint-code chapter classifier."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import joblib
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from emergency_dispatcher.triage_ml import CHAPTERS, MODEL_VERSION


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/triage_dataset/eden_ricapps_zenodo_19606710/complaint_code_chapter.csv"
DEFAULT_ARTIFACT = ROOT / "data/triage_models/eden_ricapps_code_chapter_v1.joblib"
DEFAULT_REPORT = ROOT / "data/triage_models/eden_ricapps_code_chapter_evaluation.json"
SEED = 20261029


def read_data(path: Path) -> tuple[list[str], list[str], dict[str, str]]:
    by_code: dict[str, tuple[str, str]] = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"complaint_code", "chapter_id", "chapter_label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Dataset is missing required code or chapter columns")
        for row in reader:
            code = row["complaint_code"].strip().upper()
            chapter_id = row["chapter_id"].strip()
            label = row["chapter_label"].strip()
            if not code or chapter_id not in CHAPTERS or CHAPTERS[chapter_id] != label:
                raise ValueError("Dataset contains an empty code or unknown chapter class")
            previous = by_code.setdefault(code, (chapter_id, label))
            if previous != (chapter_id, label):
                raise ValueError(f"Conflicting targets for complaint code {code}")
    codes = sorted(by_code)
    labels = [by_code[code][0] for code in codes]
    mapping = {code: by_code[code][1] for code in codes}
    return codes, labels, mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--force", action="store_true", help="Allow replacing existing model artifacts")
    args = parser.parse_args()
    if not args.force and (args.artifact.exists() or args.report.exists()):
        raise SystemExit("Model or evaluation report already exists; pass --force to replace")

    codes, labels, label_mapping = read_data(args.data)
    label_ids = sorted(CHAPTERS, key=int)
    train_codes, test_codes, train_labels, test_labels = train_test_split(
        codes,
        labels,
        test_size=0.20,
        random_state=SEED,
        stratify=labels,
    )
    # Codes are deduplicated before the stratified split: a code cannot occur in
    # both partitions and source row identifiers are never model features.
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(1, 4), lowercase=True)),
            ("classifier", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=SEED)),
        ]
    )
    model.fit(train_codes, train_labels)
    baseline = DummyClassifier(strategy="most_frequent").fit(
        [[code] for code in train_codes], train_labels
    )
    predicted = model.predict(test_codes).tolist()
    baseline_predicted = baseline.predict([[code] for code in test_codes]).tolist()
    report = {
        "dataset": {
            "source": "EDEN-RICAPPS, Zenodo record 19606710, v1.0",
            "license": "CC BY 4.0",
            "task": "diagnostic/presenting-complaint code to clinical-functional chapter",
            "unique_codes": len(codes),
            "class_counts": dict(sorted(Counter(labels).items(), key=lambda x: int(x[0]))),
            "input_features": ["complaint_code"],
            "excluded_fields": ["identificador (source row identifier)", "chapter_label text (duplicate target representation)"],
            "outcomes_used": False,
        },
        "split": {
            "strategy": "stratified 80/20 split over unique complaint codes",
            "random_state": SEED,
            "train_count": len(train_codes),
            "test_count": len(test_codes),
            "test_class_counts": dict(sorted(Counter(test_labels).items(), key=lambda x: int(x[0]))),
            "code_overlap": len(set(train_codes) & set(test_codes)),
        },
        "model": {
            "version": MODEL_VERSION,
            "pipeline": "character TF-IDF 1-4 grams + class-weighted LogisticRegression",
            "scikit_learn_version": sklearn.__version__,
            "label_mapping": CHAPTERS,
        },
        "metrics": {
            "model": {
                "accuracy": accuracy_score(test_labels, predicted),
                "macro_f1": f1_score(test_labels, predicted, labels=label_ids, average="macro", zero_division=0),
                "classification_report": classification_report(
                    test_labels, predicted, labels=label_ids, output_dict=True, zero_division=0
                ),
                "confusion_matrix_labels": label_ids,
                "confusion_matrix": confusion_matrix(test_labels, predicted, labels=label_ids).tolist(),
            },
            "majority_class_baseline": {
                "strategy": "most_frequent",
                "accuracy": accuracy_score(test_labels, baseline_predicted),
                "macro_f1": f1_score(test_labels, baseline_predicted, labels=label_ids, average="macro", zero_division=0),
                "classification_report": classification_report(
                    test_labels, baseline_predicted, labels=label_ids, output_dict=True, zero_division=0
                ),
            },
        },
        "limitations": [
            "This is a code-to-clinical-functional-chapter mapping task, not free-text symptom interpretation or emergency-severity prediction.",
            "The source is a derived methodological classification, not patient-level outcomes or a bedside triage algorithm.",
            "Single framework and local coding practices may limit transfer; every result requires human review.",
            "Model confidence is not calibrated clinical certainty and no dispatch action is automated.",
        ],
    }
    artifact = {
        "pipeline": model,
        "label_mapping": CHAPTERS,
        "known_training_codes": sorted(train_codes),
        "metadata": {
            "version": MODEL_VERSION,
            "scikit_learn_version": sklearn.__version__,
            "trained_on": date.today().isoformat(),
            "dataset_doi": "10.5281/zenodo.19606710",
            "dataset_license": "CC BY 4.0",
            "feature": "complaint_code only",
            "target": "clinical-functional chapter id",
            "test_code_count": len(test_codes),
            "random_state": SEED,
        },
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact, compress=3)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "artifact": str(args.artifact),
        "report": str(args.report),
        "unique_codes": len(codes),
        "train_count": len(train_codes),
        "test_count": len(test_codes),
        "class_counts": report["dataset"]["class_counts"],
        "test_class_counts": report["split"]["test_class_counts"],
        "code_overlap": report["split"]["code_overlap"],
        "metrics": report["metrics"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
